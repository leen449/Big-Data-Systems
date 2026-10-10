package sonicspark.rdd

import org.apache.hadoop.fs.{FileSystem, Path => HadoopPath}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import sonicspark.common._

/**
 * Phase 3 - RDD Operations (SonicSpark, IT 462 Big Data Systems)
 *
 * Question this file answers: do the 38 audio features extracted in Phase 2
 * separate GTZAN's 10 genres, and which genres are acoustically closest (and
 * therefore most likely to be confused by a classifier)? Everything below is
 * RDD-based; Spark SQL/DataFrames are used only to load the Parquet file,
 * per the assignment's rules.
 *
 * Input : data/processed/final          (Phase 2's final window-level table, 58,020 rows)
 *         data/raw/features_3_sec.csv   (3-second reference table; Analysis D only)
 * Output: results/rdd_output.txt        (full printed report, analysis by analysis)
 *         outputs/rdd/analysisX_*.csv   (one small result table per analysis)
 *         outputs/rdd/tempo_flags/      (Analysis D: flagged tempo-artefact segments)
 *
 * Run:
 *   sbt "runMain sonicspark.rdd.RDDOperations"
 */
object RDDOperations {

  // ======================================================================
  // RddConfig: every tunable constant used below, in one place
  // ======================================================================
  object RddConfig {
    // Analysis A: feature columns the genre-profile table and ranking focus on
    val KeyFeatures: Seq[String] = Seq(
      "spectral_bandwidth_mean", "mfcc1_mean", "rolloff_mean",
      "rms_mean", "zcr_mean", "loudness_variation"
    )

    // Shared "top-N" size for top/bottom/closest-pair style outputs (B and C)
    val TopBottomN = 10

    // Analysis D: must match features_3_sec.csv's own 3-second segments
    val SegmentLengthSeconds = 3.0
    // Heuristic thresholds for tempo-doubling/halving artefacts (report section 2.3).
    // Not derived from the data -- just a reasonable rule of thumb for flagging outliers.
    val TempoDoubleFactor = 1.8
    val TempoHalfFactor   = 0.55
  }

  // ======================================================================
  // Case classes (used instead of nested tuples, per the course style rules)
  // ======================================================================

  /** One window, carrying its 38 features as a plain array aligned to `featureNames`. */
  case class AudioWindow(trackId: String, genre: String, windowIdx: Int, startSec: Double, features: Array[Double])

  /** Running total used to compute mean/stdDev for one (genre, feature) pair (Analysis A). */
  case class FeatureAccumulator(count: Long, sum: Double, sumSq: Double)

  case class AnalysisAResult(
    genreFeatureStats: Map[(String, String), (Double, Double)],      // (genre, feature) -> (mean, stdDev)
    rankedKeyFeatureRows: Array[((String, String), (Double, Double))], // KeyFeatures only, ranked within each feature
    statsRowCount: Long
  )

  /** Accumulator for one track's loudness while folding its windows together (Analysis B). */
  case class TrackDynamicsAcc(genre: String, minDb: Double, maxDb: Double, sumDb: Double, count: Long, sumVariation: Double)

  /** Finished per-track loudness summary (Analysis B). */
  case class TrackDynamics(genre: String, minDb: Double, maxDb: Double, meanDb: Double, meanLoudnessVariation: Double) {
    def dynamicRange: Double = maxDb - minDb
  }

  case class AnalysisBResult(
    genreDynamicStats: Map[String, (Double, Long)],        // genre -> (sum of dynamic range, track count)
    overallMean: Double,
    overallStd: Double,
    mostDynamic: Array[(String, TrackDynamics)],
    leastDynamic: Array[(String, TrackDynamics)],
    exceptionalCountByGenre: Map[String, Long]
  )

  case class GenrePairDistance(genreA: String, genreB: String, distance: Double)

  case class AnalysisCResult(
    closestPairs: Array[GenrePairDistance],
    closestPair: GenrePairDistance,
    mostDistantPairs: Array[GenrePairDistance]
  )

  /** One row of features_3_sec.csv, parsed down to just what Analysis D needs. */
  case class Segment3Sec(trackId: String, segmentIdx: Int, tempo: Double)

  /** One 3-second segment after joining a window's genre onto its matching CSV row. */
  case class JoinedSegment(trackId: String, segmentIdx: Int, genre: String, tempo: Double)

  case class AnalysisDResult(
    joinedSegmentCount: Long,
    genreTempoStats: Map[String, (Double, Long)],   // genre -> (sum tempo, segment count)
    flaggedCountByGenre: Map[String, Long],
    basePartitions: Int,
    joinedPartitions: Int
  )

  // ======================================================================
  // Base RDD
  // ======================================================================

  /**
   * Loads data/processed/final (DataFrame API, load only) and converts every
   * Row into an AudioWindow. Feature names are read from the schema at
   * runtime (never hardcoded), so this stays correct if Phase 2 changes.
   */
  def buildBaseRdd(spark: SparkSession): (RDD[AudioWindow], Array[String]) = {
    val finalDf = spark.read.parquet(Paths.Final)

    val idColumns    = Set("track_id", "label", "window_idx", "start_sec", "label_idx")
    val featureNames = finalDf.columns.filterNot(idColumns.contains)

    // TRANSFORMATION (map): DataFrame Row -> AudioWindow case class
    val baseRdd = finalDf.rdd.map { row =>
      AudioWindow(
        trackId   = row.getAs[String]("track_id"),
        genre     = row.getAs[String]("label"),
        windowIdx = row.getAs[Int]("window_idx"),
        startSec  = row.getAs[Double]("start_sec"),
        features  = featureNames.map(name => row.getAs[Double](name))
      )
    }

    (baseRdd, featureNames)
  }

  // ======================================================================
  // Analysis A: Genre acoustic profiles
  //   Question : how does each genre differ on each feature?
  //   Input    : the base RDD (AudioWindow)
  //   RDD ops  : flatMap, aggregateByKey, mapValues, filter, sortBy
  //   Output   : (genre, feature) -> (mean, stdDev) for every pair, plus a
  //              ranking of genres on a few key features
  // ======================================================================
  def computeA(baseRdd: RDD[AudioWindow], featureNames: Array[String]): AnalysisAResult = {

    // TRANSFORMATION (flatMap): one window -> one ((genre, featureName), value) record per feature
    val genreFeatureValues: RDD[((String, String), Double)] = baseRdd.flatMap { window =>
      featureNames.indices.map(i => ((window.genre, featureNames(i)), window.features(i)))
    }

    // TRANSFORMATION (aggregateByKey): fold values into (count, sum, sumOfSquares) per (genre, feature)
    val zero = FeatureAccumulator(0L, 0.0, 0.0)
    val accumulated: RDD[((String, String), FeatureAccumulator)] = genreFeatureValues.aggregateByKey(zero)(
      seqOp  = (acc, value) => FeatureAccumulator(acc.count + 1, acc.sum + value, acc.sumSq + value * value),
      combOp = (a, b)       => FeatureAccumulator(a.count + b.count, a.sum + b.sum, a.sumSq + b.sumSq)
    )

    // TRANSFORMATION (mapValues): (count, sum, sumSq) -> (mean, stdDev)
    val stats: RDD[((String, String), (Double, Double))] = accumulated.mapValues { acc =>
      val mean     = acc.sum / acc.count
      val variance = acc.sumSq / acc.count - mean * mean
      (mean, math.sqrt(math.max(variance, 0.0)))
    }

    val statsRowCount = stats.count()  // ACTION (count): expect numGenres x numFeatures (10 x 38 = 380)

    // TRANSFORMATION (filter): keep only the handful of features used for the ranking table
    val keyFeatureStats = stats.filter { case ((_, featureName), _) => RddConfig.KeyFeatures.contains(featureName) }

    // TRANSFORMATION (sortBy): rank genres within each key feature, highest mean first
    val rankedKeyFeatureRows = keyFeatureStats
      .sortBy({ case ((_, featureName), (mean, _)) => (featureName, -mean) })
      .collect()  // ACTION (collect): tiny (6 features x 10 genres = 60 rows)

    // ACTION (collectAsMap): full (genre, feature) -> (mean, stdDev) table, reused by Analysis C
    val genreFeatureStats = stats.collectAsMap().toMap

    AnalysisAResult(genreFeatureStats, rankedKeyFeatureRows, statsRowCount)
  }

  def reportA(result: AnalysisAResult, featureNames: Array[String], genres: Seq[String]): Unit = {
    val expectedRows = genres.size * featureNames.length
    Report.log(s"(genre, feature) stat pairs computed: ${result.statsRowCount} (expected $expectedRows)")

    // ---------- Genre x key-feature means table ----------
    val header = "genre" +: RddConfig.KeyFeatures
    val rows = genres.sorted.map { genre =>
      genre +: RddConfig.KeyFeatures.map(f => f"${result.genreFeatureStats((genre, f))._1}%.4f")
    }
    Report.log("\nGenre acoustic profiles (mean of each key feature):")
    Report.table(header, rows)
    Report.saveCsv("analysisA_genre_profiles.csv", header, rows)

    // ---------- Per-feature genre ranking, from the RDD's own sortBy ----------
    Report.log("\nGenre ranking per key feature (highest mean first):")
    RddConfig.KeyFeatures.foreach { feature =>
      val rankedGenres = result.rankedKeyFeatureRows.collect {
        case ((genre, f), (mean, _)) if f == feature => f"$genre (${mean}%.4f)"
      }
      Report.log(s"  $feature: ${rankedGenres.mkString(" > ")}")
    }
  }

  // ======================================================================
  // Analysis B: Dynamics within recordings
  //   Question : does classical/jazz have wider dynamics than metal/pop? (report 3.4.1)
  //   Input    : the base RDD (AudioWindow)
  //   RDD ops  : map, reduceByKey, mapValues, filter, reduce
  //   Output   : per-genre average dynamic range, most/least dynamic recordings,
  //              and recordings whose dynamic range is an outlier
  // ======================================================================

  // rms_mean -> decibels. Epsilon (shared with audio extraction) keeps a silent
  // window's log10 finite instead of -Infinity.
  private def toDecibels(rms: Double): Double = 20.0 * math.log10(rms + AudioConfig.Epsilon)

  private def mergeTrackDynamicsAcc(a: TrackDynamicsAcc, b: TrackDynamicsAcc): TrackDynamicsAcc =
    TrackDynamicsAcc(a.genre, math.min(a.minDb, b.minDb), math.max(a.maxDb, b.maxDb),
      a.sumDb + b.sumDb, a.count + b.count, a.sumVariation + b.sumVariation)

  /** One single-window accumulator per track, ready to be folded by reduceByKey or groupByKey. */
  private def windowLoudnessAccs(baseRdd: RDD[AudioWindow], featureNames: Array[String]): RDD[(String, TrackDynamicsAcc)] = {
    val rmsMeanIdx           = featureNames.indexOf("rms_mean")
    val loudnessVariationIdx = featureNames.indexOf("loudness_variation")
    require(rmsMeanIdx >= 0 && loudnessVariationIdx >= 0, "rms_mean / loudness_variation not found in featureNames")

    // TRANSFORMATION (map): one window -> (trackId, single-window loudness accumulator)
    baseRdd.map { window =>
      val db = toDecibels(window.features(rmsMeanIdx))
      (window.trackId, TrackDynamicsAcc(window.genre, db, db, db, 1L, window.features(loudnessVariationIdx)))
    }
  }

  def computeB(baseRdd: RDD[AudioWindow], featureNames: Array[String]): AnalysisBResult = {
    val windowAccs = windowLoudnessAccs(baseRdd, featureNames)

    // TRANSFORMATION (reduceByKey): fold every window of a track into one accumulator
    val perTrackAcc = windowAccs.reduceByKey(mergeTrackDynamicsAcc)

    // TRANSFORMATION (mapValues): accumulator -> finished per-track summary
    val trackDynamicsRdd: RDD[(String, TrackDynamics)] = perTrackAcc.mapValues { acc =>
      TrackDynamics(acc.genre, acc.minDb, acc.maxDb, acc.sumDb / acc.count, acc.sumVariation / acc.count)
    }.cache()

    // TRANSFORMATION (map) + (reduceByKey): re-key per-track summaries by genre
    val genreDynamicStats = trackDynamicsRdd
      .map { case (_, td) => (td.genre, (td.dynamicRange, 1L)) }
      .reduceByKey { case ((sum1, count1), (sum2, count2)) => (sum1 + sum2, count1 + count2) }
      .collectAsMap().toMap  // ACTION (collectAsMap): one row per genre

    // ACTION (reduce): overall mean & std of dynamic range, in a single pass over all tracks
    val (sumRange, sumRangeSq, trackCount) = trackDynamicsRdd
      .map { case (_, td) => (td.dynamicRange, td.dynamicRange * td.dynamicRange, 1L) }
      .reduce { case ((s1, sq1, c1), (s2, sq2, c2)) => (s1 + s2, sq1 + sq2, c1 + c2) }
    val overallMean = sumRange / trackCount
    val overallStd  = math.sqrt(math.max(sumRangeSq / trackCount - overallMean * overallMean, 0.0))

    // TRANSFORMATION (filter): tracks whose dynamic range is an outlier (> mean + 1 std)
    val exceptional = trackDynamicsRdd.filter { case (_, td) => td.dynamicRange > overallMean + overallStd }

    // ACTION (top) / (takeOrdered): the 10 most and 10 least dynamic recordings
    val mostDynamic  = trackDynamicsRdd.top(RddConfig.TopBottomN)(Ordering.by { case (_, td) => td.dynamicRange })
    val leastDynamic = trackDynamicsRdd.takeOrdered(RddConfig.TopBottomN)(Ordering.by { case (_, td) => td.dynamicRange })

    // ACTION (countByKey): exceptional tracks per genre
    val exceptionalCountByGenre = exceptional.map { case (_, td) => (td.genre, td) }.countByKey().toMap

    trackDynamicsRdd.unpersist()

    AnalysisBResult(genreDynamicStats, overallMean, overallStd, mostDynamic, leastDynamic, exceptionalCountByGenre)
  }

  def reportB(result: AnalysisBResult, genreFeatureStats: Map[(String, String), (Double, Double)]): Unit = {
    val avgByGenre: Map[String, Double] = result.genreDynamicStats.map { case (genre, (sum, count)) => genre -> (sum / count) }

    Report.log(f"Overall dynamic range: mean = ${result.overallMean}%.2f dB, std = ${result.overallStd}%.2f dB")

    val header = Seq("genre", "avg_dynamic_range_db", "num_tracks", "num_exceptional")
    val sortedGenres = result.genreDynamicStats.toSeq.sortBy { case (_, (sum, count)) => -(sum / count) }
    val rows = sortedGenres.map { case (genre, (_, count)) =>
      Seq(genre, f"${avgByGenre(genre)}%.2f", count.toString, result.exceptionalCountByGenre.getOrElse(genre, 0L).toString)
    }
    Report.log("\nAverage dynamic range per genre (descending):")
    Report.table(header, rows)
    Report.saveCsv("analysisB_dynamics.csv", header, rows)

    // ---------- Hypothesis check (report 3.4.1): classical/jazz wider than metal/pop? ----------
    val wideGenres       = Seq("classical", "jazz")
    val compressedGenres = Seq("metal", "pop")
    Report.log("\nHypothesis check (report 3.4.1): classical & jazz wider dynamics, metal & pop more compressed")
    val loudnessVarByGenre = (wideGenres ++ compressedGenres)
      .flatMap(g => genreFeatureStats.get((g, "loudness_variation")).map { case (mean, _) => g -> mean })
      .toMap
    (wideGenres ++ compressedGenres).foreach { g =>
      avgByGenre.get(g).foreach { avg =>
        val variation = loudnessVarByGenre.get(g).map(v => f"$v%.4f").getOrElse("n/a")
        Report.log(f"  $g%-10s avg dynamic range = $avg%.2f dB, mean loudness_variation = $variation")
      }
    }
    val wideAvgs       = wideGenres.flatMap(avgByGenre.get)
    val compressedAvgs = compressedGenres.flatMap(avgByGenre.get)
    val holdsRecordingLevel = wideAvgs.nonEmpty && compressedAvgs.nonEmpty && wideAvgs.min > compressedAvgs.max
    Report.log(s"At recording level (dynamic range): holds = $holdsRecordingLevel")

    val wideLoudnessVars       = wideGenres.flatMap(loudnessVarByGenre.get)
    val compressedLoudnessVars = compressedGenres.flatMap(loudnessVarByGenre.get)
    val holdsWindowLevel = wideLoudnessVars.nonEmpty && compressedLoudnessVars.nonEmpty &&
      wideLoudnessVars.min > compressedLoudnessVars.max
    Report.log(s"At window level (mean loudness_variation from Analysis A): holds = $holdsWindowLevel")

    // ---------- Most / least dynamic recordings ----------
    val rangeHeader = Seq("track_id", "genre", "dynamic_range_db", "min_db", "max_db")
    Report.log("\n10 most dynamic recordings:")
    val mostRows = result.mostDynamic.map { case (trackId, td) =>
      Seq(trackId, td.genre, f"${td.dynamicRange}%.2f", f"${td.minDb}%.2f", f"${td.maxDb}%.2f")
    }
    Report.table(rangeHeader, mostRows)

    Report.log("\n10 least dynamic recordings:")
    val leastRows = result.leastDynamic.map { case (trackId, td) =>
      Seq(trackId, td.genre, f"${td.dynamicRange}%.2f", f"${td.minDb}%.2f", f"${td.maxDb}%.2f")
    }
    Report.table(rangeHeader, leastRows)
  }

  // ======================================================================
  // Analysis C: Genre similarity (preview of likely confusions)
  //   Question : which genre pairs are acoustically closest, and thus likely
  //              to be confused by a classifier?
  //   Input    : the base RDD (for global mean/std) + Analysis A's genre means
  //   RDD ops  : reduce, cartesian, filter, map, sortBy
  //   Output   : the 10 closest genre pairs, the single closest pair, and the
  //              5 most distant pairs
  //
  // NOTE: this is exploratory only. It does not select features, drop any
  // columns, or tune anything -- so it cannot leak into the Phase 5 train/test
  // evaluation. It simply previews which genres look acoustically similar.
  // ======================================================================

  private def elementWiseAdd(a: Array[Double], b: Array[Double]): Array[Double] = Array.tabulate(a.length)(i => a(i) + b(i))

  private def euclideanDistance(a: Array[Double], b: Array[Double]): Double =
    math.sqrt(a.indices.map(i => math.pow(a(i) - b(i), 2)).sum)

  def computeC(
    sc: SparkContext,
    baseRdd: RDD[AudioWindow],
    featureNames: Array[String],
    genreFeatureStats: Map[(String, String), (Double, Double)],
    genres: Seq[String]
  ): AnalysisCResult = {
    val numFeatures = featureNames.length

    // ACTION (reduce): global sum, sum-of-squares, and count of every feature, in one pass
    val (sumArr, sumSqArr, totalCount) = baseRdd
      .map(w => (w.features, w.features.map(x => x * x), 1L))
      .reduce { case ((s1, sq1, c1), (s2, sq2, c2)) => (elementWiseAdd(s1, s2), elementWiseAdd(sq1, sq2), c1 + c2) }

    val globalMean = sumArr.map(_ / totalCount)
    val globalStd  = Array.tabulate(numFeatures) { i =>
      math.sqrt(math.max(sumSqArr(i) / totalCount - globalMean(i) * globalMean(i), 0.0))
    }

    // Driver-side: build each genre's z-scored centroid vector from Analysis A's means
    val genreCentroids: Seq[(String, Array[Double])] = genres.map { genre =>
      val centroid = Array.tabulate(numFeatures) { i =>
        val genreMean = genreFeatureStats((genre, featureNames(i)))._1
        (genreMean - globalMean(i)) / globalStd(i)
      }
      (genre, centroid)
    }

    // RDD CREATION (parallelize): not a transformation; turns the 10 driver-side
    // centroids into an RDD so cartesian can pair them
    val centroidsRdd = sc.parallelize(genreCentroids)

    // TRANSFORMATION (cartesian): every ordered pair of genres, including self-pairs
    val allPairs = centroidsRdd.cartesian(centroidsRdd)

    // TRANSFORMATION (filter): keep each unordered pair exactly once (genreA < genreB)
    // TRANSFORMATION (map): Euclidean distance between the two z-scored centroids
    val pairDistances = allPairs
      .filter { case ((genreA, _), (genreB, _)) => genreA < genreB }
      .map { case ((genreA, vecA), (genreB, vecB)) => GenrePairDistance(genreA, genreB, euclideanDistance(vecA, vecB)) }

    // TRANSFORMATION (sortBy): ascending by distance -> closest pairs first
    val sortedAscending = pairDistances.sortBy(_.distance)

    val closestPairs = sortedAscending.take(RddConfig.TopBottomN)  // ACTION (take)
    val closestPair  = sortedAscending.first()                      // ACTION (first)

    // TRANSFORMATION (sortBy): descending by distance -> most distant pairs first
    val mostDistantPairs = sortedAscending.sortBy(p => -p.distance).take(5)  // ACTION (take)

    AnalysisCResult(closestPairs, closestPair, mostDistantPairs)
  }

  def reportC(result: AnalysisCResult): Unit = {
    val header = Seq("genre_a", "genre_b", "distance")

    Report.log("10 closest genre pairs (most likely to be confused):")
    val closestRows = result.closestPairs.map(p => Seq(p.genreA, p.genreB, f"${p.distance}%.4f"))
    Report.table(header, closestRows)

    Report.log(f"\nSingle closest pair: ${result.closestPair.genreA} <-> ${result.closestPair.genreB} " +
      f"(distance = ${result.closestPair.distance}%.4f)")

    Report.log("\n5 most distant genre pairs (least likely to be confused):")
    val distantRows = result.mostDistantPairs.map(p => Seq(p.genreA, p.genreB, f"${p.distance}%.4f"))
    Report.table(header, distantRows)

    val csvHeader = Seq("genre_a", "genre_b", "distance", "group")
    val csvRows = closestRows.map(_ :+ "closest") ++ distantRows.map(_ :+ "most_distant")
    Report.saveCsv("analysisC_genre_similarity.csv", csvHeader, csvRows)
  }

  // ======================================================================
  // Analysis D: Tempo via a join with the 3-second table
  //   Question : what is the tempo profile per genre, and how common are
  //              tempo-doubling/halving artefacts (report 2.3)?
  //   Input    : the base RDD + data/raw/features_3_sec.csv
  //   RDD ops  : filter, map, keyBy, join, distinct, reduceByKey
  //   Output   : mean tempo per genre, flagged segments per genre, and the
  //              flagged segments themselves (saved to outputs/rdd/tempo_flags)
  // ======================================================================

  private def deleteIfExists(sc: SparkContext, path: String): Unit = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val p  = new HadoopPath(path)
    if (fs.exists(p)) fs.delete(p, true)
  }

  def computeD(sc: SparkContext, baseRdd: RDD[AudioWindow]): AnalysisDResult = {
    val rawLines = sc.textFile(Paths.Raw3Sec)  // one record per CSV line, header included

    val headerLine  = rawLines.first()  // ACTION (first): read the header row to locate columns by name
    val tempoColIdx = headerLine.split(",").indexOf("tempo")
    require(tempoColIdx >= 0, "'tempo' column not found in features_3_sec.csv header")

    // Matches "blues.00000.3.wav" -> ("blues.00000", "3"); the CSV's own filename convention.
    val filenamePattern = "^([a-z]+\\.\\d{5})\\.(\\d+)\\.wav$".r

    val segments: RDD[((String, Int), Segment3Sec)] = rawLines
      .filter(_ != headerLine)  // TRANSFORMATION (filter): drop the header row
      .map { line =>            // TRANSFORMATION (map): parse one CSV line into a Segment3Sec
        val cols = line.split(",")
        val filenamePattern(trackId, segmentIdxStr) = cols(0)
        Segment3Sec(trackId, segmentIdxStr.toInt, cols(tempoColIdx).toDouble)
      }
      .keyBy(s => (s.trackId, s.segmentIdx))  // TRANSFORMATION (keyBy)

    // TRANSFORMATION (keyBy): each window's 3-second segment = floor(start_sec / 3.0)
    val windowsKeyed = baseRdd.keyBy(w => (w.trackId, math.floor(w.startSec / RddConfig.SegmentLengthSeconds).toInt))

    // TRANSFORMATION (join): windows <-> 3-second segments on (trackId, segmentIdx).
    // jazz.00054 exists in features_3_sec.csv (librosa decoded it fine) but has no rows on the
    // window side: our own RIFF parser in AudioFeatures.scala rejected it, so data/processed/final
    // never had it to begin with. The inner join drops it automatically -- same for every window
    // Cleaning removed (silence, the mislabelled pair, or duplicates): they simply aren't present
    // to join against, with no special-casing needed here.
    val joined = windowsKeyed.join(segments)
      .map { case ((trackId, segmentIdx), (window, segment)) => JoinedSegment(trackId, segmentIdx, window.genre, segment.tempo) }
      // TRANSFORMATION (distinct): a 3-second segment spans 6 half-second-hop windows, all of
      // which join to the same segment row -- collapse back down to one record per segment.
      .distinct()
      .cache()

    val joinedSegmentCount = joined.count()  // ACTION (count): sanity-check the joined segment total

    // TRANSFORMATION (map) + (reduceByKey): mean tempo per genre
    val genreTempoStats = joined.map(s => (s.genre, (s.tempo, 1L)))
      .reduceByKey { case ((sum1, count1), (sum2, count2)) => (sum1 + sum2, count1 + count2) }
      .collectAsMap().toMap  // ACTION (collectAsMap)

    // NOTE: built with .map (not .mapValues), which always returns a fully materialized,
    // plain immutable Map -- .mapValues + .toMap can silently stay a lazy view in Scala 2.12,
    // which then fails to serialize once this map is captured by the filter closure below.
    val genreMeanTempo: Map[String, Double] = genreTempoStats.map { case (genre, (sum, count)) => genre -> (sum / count) }

    // TRANSFORMATION (filter): likely tempo-doubling/halving artefacts (heuristic, see RddConfig)
    val flagged = joined.filter { s =>
      val mean = genreMeanTempo(s.genre)
      s.tempo >= RddConfig.TempoDoubleFactor * mean || s.tempo <= RddConfig.TempoHalfFactor * mean
    }

    val flaggedCountByGenre = flagged.map(s => (s.genre, s)).countByKey().toMap  // ACTION (countByKey)

    // ACTION (saveAsTextFile): one flagged segment per line
    deleteIfExists(sc, Paths.TempoFlagsOutput)
    flagged
      .map(s => s"${s.trackId},${s.segmentIdx},${s.genre},${f"${s.tempo}%.2f"}")
      .coalesce(1)  // TRANSFORMATION (coalesce, extra): a single output file
      .saveAsTextFile(Paths.TempoFlagsOutput)

    val result = AnalysisDResult(joinedSegmentCount, genreTempoStats, flaggedCountByGenre,
      baseRdd.getNumPartitions, joined.getNumPartitions)

    joined.unpersist()
    result
  }

  def reportD(result: AnalysisDResult): Unit = {
    Report.log(s"Joined segments (windows x features_3_sec.csv): ${result.joinedSegmentCount}")

    val header = Seq("genre", "mean_tempo_bpm", "segments", "flagged", "pct_flagged")
    val rows = result.genreTempoStats.toSeq.sortBy { case (_, (sumTempo, count)) => -(sumTempo / count) }.map { case (genre, (sumTempo, count)) =>
      val flagged = result.flaggedCountByGenre.getOrElse(genre, 0L)
      Seq(genre, f"${sumTempo / count}%.2f", count.toString, flagged.toString, f"${100.0 * flagged / count}%.1f%%")
    }
    Report.log("\nTempo profile per genre, with candidate tempo-doubling/halving artefacts " +
      "(segments far from their genre's mean tempo) flagged:")
    Report.table(header, rows)
    Report.saveCsv("analysisD_tempo.csv", header, rows)

    Report.log(s"\nFlagged segments saved to: ${Paths.TempoFlagsOutput}")
  }

  // ======================================================================
  // Reflection measurements
  // ======================================================================

  private def time[T](block: => T): (T, Double) = {
    val start  = System.nanoTime()
    val result = block
    (result, (System.nanoTime() - start) / 1e9)
  }

  private def accsApproxEqual(a: TrackDynamicsAcc, b: TrackDynamicsAcc, epsilon: Double = 1e-6): Boolean =
    a.genre == b.genre && a.count == b.count &&
    math.abs(a.minDb - b.minDb) < epsilon &&
    math.abs(a.maxDb - b.maxDb) < epsilon &&
    math.abs(a.sumDb - b.sumDb) < epsilon * math.max(1.0, math.abs(a.sumDb)) &&
    math.abs(a.sumVariation - b.sumVariation) < epsilon * math.max(1.0, math.abs(a.sumVariation))

  /**
   * 1. Times Analyses A+B with the base RDD cached vs. not cached.
   * 2. Compares reduceByKey vs. groupByKey for Analysis B's per-track step
   *    (timing + shuffle bytes written), after confirming they agree.
   * 3. Prints the partition count of the base RDD and of Analysis D's joined RDD.
   *
   * Calls computeA/computeB/windowLoudnessAccs directly -- no re-implemented chains.
   */
  def runReflection(spark: SparkSession, featureNames: Array[String], dResult: AnalysisDResult): Unit = {
    val sc = spark.sparkContext
    Report.section("Reflection: Performance Measurements")

    // ---------- 1. Cached vs. uncached timing for Analyses A + B ----------
    // Each "fresh" RDD is rebuilt from Parquet so "uncached" genuinely re-reads
    // from disk instead of reusing a partition left over from an earlier run.
    def freshBaseRdd(): RDD[AudioWindow] = buildBaseRdd(spark)._1
    def runCoreAAndB(rdd: RDD[AudioWindow]): Unit = { computeA(rdd, featureNames); computeB(rdd, featureNames) }

    runCoreAAndB(freshBaseRdd())  // warm-up: absorbs JIT/codegen/filesystem overhead, excluded from the timings below

    val (_, uncachedSeconds) = time { runCoreAAndB(freshBaseRdd()) }

    val cachedRdd = freshBaseRdd().cache()
    val (_, materializeSeconds) = time { cachedRdd.count() }  // ACTION (count): forces the cache to fill
    val (_, cachedSeconds)      = time { runCoreAAndB(cachedRdd) }
    cachedRdd.unpersist()

    Report.log(f"Analyses A+B, base RDD NOT cached : $uncachedSeconds%.2f s")
    Report.log(f"Cache materialization (count)      : $materializeSeconds%.2f s")
    Report.log(f"Analyses A+B, base RDD cached       : $cachedSeconds%.2f s")
    Report.log("Note: local-mode timings on a single laptop, with JVM/OS noise -- indicative only, not a benchmark.")

    // ---------- 2. reduceByKey vs. groupByKey for Analysis B's per-track step ----------
    val windowAccs = windowLoudnessAccs(freshBaseRdd(), featureNames).cache()
    windowAccs.count()  // materialize once so both timings below measure only the aggregation step

    val (reduceResult, reduceSeconds, reduceShuffleBytes) = ShuffleMetrics.timeAndShuffle(sc) {
      windowAccs.reduceByKey(mergeTrackDynamicsAcc).collectAsMap()
    }
    val (groupResult, groupSeconds, groupShuffleBytes) = ShuffleMetrics.timeAndShuffle(sc) {
      windowAccs.groupByKey().mapValues(_.reduce(mergeTrackDynamicsAcc)).collectAsMap()
    }
    windowAccs.unpersist()

    val identical = reduceResult.keySet == groupResult.keySet &&
      reduceResult.forall { case (trackId, acc) => accsApproxEqual(acc, groupResult(trackId)) }
    require(identical, "reduceByKey and groupByKey produced different per-track results -- aborting the comparison")

    Report.log(s"\nreduceByKey vs groupByKey agree on every track: $identical")
    Report.log(f"reduceByKey : $reduceSeconds%.2f s, shuffle write = $reduceShuffleBytes bytes")
    Report.log(f"groupByKey  : $groupSeconds%.2f s, shuffle write = $groupShuffleBytes bytes")
    Report.log("Note: local-mode timings/shuffle bytes -- indicative only, not a benchmark.")

    // ---------- 3. Partition counts ----------
    Report.log(s"\nBase RDD partitions (data/processed/final): ${dResult.basePartitions}")
    Report.log(s"Analysis D joined RDD partitions (windows join segments): ${dResult.joinedPartitions}")
  }

  // ======================================================================
  // main
  // ======================================================================
  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-RDDOperations")
    val sc    = spark.sparkContext

    Report.start()
    Report.section("SonicSpark Phase 3: RDD Operations")

    val (uncachedBase, featureNames) = buildBaseRdd(spark)
    val baseRdd = uncachedBase.cache()  // every analysis below reuses this RDD
    Report.log(s"Base RDD partitions: ${baseRdd.getNumPartitions}")

    val totalWindows   = baseRdd.count()                                    // ACTION (count): sanity check
    val genres         = baseRdd.map(_.genre).distinct().collect().sorted   // TRANSFORMATION (distinct) + ACTION (collect)
    val distinctTracks = baseRdd.map(_.trackId).distinct().count()          // ACTION (count)

    Report.log(s"Loaded $totalWindows windows across ${distinctTracks} tracks and ${genres.length} genres")
    require(totalWindows == 58020L, s"Expected 58,020 windows, got $totalWindows")
    require(distinctTracks == 984L, s"Expected 984 tracks, got $distinctTracks")
    require(genres.length == 10, s"Expected 10 genres, got ${genres.length}")

    Report.section("Analysis A: Genre Acoustic Profiles")
    val resultA = computeA(baseRdd, featureNames)
    reportA(resultA, featureNames, genres)

    Report.section("Analysis B: Dynamics Within Recordings")
    val resultB = computeB(baseRdd, featureNames)
    reportB(resultB, resultA.genreFeatureStats)

    Report.section("Analysis C: Genre Similarity (Preview of Likely Confusions)")
    val resultC = computeC(sc, baseRdd, featureNames, resultA.genreFeatureStats, genres)
    reportC(resultC)

    Report.section("Analysis D: Tempo via the 3-Second Table")
    val resultD = computeD(sc, baseRdd)
    reportD(resultD)

    runReflection(spark, featureNames, resultD)

    Report.log(s"\nFull report saved to: ${Paths.RddReport}")
    spark.stop()
  }
}
