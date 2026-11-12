package sonicspark.ml

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths => JPaths}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Phase 5, stage 2: TRAIN-ONLY FEATURE ANALYSIS
 *
 * Reimplements Reduction.scala's eta-squared and correlation-redundancy logic
 * independently (not imported), because Reduction.scala's job is Phase 2's
 * full-dataset feature selection, while this file's job is to redo the same
 * math on the training windows only -- a different row subset with a
 * different purpose (auditing whether Phase 2's decision still holds once a
 * proper train/test split exists, and whether anything new should be
 * dropped). Keeping it separate also means Phase 2's code is never touched
 * by Phase 5's work.
 *
 * This result is BINDING, not just descriptive: whatever analyzeTrainFeatures
 * drops becomes the actual base feature set (F0) used by every model.
 */
object FeatureAnalysis {

  /** feature -> (eta squared, decision text, for the printed/saved ranking). */
  case class FeatureAnalysisResult(
    eta2: Map[String, Double],
    ranking: Seq[(String, Double, String)],       // feature, eta2, decision -- sorted by eta2 desc
    reportedPairs: Seq[(String, String, Double)],  // every pair with |r| > ReportCorrThreshold
    droppedFeatures: Set[String],
    keptBaseFeatures: Seq[String]
  )

  /** eta² = between-genre variation / total variation, for each feature. Identical formula to Reduction.scala. */
  def etaSquared(df: DataFrame, cols: Seq[String], n: Long): Map[String, Double] = {
    val overallExprs = cols.flatMap(c => Seq(avg(c).as(s"m_$c"), var_pop(c).as(s"v_$c")))
    val overall      = df.agg(overallExprs.head, overallExprs.tail: _*).head()

    val genreExprs = count("*").as("n") +: cols.map(c => avg(c).as(c))
    val genres     = df.groupBy("label").agg(genreExprs.head, genreExprs.tail: _*).collect()

    cols.map { c =>
      val mean    = overall.getAs[Double](s"m_$c")
      val total   = overall.getAs[Double](s"v_$c") * n
      val between = genres.map(g => g.getAs[Long]("n") * math.pow(g.getAs[Double](c) - mean, 2)).sum
      c -> (if (total > 0) between / total else 0.0)
    }.toMap
  }

  /** All feature pairs whose absolute Pearson correlation exceeds `threshold`. Identical approach to Reduction.scala. */
  def correlatedPairs(df: DataFrame, cols: Seq[String], threshold: Double): Seq[(String, String, Double)] = {
    val vectors = new VectorAssembler()
      .setInputCols(cols.toArray)
      .setOutputCol("v")
      .transform(df.select(cols.map(col): _*))
    val m = Correlation.corr(vectors, "v").head.getAs[Matrix](0)

    for {
      i <- cols.indices
      j <- (i + 1) until cols.size
      r = m(i, j)
      if math.abs(r) > threshold
    } yield (cols(i), cols(j), r)
  }

  /**
   * Train-only version of Phase 2's R1 (low relevance) + R2 (redundancy) rules,
   * applied to the 38 final-dataset features. Also reports every pair with
   * |r| > ReportCorrThreshold (0.8), a wider net than the 0.95 action threshold,
   * purely for inspection.
   */
  def analyzeTrainFeatures(trainWindows: DataFrame, cols: Seq[String]): FeatureAnalysisResult = {
    val n    = trainWindows.count()
    val eta2 = etaSquared(trainWindows, cols, n)

    val lowRelevance = cols.filter(c => eta2(c) < MlConfig.MinEta2)
    val relevant     = cols.filterNot(lowRelevance.contains)

    val reportedPairs = correlatedPairs(trainWindows, relevant, MlConfig.ReportCorrThreshold)
    val actionablePairs = reportedPairs.filter(p => math.abs(p._3) > MlConfig.MaxCorr)

    val redundant = scala.collection.mutable.LinkedHashMap[String, (String, Double)]()
    actionablePairs.sortBy(p => -math.abs(p._3)).foreach { case (a, b, r) =>
      if (!redundant.contains(a) && !redundant.contains(b)) {
        val (keep, drop) = if (eta2(a) >= eta2(b)) (a, b) else (b, a)
        redundant(drop) = (keep, r)
      }
    }

    val dropped = lowRelevance.toSet ++ redundant.keySet
    val kept    = cols.filterNot(dropped.contains)

    val ranking = cols.sortBy(c => -eta2(c)).map { c =>
      val decision =
        if (lowRelevance.contains(c)) "dropped: low relevance"
        else redundant.get(c) match {
          case Some((keep, r)) => f"dropped: redundant with $keep (r = $r%.3f)"
          case None            => "kept"
        }
      (c, eta2(c), decision)
    }

    FeatureAnalysisResult(eta2, ranking, reportedPairs, dropped, kept)
  }

  /** eta² for the candidate extra features (range_p95_p5 + the 3 change columns), train rows only. */
  def etaSquaredForExtraFeatures(trainWindowsWithExtras: DataFrame, extraCols: Seq[String]): Map[String, Double] =
    etaSquared(trainWindowsWithExtras, extraCols, trainWindowsWithExtras.count())

  /**
   * Prints and saves the full ranking (38 base features + the 4 extra-feature rows,
   * clearly marked) and the |r| > 0.8 pairs. States explicitly whether anything beyond
   * spectral_centroid_mean (already absent from the 38, since Phase 2 already dropped
   * it) gets newly dropped by this train-only analysis.
   */
  def reportAndSave(result: FeatureAnalysisResult, extraEta2: Map[String, Double]): Unit = {
    val header = Seq("feature", "eta_squared", "decision")
    val baseRows  = result.ranking.map { case (c, e, d) => Seq(c, f"$e%.4f", d) }
    val extraRows = extraEta2.toSeq.sortBy(-_._2).map { case (c, e) =>
      Seq(c, f"$e%.4f", "candidate extra feature -- not part of F0, see CV feature-set comparison")
    }

    Report.log("\n--- Train-only feature analysis: eta^2 ranking (38 final-dataset features) ---")
    Report.table(header, baseRows)
    Report.log("--- Candidate extra features (computed separately, not part of F0) ---")
    Report.table(header, extraRows)
    Report.saveCsv("train_relevance.csv", header, baseRows ++ extraRows, Paths.MlOutputDir)

    val newDrops = result.droppedFeatures
    if (newDrops.isEmpty) {
      Report.log("Train-only analysis agrees with Phase 2: no feature is dropped beyond " +
        "spectral_centroid_mean, which Phase 2 had already removed from this 38-feature set.")
    } else {
      Report.log(s"Train-only analysis drops ${newDrops.size} feature(s) beyond Phase 2's original " +
        s"removal: ${newDrops.mkString(", ")}. These are removed from F0 and from every model.")
    }

    val pairHeader = Seq("feature_a", "feature_b", "pearson_r")
    val pairRows = result.reportedPairs.sortBy(p => -math.abs(p._3)).map { case (a, b, r) => Seq(a, b, f"$r%.4f") }
    Report.log(s"\n--- Pairwise correlations with |r| > ${MlConfig.ReportCorrThreshold} (train-only) ---")
    Report.table(pairHeader, pairRows)
    Report.saveCsv("train_correlation_pairs.csv", pairHeader, pairRows, Paths.MlOutputDir)
  }

  /**
   * Additional check (section 3): re-derive Phase 2's specific removal of
   * spectral_centroid_mean using train-only statistics, from data/interim/02_integrated
   * (the only stored table that still has spectral_centroid_mean -- Reduction.scala
   * already dropped it from everything downstream). Confirms whether
   * r(spectral_centroid_mean, rolloff_mean) still exceeds |r| > MaxCorr and whether
   * spectral_centroid_mean still has the lower eta^2, using train recordings only.
   */
  def recheckPhase2Selection(spark: SparkSession, trainTracks: DataFrame): Unit = {
    val integrated = DataIO.readStage(spark, Paths.Integrated)
    val trainOnly  = integrated.join(trainTracks.select("track_id"), "track_id")
    val n          = trainOnly.count()
    val numTracks  = trainTracks.count()

    val cols = Seq("spectral_centroid_mean", "rolloff_mean")
    val eta2 = etaSquared(trainOnly, cols, n)
    val r    = correlatedPairs(trainOnly, cols, 0.0).headOption.map(_._3).getOrElse(0.0)

    val stillRedundant = math.abs(r) > MlConfig.MaxCorr
    val lowerEta2       = eta2("spectral_centroid_mean") < eta2("rolloff_mean")
    val agrees          = stillRedundant && lowerEta2

    val lines = Seq(
      "Phase 2 selection recheck (train-only) -- Phase 5, Section 3 additional check",
      s"Input: ${Paths.Integrated}, filtered to the $numTracks training recordings ($n window rows).",
      "",
      f"eta^2(spectral_centroid_mean) = ${eta2("spectral_centroid_mean")}%.4f",
      f"eta^2(rolloff_mean)           = ${eta2("rolloff_mean")}%.4f",
      f"Pearson r(spectral_centroid_mean, rolloff_mean) = $r%.4f",
      "",
      s"Phase 2 (full dataset, 58,020 windows) dropped spectral_centroid_mean because " +
        s"|r| > ${MlConfig.MaxCorr} with rolloff_mean, and spectral_centroid_mean had the lower eta^2.",
      s"Train-only recheck: |r| > ${MlConfig.MaxCorr}? $stillRedundant. " +
        s"spectral_centroid_mean has the lower eta^2? $lowerEta2.",
      if (agrees)
        "Agreement: Phase 2's removal of spectral_centroid_mean still holds under train-only statistics."
      else
        "Disagreement: train-only statistics would not reach the same decision as Phase 2's " +
          "full-dataset statistics. Reported as required, not silently assumed -- this does not " +
          "change Phase 2's already-committed output, only documents the discrepancy."
    )

    val path = JPaths.get(Paths.MlOutputDir).resolve("phase2_selection_recheck.txt")
    Files.createDirectories(path.getParent)
    Files.write(path, lines.mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

    Report.log("\n--- Phase 2 selection recheck (train-only) ---")
    lines.foreach(Report.log)
    Report.log(s"Saved to: $path")
  }
}
