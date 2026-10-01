package sonicspark.preprocessing

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Stage 1 of the preprocessing pipeline: CLEANING
 *
 * Input : data/interim/00_window_features (see AudioFeatures.scala)
 * Output: data/interim/01_cleaned
 *         outputs/stats/01_cleaning.csv
 *
 * Three rules, applied in order:
 *   R1. Remove silent windows.
 *   R2. Remove both copies of the known mislabelled recording pair
 *       (metal.00058 / rock.00016 are the same audio under conflicting
 *       genre labels -- a documented GTZAN fault, not a bug in extraction).
 *   R3. Remove exact-duplicate windows: GTZAN repeats some recordings under
 *       a second track number (Sturm, 2013). Detected directly from the
 *       data (groups of rows whose all 38 features are bitwise identical
 *       across different track_ids), not from a hard-coded list, so the
 *       rule stays correct if the dataset changes. Within each such group,
 *       the window from the lower-numbered track is kept.
 *
 * R2 must run before R3: R3's generic "keep the lower track number" rule
 * would otherwise keep rock.00016 (16 < 58) and only drop metal.00058,
 * which is wrong for this specific pair -- both copies must go, because
 * the two genre labels directly conflict, not just the track numbers.
 *
 * Other quality checks (nulls, label consistency, negative variances) were
 * verified during exploration and required no action. Unlike the earlier
 * CSV-based version of this stage, there is no tempo to sanity-check: tempo
 * is not extracted from the raw audio (see CLAUDE.md).
 */
object Cleaning {

  val Stage               = "cleaning"
  val SilenceRmsThreshold = 0.001  // ≈ -60 dBFS: effectively silence

  // Confirmed by a direct duplicate-feature-vector scan: metal.00058 and
  // rock.00016 are bitwise-identical across all 38 features on every
  // shared window, yet carry different genre labels. Unlike R3's other
  // duplicates (which are legitimate track-number repeats, same genre),
  // this is a genre mislabelling -- there is no "correct" copy to keep by
  // track number, so both are removed. This is the one deliberate
  // exception to "detect duplicates from the data, not by name" (R3 still
  // detects everything else automatically).
  val MislabelledPair: Set[String] = Set("metal.00058", "rock.00016")

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Cleaning")
    run(spark)
    spark.stop()
  }

  def run(spark: SparkSession): DataFrame = {
    val raw = DataIO.readStage(spark, Paths.WindowFeatures).cache()
    val rawRows = raw.count()

    // ---------- R1: remove silent windows ----------
    val isSilent = col("rms_mean") < SilenceRmsThreshold

    println("\n=== R1: silent windows removed ===")
    raw.filter(isSilent).select("track_id", "window_idx", "label", "rms_mean").show(truncate = false)

    val afterSilence = raw.filter(!isSilent).cache()
    val afterSilenceRows = afterSilence.count()

    // ---------- R2: remove the known mislabelled pair ----------
    val mislabelledCounts = afterSilence
      .filter(col("track_id").isin(MislabelledPair.toSeq: _*))
      .groupBy("track_id").count()
      .collect()

    println("\n=== R2: mislabelled recording pair removed (identical audio, conflicting genre label) ===")
    mislabelledCounts.foreach { row =>
      println(s"  REMOVED  ${row.getString(0)}: ${row.getLong(1)} windows -- duplicate of its pair, genre labels conflict")
    }

    val afterMislabelled = afterSilence.filter(!col("track_id").isin(MislabelledPair.toSeq: _*)).cache()
    val afterMislabelledRows = afterMislabelled.count()

    // ---------- R3: remove exact-duplicate windows, detected from the data ----------
    val featureCols = Schema.windowFeatureColumns
    val trackNum    = regexp_extract(col("track_id"), "(\\d+)$", 1).cast("int")

    // Feature-vector groups that span more than one track_id are duplicates;
    // record the lowest track number in each such group.
    val dupGroupMinTrack = afterMislabelled
      .withColumn("_track_num", trackNum)
      .groupBy(featureCols.head, featureCols.tail: _*)
      .agg(countDistinct("track_id").as("_distinct_tracks"), min("_track_num").as("_min_track_num"))
      .filter(col("_distinct_tracks") > 1)

    val withTrackNum = afterMislabelled.withColumn("_track_num", trackNum)

    // The representative (kept) row of each duplicate group, to name the
    // surviving partner when logging what got removed.
    val keptRepresentative = withTrackNum
      .join(dupGroupMinTrack, featureCols, "inner")
      .filter(col("_track_num") === col("_min_track_num"))
      .select((featureCols.map(col) :+ col("track_id").as("_kept_track_id") :+ col("window_idx").as("_kept_window_idx")): _*)

    val toDrop = withTrackNum
      .join(dupGroupMinTrack, featureCols, "inner")
      .filter(col("_track_num") =!= col("_min_track_num"))
      .join(keptRepresentative, featureCols, "left")
      .cache()

    val dupRemovedRows = toDrop.count()

    println(s"\n=== R3: every removed duplicate window ($dupRemovedRows total) ===")
    toDrop
      .withColumn("reason", concat(
        lit("duplicate of "), col("_kept_track_id"), lit(" window "), col("_kept_window_idx"), lit(" (lower track number, kept)")
      ))
      .select("track_id", "window_idx", "reason")
      .orderBy("track_id", "window_idx")
      .show(math.max(dupRemovedRows.toInt, 1), truncate = false)

    println("=== R3: windows removed per track (summary) ===")
    toDrop
      .groupBy("track_id", "_kept_track_id")
      .count()
      .withColumnRenamed("_kept_track_id", "duplicate_of")
      .orderBy("track_id")
      .show(100, truncate = false)

    val clean = afterMislabelled
      .join(toDrop.select("track_id", "window_idx"), Seq("track_id", "window_idx"), "left_anti")
      .cache()

    DataIO.writeStage(clean, Paths.Cleaned)

    // ---------- Metrics for the report ----------
    val cleanRows          = clean.count()
    val mislabelledRemoved = afterSilenceRows - afterMislabelledRows

    val genreWindows = genreCounts(clean)
    val genreTracks  = clean.groupBy("label").agg(countDistinct("track_id").as("n"))
      .collect().map(r => r.getString(0) -> r.getLong(1)).toMap

    println("\n=== Windows and tracks per genre, after Cleaning ===")
    Metrics.showTable(
      Seq("genre", "windows", "tracks"),
      genreWindows.keys.toSeq.sorted.map(g => Seq(g, genreWindows(g).toString, genreTracks.getOrElse(g, 0L).toString))
    )

    val metrics = Seq(
      Metric(Stage, "rows", rawRows.toString, cleanRows.toString),
      Metric(Stage, "columns", raw.columns.length.toString, clean.columns.length.toString),
      Metric(Stage, "R1: silent windows", rawRows.toString, afterSilenceRows.toString,
        s"removed ${rawRows - afterSilenceRows}: rms_mean < $SilenceRmsThreshold"),
      Metric(Stage, "R2: mislabelled pair", afterSilenceRows.toString, afterMislabelledRows.toString,
        s"removed $mislabelledRemoved: ${MislabelledPair.mkString(", ")}"),
      Metric(Stage, "R3: exact-duplicate windows", afterMislabelledRows.toString, cleanRows.toString,
        s"removed $dupRemovedRows: identical to a lower-numbered track")
    )

    println("\n=== Cleaning summary ===")
    Metrics.show(metrics)
    println(s"Metrics saved to: ${Metrics.save(metrics, "01_cleaning.csv")}")
    println(s"Clean data saved to: ${Paths.Cleaned}")

    clean
  }

  private def genreCounts(df: DataFrame): Map[String, Long] =
    df.groupBy("label").count()
      .collect()
      .map(r => r.getString(0) -> r.getLong(1))
      .toMap
}
