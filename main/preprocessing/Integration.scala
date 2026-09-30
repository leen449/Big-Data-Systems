package sonicspark.preprocessing

import java.io.File
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Stage 2 of the preprocessing pipeline: INTEGRATION
 *
 * Connects each window to the song (track) it belongs to, and computes the
 * 3-second segment it falls in (segment_idx = floor(start_sec / 3), per
 * CLAUDE.md) so later stages or ad-hoc checks can join back to
 * features_3_sec.csv. track_id is already a column on every window (set
 * during audio extraction), so unlike the CSV-based version of this stage,
 * no filename parsing is needed to find it.
 *
 * Output 1: data/interim/02_integrated -> windows + segment_idx, track_length
 * Output 2: data/interim/02_tracks     -> one row per song (for SQL joins and track-level splits)
 * Metrics : outputs/stats/02_integration.csv
 */
object Integration {

  val Stage = "integration"

  // A track needs at least this many samples to produce a full 59 windows
  // (see AudioFeatures.windowTrack: WindowSize + 58 hops).
  val MinSamplesForFullTrack: Long = AudioConfig.WindowSize + 58L * AudioConfig.WindowHop
  val ExpectedWindowsPerTrack      = 59

  val TrackIdRegex = "^([a-z]+\\.\\d{5})"  // parses features_30_sec.csv's own filenames only

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Integration")
    run(spark)
    spark.stop()
  }

  def run(spark: SparkSession): DataFrame = {

    // ---------- 1. Windows: add the 3-second segment key ----------
    val windows = DataIO.readStage(spark, Paths.Cleaned)
      .withColumn("segment_idx", floor(col("start_sec") / 3).cast("int"))

    // ---------- 2. Build the tracks table (one row per song) ----------
    val songInfo = DataIO.readFeaturesCsv(spark, Paths.Raw30Sec).select(
      regexp_extract(col("filename"), TrackIdRegex, 1).as("track_id"),
      col("label"),
      col("length").as("track_length")
    )
    // Raw (pre-Cleaning) counts come from window_features directly, not from the
    // already-cleaned windows above: a track can have fewer windows than expected
    // for two unrelated reasons -- a genuinely short recording (fewer raw windows
    // to begin with), or the silence rule in Cleaning trimming some away -- and
    // telling those apart requires both counts.
    val rawCounts   = DataIO.readStage(spark, Paths.WindowFeatures).groupBy("track_id").agg(count("*").as("num_raw_windows"))
    val cleanCounts = windows.groupBy("track_id").agg(count("*").as("num_windows"))
    val audioFiles  = listTrackIds(spark, Paths.AudioDir, ".wav").withColumn("has_audio", lit(true))
    val imageFiles  = listTrackIds(spark, Paths.ImagesDir, ".png").withColumn("has_image", lit(true))

    val tracks = songInfo
      .join(rawCounts, Seq("track_id"), "left")
      .join(cleanCounts, Seq("track_id"), "left")
      .join(audioFiles, Seq("track_id"), "left")
      .join(imageFiles, Seq("track_id"), "left")
      .na.fill(0L, Seq("num_raw_windows", "num_windows"))
      .na.fill(false, Seq("has_audio", "has_image"))
      .withColumn("windows_removed_by_cleaning", col("num_raw_windows") - col("num_windows"))
      .cache()

    // ---------- 3. Enrich windows with the song's length ----------
    val joined = windows
      .join(tracks.select(col("track_id"), col("track_length"), col("label").as("track_label")),
        Seq("track_id"), "left")
      .cache()

    val unmatched = joined.filter(col("track_label").isNull).count()
    val conflicts = joined.filter(col("track_label") =!= col("label")).count()
    val integrated = joined.drop("track_label")   // label conflicts: window label kept

    // ---------- Save both outputs ----------
    DataIO.writeStage(integrated, Paths.Integrated)
    DataIO.writeStage(tracks, Paths.Tracks)

    // ---------- Report ----------
    // Two unrelated reasons a track can end up with fewer than 59 windows:
    // (1) it was genuinely short to begin with (fewer raw windows), or
    // (2) Cleaning's silence rule trimmed windows off an otherwise full track.
    // A track with track_length >= MinSamplesForFullTrack cannot be case (1) --
    // e.g. a corrupt/unreadable file (num_raw_windows = 0) with a normal-looking
    // CSV length is neither: it belongs to neither bucket, which is correct.
    val shortRecordings = tracks.filter(col("track_length") < MinSamplesForFullTrack)
    val silenceTrimmed  = tracks.filter(col("windows_removed_by_cleaning") > 0)

    println(s"\n=== Short recordings (track_length < $MinSamplesForFullTrack samples) ===")
    shortRecordings
      .select("track_id", "track_length", "num_raw_windows")
      .orderBy("track_id")
      .show(20, truncate = false)

    println("=== Tracks that lost windows to Cleaning's silence rule ===")
    silenceTrimmed
      .select("track_id", "num_raw_windows", "num_windows", "windows_removed_by_cleaning")
      .orderBy("track_id")
      .show(20, truncate = false)

    println("=== Songs with a missing audio file or image ===")
    tracks.filter(!col("has_audio") || !col("has_image"))
      .select("track_id", "has_audio", "has_image")
      .show(truncate = false)

    val metrics = Seq(
      Metric(Stage, "window rows", windows.count().toString, integrated.count().toString, "left join keeps all windows"),
      Metric(Stage, "window columns", windows.columns.length.toString, integrated.columns.length.toString, "+ track_length"),
      Metric(Stage, "tracks table rows", "-", tracks.count().toString, "one row per song"),
      Metric(Stage, "windows without matching song", "-", unmatched.toString),
      Metric(Stage, "label conflicts (window vs song)", "-", conflicts.toString, "window label kept"),
      Metric(Stage, s"short recordings (< $ExpectedWindowsPerTrack raw windows)", "-", shortRecordings.count().toString,
        s"track_length < $MinSamplesForFullTrack samples"),
      Metric(Stage, "tracks trimmed by Cleaning's silence rule", "-", silenceTrimmed.count().toString,
        s"${silenceTrimmed.agg(sum("windows_removed_by_cleaning")).head.getLong(0)} windows removed total"),
      Metric(Stage, "songs without audio file", "-", tracks.filter(!col("has_audio")).count().toString),
      Metric(Stage, "songs without spectrogram image", "-", tracks.filter(!col("has_image")).count().toString)
    )

    println("\n=== Integration summary ===")
    Metrics.show(metrics)
    println(s"Metrics saved to: ${Metrics.save(metrics, "02_integration.csv")}")

    integrated
  }

  /** Turns file names in <dir>/<genre>/ into track_ids ("blues.00000.wav" or "blues00000.png"). */
  private def listTrackIds(spark: SparkSession, dir: String, ext: String): DataFrame = {
    import spark.implicits._
    val names = Option(new File(dir).listFiles()).getOrElse(Array.empty[File])
      .filter(_.isDirectory)
      .flatMap(g => Option(g.listFiles()).getOrElse(Array.empty[File]))
      .map(_.getName.toLowerCase)
      .filter(_.endsWith(ext))
      .toSeq

    val pattern = "^([a-z]+)\\.?(\\d{5})"
    names.toDF("name")
      .select(concat_ws(".",
        regexp_extract(col("name"), pattern, 1),
        regexp_extract(col("name"), pattern, 2)).as("track_id"))
      .distinct()
  }
}