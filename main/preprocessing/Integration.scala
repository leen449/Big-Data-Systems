package sonicspark.preprocessing

import java.io.File
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Stage 2 of the preprocessing pipeline: INTEGRATION
 *
 * Connects each 3-second segment to the song (track) it belongs to.
 *
 * Output 1: data/interim/02_integrated  -> segments + track_id, segment_index, track_tempo
 * Output 2: data/interim/02_tracks      -> one row per song (for SQL joins and track-level splits)
 * Metrics : outputs/stats/02_integration.csv
 */
object Integration {

  val Stage         = "integration"
  val SegmentLength = 66149                    // samples per 3-second segment
  val TrackIdRegex  = "^([a-z]+\\.\\d{5})"     // "blues.00000.3.wav" -> "blues.00000"

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Integration")
    run(spark)
    spark.stop()
  }

  def run(spark: SparkSession): DataFrame = {

    // ---------- 1. Segments: add the keys ----------
    val segments = DataIO.readStage(spark, Paths.Cleaned)
      .withColumn("track_id", regexp_extract(col("filename"), TrackIdRegex, 1))
      .withColumn("segment_index", regexp_extract(col("filename"), "\\.(\\d+)\\.wav$", 1).cast("int"))

    // ---------- 2. Build the tracks table (one row per song) ----------
    val songInfo = DataIO.readFeaturesCsv(spark, Paths.Raw30Sec).select(
      regexp_extract(col("filename"), TrackIdRegex, 1).as("track_id"),
      col("label"),
      col("length").as("track_length"),
      col("tempo").as("track_tempo")
    )
    val segmentCounts = segments.groupBy("track_id").agg(count("*").as("num_segments"))
    val audioFiles    = listTrackIds(spark, Paths.AudioDir, ".wav").withColumn("has_audio", lit(true))
    val imageFiles    = listTrackIds(spark, Paths.ImagesDir, ".png").withColumn("has_image", lit(true))

    val tracks = songInfo
      .join(segmentCounts, Seq("track_id"), "left")
      .join(audioFiles, Seq("track_id"), "left")
      .join(imageFiles, Seq("track_id"), "left")
      .na.fill(0L, Seq("num_segments"))
      .na.fill(false, Seq("has_audio", "has_image"))
      .cache()

    // ---------- 3. Enrich segments with song-level tempo ----------
    val joined = segments
      .join(tracks.select(col("track_id"), col("track_tempo"), col("label").as("track_label")),
        Seq("track_id"), "left")
      .cache()

    val unmatched = joined.filter(col("track_label").isNull).count()
    val conflicts = joined.filter(col("track_label") =!= col("label")).count()
    val integrated = joined.drop("track_label")   // label conflicts: segment label kept

    // ---------- Save both outputs ----------
    DataIO.writeStage(integrated, Paths.Integrated)
    DataIO.writeStage(tracks, Paths.Tracks)

    // ---------- Report ----------
    println("\n=== Songs with fewer than 10 segments ===")
    tracks.filter(col("num_segments") < 10)
      .select("track_id", "num_segments", "track_length")
      .orderBy("track_id")
      .show(20, truncate = false)

    println("=== Songs with a missing audio file or image ===")
    tracks.filter(!col("has_audio") || !col("has_image"))
      .select("track_id", "has_audio", "has_image")
      .show(truncate = false)

    val metrics = Seq(
      Metric(Stage, "segment rows", segments.count().toString, integrated.count().toString, "left join keeps all segments"),
      Metric(Stage, "segment columns", "61", integrated.columns.length.toString, "+ track_id, segment_index, track_tempo"),
      Metric(Stage, "tracks table rows", "-", tracks.count().toString, "one row per song"),
      Metric(Stage, "segments without matching song", "-", unmatched.toString),
      Metric(Stage, "label conflicts (segment vs song)", "-", conflicts.toString, "segment label kept"),
      Metric(Stage, "songs with < 10 segments", "-", tracks.filter(col("num_segments") < 10).count().toString),
      Metric(Stage, s"songs shorter than ${SegmentLength * 10} samples", "-",
        tracks.filter(col("track_length") < SegmentLength * 10).count().toString, "explains missing segments"),
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