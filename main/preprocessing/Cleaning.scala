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
 * Other quality checks (nulls, duplicates, label consistency, negative variances)
 * were verified during exploration and required no action. Unlike the earlier
 * CSV-based version of this stage, there is no tempo to sanity-check: tempo is
 * not extracted from the raw audio (see CLAUDE.md).
 */
object Cleaning {

  val Stage               = "cleaning"
  val SilenceRmsThreshold = 0.001  // ≈ -60 dBFS: effectively silence

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Cleaning")
    run(spark)
    spark.stop()
  }

  def run(spark: SparkSession): DataFrame = {
    val raw = DataIO.readStage(spark, Paths.WindowFeatures).cache()

    // ---------- Rule 1: remove silent windows ----------
    val isSilent = col("rms_mean") < SilenceRmsThreshold

    println("\n=== Silent windows removed ===")
    raw.filter(isSilent).select("track_id", "window_idx", "label", "rms_mean").show(truncate = false)

    val clean = raw.filter(!isSilent).cache()

    DataIO.writeStage(clean, Paths.Cleaned)

    // ---------- Metrics for the report ----------
    val rawRows   = raw.count()
    val cleanRows = clean.count()
    val before    = genreCounts(raw)
    val after     = genreCounts(clean)

    val metrics = Seq(
      Metric(Stage, "rows", rawRows.toString, cleanRows.toString),
      Metric(Stage, "columns", raw.columns.length.toString, clean.columns.length.toString),
      Metric(Stage, "silent windows", (rawRows - cleanRows).toString, "0", s"rms_mean < $SilenceRmsThreshold removed")
    ) ++ before.keys.toSeq.sorted.map { g =>
      Metric(Stage, s"windows: $g", before(g).toString, after.getOrElse(g, 0L).toString)
    }

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