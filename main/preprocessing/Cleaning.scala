package sonicspark.preprocessing

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Stage 1 of the preprocessing pipeline: CLEANING
 *
 * Input : data/raw/features_3_sec.csv
 * Output: data/interim/01_cleaned
 *         outputs/stats/01_cleaning.csv
 *
 * Other quality checks (nulls, duplicates, label consistency, negative variances)
 * were verified during exploration and required no action.
 */
object Cleaning {

  val Stage               = "cleaning"
  val SilenceRmsThreshold = 0.001  // ≈ -60 dBFS: effectively silence
  val TempoMin            = 40.0   // slowest realistic musical tempo (BPM)
  val TempoMax            = 240.0  // fastest realistic musical tempo (BPM)

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Cleaning")
    run(spark)
    spark.stop()
  }

  def run(spark: SparkSession): DataFrame = {
    val raw = DataIO.readFeaturesCsv(spark, Paths.Raw3Sec).cache()

    // ---------- Rule 1: remove silent segments ----------
    val isSilent = col("rms_mean") < SilenceRmsThreshold

    println("\n=== Silent segments removed ===")
    raw.filter(isSilent).select("filename", "label", "rms_mean").show(truncate = false)

    // ---------- Rule 2: flag unrealistic tempo ----------
    val clean = raw
      .filter(!isSilent)
      .withColumn("tempo_in_range", col("tempo").between(TempoMin, TempoMax))
      .cache()

    DataIO.writeStage(clean, Paths.Cleaned)

    // ---------- Metrics for the report ----------
    val rawRows   = raw.count()
    val cleanRows = clean.count()
    val tempoOut  = clean.filter(!col("tempo_in_range")).count()
    val before    = genreCounts(raw)
    val after     = genreCounts(clean)

    val metrics = Seq(
      Metric(Stage, "rows", rawRows.toString, cleanRows.toString),
      Metric(Stage, "columns", raw.columns.length.toString, clean.columns.length.toString, "+ tempo_in_range"),
      Metric(Stage, "silent segments", (rawRows - cleanRows).toString, "0", s"rms_mean < $SilenceRmsThreshold removed"),
      Metric(Stage, s"tempo outside $TempoMin-$TempoMax BPM", tempoOut.toString, tempoOut.toString, "kept, flagged")
    ) ++ before.keys.toSeq.sorted.map { g =>
      Metric(Stage, s"segments: $g", before(g).toString, after.getOrElse(g, 0L).toString)
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