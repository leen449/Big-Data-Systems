package sonicspark

import sonicspark.common._

/**
 * Smoke test for the pipeline foundations:
 *  1. Both CSVs can be read with the explicit schema
 *  2. Data can be written to Parquet and read back unchanged
 *  3. Metrics can be printed and saved
 */
object FoundationCheck {

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-FoundationCheck")

    // 1. Read both raw files with the explicit schema
    val seg   = DataIO.readFeaturesCsv(spark, Paths.Raw3Sec)
    val track = DataIO.readFeaturesCsv(spark, Paths.Raw30Sec)
    val segRows   = seg.count()
    val trackRows = track.count()

    // 2. Parquet round-trip
    DataIO.writeStage(seg, Paths.SmokeTest)
    val back     = DataIO.readStage(spark, Paths.SmokeTest)
    val backRows = back.count()

    // 3. Metrics
    val metrics = Seq(
      Metric("smoke_test", "3-sec rows (CSV)", segRows.toString, "-", "explicit schema"),
      Metric("smoke_test", "30-sec rows (CSV)", trackRows.toString, "-", "explicit schema"),
      Metric("smoke_test", "rows after Parquet round-trip", segRows.toString, backRows.toString),
      Metric("smoke_test", "columns after Parquet round-trip",
        seg.columns.length.toString, back.columns.length.toString)
    )
    Metrics.show(metrics)
    println(s"Metrics saved to: ${Metrics.save(metrics, "00_smoke_test.csv")}")

    // Stop with an error if anything changed
    require(segRows == backRows, "Row count changed after Parquet round-trip")
    require(seg.schema == back.schema, "Schema changed after Parquet round-trip")

    spark.stop()
    println("\nFOUNDATION CHECK PASSED")
  }
}