package sonicspark.preprocessing

import org.apache.spark.sql.DataFrame
import sonicspark.common._

/**
 * Runs the full preprocessing pipeline in order:
 *   raw CSV -> Cleaning -> Integration -> Reduction -> Transformation -> final dataset
 *
 *   sbt "runMain sonicspark.preprocessing.RunPreprocessing"
 */
object RunPreprocessing {

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Preprocessing")
    val start = System.nanoTime()

    val raw = DataIO.readFeaturesCsv(spark, Paths.Raw3Sec)

    val stages: Seq[(String, () => DataFrame)] = Seq(
      "cleaning"       -> (() => Cleaning.run(spark)),
      "integration"    -> (() => Integration.run(spark)),
      "reduction"      -> (() => Reduction.run(spark)),
      "transformation" -> (() => Transformation.run(spark))
    )

    val results = stages.map { case (name, runStage) =>
      println(s"\n################ STAGE: ${name.toUpperCase} ################")
      val t0      = System.nanoTime()
      val df      = runStage()
      val seconds = (System.nanoTime() - t0) / 1e9
      Seq(name, df.count().toString, df.columns.length.toString, f"$seconds%.1f s")
    }

    val header = Seq("stage", "rows", "columns", "time")
    val rows   = Seq("raw input", raw.count().toString, raw.columns.length.toString, "-") +: results

    println("\n=== Preprocessing pipeline summary ===")
    Metrics.showTable(header, rows)
    Metrics.saveTable("00_pipeline_summary.csv", header, rows)
    println(f"Total time: ${(System.nanoTime() - start) / 1e9}%.1f s")

    spark.stop()
  }
}