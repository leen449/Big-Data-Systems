package sonicspark.preprocessing

import org.apache.spark.sql.DataFrame
import sonicspark.common._

/**
 * Runs the full preprocessing pipeline in order:
 *   window_features -> Cleaning -> Integration -> Reduction -> Transformation -> final dataset
 *
 * Assumes AudioFeatures has already been run on the full corpus and written
 * data/interim/00_window_features (that stage takes about an hour, so it is
 * not re-run here on every pipeline execution).
 *
 *   sbt "runMain sonicspark.preprocessing.RunPreprocessing"
 */
object RunPreprocessing {

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Preprocessing")
    val start = System.nanoTime()

    val raw = DataIO.readStage(spark, Paths.WindowFeatures)

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