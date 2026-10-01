package sonicspark.preprocessing

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Phase 2 deliverable: a small, reproducible snapshot of the final
 * preprocessed dataset for the report/presentation.
 *
 * Input : data/processed/final (Transformation's output -- read only, not
 *         recomputed or altered in any way)
 * Output: results/phase2_snapshot.csv -- a single, human-readable CSV
 *         (gitignored like data/ and outputs/; rerun this to regenerate it)
 *
 * Sampling: one row per genre, plus 5 additional rows, chosen with
 * Window.partitionBy("label").orderBy(rand(42)) -- rn = 1 guarantees every
 * genre appears exactly once; rn = 2, itself re-shuffled with the same
 * seed, supplies the 5 extra rows. Fixing the seed makes the snapshot
 * reproducible: rerunning this object always picks the same 15 rows.
 *
 * Deliberately a standalone object, not a stage wired into
 * RunPreprocessing: it's a reporting artifact that reads an existing
 * pipeline output, not a step in the pipeline itself.
 *
 *   sbt "runMain sonicspark.preprocessing.Phase2Snapshot"
 */
object Phase2Snapshot {

  val SampleSeed    = 42
  val ExtraRowCount = 5

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Phase2Snapshot")
    run(spark)
    spark.stop()
  }

  def run(spark: SparkSession): DataFrame = {
    val finalDf = DataIO.readStage(spark, Paths.Final)

    val byGenreShuffled = Window.partitionBy("label").orderBy(rand(SampleSeed))
    val ranked = finalDf.withColumn("rn", row_number().over(byGenreShuffled))

    val onePerGenre = ranked.filter(col("rn") === 1)                                   // 10 rows, every genre covered
    val extraRows   = ranked.filter(col("rn") === 2).orderBy(rand(SampleSeed)).limit(ExtraRowCount)

    val snapshot = onePerGenre.unionByName(extraRows)
      .drop("rn")
      .orderBy("label", "track_id", "window_idx")
      .cache()

    DataIO.writeSingleCsv(spark, snapshot, Paths.Phase2Snapshot)

    println(s"\n=== Phase 2 snapshot: ${snapshot.count()} rows, ${snapshot.columns.length} columns ===")
    snapshot.show(20, truncate = false)
    println(s"Snapshot saved to: ${Paths.Phase2Snapshot}")

    snapshot
  }
}
