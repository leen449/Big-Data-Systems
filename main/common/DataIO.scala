package sonicspark.common

import org.apache.hadoop.fs.{FileSystem, Path => HadoopPath}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

/**
 * Reading and writing for every pipeline stage.
 * - Raw CSVs are read with the explicit schema and FAILFAST mode.
 * - Stage outputs are stored as Parquet and always overwritten (idempotent).
 */
object DataIO {

  /** Read a GTZAN feature CSV, verifying the header before trusting the schema. */
  def readFeaturesCsv(spark: SparkSession, path: String): DataFrame = {
    // 1. Check the header matches our schema (reads only the first line)
    val actual   = spark.read.option("header", "true").csv(path).columns.toSeq
    val expected = Schema.features.fieldNames.toSeq
    if (actual != expected) {
      val missing = expected.diff(actual)
      val extra   = actual.diff(expected)
      throw new IllegalStateException(
        s"Unexpected header in $path. Missing: ${missing.mkString(", ")} | Extra: ${extra.mkString(", ")}"
      )
    }

    // 2. Read with explicit types; stop immediately on any malformed row
    spark.read
      .option("header", "true")
      .option("mode", "FAILFAST")
      .schema(Schema.features)
      .csv(path)
  }

  /** Save a stage output as Parquet, replacing any previous run. */
  def writeStage(df: DataFrame, path: String): Unit =
    df.write.mode(SaveMode.Overwrite).parquet(path)

  /** Load the output of a previous stage. */
  def readStage(spark: SparkSession, path: String): DataFrame =
    spark.read.parquet(path)

  /** Save a DataFrame as a single human-readable CSV file (with header), replacing any previous one. */
  def writeSingleCsv(spark: SparkSession, df: DataFrame, path: String): Unit = {
    val tmpDir = path + "_tmp"
    df.coalesce(1).write.mode(SaveMode.Overwrite).option("header", "true").csv(tmpDir)

    val fs      = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val tmpPath = new HadoopPath(tmpDir)
    val part    = fs.listStatus(tmpPath).map(_.getPath).find(_.getName.startsWith("part-")).get

    val dest = new HadoopPath(path)
    fs.delete(dest, false)
    fs.rename(part, dest)
    fs.delete(tmpPath, true)
  }
}