package sonicspark.common

import org.apache.spark.sql.SparkSession

/**
 * Creates the SparkSession used by every program in the project.
 * Having it in one place means everyone runs Spark with the same settings.
 */
object Spark {

  def session(appName: String): SparkSession = {
    val spark = SparkSession.builder()
      .appName(appName)
      .master("local[*]")                          // run on this laptop, using all CPU cores
      .config("spark.sql.shuffle.partitions", "8") // default is 200, far too many for ~10k rows
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")         // hide Spark's noisy INFO messages
    spark
  }
}