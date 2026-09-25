package sonicspark

import sonicspark.common.{Paths, Spark}

/**
 * Setup test: checks that Spark runs and can read the dataset.
 * Every team member should get the same output.
 */
object HelloSpark {

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Hello")

    // ----- DataFrame check: read the CSV -----
    val df = spark.read
      .option("header", "true")      // first line contains the column names
      .option("inferSchema", "true") // detect numbers automatically instead of reading everything as text
      .csv(Paths.Raw3Sec)

    println(s"Rows: ${df.count()} | Columns: ${df.columns.length}")

    // How many segments per genre?
    df.groupBy("label").count().orderBy("label").show()

    // ----- RDD check: tiny computation -----
    val total = spark.sparkContext
      .parallelize(1 to 100) // turn the numbers 1..100 into an RDD
      .map(_ * 2)            // double each number
      .reduce(_ + _)         // add them all up
    println(s"RDD check (should be 10100): $total")

    spark.stop()
  }
}