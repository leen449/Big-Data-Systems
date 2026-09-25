package sonicspark

import org.apache.spark.sql.functions.col
import sonicspark.common._

/**
 * Quick look inside any Parquet stage output.
 *   sbt "runMain sonicspark.Peek data/interim/02_integrated"
 *   sbt "runMain sonicspark.Peek data/interim/02_tracks"
 */
object Peek {

  def main(args: Array[String]): Unit = {
    val path  = args.headOption.getOrElse(Paths.Integrated)
    val spark = Spark.session("SonicSpark-Peek")
    val df    = DataIO.readStage(spark, path)

    println(s"\n$path: ${df.count()} rows x ${df.columns.length} columns")
    df.printSchema()

    // Small tables: show everything. Wide tables: show the key columns only.
    val keyCols = Seq("filename", "track_id", "segment_index", "label",
                      "tempo", "track_tempo", "tempo_in_range").filter(df.columns.contains)
    val shown = if (df.columns.length <= 10) df.columns.toSeq else keyCols

    df.select(shown.map(col): _*).show(10, truncate = false)
    spark.stop()
  }
}