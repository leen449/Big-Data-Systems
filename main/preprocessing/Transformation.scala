package sonicspark.preprocessing

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Stage 4 of the preprocessing pipeline: TRANSFORMATION
 *
 * Input : data/interim/03_reduced
 * Output: data/processed/final                 (analysis-ready dataset)
 *         outputs/stats/04_transformation.csv  (before/after metrics)
 *         outputs/stats/04_skewness.csv        (skewness per feature, before and after)
 *         outputs/stats/04_label_mapping.csv   (genre -> label_idx)
 *
 * Scaling is intentionally NOT done here: it belongs in the ML pipeline,
 * fitted on training data only, to avoid data leakage.
 */
object Transformation {

  val Stage         = "transformation"
  val SkewThreshold = 1.0   // T2: log-transform features more skewed than this
  val LogEpsilon    = 1e-6  // T2: avoids log(0); negligible next to typical values
  val IdColumns     = Seq("track_id", "segment_index", "label")

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Transformation")
    run(spark)
    spark.stop()
  }

  def run(spark: SparkSession): DataFrame = {
    val reduced = DataIO.readStage(spark, Paths.Reduced)
    val rows    = reduced.count()

    // ---------- T1: feature engineering ----------
    val engineered = reduced
      .withColumn("harmonic_percussive_ratio", col("harmony_var") / (col("perceptr_var") + LogEpsilon))
      .withColumn("loudness_variation", sqrt(col("rms_var")) / col("rms_mean"))
      .cache()

    val featureCols = engineered.columns.filterNot(IdColumns.contains).toSeq

    // ---------- T2: log transformation of skewed features ----------
    val (skewBefore, minValue) = skewAndMin(engineered, featureCols)
    val toLog = featureCols.filter(c => skewBefore(c) > SkewThreshold && minValue(c) > 0).toSet

    def newName(c: String): String = if (toLog.contains(c)) s"log_$c" else c

    val transformed = engineered.select(
      IdColumns.map(col) ++
      featureCols.map(c => if (toLog.contains(c)) log(col(c) + LogEpsilon).as(newName(c)) else col(c)): _*
    )
    val finalFeatures = featureCols.map(newName)
    val (skewAfter, _) = skewAndMin(transformed, finalFeatures)

    // ---------- T3: label encoding ----------
    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("label_idx")
      .setStringOrderType("alphabetAsc")   // blues = 0, classical = 1, ... rock = 9
      .fit(transformed)

    val finalDf = indexer.transform(transformed)
      .select((IdColumns :+ "label_idx") ++ finalFeatures map col: _*)

    DataIO.writeStage(finalDf, Paths.Final)

    // ---------- Report 1: skewness before/after ----------
    val skewHeader = Seq("feature", "skew_before", "transformation", "skew_after")
    val skewRows = featureCols.map { c =>
      Seq(newName(c), f"${skewBefore(c)}%.2f", if (toLog.contains(c)) "log" else "-", f"${skewAfter(newName(c))}%.2f")
    }
    println("\n=== Skewness before and after ===")
    Metrics.showTable(skewHeader, skewRows)
    Metrics.saveTable("04_skewness.csv", skewHeader, skewRows)

    // ---------- Report 2: label mapping ----------
    val mapping = indexer.labelsArray.head.zipWithIndex.map { case (g, i) => Seq(g, i.toString) }.toSeq
    println("\n=== Label encoding ===")
    Metrics.showTable(Seq("genre", "label_idx"), mapping)
    Metrics.saveTable("04_label_mapping.csv", Seq("genre", "label_idx"), mapping)

    // ---------- Report 3: summary ----------
    val skewedBefore = featureCols.count(c => math.abs(skewBefore(c)) > SkewThreshold)
    val skewedAfter  = finalFeatures.count(c => math.abs(skewAfter(c)) > SkewThreshold)

    val metrics = Seq(
      Metric(Stage, "rows", rows.toString, finalDf.count().toString),
      Metric(Stage, "columns", reduced.columns.length.toString, finalDf.columns.length.toString,
        "+ 2 engineered features + label_idx"),
      Metric(Stage, "T1 engineered features", "-", "2", "harmonic_percussive_ratio loudness_variation"),
      Metric(Stage, "T2 log-transformed features", "-", toLog.size.toString, s"skewness > $SkewThreshold"),
      Metric(Stage, "features with |skewness| > 1", skewedBefore.toString, skewedAfter.toString),
      Metric(Stage, "T3 label encoding", "10 genres", "label_idx 0-9", "alphabetical"),
      Metric(Stage, "T4 scaling", "-", "-", "deferred to ML pipeline: fit on training data only")
    )
    println("\n=== Transformation summary ===")
    Metrics.show(metrics)
    println(s"Metrics saved to: ${Metrics.save(metrics, "04_transformation.csv")}")
    println(s"Final dataset saved to: ${Paths.Final}")

    finalDf
  }

  /** Skewness and minimum of each column, in one pass. */
  private def skewAndMin(df: DataFrame, cols: Seq[String]): (Map[String, Double], Map[String, Double]) = {
    val exprs = cols.flatMap(c => Seq(skewness(c).as(s"s_$c"), min(c).cast("double").as(s"m_$c")))
    val r     = df.agg(exprs.head, exprs.tail: _*).head()
    (cols.map(c => c -> r.getAs[Double](s"s_$c")).toMap,
     cols.map(c => c -> r.getAs[Double](s"m_$c")).toMap)
  }
}