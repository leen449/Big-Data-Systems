package sonicspark.ml

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Phase 5, stage 6: EVALUATION
 *
 * Pure "predictions in, numbers out": every function here takes a DataFrame
 * that already has true (label_idx) and predicted (prediction, probability)
 * columns and computes or aggregates on it. Reused identically for every
 * model x level x set combination, so the same metric definitions apply
 * everywhere -- no per-model special-casing.
 *
 * Metrics are computed by plain counting rather than org.apache.spark.mllib's
 * MulticlassMetrics, because that class doesn't expose macro-F1 either, and a
 * readable hand-rolled version is easier for the team to explain in Q&A.
 */
object MlMetrics {

  /** Recording-level soft voting: average the window probability vectors per
   *  recording, then argmax. This is the headline aggregation (section 6). */
  def softVoteToRecording(windowPreds: DataFrame): DataFrame = {
    val recordingProbs = windowPreds.groupBy("track_id")
      .agg(first("label_idx").as("label_idx"), Summarizer.mean(col("probability")).as("probability"))

    val argmaxUdf = udf((v: Vector) => v.toArray.zipWithIndex.maxBy(_._1)._2.toDouble)
    recordingProbs.withColumn("prediction", argmaxUdf(col("probability")))
  }

  /** Recording-level majority voting: plurality of window predictions, ties broken
   *  by the smaller class index (equivalent to alphabetical, since label_idx is
   *  assigned alphabetically). Secondary check alongside soft voting. */
  def majorityVoteToRecording(windowPreds: DataFrame): DataFrame = {
    val counts = windowPreds.groupBy("track_id", "label_idx", "prediction").count()
    val byTrack = Window.partitionBy("track_id").orderBy(col("count").desc, col("prediction").asc)
    counts.withColumn("rank", row_number().over(byTrack))
      .filter(col("rank") === 1)
      .select("track_id", "label_idx", "prediction")
  }

  case class ClassMetric(genre: String, precision: Double, recall: Double, f1: Double, support: Long)

  case class MetricsResult(
    accuracy: Double,
    macroF1: Double,
    weightedF1: Double,
    perClass: Seq[ClassMetric],
    confusion: Array[Array[Long]]   // confusion(trueIdx)(predIdx)
  )

  /**
   * Accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, and the
   * confusion matrix, all from one pass over a (labelCol, predCol) count table.
   * label_idx/prediction are both the alphabetical genre index (0..9), so the
   * matrix indices line up directly with `genres`' order -- no name lookup needed.
   */
  def computeMetrics(predictions: DataFrame, labelCol: String, predCol: String, genres: Seq[String]): MetricsResult = {
    val n = genres.size
    val confusion = Array.fill(n, n)(0L)
    predictions.groupBy(labelCol, predCol).count().collect().foreach { r =>
      val trueIdx = r.getAs[Double](labelCol).toInt
      val predIdx = r.getAs[Double](predCol).toInt
      confusion(trueIdx)(predIdx) = r.getAs[Long]("count")
    }

    val total   = confusion.map(_.sum).sum
    val correct = (0 until n).map(i => confusion(i)(i)).sum
    val accuracy = if (total > 0) correct.toDouble / total else 0.0

    val perClass = (0 until n).map { i =>
      val tp            = confusion(i)(i)
      val support       = confusion(i).sum                           // row sum: true count for class i
      val predictedAsI  = (0 until n).map(j => confusion(j)(i)).sum   // column sum: predicted count for class i
      val fp = predictedAsI - tp
      val fn = support - tp
      val precision = if (tp + fp > 0) tp.toDouble / (tp + fp) else 0.0
      val recall    = if (tp + fn > 0) tp.toDouble / (tp + fn) else 0.0
      val f1        = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0.0
      ClassMetric(genres(i), precision, recall, f1, support)
    }

    val macroF1    = perClass.map(_.f1).sum / n
    val weightedF1 = if (total > 0) perClass.map(c => c.f1 * c.support).sum / total else 0.0

    MetricsResult(accuracy, macroF1, weightedF1, perClass, confusion)
  }

  /**
   * One-time correctness check (not run for every model): compares the
   * hand-counted accuracy above against Spark's own MulticlassClassificationEvaluator,
   * on the same predictions. The two must agree (within floating-point rounding).
   */
  def sanityCheckAccuracy(predictions: DataFrame, labelCol: String, predCol: String): (Double, Double) = {
    val handComputed = predictions.filter(col(labelCol) === col(predCol)).count().toDouble / predictions.count()
    val sparkComputed = new MulticlassClassificationEvaluator()
      .setLabelCol(labelCol).setPredictionCol(predCol).setMetricName("accuracy")
      .evaluate(predictions)
    (handComputed, sparkComputed)
  }

  /** Prints and saves one model/level/set's metrics: summary line, per-class table, confusion matrix.
   *  `outputDir` defaults to Phase 5's own results/ml so every existing call site (MachineLearning.scala)
   *  is byte-for-byte unaffected; Phase 6's FollowUpExperiments passes results/ml_followup instead. */
  def reportMetrics(name: String, result: MetricsResult, genres: Seq[String], outputDir: String = Paths.MlOutputDir): Unit = {
    Report.log(s"\n--- $name ---")
    Report.log(f"accuracy=${result.accuracy}%.4f  macroF1=${result.macroF1}%.4f  weightedF1=${result.weightedF1}%.4f")

    val perClassHeader = Seq("genre", "precision", "recall", "f1", "support")
    val perClassRows = result.perClass.map(c =>
      Seq(c.genre, f"${c.precision}%.4f", f"${c.recall}%.4f", f"${c.f1}%.4f", c.support.toString))
    Report.table(perClassHeader, perClassRows)
    Report.saveCsv(s"per_class_$name.csv", perClassHeader, perClassRows, outputDir)

    val confHeader = "true_genre" +: genres
    val confRows = genres.zipWithIndex.map { case (g, i) => g +: result.confusion(i).map(_.toString).toSeq }
    Report.log(s"Confusion matrix ($name), rows = true genre, columns = predicted genre:")
    Report.table(confHeader, confRows)
    Report.saveCsv(s"confusion_$name.csv", confHeader, confRows, outputDir)
  }
}
