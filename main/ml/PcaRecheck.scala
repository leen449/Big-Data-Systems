package sonicspark.ml

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths => JPaths}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Audit utility: the PCA-whitened centroid baseline's `k` (number of retained principal
 * components) is computed inside Baselines.scala's fitInternal but never logged anywhere
 * -- not in results/ml_metrics.txt, not in any CSV. This object rebuilds the exact inputs
 * MachineLearning.scala used for that baseline (same table, same split, same feature set,
 * same seed/settings), calls Baselines.CentroidModel.fitPcaWhitened UNCHANGED (no
 * reimplementation, no edit to Baselines.scala -- CentroidModel's `pca: Option[PCAModel]`
 * field is already public, so `.getK` and `.explainedVariance` are readable directly),
 * and reports k plus a proof that the refit model reproduces the original run's test
 * numbers exactly.
 *
 * Input : data/processed/final  (Paths.Final -- same table MachineLearning.scala reads)
 *         results/ml/split.csv  (train/test assignment AND cv_fold, from the original run)
 * Output: results/ml/pca_components.txt
 *
 * Run:
 *   sbt "runMain sonicspark.ml.PcaRecheck"
 */
object PcaRecheck {

  // Original run's reported numbers (results/ml/metrics_summary.csv CentroidPCA/test,
  // results/ml/centroid_column_sums.csv variant=pca_whitened), to prove this is the same model.
  private val ExpectedAccuracy = 0.6453
  private val ExpectedCounts: Map[String, Long] = Map(
    "blues" -> 55, "classical" -> 28, "country" -> 27, "disco" -> 30, "hiphop" -> 23,
    "jazz" -> 21, "metal" -> 28, "pop" -> 32, "reggae" -> 28, "rock" -> 24
  )

  private def predictedCountsPerGenre(predictions: DataFrame, genres: Seq[String]): Map[String, Long] = {
    val counts = predictions.groupBy("prediction").count().collect()
      .map(r => (r.getAs[Double]("prediction").toInt, r.getAs[Long]("count"))).toMap
    genres.zipWithIndex.map { case (g, i) => g -> counts.getOrElse(i, 0L) }.toMap
  }

  def main(args: Array[String]): Unit = {
    // Same master/shuffle-partitions/seed as the original run (MlConfig.scala), so this
    // is not just "a" PCA fit but the SAME one MachineLearning.scala would produce.
    val spark = SparkSession.builder()
      .appName("SonicSpark-PcaRecheck")
      .master(MlConfig.SparkMaster)
      .config("spark.sql.shuffle.partitions", MlConfig.ShufflePartitions)
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    // ---------- Same table as MachineLearning.scala ----------
    val windows = DataIO.readStage(spark, Paths.Final)
    val idCols  = (Schema.windowIdColumns :+ "label_idx").toSet
    val candidateBaseCols = windows.columns.filterNot(idCols.contains).toSeq

    // ---------- Same split, read from the original run's results/ml/split.csv ----------
    val splitDf  = spark.read.option("header", "true").csv("results/ml/split.csv")
    val trainIds = splitDf.filter(col("split") === "train").select("track_id")
    val testIds  = splitDf.filter(col("split") === "test").select("track_id")

    val trainWindows = windows.join(trainIds, "track_id")
    val testWindows  = windows.join(testIds, "track_id")

    // ---------- Same feature set the centroid baselines used: F0, derived the same way ----------
    // MachineLearning.scala never passes F1/F2/F3 (the +range/+change variants) to the
    // centroid baselines -- Baselines.scala's own header comment says both baselines
    // "use only the base feature set F0". F0 is whatever FeatureAnalysis.analyzeTrainFeatures
    // (existing, unmodified) keeps after the train-only eta^2/correlation pass -- reused
    // here exactly, not assumed, so this recheck can't silently drift from a changed F0.
    val analysis = FeatureAnalysis.analyzeTrainFeatures(trainWindows, candidateBaseCols)
    val baseCols = analysis.keptBaseFeatures

    val lines = scala.collection.mutable.ArrayBuffer[String]()
    lines += "PCA-whitened centroid baseline: k recheck (Baselines.scala never logs k)"
    lines += s"Input: ${Paths.Final}, split from results/ml/split.csv (688 train / 296 test recordings)."
    lines += s"Feature set used by the centroid baselines: F0 (base feature set only -- Baselines.scala " +
      s"never receives the +range/+change variants used by LR/RF). F0 = ${baseCols.size} features " +
      "(FeatureAnalysis.analyzeTrainFeatures, unchanged, applied to the same candidate columns MachineLearning.scala uses)."
    lines += s"Settings: master=${MlConfig.SparkMaster}, shuffle.partitions=${MlConfig.ShufflePartitions}, " +
      s"seed=${MlConfig.Seed}, PCA variance threshold=${MlConfig.PcaVarianceThreshold}."
    lines += ""

    // ---------- Call the EXISTING Baselines.CentroidModel.fitPcaWhitened, unmodified ----------
    val model = Baselines.CentroidModel.fitPcaWhitened(trainWindows, baseCols, MlConfig.PcaVarianceThreshold)
    val pcaModel = model.pca.getOrElse(throw new IllegalStateException("fitPcaWhitened returned no PCA model"))
    val k = pcaModel.getK
    val perComponent = pcaModel.explainedVariance.toArray   // length k, already the chosen model's own components
    val cumulative = perComponent.scanLeft(0.0)(_ + _).tail // cumulative(i) = sum of components 0..i

    lines += s"Number of input features to PCA: ${baseCols.size}"
    lines += s"k (number of retained principal components): $k"
    lines += f"Cumulative explained variance at k-1 (${k - 1} components): ${if (k >= 2) cumulative(k - 2) else 0.0}%.6f"
    lines += f"Cumulative explained variance at k (${k} components):   ${cumulative(k - 1)}%.6f"
    lines += ""
    lines += "Explained-variance share of each of the first k components:"
    perComponent.zipWithIndex.foreach { case (v, i) =>
      lines += f"  component ${i + 1}: ${v}%.6f  (cumulative ${cumulative(i)}%.6f)"
    }
    lines += ""

    // ---------- Proof: predict the test recordings with this refit model ----------
    // Same functions MachineLearning.scala itself calls (Baselines.CentroidModel.predict,
    // MlMetrics.softVoteToRecording, MlMetrics.computeMetrics) -- no reimplementation.
    val testWindowPred = Baselines.CentroidModel.predict(model, testWindows)
    val testSoftVoted  = MlMetrics.softVoteToRecording(testWindowPred)
    val testResult     = MlMetrics.computeMetrics(testSoftVoted, "label_idx", "prediction", MlConfig.Genres)

    val testRecordingPred = testWindowPred.dropDuplicates("track_id")
    val actualCounts = predictedCountsPerGenre(testRecordingPred, MlConfig.Genres)

    lines += "--- Proof: refit model vs. the original run's reported numbers ---"
    lines += f"Test accuracy (recording-level, soft voting): this run = ${testResult.accuracy}%.4f, original run = $ExpectedAccuracy%.4f"
    val accuracyMatches = f"${testResult.accuracy}%.4f" == f"$ExpectedAccuracy%.4f"

    lines += "Per-genre predicted counts (test, recording level):"
    val countDiffs = scala.collection.mutable.ArrayBuffer[String]()
    MlConfig.Genres.foreach { g =>
      val actual = actualCounts.getOrElse(g, 0L)
      val expected = ExpectedCounts(g)
      val mark = if (actual == expected) "" else "  <-- MISMATCH"
      lines += s"  $g: this run = $actual, original run = $expected$mark"
      if (actual != expected) countDiffs += s"$g (this=$actual, original=$expected)"
    }

    val overallMatch = accuracyMatches && countDiffs.isEmpty
    lines += ""
    if (overallMatch) {
      lines += "MATCH -- this refit of the PCA-whitened centroid baseline reproduces the original run's " +
        "test accuracy and per-genre predicted counts exactly. k reported above is the real value used by " +
        "the original run."
    } else {
      val diffParts = scala.collection.mutable.ArrayBuffer[String]()
      if (!accuracyMatches) diffParts += f"accuracy (this=${testResult.accuracy}%.4f, original=$ExpectedAccuracy%.4f)"
      diffParts ++= countDiffs
      lines += s"MISMATCH -- differences: ${diffParts.mkString("; ")}. Do not treat k above as confirmed to be " +
        "the same model as the original run until this is reviewed."
    }

    // ---------- Optional: k per out-of-fold model (fold != i), from split.csv's cv_fold ----------
    lines += ""
    lines += "--- Optional: k for each of the 5 grouped out-of-fold PCA-whitened centroid fits ---"
    lines += "(fit on cv_fold != i, i.e. the same 4-fold training subsets ModelTraining.runGroupedOof uses)"
    val trainWithFold = trainWindows.join(splitDf.select("track_id", "fold"), "track_id")
    (0 until MlConfig.NumFolds).foreach { fold =>
      val foldTrain = trainWithFold.filter(col("fold") =!= fold.toString)
      val foldModel = Baselines.CentroidModel.fitPcaWhitened(foldTrain, baseCols, MlConfig.PcaVarianceThreshold)
      val foldK = foldModel.pca.get.getK
      lines += s"  fold $fold held out: k = $foldK"
    }

    val path = JPaths.get(Paths.MlOutputDir).resolve("pca_components.txt")
    Files.createDirectories(path.getParent)
    Files.write(path, lines.mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

    lines.foreach(println)
    println(s"\nSaved to: $path")

    spark.stop()
  }
}
