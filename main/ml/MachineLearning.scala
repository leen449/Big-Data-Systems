package sonicspark.ml

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import sonicspark.common._

/**
 * Phase 5 - Machine Learning (SonicSpark, IT 462 Big Data Systems)
 *
 * Trains and evaluates a genre classifier on top of Phase 2's final window-level
 * table, against a majority-class baseline and a recording-level nearest-centroid
 * baseline (Phase 4's Q6, redone train-only). The whole point of every stage below
 * is to never let a test recording, or even a training recording held out for a
 * CV fold, influence anything it is later evaluated on -- see the comment at the
 * top of each stage's own file for how that stage specifically avoids it.
 *
 * Input : data/processed/final        (58,020 windows, 984 recordings, 38 features)
 *         data/interim/02_integrated  (train recordings only, for the Phase 2 recheck)
 * Output: results/ml_metrics.txt      (full printed report)
 *         results/ml (split.csv, relevance, CV, metrics, confusion, etc.)
 *
 * Run:
 *   sbt "runMain sonicspark.ml.MachineLearning"
 */
object MachineLearning {

  def main(args: Array[String]): Unit = {
    val overallStart = System.nanoTime()

    // A fixed local[N] master and a fixed shuffle-partition count (instead of
    // common.Spark.session's local[*]) so that running this program twice and
    // diffing the outputs is actually a test of determinism, not an artifact
    // of however many cores happened to be free on this run.
    val spark = SparkSession.builder()
      .appName("SonicSpark-MachineLearning")
      .master(MlConfig.SparkMaster)
      .config("spark.sql.shuffle.partitions", MlConfig.ShufflePartitions)
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    Report.start(Paths.MlReport)
    Report.section("SonicSpark Phase 5: Machine Learning")
    Report.log(s"Inputs read: ${Paths.Final} (58,020 windows, 984 recordings) for every modelling step; " +
      s"${Paths.Integrated} (filtered to training recordings only) for the Phase 2 selection recheck.")
    Report.log(s"Spark master=${MlConfig.SparkMaster}, shuffle.partitions=${MlConfig.ShufflePartitions}, seed=${MlConfig.Seed}.")

    val windows = DataIO.readStage(spark, Paths.Final)
    val idCols  = (Schema.windowIdColumns :+ "label_idx").toSet
    val candidateBaseCols = windows.columns.filterNot(idCols.contains).toSeq

    // ---------- Stage: extra features (computed once, for every recording) ----------
    // Unsupervised and strictly per-recording, so this can safely run before the
    // split even exists -- see ExtraFeatures.scala for why that is safe.
    val rangeDf  = ExtraFeatures.robustDynamicRange(windows)
    val changeDf = ExtraFeatures.temporalChangeFeatures(windows)
    val windowsWithExtras = ExtraFeatures.attachExtraFeatures(windows, rangeDf, changeDf)

    // ---------- Stage: split ----------
    Report.section("Split")
    val splitResult = Split.run(spark, windowsWithExtras)
    val trainDf = splitResult.trainWindows.withColumnRenamed("fold", MlConfig.FoldColName).cache()
    val testDf  = splitResult.testWindows.drop("fold")

    Report.log(s"Gap pairs excluded from the change features (window_idx jump != 1): " +
      s"train=${ExtraFeatures.countGapPairs(splitResult.trainWindows)}, " +
      s"test=${ExtraFeatures.countGapPairs(splitResult.testWindows)} " +
      "(Phase 4 found 5 across the full 58,020-window dataset).")

    // ---------- Stage: train-only feature analysis ----------
    Report.section("Train-only feature analysis")
    val analysis = FeatureAnalysis.analyzeTrainFeatures(trainDf, candidateBaseCols)
    val extraEta2 = FeatureAnalysis.etaSquaredForExtraFeatures(trainDf, ExtraFeatures.RangeCol +: ExtraFeatures.ChangeCols)
    FeatureAnalysis.reportAndSave(analysis, extraEta2)
    val baseCols = analysis.keptBaseFeatures   // this -- not the raw 38 -- is F0

    FeatureAnalysis.recheckPhase2Selection(spark, trainDf.select("track_id").distinct())

    // ---------- Stage: cross-validation (feature-set + hyperparameter search) ----------
    Report.section("Cross-validation: feature-set and hyperparameter search")
    Report.log("Rule (fixed in advance, before any CV score is seen): each model selects the feature " +
      "set (F0/F1/F2/F3) with its own best grouped-CV score; if LR and RF select different feature " +
      "sets, both are kept and reported separately throughout evaluation below.")
    Report.log(s"Limitation (resource constraint, decided before any test-set evaluation): the RF grid " +
      s"was trimmed from the originally planned numTrees={100,200} x maxDepth={10,15} to " +
      s"numTrees=${MlConfig.RfNumTrees.mkString("{", ",", "}")} x maxDepth=${MlConfig.RfMaxDepth.mkString("{", ",", "}")} " +
      "after the 200-tree/depth-15 configuration repeatedly caused OutOfMemoryError crashes during the " +
      "grouped out-of-fold loop on this machine's ~12GB RAM in local mode. This is a hardware limit, not a modelling choice.")

    val stage1Start = System.nanoTime()
    val stage1 = ModelTraining.stage1Search(trainDf, baseCols)
    val stage1Seconds = (System.nanoTime() - stage1Start) / 1e9
    Report.log(f"Stage 1 (LR 3-config grid + 1 fixed RF config, x4 feature sets, 5-fold grouped CV) wall time: $stage1Seconds%.1f s")
    Report.log(s"LR's winning feature set: ${stage1.lrBestVariant}")
    Report.log(s"RF's winning feature set (from the fixed stage-1 config, tuned further in stage 2): ${stage1.rfWinningVariantFromFixed}")
    if (stage1.lrBestVariant != stage1.rfWinningVariantFromFixed)
      Report.log("LR and RF selected DIFFERENT feature sets -- both are carried through separately below.")
    else
      Report.log("LR and RF selected the same feature set.")

    val stage2Start = System.nanoTime()
    val stage2 = ModelTraining.stage2TuneRf(trainDf, baseCols, stage1.rfWinningVariantFromFixed)
    val stage2Seconds = (System.nanoTime() - stage2Start) / 1e9
    Report.log(f"Stage 2 (RF numTrees x maxDepth grid on RF's winning feature set, 5-fold grouped CV) wall time: $stage2Seconds%.1f s")

    val cvHeader = Seq("model", "feature_set", "params", "cv_score_weighted_f1")
    val cvRows = stage1.cvResultRows ++ stage2.cvResultRows
    Report.log("\n--- CV results (all feature sets x configs) ---")
    Report.table(cvHeader, cvRows)
    Report.saveCsv("cv_results.csv", cvHeader, cvRows, Paths.MlOutputDir)

    val lrColumns = ExtraFeatures.featureSet(stage1.lrBestVariant, baseCols)
    val rfColumns = ExtraFeatures.featureSet(stage1.rfWinningVariantFromFixed, baseCols)

    // ---------- Stage: final fit ----------
    // CrossValidator already refits the winning configuration on the ENTIRE
    // training set as part of .fit() -- reused directly here instead of fitting
    // a third time, so "the final model" and "the model CV selected" are
    // literally the same object, not two pipelines that could silently drift.
    Report.section("Final fit")
    val lrFinalModel = stage1.lrBestModel.bestModel
    val rfFinalModel = stage2.rfBestModel.bestModel
    Report.log("LR and RF final models are CrossValidator's own best-configuration refit on all training recordings.")

    val majorityFinal      = Baselines.MajorityBaseline.fit(trainDf)
    val centroidPlainFinal = Baselines.CentroidModel.fitPlain(trainDf, baseCols)
    val centroidPcaFinal   = Baselines.CentroidModel.fitPcaWhitened(trainDf, baseCols, MlConfig.PcaVarianceThreshold)

    // ---------- Genuine out-of-fold predictions (section 6's "grouped-CV out-of-fold on train") ----------
    val lrOofPipeline = ModelTraining.finalLrPipeline(stage1.lrBestModel, lrColumns)
    val rfOofPipeline = ModelTraining.finalRfPipeline(stage2.rfBestModel, rfColumns)

    val lrOof = ModelTraining.runGroupedOof[PipelineModel](
      trainDf, MlConfig.FoldColName, MlConfig.NumFolds,
      fitFn = df => lrOofPipeline.fit(df), predictFn = (m, df) => m.transform(df))
    val rfOof = ModelTraining.runGroupedOof[PipelineModel](
      trainDf, MlConfig.FoldColName, MlConfig.NumFolds,
      fitFn = df => rfOofPipeline.fit(df), predictFn = (m, df) => m.transform(df))
    val majorityOof = ModelTraining.runGroupedOof[Baselines.MajorityModel](
      trainDf, MlConfig.FoldColName, MlConfig.NumFolds,
      fitFn = df => Baselines.MajorityBaseline.fit(df), predictFn = (m, df) => Baselines.MajorityBaseline.predict(m, df))
    val centroidPlainOof = ModelTraining.runGroupedOof[Baselines.CentroidModel](
      trainDf, MlConfig.FoldColName, MlConfig.NumFolds,
      fitFn = df => Baselines.CentroidModel.fitPlain(df, baseCols),
      predictFn = (m, df) => Baselines.CentroidModel.predict(m, df))
    val centroidPcaOof = ModelTraining.runGroupedOof[Baselines.CentroidModel](
      trainDf, MlConfig.FoldColName, MlConfig.NumFolds,
      fitFn = df => Baselines.CentroidModel.fitPcaWhitened(df, baseCols, MlConfig.PcaVarianceThreshold),
      predictFn = (m, df) => Baselines.CentroidModel.predict(m, df))

    // ---------- In-sample (train) and once-only (test) predictions ----------
    val lrTrainPred = lrFinalModel.transform(trainDf)
    val rfTrainPred = rfFinalModel.transform(trainDf)
    val majorityTrainPred      = Baselines.MajorityBaseline.predict(majorityFinal, trainDf)
    val centroidPlainTrainPred = Baselines.CentroidModel.predict(centroidPlainFinal, trainDf)
    val centroidPcaTrainPred   = Baselines.CentroidModel.predict(centroidPcaFinal, trainDf)

    val lrTestPred = lrFinalModel.transform(testDf)
    val rfTestPred = rfFinalModel.transform(testDf)
    val majorityTestPred      = Baselines.MajorityBaseline.predict(majorityFinal, testDf)
    val centroidPlainTestPred = Baselines.CentroidModel.predict(centroidPlainFinal, testDf)
    val centroidPcaTestPred   = Baselines.CentroidModel.predict(centroidPcaFinal, testDf)

    // ---------- Stage: evaluation ----------
    Report.section("Evaluation")
    val modelPredictionsBySet: Seq[(String, String, DataFrame)] = Seq(
      ("LR", "train", lrTrainPred), ("LR", "oof", lrOof), ("LR", "test", lrTestPred),
      ("RF", "train", rfTrainPred), ("RF", "oof", rfOof), ("RF", "test", rfTestPred),
      ("Majority", "train", majorityTrainPred), ("Majority", "oof", majorityOof), ("Majority", "test", majorityTestPred),
      ("CentroidPlain", "train", centroidPlainTrainPred), ("CentroidPlain", "oof", centroidPlainOof), ("CentroidPlain", "test", centroidPlainTestPred),
      ("CentroidPCA", "train", centroidPcaTrainPred), ("CentroidPCA", "oof", centroidPcaOof), ("CentroidPCA", "test", centroidPcaTestPred)
    )

    val allResults = scala.collection.mutable.Map[String, MlMetrics.MetricsResult]()
    val summaryRows = scala.collection.mutable.ArrayBuffer[Seq[String]]()

    modelPredictionsBySet.foreach { case (model, set, windowPreds) =>
      val windowResult = MlMetrics.computeMetrics(windowPreds, "label_idx", "prediction", MlConfig.Genres)
      val name1 = s"${model}_window_$set"
      MlMetrics.reportMetrics(name1, windowResult, MlConfig.Genres)
      allResults(name1) = windowResult
      summaryRows += Seq(model, "window", set, f"${windowResult.accuracy}%.4f", f"${windowResult.macroF1}%.4f", f"${windowResult.weightedF1}%.4f")

      val softVoted = MlMetrics.softVoteToRecording(windowPreds)
      val softResult = MlMetrics.computeMetrics(softVoted, "label_idx", "prediction", MlConfig.Genres)
      val name2 = s"${model}_recordingSoft_$set"
      MlMetrics.reportMetrics(name2, softResult, MlConfig.Genres)
      allResults(name2) = softResult
      summaryRows += Seq(model, "recordingSoft", set, f"${softResult.accuracy}%.4f", f"${softResult.macroF1}%.4f", f"${softResult.weightedF1}%.4f")

      val majorityVoted = MlMetrics.majorityVoteToRecording(windowPreds)
      val majorityResult = MlMetrics.computeMetrics(majorityVoted, "label_idx", "prediction", MlConfig.Genres)
      val name3 = s"${model}_recordingMajority_$set"
      MlMetrics.reportMetrics(name3, majorityResult, MlConfig.Genres)
      allResults(name3) = majorityResult
      summaryRows += Seq(model, "recordingMajority", set, f"${majorityResult.accuracy}%.4f", f"${majorityResult.macroF1}%.4f", f"${majorityResult.weightedF1}%.4f")
    }

    val summaryHeader = Seq("model", "level", "set", "accuracy", "macro_f1", "weighted_f1")
    Report.log("\n--- Results summary (model x level x set) ---")
    Report.table(summaryHeader, summaryRows.toSeq)
    Report.saveCsv("metrics_summary.csv", summaryHeader, summaryRows.toSeq, Paths.MlOutputDir)

    // Metal class, called out explicitly, on the headline set/level (test, recording-level soft voting)
    Report.log("\n--- Metal class, recording-level soft voting, test set ---")
    val metalIdx = MlConfig.Genres.indexOf("metal")
    Seq("LR", "RF", "Majority", "CentroidPlain", "CentroidPCA").foreach { model =>
      val r = allResults(s"${model}_recordingSoft_test")
      val metal = r.perClass(metalIdx)
      val avgOtherF1 = (r.perClass.map(_.f1).sum - metal.f1) / (r.perClass.size - 1)
      Report.log(f"$model: metal recall=${metal.recall}%.4f f1=${metal.f1}%.4f (other genres' average f1=$avgOtherF1%.4f)")
    }

    // Sanity check: hand-computed accuracy against Spark's own evaluator, for one model/set.
    val (handAcc, sparkAcc) = MlMetrics.sanityCheckAccuracy(lrTestPred, "label_idx", "prediction")
    Report.log(f"\nSanity check (LR, window-level, test set): hand-computed accuracy=$handAcc%.6f, " +
      f"MulticlassClassificationEvaluator accuracy=$sparkAcc%.6f, match=${math.abs(handAcc - sparkAcc) < 1e-9}")

    // ---------- Stage: interpretation ----------
    Report.section("Interpretation")
    val lrModelStage = lrFinalModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]
    val rfModelStage = rfFinalModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[RandomForestClassificationModel]
    Interpretation.reportFeatureImportance(rfModelStage, rfColumns)
    Interpretation.reportLrCoefficients(lrModelStage, lrColumns, MlConfig.Genres)

    val genrePairRows = Seq("LR", "RF", "Majority", "CentroidPlain", "CentroidPCA").flatMap { model =>
      val windowResult    = allResults(s"${model}_window_test")
      val recordingResult = allResults(s"${model}_recordingSoft_test")
      (Interpretation.genrePairCheck(windowResult, MlConfig.Genres, "window") ++
        Interpretation.genrePairCheck(recordingResult, MlConfig.Genres, "recording")).map(row => model +: row)
    }
    Report.log("\n--- Genre-pair check (test set, all models/baselines) ---")
    Report.table(Seq("model", "level", "check", "count"), genrePairRows)
    Report.saveCsv("genre_pair_check.csv", Seq("model", "level", "check", "count"), genrePairRows, Paths.MlOutputDir)

    Interpretation.reportCentroidColumnSums(
      centroidPlainTestPred.dropDuplicates("track_id"),
      centroidPcaTestPred.dropDuplicates("track_id"),
      MlConfig.Genres)

    trainDf.unpersist()

    val totalSeconds = (System.nanoTime() - overallStart) / 1e9
    Report.log(f"\nTotal run time: $totalSeconds%.1f s")
    Report.log(s"Full report saved to: ${Paths.MlReport}")
    Report.log(s"CSV outputs saved under: ${Paths.MlOutputDir}")

    spark.stop()
  }
}
