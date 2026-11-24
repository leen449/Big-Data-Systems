package sonicspark.ml

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Phase 6, Part B: one row per recording instead of one row per window.
 *
 * The PCA-whitened nearest-centroid baseline (Phase 5, Baselines.scala) works
 * on recording means and nearly matches the window-level models. This part
 * tests whether a real classifier, trained directly on recording-level summary
 * statistics (mean AND spread of each feature across a recording's windows),
 * does better than voting window-level predictions together after the fact.
 * No labels are used to build the summary features, so computing them for
 * test recordings up front is not leakage -- same reasoning as ExtraFeatures.scala.
 */
object FollowUpPartB {

  private val MeanSuffix = "_mean"
  private val StdSuffix  = "_std"

  case class RecordingTable(trainRec: DataFrame, testRec: DataFrame, featureSets: Map[String, Seq[String]])

  /** Builds the recording-level table: mean + sample stddev of each of the 38 raw window
   *  features, plus the 4 Phase 5 candidate features, one row per recording. */
  def buildRecordingTable(spark: SparkSession, splitFoldDf: DataFrame): RecordingTable = {
    val windows = DataIO.readStage(spark, Paths.Final)
    // Schema.windowFeatureColumns names the features as AudioFeatures.scala first extracts
    // them (e.g. "rms_var"), but Transformation.scala (Phase 2, stage 4) log-transforms and
    // renames every variance column (e.g. "rms_var" -> "log_rms_var") in data/processed/final.
    // Reading the real column names off the loaded DataFrame -- exactly how candidateBaseCols
    // is derived everywhere else in Phase 5/6 -- avoids hard-coding the pre-rename names.
    val idCols  = (Schema.windowIdColumns :+ "label_idx").toSet
    val rawCols = windows.columns.filterNot(idCols.contains).toSeq

    val meanAggs = rawCols.map(c => avg(col(c)).as(s"$c$MeanSuffix"))
    val stdAggs  = rawCols.map(c => stddev_samp(col(c)).as(s"$c$StdSuffix"))
    val aggExprs = Seq(first("label").as("label"), first("label_idx").as("label_idx")) ++ meanAggs ++ stdAggs
    val summary = windows.groupBy("track_id").agg(aggExprs.head, aggExprs.tail: _*)

    val rangeDf  = ExtraFeatures.robustDynamicRange(windows)
    val changeDf = ExtraFeatures.temporalChangeFeatures(windows)
    val withExtras = summary.join(rangeDf, "track_id").join(changeDf, "track_id")

    val withSplit = withExtras
      .join(splitFoldDf.select("track_id", "split", "fold"), "track_id")
      .withColumn("fold", col("fold").cast("int"))
      .withColumnRenamed("fold", MlConfig.FoldColName)

    val trainRec = withSplit.filter(col("split") === "train").drop("split").cache()
    val testRec  = withSplit.filter(col("split") === "test").drop("split")

    val s0 = rawCols.map(_ + MeanSuffix)                                   // 38
    val s1 = s0 ++ rawCols.map(_ + StdSuffix)                              // 76
    val s2 = s1 ++ (ExtraFeatures.RangeCol +: ExtraFeatures.ChangeCols)    // 80

    RecordingTable(trainRec, testRec, Map("S0" -> s0, "S1" -> s1, "S2" -> s2))
  }

  /** One row IS one recording here, so the grouped-OOF predictions are already
   *  recording-level -- no soft-vote step needed, unlike Part A's window models. */
  private def oofRecordingScore(trainRec: DataFrame, cols: Seq[String], classifier: PipelineStage): (DataFrame, Double, Double) = {
    val pipeline = ModelTraining.buildPipeline(cols, classifier)
    val oof = ModelTraining.runGroupedOof[PipelineModel](
      trainRec, MlConfig.FoldColName, MlConfig.NumFolds,
      fitFn = df => pipeline.fit(df), predictFn = (m, df) => m.transform(df))
    val result = MlMetrics.computeMetrics(oof, "label_idx", "prediction", MlConfig.Genres)
    (oof, result.macroF1, result.accuracy)
  }

  // ======================================================================
  // LR: elasticNetParam 0, maxIter 500, regParam in {0.001,0.01,0.1,1.0} x
  // {S0,S1,S2} = 12 candidates.
  // ======================================================================
  case class LrCandidate(featureSet: String, regParam: Double, cols: Seq[String], candidate: FollowUpCvUtils.Candidate)

  private def runLrGrid(trainRec: DataFrame, featureSets: Map[String, Seq[String]]): Seq[LrCandidate] =
    for {
      fsName   <- Seq("S0", "S1", "S2")
      regParam <- FollowUpConfig.SummaryLrRegParams
    } yield {
      val cols = featureSets(fsName)
      val lr = new LogisticRegression()
        .setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setElasticNetParam(FollowUpConfig.SummaryLrElasticNet).setMaxIter(FollowUpConfig.SummaryLrMaxIter)
        .setFamily("multinomial").setRegParam(regParam)
      val (_, macroF1, accuracy) = oofRecordingScore(trainRec, cols, lr)
      // Simpler = fewer features, then larger regParam (more regularisation) -- ascending tiebreak, so -regParam.
      LrCandidate(fsName, regParam, cols, FollowUpCvUtils.Candidate(s"LR $fsName regParam=$regParam", cols.size, -regParam, macroF1, accuracy))
    }

  // ======================================================================
  // RF: seed 42, numTrees in {200,500} x maxDepth in {5,10,15} x {S0,S1,S2} = 18 candidates.
  // ======================================================================
  case class RfCandidate(featureSet: String, numTrees: Int, maxDepth: Int, cols: Seq[String], candidate: FollowUpCvUtils.Candidate)

  private def runRfGrid(trainRec: DataFrame, featureSets: Map[String, Seq[String]]): Seq[RfCandidate] =
    for {
      fsName   <- Seq("S0", "S1", "S2")
      numTrees <- FollowUpConfig.SummaryRfNumTrees
      maxDepth <- FollowUpConfig.SummaryRfMaxDepth
    } yield {
      val cols = featureSets(fsName)
      val rf = new RandomForestClassifier()
        .setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setNumTrees(numTrees).setMaxDepth(maxDepth).setSeed(MlConfig.Seed)
      val (_, macroF1, accuracy) = oofRecordingScore(trainRec, cols, rf)
      // Simpler = fewer features, then fewer trees, then shallower depth.
      RfCandidate(fsName, numTrees, maxDepth, cols,
        FollowUpCvUtils.Candidate(s"RF $fsName numTrees=$numTrees,maxDepth=$maxDepth", cols.size, numTrees * 1000.0 + maxDepth, macroF1, accuracy))
    }

  /** Per-genre predicted-count column sums (every genre, not just the off-diagonal ones
   *  Interpretation.genrePairCheck reports) -- task asks for the full column-sum row. */
  private def predictedCountSums(result: MlMetrics.MetricsResult, genres: Seq[String]): Seq[Seq[String]] =
    genres.zipWithIndex.map { case (g, i) =>
      val colSum = (0 until genres.size).map(j => result.confusion(j)(i)).sum
      Seq(g, colSum.toString)
    }

  /** jazz->blues is in Phase 5 Table 8's genre-pair list but not in Interpretation.genrePairCheck
   *  (which is Phase 5's own, frozen file) -- added here instead of editing that file. */
  private def jazzToBluesCount(result: MlMetrics.MetricsResult, genres: Seq[String]): Long = {
    val idx = genres.zipWithIndex.toMap
    result.confusion(idx("jazz"))(idx("blues"))
  }

  private def flagIfSummaryOnly(feature: String): String =
    if (feature.endsWith(StdSuffix) || (ExtraFeatures.RangeCol +: ExtraFeatures.ChangeCols).contains(feature))
      " <-- summary-only feature (std or candidate), not available to the window models"
    else ""

  def run(spark: SparkSession, splitFoldDf: DataFrame): Unit = {
    Report.section("Part B: model trained on recording-level summaries")
    Report.log("FOLLOW-UP (post-hoc, test set previously seen in Phase 5).")
    Report.log("Selection rule (fixed in code before any OOF score is computed): highest OOF recording-level " +
      "macro-F1; ties (|delta| < " + FollowUpConfig.TieEpsilon + ") broken by accuracy, then by the simpler " +
      "configuration (fewer features, larger regParam for LR / fewer then shallower trees for RF). " +
      "Each model picks its own best feature set and hyperparameters independently.")
    Report.log("Feature sets: S0 = 38 means; S1 = S0 + 38 stddevs (76); S2 = S1 + 4 Phase 5 candidate features (80).")

    val table = buildRecordingTable(spark, splitFoldDf)

    // ---------- search ----------
    val lrCandidates = runLrGrid(table.trainRec, table.featureSets)
    val rfCandidates = runRfGrid(table.trainRec, table.featureSets)

    val cvHeader = Seq("model", "feature_set", "config", "oof_macroF1", "oof_accuracy")
    val lrCvRows = lrCandidates.map(c => Seq("LR", c.featureSet, f"regParam=${c.regParam}%.4f", f"${c.candidate.macroF1}%.4f", f"${c.candidate.accuracy}%.4f"))
    val rfCvRows = rfCandidates.map(c => Seq("RF", c.featureSet, s"numTrees=${c.numTrees},maxDepth=${c.maxDepth}", f"${c.candidate.macroF1}%.4f", f"${c.candidate.accuracy}%.4f"))
    Report.log("\n--- Part B CV table (12 LR + 18 RF candidates, grouped 5-fold OOF on 688 training recordings) ---")
    Report.table(cvHeader, lrCvRows ++ rfCvRows)
    Report.saveCsv("summary_cv_results.csv", cvHeader, lrCvRows ++ rfCvRows, Paths.MlFollowupOutputDir)

    val lrWinner = FollowUpCvUtils.bestByNewRule(lrCandidates, (c: LrCandidate) => c.candidate)
    val rfWinner = FollowUpCvUtils.bestByNewRule(rfCandidates, (c: RfCandidate) => c.candidate)
    Report.log(f"Selected LR: feature set ${lrWinner.featureSet}, regParam=${lrWinner.regParam}%.4f " +
      f"(OOF macroF1=${lrWinner.candidate.macroF1}%.4f, accuracy=${lrWinner.candidate.accuracy}%.4f)")
    Report.log(s"Selected RF: feature set ${rfWinner.featureSet}, numTrees=${rfWinner.numTrees}, maxDepth=${rfWinner.maxDepth} " +
      f"(OOF macroF1=${rfWinner.candidate.macroF1}%.4f, accuracy=${rfWinner.candidate.accuracy}%.4f)")

    // ---------- refit + full evaluation, each selected model once ----------
    evaluateSelected("SummaryLR", lrWinner.cols,
      new LogisticRegression().setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setElasticNetParam(FollowUpConfig.SummaryLrElasticNet).setMaxIter(FollowUpConfig.SummaryLrMaxIter)
        .setFamily("multinomial").setRegParam(lrWinner.regParam),
      table.trainRec, table.testRec,
      interpret = model => reportLrInterpretation(model.asInstanceOf[LogisticRegressionModel], lrWinner.cols))

    evaluateSelected("SummaryRF", rfWinner.cols,
      new RandomForestClassifier().setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setNumTrees(rfWinner.numTrees).setMaxDepth(rfWinner.maxDepth).setSeed(MlConfig.Seed),
      table.trainRec, table.testRec,
      interpret = model => reportRfInterpretation(model.asInstanceOf[RandomForestClassificationModel], rfWinner.cols))

    table.trainRec.unpersist()
  }

  /** Refits `classifier` on all training recordings, reports train/OOF/test metrics (the full
   *  per-genre table, confusion matrix, genre-pair counts and column sums task Part B asks for),
   *  and saves the test-set recording predictions for Part C to reuse. */
  private def evaluateSelected(
    modelLabel: String,
    cols: Seq[String],
    classifier: PipelineStage,
    trainRec: DataFrame,
    testRec: DataFrame,
    interpret: Any => Unit
  ): Unit = {
    val pipeline = ModelTraining.buildPipeline(cols, classifier)
    val fitted = pipeline.fit(trainRec)

    val trainPred = fitted.transform(trainRec)
    val oofPred   = ModelTraining.runGroupedOof[PipelineModel](
      trainRec, MlConfig.FoldColName, MlConfig.NumFolds,
      fitFn = df => pipeline.fit(df), predictFn = (m, df) => m.transform(df))
    val testPred  = fitted.transform(testRec)

    val trainResult = MlMetrics.computeMetrics(trainPred, "label_idx", "prediction", MlConfig.Genres)
    val oofResult    = MlMetrics.computeMetrics(oofPred, "label_idx", "prediction", MlConfig.Genres)
    val testResult   = MlMetrics.computeMetrics(testPred, "label_idx", "prediction", MlConfig.Genres)

    MlMetrics.reportMetrics(s"${modelLabel}_train", trainResult, MlConfig.Genres, Paths.MlFollowupOutputDir)
    MlMetrics.reportMetrics(s"${modelLabel}_oof", oofResult, MlConfig.Genres, Paths.MlFollowupOutputDir)
    MlMetrics.reportMetrics(s"${modelLabel}_test", testResult, MlConfig.Genres, Paths.MlFollowupOutputDir)

    val pairRows = Interpretation.genrePairCheck(testResult, MlConfig.Genres, "recording") :+
      Seq("recording", "jazz->blues", jazzToBluesCount(testResult, MlConfig.Genres).toString)
    Report.log(s"\n--- $modelLabel: genre-pair check (test set, recording level) ---")
    Report.table(Seq("level", "check", "count"), pairRows)
    Report.saveCsv(s"${modelLabel}_genre_pair_check.csv", Seq("level", "check", "count"), pairRows, Paths.MlFollowupOutputDir)

    val colSumRows = predictedCountSums(testResult, MlConfig.Genres)
    Report.log(s"\n--- $modelLabel: predicted-count column sums (test set) ---")
    Report.table(Seq("genre", "predicted_count"), colSumRows)
    Report.saveCsv(s"${modelLabel}_column_sums.csv", Seq("genre", "predicted_count"), colSumRows, Paths.MlFollowupOutputDir)

    interpret(fitted.stages.last)

    val predictionRows = testPred.select("track_id", "label_idx", "prediction").collect().map { r =>
      Seq(r.getAs[String]("track_id"), MlConfig.Genres(r.getAs[Double]("label_idx").toInt), modelLabel, MlConfig.Genres(r.getAs[Double]("prediction").toInt))
    }.toSeq
    Report.saveCsv(s"${modelLabel.toLowerCase}_test_predictions.csv",
      Seq("track_id", "label", "model", "prediction"), predictionRows, Paths.MlFollowupOutputDir)
  }

  private def reportLrInterpretation(model: LogisticRegressionModel, cols: Seq[String]): Unit = {
    val fullRows = Interpretation.lrCoefficients(model, cols, MlConfig.Genres)
    val topRows = Interpretation.topLrCoefficients(fullRows, MlConfig.Genres, FollowUpConfig.SummaryLrTopCoefficients)
    Report.log(s"\n--- SummaryLR: top ${FollowUpConfig.SummaryLrTopCoefficients} +/- standardised coefficients per genre ---")
    topRows.foreach { row =>
      val feature = row(2)
      Report.log(row.mkString(" | ") + flagIfSummaryOnly(feature))
    }
    Report.saveCsv("summaryLR_top_coefficients.csv", Seq("genre", "direction", "feature", "coefficient"), topRows, Paths.MlFollowupOutputDir)
  }

  private def reportRfInterpretation(model: RandomForestClassificationModel, cols: Seq[String]): Unit = {
    val ranking = Interpretation.rfFeatureImportance(model, cols).take(FollowUpConfig.SummaryRfTopImportances)
    Report.log(s"\n--- SummaryRF: top ${FollowUpConfig.SummaryRfTopImportances} feature importances ---")
    val rows = ranking.map { case (f, imp) => Seq(f, f"$imp%.6f") }
    rows.foreach(row => Report.log(row.mkString(" | ") + flagIfSummaryOnly(row.head)))
    Report.saveCsv("summaryRF_top_importances.csv", Seq("feature", "importance"), rows, Paths.MlFollowupOutputDir)
  }
}
