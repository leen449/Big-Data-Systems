package sonicspark.ml

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.sql.DataFrame
import sonicspark.common._

/**
 * Phase 6, Part A: re-rank Phase 5's CV candidates by recording-level OOF score
 * instead of Spark's window-level weighted-F1.
 *
 * Phase 5's CrossValidator.avgMetrics only ever scores window rows (section 5's
 * known limitation), because CrossValidator has no concept of "vote the windows
 * of a recording together" -- it only sees whatever row-level evaluator it is
 * given. This part redoes the same 20 candidate configurations with the manual
 * grouped-OOF loop (ModelTraining.runGroupedOof, already used for the FINAL
 * configs in MachineLearning.scala) so each candidate gets real held-out
 * predictions, which can then be soft-voted to one prediction per recording
 * and scored with MlMetrics' hand-counted macro-F1 -- the metric the headline
 * numbers are actually judged on.
 */
object FollowUpPartA {

  /** Fits the given classifier 5 times (one fold held out each time), soft-votes
   *  the pooled out-of-fold window predictions to recording level, and scores them.
   *  This is the one function this whole part is built around. */
  private def oofRecordingScore(trainDf: DataFrame, featureCols: Seq[String], classifier: PipelineStage): (Double, Double) = {
    val pipeline = ModelTraining.buildPipeline(featureCols, classifier)
    val oofWindowPreds = ModelTraining.runGroupedOof[PipelineModel](
      trainDf, MlConfig.FoldColName, MlConfig.NumFolds,
      fitFn = df => pipeline.fit(df), predictFn = (m, df) => m.transform(df))
    val oofRecordingPreds = MlMetrics.softVoteToRecording(oofWindowPreds)
    val result = MlMetrics.computeMetrics(oofRecordingPreds, "label_idx", "prediction", MlConfig.Genres)
    (result.macroF1, result.accuracy)
  }

  private def lookupWeightedF1(cvRows: Seq[Map[String, String]], model: String, featureSet: String, params: String): Option[Double] =
    cvRows.find(r => r("model") == model && r("feature_set") == featureSet && r("params") == params)
      .map(_("cv_score_weighted_f1").toDouble)

  /** Best-first rank (1 = best) under Phase 5's original rule: highest window_weightedF1. */
  private def ranksByOld[T](items: Seq[T], oldScore: T => Double): Map[T, Int] =
    items.sortBy(t => -oldScore(t)).zipWithIndex.map { case (t, i) => t -> (i + 1) }.toMap

  /** Best-first rank (1 = best) under this part's fixed new rule (FollowUpCvUtils.isBetter). */
  private def ranksByNew[T](items: Seq[T], candidateOf: T => FollowUpCvUtils.Candidate): Map[T, Int] =
    items.sortWith((a, b) => FollowUpCvUtils.isBetter(candidateOf(a), candidateOf(b))).zipWithIndex.map { case (t, i) => t -> (i + 1) }.toMap

  // ======================================================================
  // LR: 3 regParams x 4 feature sets = 12 candidates, same grid Phase 5 searched.
  // ======================================================================
  private case class LrCandidate(variant: String, regParam: Double, cols: Seq[String], windowWeightedF1: Double, candidate: FollowUpCvUtils.Candidate)

  private def runLr(trainDf: DataFrame, baseCols: Seq[String], cvRows: Seq[Map[String, String]]): Seq[LrCandidate] =
    for {
      variant  <- ModelTraining.FeatureSetVariants
      regParam <- MlConfig.LrRegParams
    } yield {
      val cols = ExtraFeatures.featureSet(variant, baseCols)
      val lr = new LogisticRegression()
        .setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setElasticNetParam(MlConfig.LrElasticNet).setMaxIter(MlConfig.LrMaxIter)
        .setFamily("multinomial").setRegParam(regParam)
      val (macroF1, accuracy) = oofRecordingScore(trainDf, cols, lr)
      val params = f"regParam=$regParam%.4f"
      val windowWeightedF1 = lookupWeightedF1(cvRows, "LR", variant, params).getOrElse(
        throw new IllegalStateException(s"results/ml/cv_results.csv is missing LR/$variant/$params -- was it regenerated since Phase 5?"))
      // Simpler LR = fewer features, then larger regParam (more regularisation) -- tiebreak ascending, so -regParam.
      LrCandidate(variant, regParam, cols, windowWeightedF1, FollowUpCvUtils.Candidate(s"LR $variant $params", cols.size, -regParam, macroF1, accuracy))
    }

  // ======================================================================
  // RF stage 1: 1 fixed config (100 trees, depth 10) x 4 feature sets -- picks
  // the feature set RF stage 2 will tune further, same two-stage shape as Phase 5.
  // ======================================================================
  private case class RfStage1Candidate(variant: String, cols: Seq[String], windowWeightedF1: Double, candidate: FollowUpCvUtils.Candidate)

  private def runRfStage1(trainDf: DataFrame, baseCols: Seq[String], cvRows: Seq[Map[String, String]]): Seq[RfStage1Candidate] =
    ModelTraining.FeatureSetVariants.map { variant =>
      val cols = ExtraFeatures.featureSet(variant, baseCols)
      val rf = new RandomForestClassifier()
        .setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setNumTrees(MlConfig.Stage1FixedRfNumTrees).setMaxDepth(MlConfig.Stage1FixedRfMaxDepth).setSeed(MlConfig.Seed)
      val (macroF1, accuracy) = oofRecordingScore(trainDf, cols, rf)
      val params = s"numTrees=${MlConfig.Stage1FixedRfNumTrees},maxDepth=${MlConfig.Stage1FixedRfMaxDepth}"
      val windowWeightedF1 = lookupWeightedF1(cvRows, "RF-stage1-fixed", variant, params).getOrElse(
        throw new IllegalStateException(s"results/ml/cv_results.csv is missing RF-stage1-fixed/$variant/$params -- was it regenerated since Phase 5?"))
      // Config is fixed here, so simplicity only varies by feature count -- tiebreak constant.
      RfStage1Candidate(variant, cols, windowWeightedF1, FollowUpCvUtils.Candidate(s"RF-stage1 $variant", cols.size, 0.0, macroF1, accuracy))
    }

  // ======================================================================
  // RF stage 2: numTrees x maxDepth grid, run ONLY on whichever feature set
  // stage 1 won under THIS part's new rule (never Phase 5's F2 by default).
  // ======================================================================
  private case class RfStage2Candidate(numTrees: Int, maxDepth: Int, windowWeightedF1: Option[Double], candidate: FollowUpCvUtils.Candidate)

  private def runRfStage2(trainDf: DataFrame, variant: String, cols: Seq[String], cvRows: Seq[Map[String, String]]): Seq[RfStage2Candidate] =
    for {
      numTrees <- MlConfig.RfNumTrees
      maxDepth <- MlConfig.RfMaxDepth
    } yield {
      val rf = new RandomForestClassifier()
        .setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setNumTrees(numTrees).setMaxDepth(maxDepth).setSeed(MlConfig.Seed)
      val (macroF1, accuracy) = oofRecordingScore(trainDf, cols, rf)
      val params = s"numTrees=$numTrees,maxDepth=$maxDepth"
      // Phase 5 only ever tuned stage 2 on ITS OWN winning variant -- if the new rule
      // picked a different variant in stage 1, there is no Phase 5 number to show here.
      val windowWeightedF1 = lookupWeightedF1(cvRows, "RF", variant, params)
      // Simpler RF = fewer features (constant here), then fewer trees, then shallower depth.
      RfStage2Candidate(numTrees, maxDepth, windowWeightedF1,
        FollowUpCvUtils.Candidate(s"RF-stage2 $variant $params", cols.size, numTrees * 1000.0 + maxDepth, macroF1, accuracy))
    }

  /** Refits one final configuration on all training recordings and evaluates it once on the test set. */
  private def refitAndEvaluateOnce(modelLabel: String, cols: Seq[String], classifier: PipelineStage, trainDf: DataFrame, testDf: DataFrame): Unit = {
    val fitted = ModelTraining.buildPipeline(cols, classifier).fit(trainDf)
    val testWindowPreds = fitted.transform(testDf)

    val windowResult = MlMetrics.computeMetrics(testWindowPreds, "label_idx", "prediction", MlConfig.Genres)
    MlMetrics.reportMetrics(s"PartA_refit_${modelLabel}_window_test", windowResult, MlConfig.Genres, Paths.MlFollowupOutputDir)

    val recordingPreds = MlMetrics.softVoteToRecording(testWindowPreds)
    val recordingResult = MlMetrics.computeMetrics(recordingPreds, "label_idx", "prediction", MlConfig.Genres)
    MlMetrics.reportMetrics(s"PartA_refit_${modelLabel}_recordingSoft_test", recordingResult, MlConfig.Genres, Paths.MlFollowupOutputDir)

    // Saved separately so Part C can pick these up without re-running Part A in the same invocation.
    val idToGenre = MlConfig.Genres
    val predictionRows = recordingPreds.select("track_id", "label_idx", "prediction").collect().map { r =>
      Seq(r.getAs[String]("track_id"), idToGenre(r.getAs[Double]("label_idx").toInt), modelLabel, idToGenre(r.getAs[Double]("prediction").toInt))
    }.toSeq
    Report.saveCsv(s"partA_refit_${modelLabel}_test_predictions.csv",
      Seq("track_id", "label", "model", "prediction"), predictionRows, Paths.MlFollowupOutputDir)
  }

  def run(trainDf: DataFrame, testDf: DataFrame, baseCols: Seq[String]): Unit = {
    Report.section("Part A: cross-validation ranked by the recording-level score")
    Report.log("FOLLOW-UP (post-hoc, test set previously seen in Phase 5).")
    Report.log("Selection rule (fixed in code before any OOF score is computed): highest OOF recording-level " +
      "macro-F1; ties (|delta| < " + FollowUpConfig.TieEpsilon + ") broken by accuracy, then by the simpler " +
      "configuration (fewer features, larger regParam for LR / fewer then shallower trees for RF).")

    val cvRows = FollowUpCvUtils.readCsvAsRows(s"${Paths.MlOutputDir}/cv_results.csv")

    // ---------- LR ----------
    val lrCandidates = runLr(trainDf, baseCols, cvRows)
    val lrRankOld = ranksByOld(lrCandidates, (c: LrCandidate) => c.windowWeightedF1)
    val lrRankNew = ranksByNew(lrCandidates, (c: LrCandidate) => c.candidate)
    val lrNewWinner = lrCandidates.find(c => lrRankNew(c) == 1).get
    val oldLrRows  = cvRows.filter(_("model") == "LR")
    val oldLrBest  = oldLrRows.maxBy(_("cv_score_weighted_f1").toDouble)
    val lrNewWinnerParams = f"regParam=${lrNewWinner.regParam}%.4f"
    val lrSameAsOld = lrNewWinner.variant == oldLrBest("feature_set") && lrNewWinnerParams == oldLrBest("params")

    Report.log(s"LR: Phase 5's winner (by window weighted-F1) was ${oldLrBest("feature_set")}/${oldLrBest("params")}. " +
      s"New rule's winner (by recording macro-F1) is ${lrNewWinner.variant}/$lrNewWinnerParams.")
    if (lrSameAsOld) Report.log("LR: new rule selects the SAME configuration as Phase 5 -- not refit.")
    else {
      Report.log("LR: new rule selects a DIFFERENT configuration -- refitting on all 688 training recordings and evaluating on the test set once.")
      val lr = new LogisticRegression().setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setElasticNetParam(MlConfig.LrElasticNet).setMaxIter(MlConfig.LrMaxIter).setFamily("multinomial").setRegParam(lrNewWinner.regParam)
      refitAndEvaluateOnce("LR", lrNewWinner.cols, lr, trainDf, testDf)
    }

    // ---------- RF stage 1 (feature-set selection) ----------
    val rfStage1Candidates = runRfStage1(trainDf, baseCols, cvRows)
    val rfStage1RankOld = ranksByOld(rfStage1Candidates, (c: RfStage1Candidate) => c.windowWeightedF1)
    val rfStage1RankNew = ranksByNew(rfStage1Candidates, (c: RfStage1Candidate) => c.candidate)
    val rfStage1NewWinner = rfStage1Candidates.find(c => rfStage1RankNew(c) == 1).get
    val rfStage1OldWinner = rfStage1Candidates.find(c => rfStage1RankOld(c) == 1).get
    Report.log(s"RF stage 1: Phase 5's feature-set winner was ${rfStage1OldWinner.variant}; " +
      s"new rule's is ${rfStage1NewWinner.variant}.")

    // ---------- RF stage 2 (hyperparameter tuning, on the NEW stage-1 winner) ----------
    val rfStage2Candidates = runRfStage2(trainDf, rfStage1NewWinner.variant, rfStage1NewWinner.cols, cvRows)
    val rfStage2RankNew = ranksByNew(rfStage2Candidates, (c: RfStage2Candidate) => c.candidate)
    val rfStage2NewWinner = rfStage2Candidates.find(c => rfStage2RankNew(c) == 1).get
    val rfStage2RankOld: Map[RfStage2Candidate, Int] =
      if (rfStage2Candidates.forall(_.windowWeightedF1.isDefined))
        ranksByOld(rfStage2Candidates, (c: RfStage2Candidate) => c.windowWeightedF1.get)
      else Map.empty // Phase 5 never evaluated this feature set's stage-2 grid -- no old ranking exists.

    val oldRfRows = cvRows.filter(_("model") == "RF")
    val oldRfVariant = oldRfRows.head("feature_set")
    val oldRfBest = oldRfRows.maxBy(_("cv_score_weighted_f1").toDouble)
    val rfSameAsOld = rfStage1NewWinner.variant == oldRfVariant &&
      s"numTrees=${rfStage2NewWinner.numTrees},maxDepth=${rfStage2NewWinner.maxDepth}" == oldRfBest("params")

    if (rfStage2RankOld.isEmpty)
      Report.log(s"RF stage 2: Phase 5 never tuned the hyperparameter grid on feature set " +
        s"${rfStage1NewWinner.variant} (it only tuned ${oldRfVariant}), so no old-rule ranking exists for " +
        "this feature set -- window_weightedF1/rank_old are reported as N/A for these 4 rows, not assumed.")
    Report.log(s"RF overall: Phase 5's final config was $oldRfVariant/${oldRfBest("params")}. " +
      s"New rule's final config is ${rfStage1NewWinner.variant}/numTrees=${rfStage2NewWinner.numTrees},maxDepth=${rfStage2NewWinner.maxDepth}.")
    if (rfSameAsOld) Report.log("RF: new rule selects the SAME configuration as Phase 5 -- not refit.")
    else {
      Report.log("RF: new rule selects a DIFFERENT configuration -- refitting on all 688 training recordings and evaluating on the test set once.")
      val rf = new RandomForestClassifier().setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setNumTrees(rfStage2NewWinner.numTrees).setMaxDepth(rfStage2NewWinner.maxDepth).setSeed(MlConfig.Seed)
      refitAndEvaluateOnce("RF", rfStage1NewWinner.cols, rf, trainDf, testDf)
    }

    // ---------- cv_recording_level.csv ----------
    val header = Seq("model", "config", "feature_set", "window_weightedF1", "rec_macroF1", "rec_accuracy", "rank_old", "rank_new")
    val lrRows = lrCandidates.map { c =>
      Seq("LR", f"regParam=${c.regParam}%.4f", c.variant, f"${c.windowWeightedF1}%.4f",
        f"${c.candidate.macroF1}%.4f", f"${c.candidate.accuracy}%.4f", lrRankOld(c).toString, lrRankNew(c).toString)
    }
    val rfStage1Rows = rfStage1Candidates.map { c =>
      Seq("RF-stage1-fixed", s"numTrees=${MlConfig.Stage1FixedRfNumTrees},maxDepth=${MlConfig.Stage1FixedRfMaxDepth}", c.variant,
        f"${c.windowWeightedF1}%.4f", f"${c.candidate.macroF1}%.4f", f"${c.candidate.accuracy}%.4f",
        rfStage1RankOld(c).toString, rfStage1RankNew(c).toString)
    }
    val rfStage2Rows = rfStage2Candidates.map { c =>
      Seq("RF", s"numTrees=${c.numTrees},maxDepth=${c.maxDepth}", rfStage1NewWinner.variant,
        c.windowWeightedF1.map(v => f"$v%.4f").getOrElse("N/A"),
        f"${c.candidate.macroF1}%.4f", f"${c.candidate.accuracy}%.4f",
        rfStage2RankOld.get(c).map(_.toString).getOrElse("N/A"), rfStage2RankNew(c).toString)
    }
    val allRows = lrRows ++ rfStage1Rows ++ rfStage2Rows
    Report.log("\n--- cv_recording_level.csv (all 20 Phase 5 candidates, re-ranked by recording-level OOF score) ---")
    Report.table(header, allRows)
    Report.saveCsv("cv_recording_level.csv", header, allRows, Paths.MlFollowupOutputDir)
  }
}
