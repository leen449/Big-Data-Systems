package sonicspark.ml

import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{
  LogisticRegression, LogisticRegressionModel, RandomForestClassifier, RandomForestClassificationModel
}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
 * Phase 5, stage 4: CROSS-VALIDATION (and stage 5's grouped out-of-fold evaluation)
 *
 * The single most leakage-sensitive file in this project. Every CrossValidator
 * call below uses `.setFoldCol(MlConfig.FoldColName)` instead of Spark's default
 * random row-level folding, so a fold boundary always falls between recordings,
 * never between windows of the same recording. The fold column itself is never
 * recomputed here -- it is threaded straight through from Split.scala.
 *
 * Two different concerns share this file because they share the same fold
 * column and the same pipeline shape:
 *   1. Hyperparameter / feature-set SEARCH (stage1Search, stage2TuneRf): uses
 *      Spark's CrossValidator, whose avgMetrics give a cheap score per
 *      candidate but no actual held-out predictions.
 *   2. Genuine out-of-fold EVALUATION (runGroupedOof): a manual 5-fold loop
 *      that fits on 4 folds and predicts the 5th, for every model/baseline
 *      with its final chosen configuration, producing the real "grouped-CV
 *      out-of-fold on train" prediction set the report needs -- something
 *      CrossValidator's avgMetrics alone cannot provide.
 */
object ModelTraining {

  val FeatureSetVariants: Seq[String] = Seq("F0", "F1", "F2", "F3")

  /** VectorAssembler -> StandardScaler(withMean=true, withStd=true) -> classifier. */
  def buildPipeline(featureCols: Seq[String], classifier: PipelineStage): Pipeline = {
    val assembler = new VectorAssembler().setInputCols(featureCols.toArray).setOutputCol("features")
    val scaler = new StandardScaler()
      .setInputCol("features").setOutputCol("scaledFeatures")
      .setWithMean(true).setWithStd(true)
    new Pipeline().setStages(Array(assembler, scaler, classifier))
  }

  private def evaluator: MulticlassClassificationEvaluator =
    // Spark has no built-in macro-F1, only weighted ("f1") -- used here only to
    // rank candidates during CV. True macro-F1 for the report is computed
    // separately, from actual predictions, by MlMetrics.computeMetrics.
    new MulticlassClassificationEvaluator()
      .setLabelCol("label_idx").setPredictionCol("prediction").setMetricName("f1")

  // ======================================================================
  // Stage 1: pick each model's own best feature set (section 5's 4-feature-set
  // comparison), cheaply: LR gets its full 3-point grid, RF gets one fixed
  // config, both run across all four feature sets.
  // ======================================================================
  case class Stage1Result(
    cvResultRows: Seq[Seq[String]],     // model, feature_set, params, cv_score -- for cv_results.csv
    lrBestVariant: String,
    lrBestModel: CrossValidatorModel,   // .bestModel is LR already refit on all of trainDf for lrBestVariant
    rfWinningVariantFromFixed: String   // which feature set RF should be tuned further on, in stage 2
  )

  def stage1Search(trainDf: DataFrame, baseCols: Seq[String]): Stage1Result = {
    case class VariantRun(variant: String, cvModel: CrossValidatorModel, rows: Seq[Seq[String]], bestScore: Double)

    val lrRuns = FeatureSetVariants.map { variant =>
      val cols = ExtraFeatures.featureSet(variant, baseCols)
      val lr = new LogisticRegression()
        .setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setElasticNetParam(MlConfig.LrElasticNet).setMaxIter(MlConfig.LrMaxIter)
        .setFamily("multinomial")
      val grid = new ParamGridBuilder().addGrid(lr.regParam, MlConfig.LrRegParams.toArray).build()
      val cv = new CrossValidator()
        .setEstimator(buildPipeline(cols, lr)).setEvaluator(evaluator).setEstimatorParamMaps(grid)
        .setFoldCol(MlConfig.FoldColName).setNumFolds(MlConfig.NumFolds).setSeed(MlConfig.Seed)
      val cvModel = cv.fit(trainDf)

      val rows = grid.indices.map { i =>
        val regParam = grid(i)(lr.regParam)
        Seq("LR", variant, f"regParam=$regParam%.4f", f"${cvModel.avgMetrics(i)}%.4f")
      }
      VariantRun(variant, cvModel, rows, cvModel.avgMetrics.max)
    }

    val rfRuns = FeatureSetVariants.map { variant =>
      val cols = ExtraFeatures.featureSet(variant, baseCols)
      val rf = new RandomForestClassifier()
        .setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
        .setNumTrees(MlConfig.Stage1FixedRfNumTrees).setMaxDepth(MlConfig.Stage1FixedRfMaxDepth)
        .setSeed(MlConfig.Seed)
      val grid = new ParamGridBuilder().build()   // single point: the fixed stage-1 RF config above
      val cv = new CrossValidator()
        .setEstimator(buildPipeline(cols, rf)).setEvaluator(evaluator).setEstimatorParamMaps(grid)
        .setFoldCol(MlConfig.FoldColName).setNumFolds(MlConfig.NumFolds).setSeed(MlConfig.Seed)
      val cvModel = cv.fit(trainDf)

      val rows = Seq(Seq("RF-stage1-fixed", variant,
        s"numTrees=${MlConfig.Stage1FixedRfNumTrees},maxDepth=${MlConfig.Stage1FixedRfMaxDepth}",
        f"${cvModel.avgMetrics(0)}%.4f"))
      VariantRun(variant, cvModel, rows, cvModel.avgMetrics(0))
    }

    // Tie-break: Seq.maxBy keeps the first element reaching the maximum, so ties
    // fall to F0 (the variants are visited in F0,F1,F2,F3 order) -- deterministic.
    val lrBest = lrRuns.maxBy(_.bestScore)
    val rfBest = rfRuns.maxBy(_.bestScore)

    Stage1Result(
      cvResultRows = lrRuns.flatMap(_.rows) ++ rfRuns.flatMap(_.rows),
      lrBestVariant = lrBest.variant,
      lrBestModel = lrBest.cvModel,
      rfWinningVariantFromFixed = rfBest.variant
    )
  }

  // ======================================================================
  // Stage 2: RF only. LR is not re-run -- its best config and feature set
  // were already decided in stage 1. Full RF grid (numTrees x maxDepth),
  // run only on RF's own winning feature set from stage 1.
  // ======================================================================
  case class Stage2Result(cvResultRows: Seq[Seq[String]], rfBestModel: CrossValidatorModel)

  def stage2TuneRf(trainDf: DataFrame, baseCols: Seq[String], rfVariant: String): Stage2Result = {
    val cols = ExtraFeatures.featureSet(rfVariant, baseCols)
    val rf = new RandomForestClassifier()
      .setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
      .setSeed(MlConfig.Seed)
    val grid = new ParamGridBuilder()
      .addGrid(rf.numTrees, MlConfig.RfNumTrees.toArray)
      .addGrid(rf.maxDepth, MlConfig.RfMaxDepth.toArray)
      .build()
    val cv = new CrossValidator()
      .setEstimator(buildPipeline(cols, rf)).setEvaluator(evaluator).setEstimatorParamMaps(grid)
      .setFoldCol(MlConfig.FoldColName).setNumFolds(MlConfig.NumFolds).setSeed(MlConfig.Seed)
    val cvModel = cv.fit(trainDf)

    val rows = grid.indices.map { i =>
      val numTrees = grid(i)(rf.numTrees)
      val maxDepth = grid(i)(rf.maxDepth)
      Seq("RF", rfVariant, s"numTrees=$numTrees,maxDepth=$maxDepth", f"${cvModel.avgMetrics(i)}%.4f")
    }
    Stage2Result(rows, cvModel)
  }

  // ======================================================================
  // Generic grouped out-of-fold loop, reused identically for LR, RF, and every
  // baseline: fit on 4 folds, predict the held-out 5th, union the 5 results.
  // This -- not CrossValidator's avgMetrics -- is what produces real held-out
  // predictions for the "grouped-CV out-of-fold on train" report rows.
  // ======================================================================
  def runGroupedOof[M](
    trainDf: DataFrame,
    foldCol: String,
    numFolds: Int,
    fitFn: DataFrame => M,
    predictFn: (M, DataFrame) => DataFrame
  ): DataFrame = {
    val perFoldPredictions = (0 until numFolds).map { fold =>
      val foldTrain   = trainDf.filter(col(foldCol) =!= fold)
      val foldHoldout = trainDf.filter(col(foldCol) === fold)
      val model = fitFn(foldTrain)
      predictFn(model, foldHoldout)
    }
    perFoldPredictions.reduce(_ unionByName _)
  }

  /** Builds the final LR pipeline (winning feature set + regParam from stage 1) for use in the OOF loop. */
  def finalLrPipeline(lrBestModel: CrossValidatorModel, lrColumns: Seq[String]): Pipeline = {
    // The fitted PipelineModel's last stage is a LogisticRegressionModel (a Transformer),
    // which still exposes the Param getters (getRegParam, ...) the original Estimator had.
    val bestLr = lrBestModel.bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]
    val lr = new LogisticRegression()
      .setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
      .setElasticNetParam(MlConfig.LrElasticNet).setMaxIter(MlConfig.LrMaxIter)
      .setFamily("multinomial").setRegParam(bestLr.getRegParam)
    buildPipeline(lrColumns, lr)
  }

  /** Builds the final RF pipeline (winning feature set + numTrees/maxDepth from stage 2) for use in the OOF loop. */
  def finalRfPipeline(rfBestModel: CrossValidatorModel, rfColumns: Seq[String]): Pipeline = {
    val bestRf = rfBestModel.bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[RandomForestClassificationModel]
    val rf = new RandomForestClassifier()
      .setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
      .setSeed(MlConfig.Seed).setNumTrees(bestRf.getNumTrees).setMaxDepth(bestRf.getMaxDepth)
    buildPipeline(rfColumns, rf)
  }
}
