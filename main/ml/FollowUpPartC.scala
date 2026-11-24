package sonicspark.ml

import java.nio.file.{Files, Paths => JPaths}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Phase 6, Part C: paired comparison of every model pair on the same 296 test
 * recordings. Phase 5 never saved recording-level test predictions per model
 * (only aggregated confusion matrices/metrics), so this part's first job is to
 * regenerate them by refitting Phase 5's exact final configurations -- same
 * seed, same training data, same hyperparameters, read back from
 * results/ml/cv_results.csv rather than retyped by hand -- and to prove that
 * refit reproduces Phase 5's own numbers before trusting it for anything else.
 */
object FollowUpPartC {

  private def readPredictionsCsvIfExists(path: String): Option[Seq[Map[String, String]]] =
    if (Files.exists(JPaths.get(path))) Some(FollowUpCvUtils.readCsvAsRows(path)) else None

  private def toPredictionMapFromDf(df: DataFrame): Map[String, String] =
    df.select("track_id", "prediction").collect()
      .map(r => r.getAs[String]("track_id") -> MlConfig.Genres(r.getAs[Double]("prediction").toInt)).toMap

  private def toPredictionMapFromCsvRows(rows: Seq[Map[String, String]]): Map[String, String] =
    rows.map(r => r("track_id") -> r("prediction")).toMap

  /** Refits Phase 5's exact final LR config (read from cv_results.csv, not retyped) and
   *  returns its recording-level (soft-voted) test predictions. */
  private def refitPhase5Lr(trainDf: DataFrame, testDf: DataFrame, baseCols: Seq[String], cvRows: Seq[Map[String, String]]): DataFrame = {
    val best = cvRows.filter(_("model") == "LR").maxBy(_("cv_score_weighted_f1").toDouble)
    val variant = best("feature_set")
    val regParam = best("params").stripPrefix("regParam=").toDouble
    val cols = ExtraFeatures.featureSet(variant, baseCols)
    val lr = new LogisticRegression().setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
      .setElasticNetParam(MlConfig.LrElasticNet).setMaxIter(MlConfig.LrMaxIter).setFamily("multinomial").setRegParam(regParam)
    val fitted = ModelTraining.buildPipeline(cols, lr).fit(trainDf)
    MlMetrics.softVoteToRecording(fitted.transform(testDf))
  }

  /** Same idea for RF: variant + numTrees/maxDepth read back from cv_results.csv's "RF" rows. */
  private def refitPhase5Rf(trainDf: DataFrame, testDf: DataFrame, baseCols: Seq[String], cvRows: Seq[Map[String, String]]): DataFrame = {
    val rfRows = cvRows.filter(_("model") == "RF")
    val variant = rfRows.head("feature_set")
    val best = rfRows.maxBy(_("cv_score_weighted_f1").toDouble)
    val params = best("params").split(",").map(_.split("=")).map(a => a(0) -> a(1).toInt).toMap
    val cols = ExtraFeatures.featureSet(variant, baseCols)
    val rf = new RandomForestClassifier().setFeaturesCol("scaledFeatures").setLabelCol("label_idx")
      .setNumTrees(params("numTrees")).setMaxDepth(params("maxDepth")).setSeed(MlConfig.Seed)
    val fitted = ModelTraining.buildPipeline(cols, rf).fit(trainDf)
    MlMetrics.softVoteToRecording(fitted.transform(testDf))
  }

  /** Centroid baselines already predict at the recording level, broadcast onto every window --
   *  dropDuplicates collapses that back to one row per recording. */
  private def refitPhase5Centroid(trainDf: DataFrame, testDf: DataFrame, baseCols: Seq[String], pca: Boolean): DataFrame = {
    val model = if (pca) Baselines.CentroidModel.fitPcaWhitened(trainDf, baseCols, MlConfig.PcaVarianceThreshold)
                else Baselines.CentroidModel.fitPlain(trainDf, baseCols)
    Baselines.CentroidModel.predict(model, testDf).dropDuplicates("track_id")
  }

  private def assertMatchesPhase5(name: String, recordingPred: DataFrame, expected: Double): Unit = {
    val accuracy = MlMetrics.computeMetrics(recordingPred, "label_idx", "prediction", MlConfig.Genres).accuracy
    require(math.abs(accuracy - expected) < FollowUpConfig.Phase5AccuracyCheckTolerance,
      f"$name: recomputed test accuracy $accuracy%.4f does not match ml_metrics.txt's $expected%.4f to 4 decimals -- " +
        "the refit is not reproducing Phase 5, something upstream must have changed.")
    Report.log(f"$name: recomputed test accuracy = $accuracy%.4f, matches ml_metrics.txt's $expected%.4f to 4dp. OK.")
  }

  def run(spark: SparkSession, trainDf: DataFrame, testDf: DataFrame, baseCols: Seq[String], splitFoldDf: DataFrame): Unit = {
    Report.section("Part C: paired comparison on the test recordings")
    Report.log("FOLLOW-UP (post-hoc, test set previously seen in Phase 5).")
    Report.log(s"Phase 5 saved no per-model recording-level test predictions, only aggregated metrics -- " +
      s"regenerating them by refitting Phase 5's exact final configurations (read from results/ml/cv_results.csv, " +
      s"same seed ${MlConfig.Seed}, same training data) and checking the recomputed test accuracy against " +
      s"results/ml_metrics.txt to 4 decimals before using them for anything else.")

    val cvRows = FollowUpCvUtils.readCsvAsRows(s"${Paths.MlOutputDir}/cv_results.csv")

    val lrPred      = refitPhase5Lr(trainDf, testDf, baseCols, cvRows)
    val rfPred      = refitPhase5Rf(trainDf, testDf, baseCols, cvRows)
    val plainPred   = refitPhase5Centroid(trainDf, testDf, baseCols, pca = false)
    val pcaPred     = refitPhase5Centroid(trainDf, testDf, baseCols, pca = true)

    assertMatchesPhase5("LR_window",       lrPred,    FollowUpConfig.Phase5TestAccuracyLr)
    assertMatchesPhase5("RF_window",       rfPred,    FollowUpConfig.Phase5TestAccuracyRf)
    assertMatchesPhase5("Centroid_Plain",  plainPred, FollowUpConfig.Phase5TestAccuracyCentroidPlain)
    assertMatchesPhase5("Centroid_PCA",    pcaPred,   FollowUpConfig.Phase5TestAccuracyCentroidPca)

    val phase5Models: Seq[(String, Map[String, String])] = Seq(
      "LR_window"      -> toPredictionMapFromDf(lrPred),
      "RF_window"      -> toPredictionMapFromDf(rfPred),
      "Centroid_Plain" -> toPredictionMapFromDf(plainPred),
      "Centroid_PCA"   -> toPredictionMapFromDf(pcaPred)
    )

    // Part A only refits+saves a model if its new rule picked a different config than Phase 5 --
    // these files may legitimately not exist. Part B always saves its two winners; still read
    // defensively so `--part C` can also be run on its own against a previous Part B run's output.
    val partAModels: Seq[(String, Map[String, String])] =
      Seq("LR", "RF").flatMap { model =>
        readPredictionsCsvIfExists(s"${Paths.MlFollowupOutputDir}/partA_refit_${model}_test_predictions.csv") match {
          case Some(rows) =>
            Report.log(s"Part A refit found for $model -- included in the pairwise comparison.")
            Some(s"PartA_refit_$model" -> toPredictionMapFromCsvRows(rows))
          case None =>
            Report.log(s"No Part A refit file for $model (new rule matched Phase 5, or Part A has not been run) -- skipped.")
            None
        }
      }

    val partBModels: Seq[(String, Map[String, String])] =
      Seq("summarylr" -> "SummaryLR", "summaryrf" -> "SummaryRF").flatMap { case (fileStem, label) =>
        readPredictionsCsvIfExists(s"${Paths.MlFollowupOutputDir}/${fileStem}_test_predictions.csv") match {
          case Some(rows) => Some(label -> toPredictionMapFromCsvRows(rows))
          case None =>
            Report.log(s"No Part B predictions file for $label found -- run Part B first to include it here.")
            None
        }
      }

    val allModels = phase5Models ++ partAModels ++ partBModels

    // ---------- test_predictions_recording.csv ----------
    val testLabels = splitFoldDf.filter(col("split") === "test").select("track_id", "label")
      .collect().map(r => r.getAs[String]("track_id") -> r.getAs[String]("label")).toMap
    val testIds = testLabels.keys.toSeq.sorted

    val predictionRows = allModels.flatMap { case (model, preds) =>
      testIds.map(id => Seq(id, testLabels(id), model, preds(id)))
    }
    Report.saveCsv("test_predictions_recording.csv", Seq("track_id", "label", "model", "prediction"), predictionRows, Paths.MlFollowupOutputDir)
    Report.log(s"\nSaved recording-level test predictions for ${allModels.size} models (${allModels.map(_._1).mkString(", ")}) " +
      s"to results/ml_followup/test_predictions_recording.csv.")

    // ---------- pairwise comparison ----------
    def correctnessVector(preds: Map[String, String]): Array[Boolean] = testIds.map(id => preds(id) == testLabels(id)).toArray

    val pairRows = allModels.combinations(2).toSeq.map {
      case Seq((nameA, predsA), (nameB, predsB)) =>
        val correctA = correctnessVector(predsA)
        val correctB = correctnessVector(predsB)
        val accA = correctA.count(identity).toDouble / correctA.length
        val accB = correctB.count(identity).toDouble / correctB.length
        val ci = FollowUpStats.bootstrapCi(correctA, correctB, FollowUpConfig.BootstrapResamples, FollowUpConfig.BootstrapSeed)
        val (b, c) = FollowUpStats.discordantCounts(correctA, correctB)
        val p = FollowUpStats.exactMcNemarP(b, c)
        Seq(nameA, nameB, f"$accA%.4f", f"$accB%.4f", f"${accA - accB}%.4f", f"${ci.lower}%.4f", f"${ci.upper}%.4f", b.toString, c.toString, f"$p%.6f")
    }

    val pairHeader = Seq("model_a", "model_b", "acc_a", "acc_b", "diff", "ci_lower", "ci_upper", "b", "c", "mcnemar_p")
    Report.log(s"\n--- Pairwise comparison (${pairRows.size} pairs, paired bootstrap 95% CI " +
      s"${FollowUpConfig.BootstrapResamples} resamples seed ${FollowUpConfig.BootstrapSeed}, exact McNemar) ---")
    Report.table(pairHeader, pairRows)
    Report.saveCsv("pairwise_comparison.csv", pairHeader, pairRows, Paths.MlFollowupOutputDir)
  }
}
