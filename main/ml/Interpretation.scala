package sonicspark.ml

import org.apache.spark.ml.classification.{LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Phase 5, stage 7: INTERPRETATION
 *
 * Derives report tables from already-fitted models and already-computed
 * predictions: RF feature importances, LR standardised coefficients, the
 * genre-pair confusion check, and the centroid baselines' per-genre predicted
 * counts. None of this feeds back into model selection -- it is produced
 * only after the final models are fixed, purely for the write-up.
 */
object Interpretation {

  /** Full feature-importance ranking, highest first. */
  def rfFeatureImportance(rfModel: RandomForestClassificationModel, featureCols: Seq[String]): Seq[(String, Double)] =
    featureCols.zip(rfModel.featureImportances.toArray).sortBy(-_._2)

  def reportFeatureImportance(rfModel: RandomForestClassificationModel, featureCols: Seq[String]): Unit = {
    val ranking = rfFeatureImportance(rfModel, featureCols)
    val header  = Seq("feature", "importance")
    val rows    = ranking.map { case (f, imp) => Seq(f, f"$imp%.6f") }

    Report.log("\n--- RandomForest feature importances (full ranking) ---")
    Report.log("Importance is split across correlated features (e.g. an MFCC mean/variance pair, or " +
      "spectral_bandwidth vs rolloff): a feature ranked lower may still matter jointly with its correlated partner.")
    Report.table(header, rows)
    Report.saveCsv("rf_feature_importance.csv", header, rows, Paths.MlOutputDir)
  }

  /** Long format: one row per (genre, feature) standardised coefficient. */
  def lrCoefficients(lrModel: LogisticRegressionModel, featureCols: Seq[String], genres: Seq[String]): Seq[Seq[String]] = {
    val matrix = lrModel.coefficientMatrix
    for {
      (genre, classIdx) <- genres.zipWithIndex
      (feature, featIdx) <- featureCols.zipWithIndex
    } yield Seq(genre, feature, f"${matrix(classIdx, featIdx)}%.6f")
  }

  /** Top-N positive and top-N negative coefficients per genre, for the printed report. */
  def topLrCoefficients(rows: Seq[Seq[String]], genres: Seq[String], topN: Int = 5): Seq[Seq[String]] =
    genres.flatMap { genre =>
      val genreRows = rows.filter(_.head == genre).map(r => (r(1), r(2).toDouble))
      val topPos = genreRows.sortBy(-_._2).take(topN).map { case (f, c) => Seq(genre, "top_positive", f, f"$c%.6f") }
      val topNeg = genreRows.sortBy(_._2).take(topN).map { case (f, c) => Seq(genre, "top_negative", f, f"$c%.6f") }
      topPos ++ topNeg
    }

  def reportLrCoefficients(lrModel: LogisticRegressionModel, featureCols: Seq[String], genres: Seq[String]): Unit = {
    val fullRows = lrCoefficients(lrModel, featureCols, genres)
    Report.saveCsv("lr_coefficients.csv", Seq("genre", "feature", "coefficient"), fullRows, Paths.MlOutputDir)

    val topRows = topLrCoefficients(fullRows, genres)
    Report.log("\n--- LogisticRegression top +/- standardised coefficients per genre ---")
    Report.log("Coefficients are on the StandardScaler's standardised scale (zero mean, unit variance), so " +
      "they are comparable across features. As with the RF ranking, importance here is split across " +
      "correlated features, since the pipeline removes no correlation beyond what Section 3 already dropped.")
    Report.table(Seq("genre", "direction", "feature", "coefficient"), topRows)
  }

  /**
   * Named genre-pair checks (test set, both levels, counts in both directions
   * where relevant): Phase 3's closest pair (hiphop<->reggae), Phase 3's other
   * close pair (country<->rock), rock's and classical's total off-diagonal
   * confusion, the Phase 4 Q6 candidate confusions, and whether pop/blues act
   * as "attractors" (off-diagonal column sums).
   */
  def genrePairCheck(result: MlMetrics.MetricsResult, genres: Seq[String], level: String): Seq[Seq[String]] = {
    val idx = genres.zipWithIndex.toMap

    def count(trueGenre: String, predGenre: String): Long = result.confusion(idx(trueGenre))(idx(predGenre))

    def offDiagonalRowSum(genre: String): Long = {
      val i = idx(genre)
      (0 until genres.size).filter(_ != i).map(j => result.confusion(i)(j)).sum
    }

    def offDiagonalColSum(genre: String): Long = {
      val i = idx(genre)
      (0 until genres.size).filter(_ != i).map(j => result.confusion(j)(i)).sum
    }

    Seq(
      Seq(level, "hiphop->reggae", count("hiphop", "reggae").toString),
      Seq(level, "reggae->hiphop", count("reggae", "hiphop").toString),
      Seq(level, "country->rock", count("country", "rock").toString),
      Seq(level, "rock->country", count("rock", "country").toString),
      Seq(level, "rock_total_offdiagonal", offDiagonalRowSum("rock").toString),
      Seq(level, "classical_total_offdiagonal", offDiagonalRowSum("classical").toString),
      Seq(level, "country->blues", count("country", "blues").toString),
      Seq(level, "disco->pop", count("disco", "pop").toString),
      Seq(level, "hiphop->pop", count("hiphop", "pop").toString),
      Seq(level, "rock->metal", count("rock", "metal").toString),
      Seq(level, "rock->blues", count("rock", "blues").toString),
      Seq(level, "pop_offdiagonal_column_sum", offDiagonalColSum("pop").toString),
      Seq(level, "blues_offdiagonal_column_sum", offDiagonalColSum("blues").toString)
    )
  }

  def reportGenrePairCheck(
    windowResult: MlMetrics.MetricsResult,
    recordingResult: MlMetrics.MetricsResult,
    genres: Seq[String]
  ): Unit = {
    val rows = genrePairCheck(windowResult, genres, "window") ++ genrePairCheck(recordingResult, genres, "recording")
    Report.log("\n--- Genre-pair check (test set) ---")
    Report.table(Seq("level", "check", "count"), rows)
    Report.saveCsv("genre_pair_check.csv", Seq("level", "check", "count"), rows, Paths.MlOutputDir)
  }

  /** How many recordings each genre's centroid "attracted" (predicted count), for one centroid variant. */
  private def predictedCountsPerGenre(predictions: DataFrame, genres: Seq[String]): Seq[(String, Long)] = {
    val counts = predictions.groupBy("prediction").count().collect()
      .map(r => (r.getAs[Double]("prediction").toInt, r.getAs[Long]("count"))).toMap
    genres.zipWithIndex.map { case (g, i) => (g, counts.getOrElse(i, 0L)) }
  }

  def reportCentroidColumnSums(
    plainRecordingPredictions: DataFrame,
    pcaRecordingPredictions: DataFrame,
    genres: Seq[String]
  ): Unit = {
    val plainCounts = predictedCountsPerGenre(plainRecordingPredictions, genres).map { case (g, c) => Seq("plain", g, c.toString) }
    val pcaCounts   = predictedCountsPerGenre(pcaRecordingPredictions, genres).map { case (g, c) => Seq("pca_whitened", g, c.toString) }
    val rows = plainCounts ++ pcaCounts

    Report.log("\n--- Centroid baseline: per-genre predicted counts (column sums) ---")
    Report.table(Seq("variant", "genre", "predicted_count"), rows)
    Report.saveCsv("centroid_column_sums.csv", Seq("variant", "genre", "predicted_count"), rows, Paths.MlOutputDir)
  }
}
