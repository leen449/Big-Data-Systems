package sonicspark.ml

import org.apache.spark.ml.feature.{PCA, PCAModel, StandardScaler, StandardScalerModel, VectorAssembler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
 * Phase 5, stage 5 (part 1): BASELINES
 *
 * Baseline A (majority class) and Baseline B (recording-level nearest centroid,
 * ported from Phase 4's Q6, train-only) both use only the base feature set F0 --
 * they are reference points, not part of the F0-F3 comparison that the real
 * classifiers go through. Both baselines emit "prediction"/"probability" columns
 * with the same shape Spark's own classifiers use (Double prediction, one-hot
 * probability vector), so MlMetrics and the soft/majority voting aggregators can
 * treat baselines and real models identically.
 */
object Baselines {

  /** Converts a predicted class index into a one-hot probability vector, for voting. */
  private def oneHotUdf(numClasses: Int) =
    udf((classIdx: Double) => {
      val arr = Array.fill(numClasses)(0.0)
      arr(classIdx.toInt) = 1.0
      Vectors.dense(arr)
    })

  private def attachProbability(df: DataFrame, numClasses: Int): DataFrame =
    df.withColumn("probability", oneHotUdf(numClasses)(col("prediction")))

  private def euclidean(a: Vector, b: Vector): Double =
    math.sqrt(a.toArray.zip(b.toArray).map { case (x, y) => math.pow(x - y, 2) }.sum)

  // ======================================================================
  // Baseline A: majority class of the training windows
  // ======================================================================
  case class MajorityModel(majorityGenreIdx: Int)

  object MajorityBaseline {

    /** Mode of the training windows' genre, ties broken alphabetically. */
    def fit(trainWindows: DataFrame): MajorityModel = {
      val counts = trainWindows.groupBy("label").count().collect()
        .map(r => (r.getString(0), r.getLong(1)))
      val majorityGenre = counts.sortBy { case (genre, cnt) => (-cnt, genre) }.head._1
      MajorityModel(MlConfig.Genres.indexOf(majorityGenre))
    }

    /** Every row (window or recording) gets the same constant predicted class. */
    def predict(model: MajorityModel, df: DataFrame): DataFrame = {
      val withPrediction = df.withColumn("prediction", lit(model.majorityGenreIdx.toDouble))
      attachProbability(withPrediction, MlConfig.Genres.size)
    }
  }

  // ======================================================================
  // Baseline B: recording-level nearest centroid (Phase 4 Q6, train-only)
  //   - fitPlain:      z-score (StandardScaler) -> recording mean -> genre centroid
  //   - fitPcaWhitened: same, then PCA (>=95% variance) with components whitened
  //                     by 1/sqrt(eigenvalue), to test whether correlated MFCC
  //                     features let pop/blues centroids "attract" other genres.
  // ======================================================================
  case class CentroidModel(
    featureCols: Seq[String],
    assembler: VectorAssembler,
    scaler: StandardScalerModel,
    pca: Option[PCAModel],
    eigenvalues: Option[Array[Double]],   // defined only when pca is defined; used to whiten
    centroids: Map[String, Vector]        // genre -> centroid, in the same space predict() must use
  )

  object CentroidModel {

    def fitPlain(train: DataFrame, featureCols: Seq[String]): CentroidModel =
      fitInternal(train, featureCols, usePca = false, MlConfig.PcaVarianceThreshold)

    def fitPcaWhitened(train: DataFrame, featureCols: Seq[String], varianceThreshold: Double): CentroidModel =
      fitInternal(train, featureCols, usePca = true, varianceThreshold)

    private def fitInternal(
      train: DataFrame,
      featureCols: Seq[String],
      usePca: Boolean,
      varianceThreshold: Double
    ): CentroidModel = {
      val assembler = new VectorAssembler().setInputCols(featureCols.toArray).setOutputCol("assembled")
      val assembled = assembler.transform(train)
      val scaler = new StandardScaler()
        .setInputCol("assembled").setOutputCol("scaled")
        .setWithMean(true).setWithStd(true)
        .fit(assembled)
      val scaled = scaler.transform(assembled)

      val (pcaModelOpt, eigenvaluesOpt, vecCol, vecDf) =
        if (usePca) {
          // Fit once with every component to read off how much variance each one explains,
          // then refit with just enough components to reach the variance threshold.
          val full = new PCA().setInputCol("scaled").setOutputCol("pca").setK(featureCols.size).fit(scaled)
          val cumulative = full.explainedVariance.toArray.scanLeft(0.0)(_ + _).tail
          val k = math.max(1, cumulative.indexWhere(_ >= varianceThreshold) + 1)
          val pcaModel = new PCA().setInputCol("scaled").setOutputCol("pca").setK(k).fit(scaled)

          // StandardScaler gives every input feature unit variance, so the total variance
          // of the scaled space is simply the number of input features. Each component's
          // eigenvalue is its share of that total, per explainedVariance.
          val totalVariance = featureCols.size.toDouble
          val eigen = pcaModel.explainedVariance.toArray.map(_ * totalVariance)

          val whitenUdf = udf((v: Vector) =>
            Vectors.dense(v.toArray.zip(eigen).map { case (x, e) => x / math.sqrt(e) }))
          val whitenedDf = pcaModel.transform(scaled).withColumn("whitened", whitenUdf(col("pca")))
          (Some(pcaModel), Some(eigen), "whitened", whitenedDf)
        } else {
          (None, None, "scaled", scaled)
        }

      // Recording profile: mean of (possibly whitened) feature vectors across that
      // recording's windows. Genre centroid: mean of recording profiles within the
      // genre -- NOT a window-level mean -- exactly mirroring Q6.
      val recordingProfile = vecDf.groupBy("track_id", "label")
        .agg(Summarizer.mean(col(vecCol)).as("profile"))
      val centroidRows = recordingProfile.groupBy("label")
        .agg(Summarizer.mean(col("profile")).as("centroid"))
        .collect()
      val centroids = centroidRows.map(r => r.getAs[String]("label") -> r.getAs[Vector]("centroid")).toMap

      CentroidModel(featureCols, assembler, scaler, pcaModelOpt, eigenvaluesOpt, centroids)
    }

    /**
     * Recording-level prediction, broadcast onto every window of that recording.
     * This is deliberately not a per-window prediction: the centroid model only
     * ever decides at the recording level, so every window of a recording gets
     * an identical copy of that recording's one verdict.
     */
    def predict(model: CentroidModel, df: DataFrame): DataFrame = {
      val assembled = model.assembler.transform(df)
      val scaled    = model.scaler.transform(assembled)

      val (vecCol, vecDf) = model.pca match {
        case Some(pcaModel) =>
          val eigen = model.eigenvalues.get
          val whitenUdf = udf((v: Vector) =>
            Vectors.dense(v.toArray.zip(eigen).map { case (x, e) => x / math.sqrt(e) }))
          ("whitened", pcaModel.transform(scaled).withColumn("whitened", whitenUdf(col("pca"))))
        case None =>
          ("scaled", scaled)
      }

      val recordingProfile = vecDf.groupBy("track_id").agg(Summarizer.mean(col(vecCol)).as("profile"))

      val genreOrder   = MlConfig.Genres
      val centroidsRef = model.centroids
      val nearestIdxUdf = udf((profile: Vector) => {
        val distances = genreOrder.map(g => (g, euclidean(profile, centroidsRef(g))))
        genreOrder.indexOf(distances.minBy(_._2)._1).toDouble
      })

      val recordingPrediction = recordingProfile
        .withColumn("prediction", nearestIdxUdf(col("profile")))
        .select("track_id", "prediction")

      val windowPredictions = df.join(recordingPrediction, "track_id")
      attachProbability(windowPredictions, MlConfig.Genres.size)
    }
  }
}
