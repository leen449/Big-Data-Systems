package sonicspark.ml

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Phase 5, stage 1: SPLIT
 *
 * Everything here exists to satisfy one hard rule: no recording may have
 * windows on both sides of train/test, and every grouped-CV fold must hold
 * out whole recordings, never individual windows. `df.sample`/`randomSplit`
 * operate row-by-row and would put different windows of the same recording
 * on different sides -- that is exactly the leakage this project must avoid
 * (a model could "recognise" a recording from one window and get the rest
 * of its windows right for free). So every random choice below is made on
 * the driver, over a small, explicitly sorted array of track_ids, using a
 * seeded scala.util.Random -- never Spark's own row ordering, which is not
 * guaranteed stable across runs.
 */
object Split {

  /** One row per recording: which side of the split it's on, and (train only) its CV fold. */
  case class SplitResult(
    trainWindows: DataFrame,   // data/processed/final rows for training recordings only
    testWindows: DataFrame,    // data/processed/final rows for test recordings only
    splitTable: DataFrame      // track_id, label, split, fold (fold = -1 for test)
  )

  /**
   * Recording-level 70/30 split, stratified by genre, seed 42.
   * Per genre: sort track_ids alphabetically (so the input order never matters),
   * shuffle with one seeded Random, take round(trainFrac * n) for train.
   */
  def stratifiedTrainTestSplit(
    spark: SparkSession,
    trackTable: DataFrame,
    trainFrac: Double,
    seed: Long
  ): DataFrame = {
    val rows: Array[(String, String)] =
      trackTable.select("track_id", "label").collect().map(r => (r.getString(0), r.getString(1)))

    val byGenre: Seq[(String, Seq[String])] =
      rows.groupBy(_._2).toSeq.sortBy(_._1).map { case (genre, trackRows) =>
        (genre, trackRows.map(_._1).sorted.toSeq)
      }

    // One Random instance, consumed genre by genre in a fixed (alphabetical) order,
    // so the result is reproducible regardless of how Spark happened to order rows.
    val rng = new scala.util.Random(seed)
    val splitRows: Seq[(String, String, String)] = byGenre.flatMap { case (genre, sortedTrackIds) =>
      val shuffled = rng.shuffle(sortedTrackIds)
      val numTrain = math.round(trainFrac * sortedTrackIds.size).toInt
      val (trainIds, testIds) = shuffled.splitAt(numTrain)
      trainIds.map(id => (id, genre, "train")) ++ testIds.map(id => (id, genre, "test"))
    }

    spark.createDataFrame(splitRows).toDF("track_id", "label", "split")
  }

  /**
   * Grouped 5-fold assignment over training recordings only, stratified by genre,
   * seed 42. Same driver-side seeded-shuffle technique as the train/test split,
   * so that a recording's fold number is reproducible and every fold contains
   * whole recordings.
   */
  def assignGroupedFolds(
    spark: SparkSession,
    trainTracks: DataFrame,
    numFolds: Int,
    seed: Long
  ): DataFrame = {
    val rows: Array[(String, String)] =
      trainTracks.select("track_id", "label").collect().map(r => (r.getString(0), r.getString(1)))

    val byGenre: Seq[(String, Seq[String])] =
      rows.groupBy(_._2).toSeq.sortBy(_._1).map { case (genre, trackRows) =>
        (genre, trackRows.map(_._1).sorted.toSeq)
      }

    val rng = new scala.util.Random(seed)
    val foldRows: Seq[(String, String, Int)] = byGenre.flatMap { case (genre, sortedTrackIds) =>
      val shuffled = rng.shuffle(sortedTrackIds)
      shuffled.zipWithIndex.map { case (trackId, i) => (trackId, genre, i % numFolds) }
    }

    spark.createDataFrame(foldRows).toDF("track_id", "label", "fold")
  }

  /** Merges the train/test split with the fold assignment; test rows get fold = -1. */
  def combineSplitAndFolds(splitDf: DataFrame, foldDf: DataFrame): DataFrame =
    splitDf
      .join(foldDf.select(col("track_id").as("fold_track_id"), col("fold")), col("track_id") === col("fold_track_id"), "left")
      .drop("fold_track_id")
      .withColumn("fold", when(col("split") === "test", lit(-1)).otherwise(col("fold")))

  /** Hard check: no track_id may appear on both sides of the split. */
  def assertDisjoint(splitFoldDf: DataFrame): Unit = {
    val trainIds = splitFoldDf.filter(col("split") === "train")
      .select("track_id").collect().map(_.getString(0)).toSet
    val testIds = splitFoldDf.filter(col("split") === "test")
      .select("track_id").collect().map(_.getString(0)).toSet
    val overlap = trainIds.intersect(testIds)
    require(overlap.isEmpty,
      s"Leakage: ${overlap.size} track_id(s) appear in both train and test: ${overlap.mkString(", ")}")
  }

  /** Joins split/fold onto every window row of the recording it belongs to. */
  def attachToWindows(windows: DataFrame, splitFoldDf: DataFrame): DataFrame =
    windows.join(splitFoldDf.select("track_id", "split", "fold"), Seq("track_id"))

  /** Per genre, per split: how many recordings and how many windows. For the report. */
  def summarize(windowsWithSplit: DataFrame): Seq[Seq[String]] =
    windowsWithSplit.groupBy("label", "split")
      .agg(countDistinct("track_id").as("num_recordings"), count("*").as("num_windows"))
      .orderBy("label", "split")
      .collect()
      .map(r => Seq(
        r.getAs[String]("label"),
        r.getAs[String]("split"),
        r.getAs[Long]("num_recordings").toString,
        r.getAs[Long]("num_windows").toString
      ))

  /**
   * Runs the full split stage: track-level split, grouped folds on the training
   * recordings, the disjointness assertion, and the per-genre/per-split summary.
   * Also writes results/ml/split.csv (one row per recording).
   */
  def run(spark: SparkSession, windows: DataFrame): SplitResult = {
    val trackTable = windows.select("track_id", "label").distinct()

    val splitDf = stratifiedTrainTestSplit(spark, trackTable, MlConfig.TrainFraction, MlConfig.Seed)
    val trainTracks = splitDf.filter(col("split") === "train").select("track_id", "label")
    val foldDf = assignGroupedFolds(spark, trainTracks, MlConfig.NumFolds, MlConfig.Seed)
    val splitFoldDf = combineSplitAndFolds(splitDf, foldDf).cache()

    assertDisjoint(splitFoldDf)

    val windowsWithSplit = attachToWindows(windows, splitFoldDf)
    val trainWindows = windowsWithSplit.filter(col("split") === "train").drop("split")
    val testWindows  = windowsWithSplit.filter(col("split") === "test").drop("split")

    val summaryRows = Split.summarize(windowsWithSplit)
    Report.log(s"Track-level split: ${MlConfig.TrainFraction} train / ${1 - MlConfig.TrainFraction} test, " +
      s"stratified by genre, seed ${MlConfig.Seed}. Grouped ${MlConfig.NumFolds}-fold CV on training recordings only.")
    Report.table(Seq("genre", "split", "num_recordings", "num_windows"), summaryRows)
    Report.log("Disjointness check passed: no track_id appears in both train and test.")

    val splitRowsForCsv = splitFoldDf.orderBy("label", "track_id").collect().map(r =>
      Seq(r.getAs[String]("track_id"), r.getAs[String]("label"), r.getAs[String]("split"), r.getAs[Int]("fold").toString)
    )
    Report.saveCsv("split.csv", Seq("track_id", "label", "split", "fold"), splitRowsForCsv, Paths.MlOutputDir)

    SplitResult(trainWindows, testWindows, splitFoldDf)
  }
}
