package sonicspark.ml

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import sonicspark.common._
import scala.io.Source

/**
 * Phase 6 (follow-up experiments): shared setup and ranking helpers used by
 * Part A and Part B. Kept separate from Phase 5's own files so nothing here
 * can accidentally change what MachineLearning.scala produces.
 */
object FollowUpCvUtils {

  // ======================================================================
  // Setup: re-load the EXACT Phase 5 split/fold assignment from disk, never
  // recompute it. This is the one hard rule every part of this file depends on.
  // ======================================================================
  case class FollowUpSplit(trainDf: DataFrame, testDf: DataFrame, splitFoldDf: DataFrame, baseCols: Seq[String])

  def loadSplitAndBaseCols(spark: SparkSession): FollowUpSplit = {
    val windows = DataIO.readStage(spark, Paths.Final)
    val idCols  = (Schema.windowIdColumns :+ "label_idx").toSet
    val candidateBaseCols = windows.columns.filterNot(idCols.contains).toSeq

    val rangeDf  = ExtraFeatures.robustDynamicRange(windows)
    val changeDf = ExtraFeatures.temporalChangeFeatures(windows)
    val windowsWithExtras = ExtraFeatures.attachExtraFeatures(windows, rangeDf, changeDf)

    // results/ml/split.csv, read as-is -- NOT recomputed. "fold" is text in the
    // CSV and must be cast back to Int for CrossValidator/runGroupedOof to use it.
    val splitFoldDf = spark.read.option("header", "true").csv(s"${Paths.MlOutputDir}/split.csv")
      .withColumn("fold", col("fold").cast("int"))

    Split.assertDisjoint(splitFoldDf)
    val numTrain = splitFoldDf.filter(col("split") === "train").count()
    val numTest  = splitFoldDf.filter(col("split") === "test").count()
    require(numTrain == 688 && numTest == 296,
      s"Expected 688 train / 296 test recordings from results/ml/split.csv, found $numTrain / $numTest.")

    val windowsWithSplit = Split.attachToWindows(windowsWithExtras, splitFoldDf)
    val trainDf = windowsWithSplit.filter(col("split") === "train").drop("split")
      .withColumnRenamed("fold", MlConfig.FoldColName).cache()
    val testDf  = windowsWithSplit.filter(col("split") === "test").drop("split", "fold")

    // Re-derives F0 deterministically from the same training windows Phase 5 used --
    // this is reuse of a pure function, not a second, possibly-different decision.
    val baseCols = FeatureAnalysis.analyzeTrainFeatures(trainDf, candidateBaseCols).keptBaseFeatures

    FollowUpSplit(trainDf, testDf, splitFoldDf, baseCols)
  }

  // ======================================================================
  // Minimal CSV reader for results/ml/cv_results.csv, Report.saveCsv's own
  // format (fields containing a comma -- e.g. "numTrees=100,maxDepth=10" --
  // are wrapped in double quotes). Good enough for the small files this
  // project writes; not a general-purpose CSV library.
  // ======================================================================
  private def parseCsvLine(line: String): Seq[String] = {
    val fields = scala.collection.mutable.ArrayBuffer[String]()
    val current = new StringBuilder
    var inQuotes = false
    var i = 0
    while (i < line.length) {
      val c = line(i)
      if (inQuotes) {
        if (c == '"') inQuotes = false else current.append(c)
      } else if (c == '"') {
        inQuotes = true
      } else if (c == ',') {
        fields += current.toString; current.clear()
      } else {
        current.append(c)
      }
      i += 1
    }
    fields += current.toString
    fields.toSeq
  }

  /** Reads a Report.saveCsv-written file into a list of header-keyed rows. */
  def readCsvAsRows(path: String): Seq[Map[String, String]] = {
    val lines = Source.fromFile(path).getLines().toSeq
    val header = parseCsvLine(lines.head)
    lines.tail.map(line => header.zip(parseCsvLine(line)).toMap)
  }

  // ======================================================================
  // Selection rule (fixed here, in code, before any score is computed):
  // highest recording macro-F1; ties (|delta| < TieEpsilon) broken by
  // accuracy; still tied, broken by the simpler configuration. "Simpler"
  // is caller-supplied (fewer features, larger regParam / fewer-shallower
  // trees), as a single ascending-is-simpler Double so the same comparator
  // works for every candidate type below.
  // ======================================================================
  case class Candidate(
    label: String,            // human-readable config description, for the report
    featureCount: Int,
    simplicityTiebreak: Double,
    macroF1: Double,
    accuracy: Double
  )

  /** True if `a` is selected over `b` under the fixed rule above. */
  def isBetter(a: Candidate, b: Candidate): Boolean = {
    if (math.abs(a.macroF1 - b.macroF1) >= FollowUpConfig.TieEpsilon) a.macroF1 > b.macroF1
    else if (a.accuracy != b.accuracy) a.accuracy > b.accuracy
    else if (a.featureCount != b.featureCount) a.featureCount < b.featureCount
    else a.simplicityTiebreak < b.simplicityTiebreak
  }

  /** Candidates ordered best-first under the fixed rule; index 0 is the winner. */
  def rankByNewRule(candidates: Seq[Candidate]): Seq[Candidate] =
    candidates.sortWith(isBetter)

  /** The single best item under the fixed rule, for callers that carry extra fields (e.g. hyperparameters)
   *  alongside each Candidate and just need to pick the winning one. */
  def bestByNewRule[T](items: Seq[T], candidateOf: T => Candidate): T =
    items.reduce((a, b) => if (isBetter(candidateOf(a), candidateOf(b))) a else b)
}
