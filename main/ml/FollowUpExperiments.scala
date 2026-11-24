package sonicspark.ml

import org.apache.spark.sql.SparkSession
import sonicspark.common._

/**
 * Phase 6 - Follow-up experiments (SonicSpark, IT 462 Big Data Systems)
 *
 * A separate program from Phase 5's MachineLearning.scala, which this file
 * never touches: every output here goes to results/ml_followup/ and
 * results/ml_followup_metrics.txt, and the Phase 5 split/fold assignment
 * (results/ml/split.csv) is read back exactly as written, never recomputed.
 *
 *   Part A: re-ranks Phase 5's 20 CV candidates by recording-level OOF
 *           macro-F1 instead of Spark's window-level weighted-F1.
 *   Part B: trains a classifier directly on one row per recording (mean +
 *           stddev of each feature, plus Phase 5's 4 candidate features).
 *   Part C: paired bootstrap + exact McNemar comparison of every model pair
 *           on the 296 test recordings, regenerating Phase 5's own
 *           recording-level test predictions first (and checking them
 *           against results/ml_metrics.txt) since Phase 5 never saved them.
 *
 * All results here are FOLLOW-UP: post-hoc, on a test set already seen once
 * in Phase 5's own evaluation.
 *
 * Run:
 *   sbt "runMain sonicspark.ml.FollowUpExperiments"            (all three parts)
 *   sbt "runMain sonicspark.ml.FollowUpExperiments --part A"   (one part only;
 *   sbt "runMain sonicspark.ml.FollowUpExperiments --part B"    Part C reads
 *   sbt "runMain sonicspark.ml.FollowUpExperiments --part C"    A/B's saved
 *                                                                CSVs if present)
 */
object FollowUpExperiments {

  private def parsePart(args: Array[String]): Option[String] = {
    val idx = args.indexOf("--part")
    if (idx >= 0 && idx + 1 < args.length) Some(args(idx + 1).toUpperCase) else None
  }

  def main(args: Array[String]): Unit = {
    val overallStart = System.nanoTime()
    val part = parsePart(args)   // None = run all three parts

    // Same fixed master/shuffle-partition settings as Phase 5's MachineLearning.scala,
    // for the same reason: deterministic output, not an artifact of free core count.
    val spark = SparkSession.builder()
      .appName("SonicSpark-FollowUpExperiments")
      .master(MlConfig.SparkMaster)
      .config("spark.sql.shuffle.partitions", MlConfig.ShufflePartitions)
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    Report.start(Paths.MlFollowupReport)
    Report.section("SonicSpark Phase 6: Follow-up Experiments")
    Report.log("FOLLOW-UP (post-hoc, test set previously seen in Phase 5). Every result below is labeled " +
      "FOLLOW-UP again at the top of its own section -- this is not a new Phase 5 run.")
    Report.log(s"Part requested: ${part.getOrElse("ALL (A, B, C)")}")
    Report.log(s"Spark master=${MlConfig.SparkMaster}, shuffle.partitions=${MlConfig.ShufflePartitions}, seed=${MlConfig.Seed}.")

    val setup = FollowUpCvUtils.loadSplitAndBaseCols(spark)
    Report.log(s"Loaded results/ml/split.csv unchanged: 688 train / 296 test recordings, no shared track_id " +
      s"(assertion passed). F0 re-derived from training windows only: ${setup.baseCols.size} features kept.")

    if (part.isEmpty || part.contains("A")) {
      val t0 = System.nanoTime()
      FollowUpPartA.run(setup.trainDf, setup.testDf, setup.baseCols)
      Report.log(f"\nPart A wall time: ${(System.nanoTime() - t0) / 1e9}%.1f s")
    }

    if (part.isEmpty || part.contains("B")) {
      val t0 = System.nanoTime()
      FollowUpPartB.run(spark, setup.splitFoldDf)
      Report.log(f"\nPart B wall time: ${(System.nanoTime() - t0) / 1e9}%.1f s")
    }

    if (part.isEmpty || part.contains("C")) {
      val t0 = System.nanoTime()
      FollowUpPartC.run(spark, setup.trainDf, setup.testDf, setup.baseCols, setup.splitFoldDf)
      Report.log(f"\nPart C wall time: ${(System.nanoTime() - t0) / 1e9}%.1f s")
    }

    setup.trainDf.unpersist()
    Report.log(f"\nTotal run time: ${(System.nanoTime() - overallStart) / 1e9}%.1f s")
    Report.log(s"Full report saved to: ${Paths.MlFollowupReport}")
    Report.log(s"CSV outputs saved under: ${Paths.MlFollowupOutputDir}")

    spark.stop()
  }
}
