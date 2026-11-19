package sonicspark.ml

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths => JPaths}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._
import sonicspark.preprocessing.Transformation

/**
 * Audit utility (Phase 5 audit item C4) -- NOT part of the modelling pipeline and not
 * wired into MachineLearning.scala. Standalone check: Transformation.scala (Phase 2)
 * decided its log-transform (which of the 38 features become log_X) using the FULL,
 * pre-split dataset. This object re-applies that exact same rule using only the
 * training recordings (from results/ml/split.csv, already written by a prior
 * MachineLearning run) and reports whether the decision would have come out the same.
 *
 * Input : data/interim/03_reduced   (Transformation.scala's own input -- see its header)
 *         results/ml/split.csv      (train/test assignment, to filter to train track_ids)
 *         outputs/stats/04_skewness.csv (Phase 2's own full-dataset decision, for comparison)
 * Output: results/ml/phase2_transform_recheck.txt
 *
 * Run:
 *   sbt "runMain sonicspark.ml.TransformRecheck"
 */
object TransformRecheck {

  private val SkewThreshold = Transformation.SkewThreshold   // 1.0
  private val LogEpsilon    = Transformation.LogEpsilon      // 1e-6
  private val IdColumns     = Transformation.IdColumns       // track_id, label, window_idx, start_sec

  /** Same formula as Transformation.skewAndMin, reimplemented here (that method is private there). */
  private def skewAndMin(df: DataFrame, cols: Seq[String]): (Map[String, Double], Map[String, Double]) = {
    val exprs = cols.flatMap(c => Seq(skewness(c).as(s"s_$c"), min(c).cast("double").as(s"m_$c")))
    val r     = df.agg(exprs.head, exprs.tail: _*).head()
    (cols.map(c => c -> r.getAs[Double](s"s_$c")).toMap,
     cols.map(c => c -> r.getAs[Double](s"m_$c")).toMap)
  }

  /** Reads outputs/stats/04_skewness.csv (Phase 2's own full-dataset report) and returns,
   *  per ORIGINAL (pre-rename) feature name, its decision: "log", "log_rejected", or "kept". */
  private def readPhase2Decisions(spark: SparkSession): Map[String, (String, Double, Double)] = {
    val df = spark.read.option("header", "true").csv("outputs/stats/04_skewness.csv")
    df.collect().map { (r: Row) =>
      val reportedName = r.getAs[String]("feature")
      val transformNote = r.getAs[String]("transformation")
      val skewBefore = r.getAs[String]("skew_before").toDouble
      val skewAfter  = r.getAs[String]("skew_after").toDouble
      val (origName, decision) =
        if (transformNote == "log") (reportedName.stripPrefix("log_"), "log")
        else if (transformNote.startsWith("log rejected")) (reportedName, "log_rejected")
        else (reportedName, "kept")
      origName -> (decision, skewBefore, skewAfter)
    }.toMap
  }

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-TransformRecheck")
    spark.sparkContext.setLogLevel("WARN")

    // ---------- Load train-only recordings from the already-written split ----------
    val splitDf   = spark.read.option("header", "true").csv("results/ml/split.csv")
    val trainIds  = splitDf.filter(col("split") === "train").select("track_id")
    val numTrain  = trainIds.count()

    // ---------- Input: data/interim/03_reduced, filtered to training recordings ----------
    val reduced   = DataIO.readStage(spark, Paths.Reduced)
    val trainOnly = reduced.join(trainIds, "track_id")
    val n         = trainOnly.count()

    // ---------- T1: same feature engineering as Transformation.scala ----------
    val engineered = trainOnly.withColumn("loudness_variation", sqrt(col("rms_var")) / col("rms_mean")).cache()
    val featureCols = engineered.columns.filterNot(IdColumns.contains).toSeq

    // ---------- T2: same log-transform rule as Transformation.scala, train-only ----------
    val (skewBefore, minValue) = skewAndMin(engineered, featureCols)
    val logCandidates = featureCols.filter(c => skewBefore(c) > SkewThreshold && minValue(c) >= 0)

    val logProbe = engineered.select(logCandidates.map(c => log(col(c) + LogEpsilon).as(c)): _*)
    val (skewAfterProbe, _) = skewAndMin(logProbe, logCandidates)
    val toLog    = logCandidates.filter(c => math.abs(skewAfterProbe(c)) < math.abs(skewBefore(c))).toSet
    val rejected = logCandidates.filterNot(toLog.contains).toSet

    def trainDecision(c: String): String =
      if (toLog.contains(c)) "log" else if (rejected.contains(c)) "log_rejected" else "kept"

    def trainSkewAfter(c: String): Double =
      if (toLog.contains(c)) skewAfterProbe(c) else skewBefore(c)   // unchanged if not logged, same convention as Transformation's own report

    // ---------- Compare against Phase 2's full-dataset decision ----------
    val phase2 = readPhase2Decisions(spark)

    val header = Seq("feature", "skew_before_train", "skew_after_train", "decision_train",
      "skew_before_phase2", "skew_after_phase2", "decision_phase2", "match")
    val rows = featureCols.sorted.map { c =>
      val (p2Decision, p2SkewBefore, p2SkewAfter) = phase2.getOrElse(c,
        throw new IllegalStateException(s"Feature $c not found in outputs/stats/04_skewness.csv -- " +
          "re-run RunPreprocessing before this check."))
      val tDecision = trainDecision(c)
      val tSkewAfter = trainSkewAfter(c)
      val matches = tDecision == p2Decision
      Seq(c, f"${skewBefore(c)}%.4f", f"$tSkewAfter%.4f", tDecision,
        f"$p2SkewBefore%.4f", f"$p2SkewAfter%.4f", p2Decision, matches.toString)
    }

    val differences = rows.filter(_.last == "false")

    val lines = Seq(
      "Phase 2 log-transform recheck (train-only) -- Phase 5 audit, item C4",
      s"Input: ${Paths.Reduced}, filtered to the $numTrain training recordings ($n window rows), " +
        "per results/ml/split.csv.",
      s"Rule re-applied exactly as Transformation.scala: candidate if skewness > $SkewThreshold and min >= 0; " +
        s"transform = log(x + $LogEpsilon); keep the log version only if |skewness after| < |skewness before|.",
      s"Phase 2's own full-dataset decision read from outputs/stats/04_skewness.csv.",
      "",
      header.mkString(","),
    ) ++ rows.map(_.mkString(",")) ++ Seq(
      "",
      if (differences.isEmpty)
        "IDENTICAL -- every feature's train-only log-transform decision matches Phase 2's full-dataset " +
          "decision. The train-only feature space equals data/processed/final: the existing Phase 5 " +
          "results (built on data/processed/final) stand unchanged with respect to this specific check."
      else
        s"DIFFERENCES FOUND (${differences.size} feature(s)): " +
          differences.map(r => s"${r(0)} (train=${r(3)}, phase2=${r(6)})").mkString("; ") +
          " -- STOP: do not treat the existing Phase 5 results as unaffected until this is reviewed."
    )

    val path = JPaths.get(Paths.MlOutputDir).resolve("phase2_transform_recheck.txt")
    Files.createDirectories(path.getParent)
    Files.write(path, lines.mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

    lines.foreach(println)
    println(s"\nSaved to: $path")

    spark.stop()
  }
}
