package sonicspark.ml

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths => JPaths, StandardOpenOption}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import sonicspark.common._
import sonicspark.preprocessing.Reduction

/**
 * Audit utility (Phase 5 audit, remaining Reduction caveat) -- NOT part of the
 * modelling pipeline, not wired into MachineLearning.scala, does not modify any
 * existing main/ml file. Phase 2's Reduction.scala dropped `spectral_centroid_mean`
 * (R2: |r| > 0.95 with rolloff_mean, lower eta^2) using the FULL, pre-split dataset.
 * This object re-runs Reduction's exact two-step rule (R1 relevance, then R2
 * redundancy: process pairs by descending |r|, drop the lower-eta^2 feature of each
 * pair) on the 38 pre-reduction features, restricted to training recordings only, to
 * check whether the same single feature would be dropped.
 *
 * Reuses FeatureAnalysis.etaSquared / FeatureAnalysis.correlatedPairs (existing,
 * unmodified public methods -- the same formulas Reduction.scala itself uses).
 *
 * Input : data/interim/02_integrated   (Reduction.scala's own input, still has all 38 features)
 *         results/ml/split.csv         (train/test assignment, to filter to train track_ids)
 * Output: appended to results/ml/phase2_selection_recheck.txt
 *
 * Run:
 *   sbt "runMain sonicspark.ml.ReductionRecheck"
 */
object ReductionRecheck {

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-ReductionRecheck")
    spark.sparkContext.setLogLevel("WARN")

    // ---------- Load train-only recordings from the already-written split ----------
    val splitDf  = spark.read.option("header", "true").csv("results/ml/split.csv")
    val trainIds = splitDf.filter(col("split") === "train").select("track_id")
    val numTrain = trainIds.count()

    // ---------- Input: data/interim/02_integrated, filtered to training recordings ----------
    val integrated = DataIO.readStage(spark, Paths.Integrated)
    val trainOnly  = integrated.join(trainIds, "track_id").cache()
    val n          = trainOnly.count()

    val candidates = Schema.windowFeatureColumns   // all 38 pre-reduction features
    require(candidates.size == 38, s"Expected 38 pre-reduction features, got ${candidates.size}")

    // ---------- R1: relevance (eta^2 < 0.01), train-only, same rule as Reduction.scala ----------
    val eta2         = FeatureAnalysis.etaSquared(trainOnly, candidates, n)
    val lowRelevance = candidates.filter(c => eta2(c) < Reduction.MinEta2)
    val relevant     = candidates.filterNot(lowRelevance.contains)

    // ---------- Full pairwise correlation among the 38 features, train-only ----------
    // threshold = 0.0 so every pair is returned; we filter for >0.95 ourselves below,
    // and separately pull out every pair involving spectral_centroid_mean.
    val allPairs = FeatureAnalysis.correlatedPairs(trainOnly, candidates, 0.0)

    val targetFeature = "spectral_centroid_mean"
    val targetPairs = allPairs
      .filter(p => p._1 == targetFeature || p._2 == targetFeature)
      .map { case (a, b, r) => (if (a == targetFeature) b else a, r) }
      .sortBy { case (_, r) => -math.abs(r) }

    val actionablePairs = allPairs.filter(p => math.abs(p._3) > Reduction.MaxCorr)

    // ---------- R2: redundancy, train-only, Reduction.scala's exact greedy rule ----------
    // (restricted to `relevant`, matching Reduction.scala's own `correlatedPairs(df, relevant)` scope)
    val actionableAmongRelevant = actionablePairs.filter(p => relevant.contains(p._1) && relevant.contains(p._2))
    val redundant = scala.collection.mutable.LinkedHashMap[String, (String, Double)]()
    actionableAmongRelevant.sortBy(p => -math.abs(p._3)).foreach { case (a, b, r) =>
      if (!redundant.contains(a) && !redundant.contains(b)) {
        val (keep, drop) = if (eta2(a) >= eta2(b)) (a, b) else (b, a)
        redundant(drop) = (keep, r)
      }
    }

    val droppedSet = lowRelevance.toSet ++ redundant.keySet
    val expectedSet = Set(targetFeature)
    val matchesExactly = droppedSet == expectedSet

    // ---------- Build the appended report ----------
    val lines = scala.collection.mutable.ArrayBuffer[String]()
    lines += ""
    lines += "--- Reduction full-rule recheck (train-only) -- closes the audit's remaining Reduction caveat ---"
    lines += s"Input: ${Paths.Integrated}, filtered to the $numTrain training recordings ($n window rows), " +
      "per results/ml/split.csv."
    lines += s"Candidates: all ${candidates.size} pre-reduction features (Schema.windowFeatureColumns)."
    lines += ""
    lines += s"R1 (relevance, eta^2 < ${Reduction.MinEta2}), train-only: " +
      (if (lowRelevance.isEmpty) "0 features dropped (matches the full-dataset result: 0 dropped)."
       else s"${lowRelevance.size} feature(s) dropped: ${lowRelevance.mkString(", ")} (full dataset dropped 0 -- DIFFERS).")
    lines += ""
    lines += s"Pearson r($targetFeature, X), train-only, for all other ${candidates.size - 1} pre-reduction features " +
      "(sorted by |r| descending):"
    targetPairs.foreach { case (other, r) =>
      val flag = if (math.abs(r) > Reduction.MaxCorr) "  <-- |r| > 0.95" else ""
      lines += f"  $targetFeature, $other, r = $r%.4f$flag"
    }
    lines += ""
    lines += s"All pairs among the ${candidates.size} pre-reduction features with |r| > ${Reduction.MaxCorr} (train-only):"
    if (actionablePairs.isEmpty) {
      lines += "  (none)"
    } else {
      actionablePairs.sortBy(p => -math.abs(p._3)).foreach { case (a, b, r) =>
        lines += f"  $a, $b, r = $r%.4f"
      }
    }
    lines += ""
    lines += "R2 (redundancy), train-only, Reduction's exact greedy rule (process actionable pairs by descending " +
      "|r|, drop the lower-eta^2 feature, skip a pair if either side is already dropped):"
    if (redundant.isEmpty) {
      lines += "  (nothing dropped)"
    } else {
      redundant.foreach { case (drop, (keep, r)) =>
        lines += f"  drop $drop (eta^2=${eta2(drop)}%.4f), keep $keep (eta^2=${eta2(keep)}%.4f), r=$r%.4f"
      }
    }
    lines += ""
    lines += s"Train-only R1+R2 would drop: {${droppedSet.toSeq.sorted.mkString(", ")}}"
    lines += s"Phase 2's full-dataset rule dropped: {$targetFeature}"
    lines += ""
    if (matchesExactly) {
      lines += "Phase 2 Reduction fully re-verified on training data."
    } else {
      lines += s"DIFFERENCE FOUND: train-only R1+R2 drops {${droppedSet.toSeq.sorted.mkString(", ")}}, " +
        s"not exactly {$targetFeature}. STOP -- do not treat Phase 2's Reduction output as unaffected " +
        "until this is reviewed."
    }

    val path = JPaths.get(Paths.MlOutputDir).resolve("phase2_selection_recheck.txt")
    Files.write(path, (System.lineSeparator() + lines.mkString(System.lineSeparator())).getBytes(StandardCharsets.UTF_8),
      StandardOpenOption.APPEND)

    lines.foreach(println)
    println(s"\nAppended to: $path")

    spark.stop()
  }
}
