package sonicspark.ml

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Phase 5, stage 3: EXTRA FEATURES
 *
 * Two candidate features, ported from Phase 4's SQL definitions (Q2 and Q4)
 * but aggregated by recording instead of by genre, so each becomes a single
 * recording-constant value attached to every window of that recording:
 *
 *   - robust dynamic range  (Q2): per-recording p95-p5 of window loudness.
 *   - temporal change       (Q4): mean absolute change between consecutive
 *     windows, for loudness/bandwidth/mfcc1, same LAG + exclusion rules as Q4.
 *
 * Both use no labels and no information from any other recording, so they
 * can legitimately be computed over every recording -- train and test alike
 * -- without leaking test information into training. What they CAN leak is
 * subtler: because the value is identical across all of a recording's
 * windows, a model could learn to "fingerprint" a specific training
 * recording (memorise its exact range/change value) instead of learning
 * anything about genre. Row-level cross-validation would hide this, because
 * a fingerprinted recording's windows would appear in both the training and
 * validation side of a fold. Grouped cross-validation (whole recordings held
 * out per fold, see Split.scala/ModelTraining.scala) is exactly what would
 * expose it: a fingerprint learned from training recordings cannot help on
 * recordings the model has never seen a single window of.
 */
object ExtraFeatures {

  val RangeCol: String = "range_p95_p5"
  val ChangeCols: Seq[String] = Seq("d_loudness_mean_abs", "d_bandwidth_mean_abs", "d_mfcc1_mean_abs")

  /**
   * Q2-style robust dynamic range: per window, loudness_db = 20*log10(rms_mean + eps);
   * per recording, range_p95_p5 = percentile_approx(0.95) - percentile_approx(0.05) of
   * loudness_db across that recording's own windows. Grouped strictly by track_id --
   * never genre-wide or dataset-wide -- exactly mirroring Q2's GROUP BY.
   */
  def robustDynamicRange(windows: DataFrame): DataFrame = {
    val withLoudness = windows.withColumn(
      "loudness_db",
      lit(20.0) * log10(col("rms_mean") + AudioConfig.Epsilon)
    )

    withLoudness.groupBy("track_id").agg(
      (percentile_approx(col("loudness_db"), lit(0.95), lit(10000))
        - percentile_approx(col("loudness_db"), lit(0.05), lit(10000))
      ).as(RangeCol)
    )
  }

  /**
   * Q4-style temporal change, aggregated to one mean-absolute-change value per
   * recording (instead of Q4's per-genre aggregation). Same per-window delta
   * formula and the same two exclusion rules as Q4:
   *   - a track's first window has no predecessor (LAG is null) -> excluded
   *   - a pair whose window_idx gap isn't exactly 1 (a window was removed by
   *     upstream cleaning) is not truly consecutive -> excluded
   */
  def temporalChangeFeatures(windows: DataFrame): DataFrame = {
    val byTrack = Window.partitionBy("track_id").orderBy("window_idx")

    val withDeltas = windows
      .withColumn("loudness_db", lit(20.0) * log10(col("rms_mean") + AudioConfig.Epsilon))
      .withColumn("prev_idx", lag(col("window_idx"), 1).over(byTrack))
      .withColumn("d_loudness", abs(col("loudness_db") - lag(col("loudness_db"), 1).over(byTrack)))
      .withColumn("d_bandwidth", abs(col("spectral_bandwidth_mean") - lag(col("spectral_bandwidth_mean"), 1).over(byTrack)))
      .withColumn("d_mfcc1", abs(col("mfcc1_mean") - lag(col("mfcc1_mean"), 1).over(byTrack)))
      // drop each track's first window (no predecessor) and any non-consecutive gap pair
      .filter(col("prev_idx").isNotNull && (col("window_idx") - col("prev_idx") === 1))

    withDeltas.groupBy("track_id").agg(
      avg("d_loudness").as(ChangeCols(0)),
      avg("d_bandwidth").as(ChangeCols(1)),
      avg("d_mfcc1").as(ChangeCols(2))
    )
  }

  /**
   * Diagnostic matching Q4's own gap-pair count: how many consecutive-window
   * pairs were excluded because the predecessor window was removed upstream
   * (window_idx gap != 1). Phase 4 found 5 in the full 58,020-window dataset;
   * a train-only or test-only subset should find at most that many.
   */
  def countGapPairs(windows: DataFrame): Long = {
    val byTrack = Window.partitionBy("track_id").orderBy("window_idx")
    windows
      .withColumn("prev_idx", lag(col("window_idx"), 1).over(byTrack))
      .filter(col("prev_idx").isNotNull && (col("window_idx") - col("prev_idx") =!= 1))
      .count()
  }

  /** Joins both extra-feature tables onto every window of the recordings they describe. */
  def attachExtraFeatures(windows: DataFrame, range: DataFrame, change: DataFrame): DataFrame =
    windows.join(range, "track_id").join(change, "track_id")

  /**
   * F0 = base features only (post train-only-analysis). F1 = +range. F2 = +change.
   * F3 = +both. The decision between these four is made only by comparing their
   * grouped-CV scores (see ModelTraining.scala) -- never by looking at which one
   * "sounds like it should help".
   */
  def featureSet(variant: String, baseCols: Seq[String]): Seq[String] = variant match {
    case "F0" => baseCols
    case "F1" => baseCols :+ RangeCol
    case "F2" => baseCols ++ ChangeCols
    case "F3" => baseCols ++ (RangeCol +: ChangeCols)
    case other => throw new IllegalArgumentException(s"Unknown feature-set variant: $other")
  }
}
