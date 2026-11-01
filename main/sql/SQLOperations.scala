package sonicspark.sql

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths => JPaths}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import sonicspark.common._

/**
 * Phase 4 - Spark SQL Operations (SonicSpark, IT 462 Big Data Systems)
 *
 * EXPLORATORY ONLY: every query below is descriptive analysis over the already-finished
 * Phase 2 dataset. No result here feeds feature selection, scaling, or model training --
 * Phase 5 must derive its own statistics from the training split only.
 *
 * Question this file answers: beyond Phase 3's RDD-level profiles, what do genre-level
 * variance structure, recording dynamics, position-within-excerpt effects, frame-to-frame
 * stability, segment-level tempo/loudness correlation, and a simple nearest-centroid
 * exploration reveal about how separable GTZAN's 10 genres are?
 *
 * Input : data/processed/final        (Phase 2's final window-level table, 58,020 rows)
 *         data/raw/features_3_sec.csv (3-second reference table; tempo + join keys only)
 * Output: results/sql_output.txt      (full printed report: question + SQL + result per query)
 *         results/sql/q*.csv          (one small CSV per query result)
 *         results/sql_results.csv     (every query's result, combined in long format)
 *
 * Run:
 *   sbt "runMain sonicspark.sql.SQLOperations"
 */
object SQLOperations {

  // ======================================================================
  // SqlConfig: every tunable constant used below, in one place
  // ======================================================================
  object SqlConfig {
    // Q1: same 6 features as RDDOperations.RddConfig.KeyFeatures, reused here for
    // cross-phase comparability: three most genre-relevant spectral features plus
    // three energy features
    val VarianceFeatures: Seq[String] = Seq(
      "spectral_bandwidth_mean", "mfcc1_mean", "rolloff_mean",
      "loudness_variation", "rms_mean", "zcr_mean"
    )

    // Q2/Q3/Q4: fixed accuracy parameter for percentile_approx, so results are
    // deterministic and reproducible run to run (same input -> same output)
    val PercentileAccuracy = 10000

    // Q2: recording with the largest max-min range, checked under both measures
    val Q2CheckTrack = "hiphop.00038"

    // Q6: how many off-diagonal (confused) genre pairs to report
    val TopOffDiagonalPairs = 10
  }

  // ======================================================================
  // Combined long-format accumulator: (query_id, question, row_num, column, value)
  // Collected across every query and flushed once to results/sql_results.csv
  // ======================================================================
  private val longRows = scala.collection.mutable.ArrayBuffer.empty[Seq[String]]

  private def escape(s: String): String =
    if (s.exists(c => c == ',' || c == '"' || c == '\n')) "\"" + s.replace("\"", "\"\"") + "\""
    else s

  /** Appends every cell of `rows` (header-aligned) to the combined long-format buffer. */
  private def recordLong(queryId: String, question: String, header: Seq[String], rows: Seq[Seq[String]]): Unit = {
    rows.zipWithIndex.foreach { case (row, i) =>
      header.zip(row).foreach { case (col, value) =>
        longRows += Seq(queryId, question, (i + 1).toString, col, value)
      }
    }
  }

  /** Writes the combined long-format CSV (collect-then-Java-IO, no Spark part-files). */
  private def writeLongCsv(): Unit = {
    val header = Seq("query_id", "question", "row_num", "column", "value")
    val lines  = (header +: longRows).map(_.map(escape).mkString(","))
    val path   = JPaths.get(Paths.SqlResultsCsv)
    Files.write(path, lines.mkString("\n").getBytes(StandardCharsets.UTF_8))
  }

  // Formats one Spark Row as strings, in column order, for printing/CSV/long-format reuse.
  private def rowToStrings(row: Row): Seq[String] =
    row.schema.fields.indices.map { i =>
      val v = row.get(i)
      if (v == null) "" else v.toString
    }

  /**
   * Runs one query end to end: logs the question + SQL, prints/saves the result table,
   * and records it into the combined long-format buffer. Returns the collected rows so
   * callers that need a second pass (e.g. Q7's EXPLAIN on Q5) can reuse the SQL text.
   */
  private def runQuery(spark: SparkSession, queryId: String, question: String, sql: String, csvName: String): Array[Row] =
    runQueryDf(queryId, question, sql, spark.sql(sql), csvName)

  /** Same as runQuery, but takes an already-built DataFrame (e.g. after na.fill) instead of running sql directly. */
  private def runQueryDf(queryId: String, question: String, sql: String, df: DataFrame, csvName: String): Array[Row] = {
    Report.log(s"\n--- $queryId: $question ---")
    Report.log(sql)

    val header = df.columns.toSeq
    val rows   = df.collect()
    val strRows = rows.map(rowToStrings).toSeq

    Report.table(header, strRows)
    Report.saveCsv(csvName, header, strRows, Paths.SqlOutputDir)
    recordLong(queryId, question, header, strRows)

    rows
  }

  // ======================================================================
  // View setup
  // ======================================================================

  /**
   * Registers windows / segments / tracks as temporary views:
   *   windows  : data/processed/final + segment_idx = FLOOR(start_sec / 3)
   *   segments : features_3_sec.csv (filename, tempo only) parsed to (track_id, segment_idx),
   *              restricted to segments that match at least one cleaned window (inner join)
   *   tracks   : built from windows only (track_id, label, num_windows) -- see the report-back
   *              note on why track_length is NOT included (features_30_sec.csv is off-limits here)
   */
  def registerViews(spark: SparkSession): Unit = {
    val finalDf = spark.read.parquet(Paths.Final)
    finalDf.createOrReplaceTempView("final_raw")

    spark.sql(
      """
        |SELECT *, CAST(FLOOR(start_sec / 3) AS INT) AS segment_idx
        |FROM final_raw
      """.stripMargin
    ).createOrReplaceTempView("windows")

    // Only filename + tempo are kept: features_3_sec.csv's other feature columns are not used.
    DataIO.readFeaturesCsv(spark, Paths.Raw3Sec)
      .select("filename", "tempo")
      .createOrReplaceTempView("segments_raw")

    spark.sql(
      """
        |SELECT
        |  regexp_extract(filename, '^([a-z]+\\.\\d{5})\\.', 1) AS track_id,
        |  CAST(regexp_extract(filename, '^[a-z]+\\.\\d{5}\\.(\\d+)\\.wav$', 1) AS INT) AS segment_idx,
        |  tempo
        |FROM segments_raw
      """.stripMargin
    ).createOrReplaceTempView("segments_all")

    spark.sql(
      """
        |SELECT s.track_id, s.segment_idx, s.tempo
        |FROM segments_all s
        |INNER JOIN (SELECT DISTINCT track_id, segment_idx FROM windows) w
        |  ON s.track_id = w.track_id AND s.segment_idx = w.segment_idx
      """.stripMargin
    ).createOrReplaceTempView("segments")

    spark.sql(
      """
        |SELECT track_id, label, COUNT(*) AS num_windows
        |FROM windows
        |GROUP BY track_id, label
      """.stripMargin
    ).createOrReplaceTempView("tracks")
  }

  /** The 3 required count assertions, run against the views just registered. */
  def assertCounts(spark: SparkSession): Unit = {
    val windowCount  = spark.sql("SELECT COUNT(*) AS n FROM windows").head().getAs[Long]("n")
    val trackCount   = spark.sql("SELECT COUNT(*) AS n FROM tracks").head().getAs[Long]("n")
    val segmentCount = spark.sql("SELECT COUNT(*) AS n FROM segments").head().getAs[Long]("n")

    Report.log(s"windows : $windowCount (expected 58,020)")
    Report.log(s"tracks  : $trackCount (expected 984 recordings)")
    Report.log(s"segments: $segmentCount (Phase 3's RDD join reported 9,827 joined segments)")

    require(windowCount == 58020L, s"Expected 58,020 windows, got $windowCount")
    require(trackCount == 984L, s"Expected 984 recordings, got $trackCount")

    if (segmentCount == 9827L) {
      Report.log("segments view matches Phase 3's RDD join exactly: both inner-join features_3_sec.csv " +
        "against the same cleaned base (data/processed/final) on (track_id, segment_idx).")
    } else {
      Report.log(s"segments view differs from Phase 3 by ${segmentCount - 9827L} -- " +
        "if this ever shows up, re-check the filename regex against features_3_sec.csv's actual names.")
    }
  }

  // ======================================================================
  // Q1: Variance decomposition
  //   Question: for a handful of key features, how much of the total variance across all
  //   windows is within-recording, between-recording-within-genre, or between-genre?
  // ======================================================================
  def runQ1(spark: SparkSession): Unit = {
    val n = SqlConfig.VarianceFeatures.size
    val stackArgs = SqlConfig.VarianceFeatures.map(f => s"'$f', $f").mkString(", ")

    spark.sql(
      s"""
         |SELECT track_id, label AS genre, stack($n, $stackArgs) AS (feature, value)
         |FROM windows
      """.stripMargin
    ).createOrReplaceTempView("q1_long")

    val decompositionSql =
      """
        |WITH grand AS (
        |  SELECT feature, AVG(value) AS grand_mean
        |  FROM q1_long GROUP BY feature
        |),
        |genre_stats AS (
        |  SELECT feature, genre, AVG(value) AS genre_mean, COUNT(*) AS n_genre
        |  FROM q1_long GROUP BY feature, genre
        |),
        |recording_stats AS (
        |  SELECT feature, track_id, genre, AVG(value) AS rec_mean,
        |         VAR_SAMP(value) AS rec_var, COUNT(*) AS n_rec
        |  FROM q1_long GROUP BY feature, track_id, genre
        |),
        |ss_within AS (
        |  -- within-recording: sum over recordings of (n_rec - 1) * rec_var
        |  -- (VAR_SAMP * (n-1) = sum of squared deviations from the recording's own mean)
        |  SELECT feature, SUM((n_rec - 1) * COALESCE(rec_var, 0.0)) AS ss_within
        |  FROM recording_stats GROUP BY feature
        |),
        |ss_between_rec AS (
        |  -- between recordings, within genre: sum over recordings of n_rec * (rec_mean - genre_mean)^2
        |  SELECT r.feature, SUM(r.n_rec * POWER(r.rec_mean - g.genre_mean, 2)) AS ss_between_rec
        |  FROM recording_stats r JOIN genre_stats g
        |    ON r.feature = g.feature AND r.genre = g.genre
        |  GROUP BY r.feature
        |),
        |ss_between_genre AS (
        |  -- between genres: sum over genres of n_genre * (genre_mean - grand_mean)^2
        |  SELECT g.feature, SUM(g.n_genre * POWER(g.genre_mean - gr.grand_mean, 2)) AS ss_between_genre
        |  FROM genre_stats g JOIN grand gr ON g.feature = gr.feature
        |  GROUP BY g.feature
        |)
        |SELECT
        |  w.feature,
        |  ROUND(w.ss_within, 2)             AS ss_within,
        |  ROUND(br.ss_between_rec, 2)        AS ss_between_recording,
        |  ROUND(bg.ss_between_genre, 2)      AS ss_between_genre,
        |  ROUND(w.ss_within + br.ss_between_rec + bg.ss_between_genre, 2) AS ss_total,
        |  ROUND(w.ss_within / (w.ss_within + br.ss_between_rec + bg.ss_between_genre), 4)        AS within_recording_share,
        |  ROUND(br.ss_between_rec / (w.ss_within + br.ss_between_rec + bg.ss_between_genre), 4)  AS between_recording_share,
        |  ROUND(bg.ss_between_genre / (w.ss_within + br.ss_between_rec + bg.ss_between_genre), 4) AS between_genre_share
        |FROM ss_within w
        |JOIN ss_between_rec br ON w.feature = br.feature
        |JOIN ss_between_genre bg ON w.feature = bg.feature
        |ORDER BY between_genre_share DESC, w.feature ASC
      """.stripMargin

    runQuery(spark, "Q1_decomp",
      "For each key feature, what share of total variance across all windows is within-recording, " +
        "between-recording-within-genre, and between-genre?",
      decompositionSql, "q1_variance_decomposition.csv")

    val withinByGenreSql =
      """
        |WITH recording_stats AS (
        |  SELECT feature, track_id, genre, VAR_SAMP(value) AS rec_var, COUNT(*) AS n_rec
        |  FROM q1_long GROUP BY feature, track_id, genre
        |)
        |SELECT
        |  feature, genre,
        |  ROUND(SUM((n_rec - 1) * COALESCE(rec_var, 0.0)), 2) AS ss_within_recording,
        |  SUM(n_rec) AS n_windows
        |FROM recording_stats
        |GROUP BY feature, genre
        |ORDER BY feature ASC, genre ASC
      """.stripMargin

    runQuery(spark, "Q1_within_by_genre",
      "Within the within-recording variance share, how is it distributed across genres, per feature?",
      withinByGenreSql, "q1_within_recording_by_genre.csv")
  }

  // ======================================================================
  // Q2: Robust recording-level dynamic range
  //   Question: per recording, how wide is the dynamic range in dB, and does a robust
  //   (p95-p5) estimate change which recordings/genres look "most dynamic" vs. a raw max-min?
  // ======================================================================
  def runQ2(spark: SparkSession): Unit = {
    spark.sql(
      s"""
         |SELECT track_id, label AS genre, window_idx,
         |       20 * LOG10(rms_mean + ${AudioConfig.Epsilon}) AS loudness_db
         |FROM windows
      """.stripMargin
    ).createOrReplaceTempView("q2_window_db")

    spark.sql(
      s"""
         |SELECT track_id, genre,
         |       MAX(loudness_db) - MIN(loudness_db) AS range_minmax,
         |       PERCENTILE_APPROX(loudness_db, 0.95, ${SqlConfig.PercentileAccuracy})
         |         - PERCENTILE_APPROX(loudness_db, 0.05, ${SqlConfig.PercentileAccuracy}) AS range_p95_p5
         |FROM q2_window_db
         |GROUP BY track_id, genre
      """.stripMargin
    ).createOrReplaceTempView("q2_per_recording")

    val summarySql =
      """
        |WITH threshold AS (
        |  SELECT AVG(range_minmax) + STDDEV_SAMP(range_minmax)   AS thresh_minmax,
        |         AVG(range_p95_p5) + STDDEV_SAMP(range_p95_p5)   AS thresh_p95_p5
        |  FROM q2_per_recording
        |)
        |SELECT
        |  genre,
        |  ROUND(AVG(range_minmax), 2)                                    AS mean_range_minmax,
        |  ROUND(PERCENTILE_APPROX(range_minmax, 0.5, 10000), 2)          AS median_range_minmax,
        |  ROUND(AVG(range_p95_p5), 2)                                    AS mean_range_p95_p5,
        |  ROUND(PERCENTILE_APPROX(range_p95_p5, 0.5, 10000), 2)          AS median_range_p95_p5,
        |  SUM(CASE WHEN range_minmax > (SELECT thresh_minmax FROM threshold) THEN 1 ELSE 0 END) AS n_above_threshold_minmax,
        |  SUM(CASE WHEN range_p95_p5 > (SELECT thresh_p95_p5 FROM threshold) THEN 1 ELSE 0 END) AS n_above_threshold_p95_p5
        |FROM q2_per_recording
        |GROUP BY genre
        |ORDER BY mean_range_p95_p5 DESC, genre ASC
      """.stripMargin

    runQuery(spark, "Q2_summary",
      "Per genre, what is the mean/median recording-level dynamic range (max-min vs. robust p95-p5), " +
        "and how many recordings exceed a mean+1SD threshold under each metric?",
      summarySql, "q2_dynamic_range.csv")

    val checkSql =
      s"""
         |WITH ranked AS (
         |  SELECT track_id, genre, range_minmax, range_p95_p5,
         |         RANK() OVER (ORDER BY range_minmax DESC) AS rank_minmax,
         |         RANK() OVER (ORDER BY range_p95_p5 DESC) AS rank_p95_p5
         |  FROM q2_per_recording
         |)
         |SELECT track_id, genre, ROUND(range_minmax, 2) AS range_minmax, ROUND(range_p95_p5, 2) AS range_p95_p5,
         |       rank_minmax, rank_p95_p5
         |FROM ranked
         |WHERE track_id = '${SqlConfig.Q2CheckTrack}'
      """.stripMargin

    runQuery(spark, "Q2_check",
      s"Where does ${SqlConfig.Q2CheckTrack} rank among all 984 recordings under max-min vs. the robust p95-p5 metric?",
      checkSql, "q2_hiphop00038_check.csv")
  }

  // ======================================================================
  // Q3: Position effect
  //   Question: does a window's position within the 30-second excerpt affect its loudness
  //   and spectral features? (GTZAN clips are 30-second excerpts, not full songs, so the
  //   first/last windows are not song intros/outros -- just the start/end of the excerpt.)
  // ======================================================================
  def runQ3(spark: SparkSession): Unit = {
    val sql =
      s"""
         |WITH positioned AS (
         |  SELECT label AS genre,
         |         20 * LOG10(rms_mean + ${AudioConfig.Epsilon}) AS loudness_db,
         |         spectral_bandwidth_mean,
         |         window_idx * 1.0 / MAX(window_idx) OVER (PARTITION BY track_id) AS rel_pos
         |  FROM windows
         |),
         |bucketed AS (
         |  SELECT genre, loudness_db, spectral_bandwidth_mean,
         |         CASE WHEN rel_pos <= 0.10 THEN 'intro'
         |              WHEN rel_pos >= 0.90 THEN 'outro'
         |              ELSE 'middle' END AS position_bucket
         |  FROM positioned
         |)
         |SELECT genre, position_bucket,
         |       ROUND(AVG(loudness_db), 2) AS avg_loudness_db,
         |       ROUND(AVG(spectral_bandwidth_mean), 2) AS avg_spectral_bandwidth_mean,
         |       COUNT(*) AS n_windows
         |FROM bucketed
         |GROUP BY genre, position_bucket
         |ORDER BY genre ASC,
         |  CASE position_bucket WHEN 'intro' THEN 1 WHEN 'middle' THEN 2 WHEN 'outro' THEN 3 END ASC
      """.stripMargin

    runQuery(spark, "Q3",
      "Does a window's position within the 30-second excerpt affect its loudness and spectral features?",
      sql, "q3_position_effect.csv")
  }

  // ======================================================================
  // Q4: Temporal stability
  //   Question: which genres change the most/least from one window to the next
  //   (loudness, bandwidth, mfcc1), and how do genres rank against each other?
  // ======================================================================
  def runQ4(spark: SparkSession): Unit = {
    val excludedPairs = spark.sql(
      """
        |WITH deltas AS (
        |  SELECT track_id, window_idx,
        |    LAG(window_idx) OVER (PARTITION BY track_id ORDER BY window_idx) AS prev_idx
        |  FROM windows
        |)
        |SELECT COUNT(*) AS n
        |FROM deltas
        |WHERE prev_idx IS NOT NULL AND window_idx - prev_idx <> 1
      """.stripMargin
    ).head().getAs[Long]("n")
    Report.log(s"Q4: $excludedPairs consecutive-window pair(s) excluded " +
      "(predecessor window removed in cleaning, so the pair is not truly consecutive).")

    val sql =
      s"""
         |WITH deltas AS (
         |  SELECT track_id, label AS genre, window_idx,
         |    LAG(window_idx) OVER (PARTITION BY track_id ORDER BY window_idx) AS prev_idx,
         |    ABS(20 * LOG10(rms_mean + ${AudioConfig.Epsilon})
         |        - LAG(20 * LOG10(rms_mean + ${AudioConfig.Epsilon})) OVER (PARTITION BY track_id ORDER BY window_idx)) AS d_loudness,
         |    ABS(spectral_bandwidth_mean
         |        - LAG(spectral_bandwidth_mean) OVER (PARTITION BY track_id ORDER BY window_idx)) AS d_bandwidth,
         |    ABS(mfcc1_mean
         |        - LAG(mfcc1_mean) OVER (PARTITION BY track_id ORDER BY window_idx)) AS d_mfcc1
         |  FROM windows
         |),
         |genre_stability AS (
         |  SELECT genre,
         |    ROUND(AVG(d_loudness), 4) AS mean_abs_change_loudness,
         |    ROUND(AVG(d_bandwidth), 4) AS mean_abs_change_bandwidth,
         |    ROUND(AVG(d_mfcc1), 4) AS mean_abs_change_mfcc1
         |  FROM deltas
         |  WHERE d_loudness IS NOT NULL        -- drops each track's first window (no predecessor)
         |    AND window_idx - prev_idx = 1     -- drops pairs whose predecessor was removed in cleaning (not truly consecutive)
         |  GROUP BY genre
         |)
         |SELECT genre, mean_abs_change_loudness, mean_abs_change_bandwidth, mean_abs_change_mfcc1,
         |  RANK() OVER (ORDER BY mean_abs_change_loudness DESC) AS rank_loudness,
         |  RANK() OVER (ORDER BY mean_abs_change_bandwidth DESC) AS rank_bandwidth,
         |  RANK() OVER (ORDER BY mean_abs_change_mfcc1 DESC) AS rank_mfcc1
         |FROM genre_stability
         |ORDER BY rank_loudness ASC, genre ASC
      """.stripMargin

    runQuery(spark, "Q4",
      "Per genre, how much does loudness/bandwidth/mfcc1 change on average between consecutive " +
        "windows, and how do genres rank by that instability?",
      sql, "q4_temporal_stability.csv")
  }

  // ======================================================================
  // Q5: Three-level join (windows join segments join tracks)
  //   Question: at the 3-second-segment level, how correlated is loudness variation with tempo,
  //   overall and per genre -- using both Pearson and a Spearman rank correlation (less
  //   sensitive to extreme tempo values)?
  // ======================================================================
  val Q5JoinSql: String =
    """
      |WITH matched_windows AS (
      |  SELECT w.track_id, w.segment_idx, w.label AS genre, w.loudness_variation
      |  FROM windows w
      |  WHERE w.start_sec - 3 * w.segment_idx <= 2  -- keep only windows fully inside their 3-second segment
      |),
      |segment_loudness AS (
      |  SELECT m.track_id, m.segment_idx, m.genre, AVG(m.loudness_variation) AS avg_loudness_variation
      |  FROM matched_windows m
      |  GROUP BY m.track_id, m.segment_idx, m.genre
      |)
      |SELECT sl.track_id, sl.segment_idx, sl.genre, sl.avg_loudness_variation, s.tempo
      |FROM segment_loudness sl
      |JOIN segments s ON sl.track_id = s.track_id AND sl.segment_idx = s.segment_idx
      |JOIN tracks t   ON sl.track_id = t.track_id
    """.stripMargin

  def runQ5(spark: SparkSession): Unit = {
    spark.sql(Q5JoinSql).createOrReplaceTempView("q5_joined")

    val sql =
      """
        |WITH ranked AS (
        |  -- average ranks (not RANK()'s minimum rank) for ties, as required by Spearman --
        |  -- tempo only has 61 distinct values, so ties are common
        |  SELECT *,
        |    RANK() OVER (ORDER BY avg_loudness_variation)
        |      + (COUNT(*) OVER (PARTITION BY avg_loudness_variation) - 1) / 2.0 AS rank_loudness,
        |    RANK() OVER (ORDER BY tempo)
        |      + (COUNT(*) OVER (PARTITION BY tempo) - 1) / 2.0 AS rank_tempo,
        |    RANK() OVER (PARTITION BY genre ORDER BY avg_loudness_variation)
        |      + (COUNT(*) OVER (PARTITION BY genre, avg_loudness_variation) - 1) / 2.0 AS rank_loudness_genre,
        |    RANK() OVER (PARTITION BY genre ORDER BY tempo)
        |      + (COUNT(*) OVER (PARTITION BY genre, tempo) - 1) / 2.0 AS rank_tempo_genre
        |  FROM q5_joined
        |)
        |SELECT genre,
        |  ROUND(CORR(avg_loudness_variation, tempo), 4) AS pearson_corr,
        |  ROUND(CORR(rank_loudness_genre, rank_tempo_genre), 4) AS spearman_corr,
        |  COUNT(*) AS n_segments
        |FROM ranked
        |GROUP BY genre
        |UNION ALL
        |SELECT 'OVERALL' AS genre,
        |  ROUND(CORR(avg_loudness_variation, tempo), 4),
        |  ROUND(CORR(rank_loudness, rank_tempo), 4),
        |  COUNT(*)
        |FROM ranked
        |ORDER BY CASE WHEN genre = 'OVERALL' THEN 0 ELSE 1 END ASC, genre ASC
      """.stripMargin

    runQuery(spark, "Q5",
      "At the 3-second-segment level, how correlated is average loudness variation with tempo " +
        "(Pearson, and Spearman rank correlation, which is less sensitive to extreme tempo " +
        "values), per genre and overall?",
      sql, "q5_loudness_tempo_corr.csv")
  }

  // ======================================================================
  // Q7: EXPLAIN for Q5's three-level join
  //   Question: which physical join strategy does Spark choose for Q5's windows/segments/tracks join?
  // ======================================================================
  def runQ7(spark: SparkSession): Unit = {
    val plan = spark.sql(s"EXPLAIN FORMATTED $Q5JoinSql").collect().map(_.getString(0)).mkString("\n")

    Report.log("\n--- Q7: What join strategy does Spark choose for Q5's three-level join? ---")
    Report.log(s"EXPLAIN FORMATTED $Q5JoinSql")
    Report.log(plan)

    val strategy =
      if (plan.contains("BroadcastHashJoin")) "BroadcastHashJoin (small side broadcast to every executor)"
      else if (plan.contains("SortMergeJoin")) "SortMergeJoin (both sides shuffled and sorted)"
      else if (plan.contains("ShuffledHashJoin")) "ShuffledHashJoin"
      else "Could not identify a standard join operator name in the plan -- see full text above."

    Report.log(s"Join strategy chosen: $strategy")

    val question = "Which physical join strategy does Spark choose for Q5's windows/segments/tracks join?"
    recordLong("Q7", question, Seq("column", "value"), Seq(Seq("join_strategy", strategy)))
    recordLong("Q7", question, Seq("column", "value"), Seq(Seq("explain_formatted", plan)))
  }

  // ======================================================================
  // Q6: Recording-level nearest centroid (exploratory, full data, NOT a model baseline)
  //   Question: if every recording is z-scored and compared to each genre's average profile,
  //   how often does its nearest centroid match its true genre, and which genres get confused?
  // ======================================================================
  def runQ6(spark: SparkSession, featureCols: Seq[String]): Unit = {
    val n = featureCols.size
    val stackArgs = featureCols.map(f => s"'$f', $f").mkString(", ")

    spark.sql(
      s"""
         |SELECT track_id, label AS genre, stack($n, $stackArgs) AS (feature, value)
         |FROM windows
      """.stripMargin
    ).createOrReplaceTempView("q6_long")

    spark.sql(
      """
        |SELECT track_id, genre, feature,
        |  (value - AVG(value) OVER (PARTITION BY feature)) / STDDEV_SAMP(value) OVER (PARTITION BY feature) AS z_value
        |FROM q6_long
      """.stripMargin
    ).createOrReplaceTempView("q6_long_z")

    spark.sql(
      """
        |SELECT track_id, genre, feature, AVG(z_value) AS avg_z
        |FROM q6_long_z
        |GROUP BY track_id, genre, feature
      """.stripMargin
    ).createOrReplaceTempView("q6_recording_profile")

    spark.sql(
      """
        |SELECT genre AS centroid_genre, feature, AVG(avg_z) AS centroid_z
        |FROM q6_recording_profile
        |GROUP BY genre, feature
      """.stripMargin
    ).createOrReplaceTempView("q6_genre_centroid")

    spark.sql(
      """
        |WITH recording_genres AS (SELECT DISTINCT track_id, genre FROM q6_recording_profile),
        |centroid_genres AS (SELECT DISTINCT centroid_genre FROM q6_genre_centroid),
        |pairs AS (
        |  SELECT rg.track_id, rg.genre AS true_genre, cg.centroid_genre
        |  FROM recording_genres rg
        |  CROSS JOIN centroid_genres cg
        |),
        |dist AS (
        |  SELECT p.track_id, p.true_genre, p.centroid_genre,
        |    SQRT(SUM(POWER(r.avg_z - c.centroid_z, 2))) AS distance
        |  FROM pairs p
        |  JOIN q6_recording_profile r ON p.track_id = r.track_id
        |  JOIN q6_genre_centroid c ON p.centroid_genre = c.centroid_genre AND r.feature = c.feature
        |  GROUP BY p.track_id, p.true_genre, p.centroid_genre
        |),
        |nearest AS (
        |  SELECT *, ROW_NUMBER() OVER (PARTITION BY track_id ORDER BY distance ASC, centroid_genre ASC) AS rn
        |  FROM dist
        |)
        |SELECT track_id, true_genre, centroid_genre AS nearest_genre, ROUND(distance, 4) AS distance
        |FROM nearest WHERE rn = 1
      """.stripMargin
    ).createOrReplaceTempView("q6_nearest")

    val genreList = spark.sql("SELECT DISTINCT label AS genre FROM windows ORDER BY genre")
      .collect().map(_.getString(0)).toSeq
    val pivotList = genreList.map(g => s"'$g'").mkString(", ")

    val pivotSql =
      s"""
         |SELECT * FROM (
         |  SELECT true_genre, nearest_genre FROM q6_nearest
         |)
         |PIVOT (
         |  COUNT(*) FOR nearest_genre IN ($pivotList)
         |)
         |ORDER BY true_genre ASC
      """.stripMargin

    // Genre pairs with zero recordings land as null cells from the pivot; fill with 0 so every cell is a count.
    val pivotDf = spark.sql(pivotSql).na.fill(0)
    runQueryDf("Q6_confusion",
      "(exploratory, not a model) For each true genre, how many recordings land nearest to each genre's z-scored centroid?",
      pivotSql, pivotDf, "q6_confusion_pivot.csv")

    val selfShareSql =
      """
        |SELECT true_genre,
        |  SUM(CASE WHEN nearest_genre = true_genre THEN 1 ELSE 0 END) AS n_self,
        |  COUNT(*) AS n_total,
        |  ROUND(SUM(CASE WHEN nearest_genre = true_genre THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS self_share
        |FROM q6_nearest
        |GROUP BY true_genre
        |ORDER BY self_share DESC, true_genre ASC
      """.stripMargin

    runQuery(spark, "Q6_self_share",
      "(exploratory, not a model) Per genre, what share of recordings land nearest to their own centroid?",
      selfShareSql, "q6_self_match_share.csv")

    val topOffDiagonalSql =
      s"""
         |SELECT true_genre, nearest_genre, COUNT(*) AS n
         |FROM q6_nearest
         |WHERE true_genre <> nearest_genre
         |GROUP BY true_genre, nearest_genre
         |ORDER BY n DESC, true_genre ASC, nearest_genre ASC
         |LIMIT ${SqlConfig.TopOffDiagonalPairs}
      """.stripMargin

    runQuery(spark, "Q6_top_pairs",
      "(exploratory, not a model) Which off-diagonal (true genre, nearest genre) pairs are most common -- candidate confusions?",
      topOffDiagonalSql, "q6_top_offdiagonal.csv")
  }

  // ======================================================================
  // main
  // ======================================================================
  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-SQLOperations")

    Report.start(Paths.SqlReport)
    Report.section("SonicSpark Phase 4: Spark SQL Operations")
    Report.log("Exploratory only: no result below feeds Phase 5 feature selection, scaling, or modelling.")

    registerViews(spark)

    Report.section("View counts (assertions)")
    assertCounts(spark)

    val featureCols = spark.read.parquet(Paths.Final).columns
      .filterNot(Set("track_id", "label", "window_idx", "start_sec", "label_idx").contains)
      .toSeq

    Report.section("Q1: Variance Decomposition")
    runQ1(spark)

    Report.section("Q2: Robust Recording-Level Dynamic Range")
    runQ2(spark)

    Report.section("Q3: Position Effect (Intro/Middle/Outro)")
    runQ3(spark)

    Report.section("Q4: Temporal Stability")
    runQ4(spark)

    Report.section("Q5: Three-Level Join (windows -> segments -> tracks)")
    runQ5(spark)

    Report.section("Q6: Recording-Level Nearest Centroid (exploratory)")
    runQ6(spark, featureCols)

    Report.section("Q7: EXPLAIN for Q5")
    runQ7(spark)

    writeLongCsv()

    Report.log(s"\nFull report saved to: ${Paths.SqlReport}")
    Report.log(s"Combined long-format results saved to: ${Paths.SqlResultsCsv}")
    spark.stop()
  }
}
