package sonicspark.preprocessing

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import sonicspark.common._

/**
 * Stage 3 of the preprocessing pipeline: REDUCTION
 *
 * Input : data/interim/02_integrated
 * Output: data/interim/03_reduced
 *         outputs/stats/03_reduction.csv          (before/after metrics)
 *         outputs/stats/03_feature_relevance.csv  (every feature, its eta², and the decision)
 */
object Reduction {

  val Stage   = "reduction"
  val MinEta2 = 0.01  // R4: below this, genre explains < 1% of a feature's variation
  val MaxCorr = 0.95  // R5: above this, two features carry the same information

  def main(args: Array[String]): Unit = {
    val spark = Spark.session("SonicSpark-Reduction")
    run(spark)
    spark.stop()
  }

  def run(spark: SparkSession): DataFrame = {
    val df   = DataIO.readStage(spark, Paths.Integrated).cache()
    val rows = df.count()

    // ---------- R1–R3: structural drops ----------
    val lengthValues = df.select("length").distinct().count()
    require(lengthValues == 1, s"'length' has $lengthValues distinct values; review rule R1")
    val tempoCorr = df.stat.corr("tempo", "track_tempo")

    // Audio features considered from here on: segment tempo replaced by track tempo
    val candidates = Schema.featureColumns.filterNot(_ == "tempo") :+ "track_tempo"

    // ---------- R4: relevance (eta squared) ----------
    val eta2         = etaSquared(df, candidates, rows)
    val lowRelevance = candidates.filter(c => eta2(c) < MinEta2)
    val relevant     = candidates.filterNot(lowRelevance.contains)

    // ---------- R5: redundancy (correlation) ----------
    val pairs     = correlatedPairs(df, relevant)
    val redundant = scala.collection.mutable.LinkedHashMap[String, (String, Double)]()
    pairs.sortBy(p => -math.abs(p._3)).foreach { case (a, b, r) =>
      if (!redundant.contains(a) && !redundant.contains(b)) {
        val (keep, drop) = if (eta2(a) >= eta2(b)) (a, b) else (b, a)
        redundant(drop) = (keep, r)
      }
    }
    val kept = relevant.filterNot(redundant.contains)

    // ---------- Output: identifiers + label + kept features ----------
    val reduced = df.select((Seq("track_id", "segment_index", "label") ++ kept).map(col): _*)
    DataIO.writeStage(reduced, Paths.Reduced)

    // ---------- Report 1: feature relevance ranking ----------
    val ranking = candidates.sortBy(c => -eta2(c)).map { c =>
      val decision =
        if (lowRelevance.contains(c)) "dropped: low relevance"
        else redundant.get(c) match {
          case Some((keep, r)) => f"dropped: redundant with $keep (r = $r%.3f)"
          case None            => "kept"
        }
      Seq(c, f"${eta2(c)}%.4f", decision)
    }
    val rankingHeader = Seq("feature", "eta_squared", "decision")
    println("\n=== Feature relevance (eta squared, highest first) ===")
    Metrics.showTable(rankingHeader, ranking)
    Metrics.saveTable("03_feature_relevance.csv", rankingHeader, ranking)

    // ---------- Report 2: before/after metrics ----------
    val metrics = Seq(
      Metric(Stage, "rows", rows.toString, rows.toString, "no sampling: data fits in memory"),
      Metric(Stage, "columns", df.columns.length.toString, reduced.columns.length.toString),
      Metric(Stage, "R1 dropped: constant", "length", "-", "1 distinct value"),
      Metric(Stage, "R2 dropped: identifier", "filename", "-", "replaced by track_id + segment_index"),
      Metric(Stage, "R3 dropped: unreliable", "tempo tempo_in_range", "-",
        f"replaced by track_tempo (correlation $tempoCorr%.2f)"),
      Metric(Stage, s"R4 dropped: eta2 < $MinEta2", lowRelevance.size.toString, "-", lowRelevance.mkString(" ")),
      Metric(Stage, s"R5 dropped: |r| > $MaxCorr", redundant.size.toString, "-", redundant.keys.mkString(" ")),
      Metric(Stage, "audio features", candidates.size.toString, kept.size.toString)
    )
    println("\n=== Reduction summary ===")
    Metrics.show(metrics)
    println(s"Metrics saved to: ${Metrics.save(metrics, "03_reduction.csv")}")

    reduced
  }

  /** eta² = between-genre variation / total variation, for each feature. */
  private def etaSquared(df: DataFrame, cols: Seq[String], n: Long): Map[String, Double] = {
    val overallExprs = cols.flatMap(c => Seq(avg(c).as(s"m_$c"), var_pop(c).as(s"v_$c")))
    val overall      = df.agg(overallExprs.head, overallExprs.tail: _*).head()

    val genreExprs = count("*").as("n") +: cols.map(c => avg(c).as(c))
    val genres     = df.groupBy("label").agg(genreExprs.head, genreExprs.tail: _*).collect()

    cols.map { c =>
      val mean    = overall.getAs[Double](s"m_$c")
      val total   = overall.getAs[Double](s"v_$c") * n
      val between = genres.map(g => g.getAs[Long]("n") * math.pow(g.getAs[Double](c) - mean, 2)).sum
      c -> (if (total > 0) between / total else 0.0)
    }.toMap
  }

  /** All feature pairs whose absolute Pearson correlation exceeds MaxCorr. */
  private def correlatedPairs(df: DataFrame, cols: Seq[String]): Seq[(String, String, Double)] = {
    val vectors = new VectorAssembler()
      .setInputCols(cols.toArray)
      .setOutputCol("v")
      .transform(df.select(cols.map(col): _*))
    val m = Correlation.corr(vectors, "v").head.getAs[Matrix](0)

    for {
      i <- cols.indices
      j <- (i + 1) until cols.size
      r = m(i, j)
      if math.abs(r) > MaxCorr
    } yield (cols(i), cols(j), r)
  }
}