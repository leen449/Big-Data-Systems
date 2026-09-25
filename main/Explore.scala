package sonicspark

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import sonicspark.common.{Paths, Spark}

/**
 * Stage 1 exploration of features_3_sec.csv.
 *
 * Run everything at once:
 *   sbt "runMain sonicspark.exploration.Explore all"
 *
 * Or one feature group at a time:
 *   sbt "runMain sonicspark.exploration.Explore identity"   (Shahad)
 *   sbt "runMain sonicspark.exploration.Explore energy"     (Aryam)
 *   sbt "runMain sonicspark.exploration.Explore spectral"   (Leen)
 *   sbt "runMain sonicspark.exploration.Explore timbre"     (Ryouf)
 */
object Explore {

  // ---------- Feature groups (one per team member) ----------
  val groups: Map[String, Seq[String]] = Map(
    "energy" -> Seq(
      "chroma_stft_mean", "chroma_stft_var",
      "rms_mean", "rms_var",
      "tempo"
    ),
    "spectral" -> Seq(
      "spectral_centroid_mean", "spectral_centroid_var",
      "spectral_bandwidth_mean", "spectral_bandwidth_var",
      "rolloff_mean", "rolloff_var",
      "zero_crossing_rate_mean", "zero_crossing_rate_var"
    ),
    "timbre" -> (
      Seq("harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var") ++
      (1 to 20).flatMap(i => Seq(s"mfcc${i}_mean", s"mfcc${i}_var"))
    )
  )

  // ---------- Entry point ----------
  def main(args: Array[String]): Unit = {
    val group = args.headOption.getOrElse("all").toLowerCase
    val spark = Spark.session(s"SonicSpark-Explore-$group")

    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(Paths.Raw3Sec)
      .cache() // keep in memory: we query it many times

    println(s"\n=== Dataset: ${df.count()} rows x ${df.columns.length} columns ===")

    // "all" runs every group in order; otherwise run just the one requested
    val toRun =
      if (group == "all") Seq("identity") ++ groups.keys.toSeq.sorted
      else Seq(group)

    toRun.foreach(g => runGroup(df, g))

    spark.stop()
  }

  // ---------- Runs the exploration for one group ----------
  def runGroup(df: DataFrame, group: String): Unit = {
    println(s"\n\n################ GROUP: ${group.toUpperCase} ################")

    group match {
      case "identity" =>
        df.printSchema()
        exploreIdentity(df)

      case g if groups.contains(g) =>
        val cols = groups(g)
        val missing = cols.filterNot(df.columns.contains)
        if (missing.nonEmpty) println(s"WARNING: columns not found: ${missing.mkString(", ")}")
        val present = cols.filter(df.columns.contains)

        println(s"\n=== Column profile: $g (${present.size} columns) ===")
        profile(df, present).show(present.size, truncate = false)

        println(s"\n=== Average per genre: $g ===")
        perGenre(df, present)

      case other =>
        println(s"Unknown group '$other'. Use: all, identity, ${groups.keys.mkString(", ")}")
    }
  }

  // ---------- Identity group: identifiers, label, and the 9,990 question ----------
  def exploreIdentity(df: DataFrame): Unit = {
    val total = df.count()

    println("\n=== filename ===")
    val distinctFiles = df.select("filename").distinct().count()
    println(s"Distinct filenames: $distinctFiles | Duplicate filenames: ${total - distinctFiles}")
    println(s"Fully duplicated rows: ${total - df.distinct().count()}")

    println("\n=== length (samples per segment) ===")
    df.groupBy("length").count().orderBy(desc("count")).show(10)

    println("\n=== label ===")
    df.groupBy("label").count().orderBy("label").show()

    val mismatch = df
      .filter(split(col("filename"), "\\.").getItem(0) =!= col("label"))
      .count()
    println(s"Rows where label does not match the filename prefix: $mismatch")

    println("\n=== Segments per track ===")
    val withTrack = df.withColumn(
      "track_id", regexp_extract(col("filename"), "^([a-z]+\\.\\d{5})", 1)
    )
    val perTrack = withTrack.groupBy("track_id").agg(count("*").as("segments"))

    println(s"Distinct tracks: ${perTrack.count()}")
    println("How many tracks have N segments:")
    perTrack.groupBy("segments").agg(count("*").as("num_tracks")).orderBy("segments").show()

    println("Tracks with fewer than 10 segments:")
    perTrack.filter(col("segments") < 10).orderBy("track_id").show(50, truncate = false)

    println("Tracks per genre:")
    withTrack.select("label", "track_id").distinct()
      .groupBy("label").count().orderBy("label").show()
  }

  // ---------- Feature groups: one summary row per column ----------
  def profile(df: DataFrame, cols: Seq[String]): DataFrame = {
    // Build all statistics in ONE pass over the data
    val exprs = cols.flatMap { c =>
      Seq(
        sum(when(col(c).isNull || isnan(col(c)), 1).otherwise(0)).as(s"${c}__nulls"),
        min(c).cast("double").as(s"${c}__min"),
        max(c).cast("double").as(s"${c}__max"),
        avg(c).as(s"${c}__mean"),
        stddev(c).as(s"${c}__std"),
        skewness(c).as(s"${c}__skew"),
        sum(when(col(c) < 0, 1).otherwise(0)).as(s"${c}__neg"),
        sum(when(col(c) === 0, 1).otherwise(0)).as(s"${c}__zero")
      )
    }
    val r = df.agg(exprs.head, exprs.tail: _*).head()

    // Reshape: one row per feature
    val rows = cols.map { c =>
      (c,
        r.getAs[Long](s"${c}__nulls"),
        r.getAs[Double](s"${c}__min"),
        r.getAs[Double](s"${c}__max"),
        r.getAs[Double](s"${c}__mean"),
        r.getAs[Double](s"${c}__std"),
        r.getAs[Double](s"${c}__skew"),
        r.getAs[Long](s"${c}__neg"),
        r.getAs[Long](s"${c}__zero"))
    }

    val spark = df.sparkSession
    import spark.implicits._
    rows.toDF("feature", "nulls", "min", "max", "mean", "stddev", "skewness", "negatives", "zeros")
  }

  // ---------- Average of each feature per genre (5 columns per table) ----------
  def perGenre(df: DataFrame, cols: Seq[String]): Unit = {
    cols.grouped(5).foreach { chunk =>
      val aggs = chunk.map(c => avg(c).as(c))
      df.groupBy("label").agg(aggs.head, aggs.tail: _*).orderBy("label").show(truncate = false)
    }
  }
}