package sonicspark.ml

/**
 * Stage 5 of the pipeline: MACHINE LEARNING
 *
 * Every tunable number used by the ML phase, in one place -- mirrors how
 * AudioConfig.scala centralises extraction constants and Reduction.scala
 * centralises its thresholds. Nothing in main/ml/ should hard-code these
 * values directly.
 */
object MlConfig {

  // ---------- Reproducibility ----------
  // No shared seed constant exists elsewhere in the project (Phase2Snapshot's
  // SampleSeed is local to a reporting sample, not a modelling seed). 42 is
  // this phase's own fixed seed, used for every random choice: the track-level
  // split, the grouped-fold assignment, and RandomForest's internal randomness.
  val Seed: Long = 42L

  // ---------- SparkSession ----------
  // common.Spark.session uses local[*], whose degree of parallelism depends on
  // the machine's core count and can vary run to run on a loaded machine. This
  // phase fixes both the master and the shuffle partition count explicitly, so
  // that running the whole program twice and diffing the outputs is actually a
  // test of determinism, not an artifact of different parallelism.
  // local[3], not local[4]: fewer concurrent tasks means fewer simultaneous
  // RandomForest split-finding buffers in memory at once, which matters at this
  // dataset size with a 200-tree/depth-15 grid point over a multi-hour single-JVM run.
  val SparkMaster       = "local[3]"
  val ShufflePartitions = "8"

  // ---------- Track-level split (section 2) ----------
  val TrainFraction: Double = 0.7
  val NumFolds: Int         = 5
  val FoldColName           = "cv_fold"   // -1 for test rows, 0..NumFolds-1 for train rows

  // ---------- Train-only feature analysis (section 3), same rules as Reduction.scala ----------
  val MinEta2  = 0.01   // drop if eta² < this
  val MaxCorr  = 0.95   // flag a pair if |r| > this; drop the lower-eta² feature of the pair
  val ReportCorrThreshold = 0.8   // report (but do not act on) pairs above this

  // ---------- Models (section 5) ----------
  val LrRegParams: Seq[Double] = Seq(0.001, 0.01, 0.1)
  val LrElasticNet: Double     = 0.0   // pure L2 (ridge), per the task's model spec
  val LrMaxIter: Int           = 200

  // Trimmed from the originally planned {100,200} x {10,15} after two repeated
  // OutOfMemoryError crashes (~2 hours of wall time each) at the 200-tree/depth-15
  // corner, on this machine's ~12GB RAM in local mode. Decided from a resource
  // constraint alone, before any test-set evaluation -- not a modelling choice.
  val RfNumTrees: Seq[Int] = Seq(100, 150)
  val RfMaxDepth: Seq[Int] = Seq(10, 12)
  // Stage 1 only needs one fixed RandomForest configuration to compare the four
  // feature sets cheaply, before stage 2 tunes the winning set's full RF grid.
  val Stage1FixedRfNumTrees = 100
  val Stage1FixedRfMaxDepth = 10

  // ---------- PCA-whitened centroid baseline (section 5, Baseline B) ----------
  val PcaVarianceThreshold: Double = 0.95

  // ---------- Genres, alphabetical -- must match label_idx (StringIndexer, alphabetAsc) ----------
  val Genres: Seq[String] = Seq(
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
  )
}
