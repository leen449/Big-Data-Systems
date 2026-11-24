package sonicspark.ml

/**
 * Phase 6 (follow-up experiments): every tunable number used by FollowUpExperiments,
 * in one place -- same idea as MlConfig.scala, but kept in its own file so that
 * MlConfig.scala (Phase 5's own constants, frozen alongside its checksummed output)
 * never has to change for this follow-up work.
 */
object FollowUpConfig {

  // Tie rule shared by Part A and Part B: two candidates are "tied" on recording
  // macro-F1 if they differ by less than this, and the tie is then broken by
  // accuracy, then by which configuration is simpler. Fixed here, in code, before
  // any OOF score is computed -- the same number is quoted in the printed report.
  val TieEpsilon: Double = 0.0005

  // ---------- Part B: recording-level summary model grids ----------
  // Only 688 training recordings, so an LR regParam of 1.0 is added relative to
  // Phase 5's window-level grid (MlConfig.LrRegParams), which never needed it.
  val SummaryLrRegParams: Seq[Double] = Seq(0.001, 0.01, 0.1, 1.0)
  val SummaryLrElasticNet: Double     = 0.0
  val SummaryLrMaxIter: Int           = 500

  // Cheap at 688 rows -- no OutOfMemoryError risk like Phase 5's window-level RF grid.
  val SummaryRfNumTrees: Seq[Int] = Seq(200, 500)
  val SummaryRfMaxDepth: Seq[Int] = Seq(5, 10, 15)

  val SummaryLrTopCoefficients: Int = 3     // Part B asks for top 3, not Phase 5's top 5
  val SummaryRfTopImportances: Int  = 15

  // ---------- Part C: paired comparison on the test recordings ----------
  val BootstrapResamples: Int = 10000
  val BootstrapSeed: Long     = 42L   // same project-wide seed as MlConfig.Seed
  val BootstrapCiLower: Double = 0.025
  val BootstrapCiUpper: Double = 0.975

  // Phase 5's published recording-level soft-vote test accuracies (results/ml_metrics.txt,
  // README "Machine learning (Phase 5)" table). Part C must reproduce these to 4dp before
  // trusting its own refit -- not a new result, just the number already on record.
  val Phase5TestAccuracyLr             = 0.7027
  val Phase5TestAccuracyRf             = 0.6824
  val Phase5TestAccuracyCentroidPca    = 0.6453
  val Phase5TestAccuracyCentroidPlain  = 0.5236
  val Phase5AccuracyCheckTolerance     = 0.00005   // match "to 4 decimals"
}
