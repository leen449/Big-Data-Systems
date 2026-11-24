package sonicspark.ml

/**
 * Phase 6, Part C: paired statistics for comparing two models on the SAME
 * 296 test recordings. A single accuracy number on 296 recordings carries a
 * wide confidence interval; both tests below exploit the pairing (same
 * recordings, same true labels) to say something sharper than that.
 */
object FollowUpStats {

  case class BootstrapCi(lower: Double, upper: Double)

  /**
   * Paired bootstrap 95% CI of the accuracy difference (A - B), percentile method.
   * Each resample draws the SAME set of resampled recording indices for both models
   * (that is what "paired" means here) -- an unpaired bootstrap would compare two
   * independently-resampled populations and throw away the fact that both models
   * were scored on the identical recordings.
   */
  def bootstrapCi(correctA: Array[Boolean], correctB: Array[Boolean], resamples: Int, seed: Long): BootstrapCi = {
    require(correctA.length == correctB.length, "Paired vectors must be the same length")
    val n = correctA.length
    val rng = new scala.util.Random(seed)
    val diffs = Array.fill(resamples) {
      var sumA, sumB, i = 0
      while (i < n) {
        val idx = rng.nextInt(n)
        if (correctA(idx)) sumA += 1
        if (correctB(idx)) sumB += 1
        i += 1
      }
      (sumA - sumB).toDouble / n
    }
    val sorted = diffs.sorted
    val lowerIdx = math.floor(FollowUpConfig.BootstrapCiLower * (resamples - 1)).toInt
    val upperIdx = math.ceil(FollowUpConfig.BootstrapCiUpper * (resamples - 1)).toInt
    BootstrapCi(sorted(lowerIdx), sorted(upperIdx))
  }

  /** b = recordings A got right and B got wrong; c = the reverse. McNemar only looks
   *  at these discordant pairs -- recordings both models got right or both got wrong
   *  carry no information about which model is better. */
  def discordantCounts(correctA: Array[Boolean], correctB: Array[Boolean]): (Int, Int) = {
    var b, c = 0
    correctA.indices.foreach { i =>
      if (correctA(i) && !correctB(i)) b += 1
      else if (!correctA(i) && correctB(i)) c += 1
    }
    (b, c)
  }

  /**
   * Exact two-sided McNemar test: under H0 (the two models are equally likely to be
   * the one that's right on a discordant pair), b ~ Binomial(b+c, 0.5). The exact
   * p-value is 2 * P(X <= min(b,c)), capped at 1 -- not the chi-square approximation,
   * which is unreliable once b+c is small. Computed with BigInt/BigDecimal because
   * b+c can be up to 296, far beyond where Double factorials stay accurate.
   */
  def exactMcNemarP(b: Int, c: Int): Double = {
    val n = b + c
    if (n == 0) 1.0
    else {
      val k = math.min(b, c)

      def choose(n: Int, k: Int): BigInt = {
        val numerator = (0 until k).foldLeft(BigInt(1)) { (acc, i) => acc * (n - i) }
        val denominator = (1 to k).foldLeft(BigInt(1)) { (acc, i) => acc * i }
        numerator / denominator
      }

      val total = BigDecimal(2).pow(n)
      val cumulative = (0 to k).foldLeft(BigDecimal(0)) { (acc, i) => acc + BigDecimal(choose(n, i)) }
      val oneSided = cumulative / total
      math.min(1.0, (oneSided * 2).toDouble)
    }
  }
}
