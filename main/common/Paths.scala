package sonicspark.common

/**
 * All file locations in one place.
 * If a folder ever moves, we change it here only.
 */
object Paths {

  // ---------- Raw data (from Kaggle, never committed to Git) ----------
  val Raw3Sec   = "data/raw/features_3_sec.csv"
  val Raw30Sec  = "data/raw/features_30_sec.csv"
  val AudioDir  = "data/raw/genres_original"
  val ImagesDir = "data/raw/images_original"

  // ---------- Pipeline outputs (one per preprocessing step) ----------
  val Cleaned    = "data/interim/01_cleaned"
  val Integrated = "data/interim/02_integrated"
  val Reduced    = "data/interim/03_reduced"
  val Final      = "data/processed/final"

   // ---------- Technical checks ----------
  val SmokeTest = "data/interim/_smoke_test"
  val cleaningStats = "outputs/stats/01_cleaning.csv"
  val Tracks = "data/interim/02_tracks"
  val reductionStats = "outputs/stats/03_reduction.csv"
  val featureRelevanceStats = "outputs/stats/03_feature_relevance.csv"
  val transformationStats = "outputs/stats/04_transformation.csv"
  val skewnessStats = "outputs/stats/04_skewness.csv"
  val labelMappingStats = "outputs/stats/04_label_mapping.csv"

  // ---------- Material for the report ----------
  val Stats   = "outputs/stats"
  val Figures = "outputs/figures"
}