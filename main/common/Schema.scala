package sonicspark.common

import org.apache.spark.sql.types._

/**
 * The single definition of the GTZAN feature table.
 * Both features_3_sec.csv and features_30_sec.csv share this structure.
 * Column order matches the CSV header exactly.
 */
object Schema {

  // Features stored as a mean/variance pair, in CSV order
  private val pairedFeatures: Seq[String] = Seq(
    "chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth",
    "rolloff", "zero_crossing_rate", "harmony", "perceptr"
  ).flatMap(f => Seq(s"${f}_mean", s"${f}_var"))

  // MFCC 1..20, each as a mean/variance pair
  private val mfccFeatures: Seq[String] =
    (1 to 20).flatMap(i => Seq(s"mfcc${i}_mean", s"mfcc${i}_var"))

  /** All 57 numeric audio features, in CSV order. Reused by every stage. */
  val featureColumns: Seq[String] = pairedFeatures ++ Seq("tempo") ++ mfccFeatures

  /** Variance columns only (candidates for log transformation later). */
  val varianceColumns: Seq[String] = featureColumns.filter(_.endsWith("_var"))

  /** Full table schema: identifier, length, 57 features, label (60 columns). */
  val features: StructType = StructType(
    Seq(
      StructField("filename", StringType),
      StructField("length", IntegerType)
    ) ++
    featureColumns.map(c => StructField(c, DoubleType)) ++
    Seq(StructField("label", StringType))
  )
}