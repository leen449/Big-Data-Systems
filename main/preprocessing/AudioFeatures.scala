package sonicspark.preprocessing

import breeze.linalg.DenseVector
import breeze.signal.fourierTr
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import scala.util.{Failure, Success, Try}
import sonicspark.common._

/**
 * Extraction stage: turns raw GTZAN WAV files into the window_features table
 * that replaces features_3_sec.csv as the analytical input for the rest of
 * the pipeline (see CLAUDE.md, "Agreed pipeline").
 *
 * Covers stages (a)-(e): WAV loading, windowing, time-domain features (RMS,
 * ZCR), spectral features (centroid, bandwidth, roll-off, flatness), MFCC
 * 1-13, and validation against features_3_sec.csv.
 */
object AudioFeatures {

  /** One fully-decoded track: mono PCM samples in [-1, 1], plus its identity. */
  case class WavAudio(trackId: String, label: String, sampleRate: Int, samples: Array[Double])

  /** One 1-second, half-overlapping window cut from a track. */
  case class AudioWindow(
    trackId: String,
    label: String,
    windowIdx: Int,
    startSec: Double,
    sampleRate: Int,
    samples: Array[Double]
  )

  /**
   * One row of the final window_features table: identity columns plus the
   * full 38 feature columns from CLAUDE.md (time-domain, spectral, MFCC).
   * This is the complete schema; every stage from (b) onward fills in more
   * of it, but the shape was fixed once MFCC (the last feature group) was
   * added in stage (d).
   */
  case class WindowFeatures(
    trackId: String,
    label: String,
    windowIdx: Int,
    startSec: Double,
    rmsMean: Double,
    rmsVar: Double,
    zcrMean: Double,
    zcrVar: Double,
    spectralCentroidMean: Double,
    spectralCentroidVar: Double,
    spectralBandwidthMean: Double,
    spectralBandwidthVar: Double,
    rolloffMean: Double,
    rolloffVar: Double,
    flatnessMean: Double,
    flatnessVar: Double,
    mfcc1Mean: Double, mfcc1Var: Double,
    mfcc2Mean: Double, mfcc2Var: Double,
    mfcc3Mean: Double, mfcc3Var: Double,
    mfcc4Mean: Double, mfcc4Var: Double,
    mfcc5Mean: Double, mfcc5Var: Double,
    mfcc6Mean: Double, mfcc6Var: Double,
    mfcc7Mean: Double, mfcc7Var: Double,
    mfcc8Mean: Double, mfcc8Var: Double,
    mfcc9Mean: Double, mfcc9Var: Double,
    mfcc10Mean: Double, mfcc10Var: Double,
    mfcc11Mean: Double, mfcc11Var: Double,
    mfcc12Mean: Double, mfcc12Var: Double,
    mfcc13Mean: Double, mfcc13Var: Double
  )

  def main(args: Array[String]): Unit = {
    runSyntheticChecks()

    val audioDir = args.headOption.getOrElse(Paths.SampleAudioDir)
    val spark    = Spark.session("SonicSpark-AudioFeatures")
    val features = run(spark, audioDir)
    validate(spark, features)

    // Persist only the full-corpus run: sample runs are for quick inspection,
    // not meant to overwrite the real window_features output.
    if (audioDir == Paths.AudioDir) {
      DataIO.writeStage(features, Paths.WindowFeatures)
      println(s"\nWindow features saved to: ${Paths.WindowFeatures}")
    }

    spark.stop()
  }

  // ==================================================================
  // Stage (a) driver: load every WAV file under audioDir and window it
  // ==================================================================

  /**
   * Loads every WAV file under audioDir, windows it, extracts every
   * feature (time-domain, spectral, MFCC), prints diagnostics, and returns
   * the complete window_features table.
   */
  def run(spark: SparkSession, audioDir: String = Paths.AudioDir): DataFrame = {
    import spark.implicits._

    // audioRdd and its windowed form are NOT cached: a window is 22,050
    // samples (~176 KB), and the full corpus has ~59,000 of them, so
    // caching either full-resolution RDD needs 5-10+ GB of heap. Instead,
    // windowing and feature extraction below are chained into a single
    // pass, so raw samples are only ever held transiently per-track/
    // per-window, never materialized all at once. Only the final
    // feature table -- four orders of magnitude smaller -- is cached.
    val (audioRdd, failures) = loadAudioFiles(spark, audioDir)
    val loadedCount = audioRdd.count()

    println(s"\n=== WAV loading: $audioDir ===")
    println(s"Successfully parsed: $loadedCount")
    println(s"Excluded (failed to parse): ${failures.size}")
    failures.foreach { case (path, reason) => println(s"  EXCLUDED  $path  ->  $reason") }

    val sampleRates = audioRdd.map(_.sampleRate).distinct().collect()  // tiny: at most a handful of distinct values
    println(s"Distinct sample rates observed: ${sampleRates.mkString(", ")}")

    // ---------- Windowing + feature extraction (time-domain + spectral + MFCC), one pass ----------
    // Computed once on the driver and broadcast, not recomputed per task.
    val hannBroadcast = spark.sparkContext.broadcast(computeHannWindow(AudioConfig.FrameSize))
    val melFilterbankBroadcast = spark.sparkContext.broadcast(
      computeMelFilterbank(AudioConfig.ExpectedSampleRate, AudioConfig.FrameSize, AudioConfig.MelBands)
    )

    val features = audioRdd
      .flatMap(windowTrack)
      .map(w => extractFeatures(w, hannBroadcast.value, melFilterbankBroadcast.value))
      .toDF((Schema.windowIdColumns ++ Schema.windowFeatureColumns): _*)
      .cache()

    val windowCount = features.count()
    println(s"\n=== Windowing (size=${AudioConfig.WindowSize}, hop=${AudioConfig.WindowHop}) ===")
    println(s"Total windows: $windowCount")

    val perTrack = features.groupBy("track_id").count().withColumnRenamed("count", "num_windows")

    println("\n=== Windows per track ===")
    perTrack.orderBy("track_id").show(20, truncate = false)

    println("\n=== Windows-per-track summary ===")
    perTrack.agg(
      min("num_windows").as("min"),
      max("num_windows").as("max"),
      avg("num_windows").as("avg")
    ).show()

    val perGenre = features.groupBy("label").count().withColumnRenamed("count", "num_windows")

    println("\n=== Windows per genre ===")
    perGenre.orderBy("label").show(20, truncate = false)

    println("\n=== Schema ===")
    features.printSchema()

    println("\n=== Sample rows ===")
    features.orderBy("track_id", "window_idx").show(10, truncate = false)

    println("\n=== Feature summary (all windows) ===")
    features.select(
      "rms_mean", "zcr_mean",
      "spectral_centroid_mean", "spectral_bandwidth_mean",
      "rolloff_mean", "flatness_mean",
      "mfcc1_mean", "mfcc2_mean", "mfcc13_mean"
    ).describe().show()

    reportAnomalies(features, Schema.windowFeatureColumns)

    features
  }

  /**
   * Scans every feature column for NaN, +/-Infinity, and implausibly large
   * magnitudes in a single pass, and prints how many of each were found.
   * With the epsilon-guarded math in this file, all three counts are
   * expected to be zero; this is the check that confirms it on real data.
   *
   * The threshold is deliberately generous: Hz-based variances (centroid,
   * bandwidth, roll-off) are bounded by roughly Nyquist^2 = 11,025^2 ~= 1.2e8
   * in the most extreme legitimate case (a window whose energy alternates
   * between 0 Hz and Nyquist every frame), so 1e9 comfortably clears any
   * real feature value while still catching an actual numerical blow-up.
   */
  private def reportAnomalies(df: DataFrame, featureColumns: Seq[String]): Unit = {
    val ExtremeThreshold = 1e9

    val exprs = featureColumns.flatMap { c =>
      Seq(
        sum(when(isnan(col(c)) || col(c).isNull, 1).otherwise(0)).as(s"${c}__nan"),
        sum(when(col(c) === Double.PositiveInfinity || col(c) === Double.NegativeInfinity, 1).otherwise(0)).as(s"${c}__inf"),
        sum(when(abs(col(c)) > ExtremeThreshold, 1).otherwise(0)).as(s"${c}__extreme")
      )
    }
    val row = df.agg(exprs.head, exprs.tail: _*).head()

    val nanCount     = featureColumns.map(c => row.getAs[Long](s"${c}__nan")).sum
    val infCount     = featureColumns.map(c => row.getAs[Long](s"${c}__inf")).sum
    val extremeCount = featureColumns.map(c => row.getAs[Long](s"${c}__extreme")).sum
    val affectedColumns = featureColumns.filter { c =>
      row.getAs[Long](s"${c}__nan") > 0 || row.getAs[Long](s"${c}__inf") > 0 || row.getAs[Long](s"${c}__extreme") > 0
    }

    println("\n=== Anomaly scan (all feature columns) ===")
    println(s"NaN values: $nanCount")
    println(s"Infinite values: $infCount")
    println(s"Extreme values (|value| > $ExtremeThreshold): $extremeCount")
    if (affectedColumns.nonEmpty) println(s"Affected columns: ${affectedColumns.mkString(", ")}")
  }

  // ==================================================================
  // Validation (stage e): compare against features_3_sec.csv
  // ==================================================================

  /**
   * Sanity-checks the extraction against librosa's own numbers. Window
   * features are averaged up to 3-second segments (segment_idx = floor
   * (start_sec / 3), same join key CLAUDE.md defines for window -> segment
   * joins), matched to the corresponding row of features_3_sec.csv by
   * track_id + segment_idx, and compared via Pearson correlation on RMS
   * and spectral centroid. Exact equality is not expected -- the CSV's
   * values come from librosa's own frame-by-frame extraction over the raw
   * 3-second clip, ours from averaging half-overlapping 1-second windows --
   * but a strong correlation confirms the extraction is measuring the same
   * underlying signal.
   */
  def validate(spark: SparkSession, windowFeatures: DataFrame): Unit = {
    val extractedSegments = windowFeatures
      .withColumn("segment_idx", floor(col("start_sec") / 3).cast("int"))
      .groupBy("track_id", "segment_idx")
      .agg(
        avg("rms_mean").as("extracted_rms_mean"),
        avg("spectral_centroid_mean").as("extracted_centroid_mean")
      )

    val csvSegments = DataIO.readFeaturesCsv(spark, Paths.Raw3Sec)
      .withColumn("track_id", regexp_extract(col("filename"), "^([a-z]+\\.\\d{5})", 1))
      .withColumn("segment_idx", regexp_extract(col("filename"), "\\.(\\d+)\\.wav$", 1).cast("int"))
      .select(
        col("track_id"), col("segment_idx"),
        col("rms_mean").as("csv_rms_mean"),
        col("spectral_centroid_mean").as("csv_centroid_mean")
      )

    val joined = extractedSegments
      .join(csvSegments, Seq("track_id", "segment_idx"), "inner")
      .cache()
    val matched = joined.count()

    println(s"\n=== Validation against features_3_sec.csv: $matched matched segments ===")
    joined.orderBy("track_id", "segment_idx").show(20, truncate = false)

    if (matched < 2) {
      println("Fewer than 2 matched segments: not enough data to compute a correlation.")
    } else {
      val rmsCorr      = joined.stat.corr("extracted_rms_mean", "csv_rms_mean")
      val centroidCorr = joined.stat.corr("extracted_centroid_mean", "csv_centroid_mean")
      println(f"Pearson correlation, RMS mean (ours vs. librosa):               $rmsCorr%.4f")
      println(f"Pearson correlation, spectral centroid mean (ours vs. librosa): $centroidCorr%.4f")
    }
  }

  // ==================================================================
  // WAV loading
  // ==================================================================

  /**
   * Reads every *.wav file under audioDir/<genre>/ with sc.binaryFiles,
   * parses each one, and splits the results into successes and failures.
   * One bad file is logged and excluded; it never crashes the job.
   *
   * NOT cached: a decoded track is ~5 MB (661,794 samples x 8 bytes), so
   * caching all ~1,000 of them needs ~5 GB of heap, which this job doesn't
   * have to spare once the (much larger) windowed feature extraction is
   * running too. Re-parsing WAV files from disk is cheap; re-running the
   * FFT/mel/DCT feature pipeline would not be, so that part is cached
   * instead (see run()).
   */
  def loadAudioFiles(spark: SparkSession, audioDir: String): (RDD[WavAudio], Seq[(String, String)]) = {
    val files = spark.sparkContext.binaryFiles(s"$audioDir/*/*.wav")

    val parsed = files.map { case (path, stream) =>
      val (trackId, label) = trackIdAndLabel(path)
      parseWav(stream.toArray(), trackId, label) match {
        case Success(audio) => Right(audio)
        case Failure(e)      => Left((path, Option(e.getMessage).getOrElse(e.toString)))
      }
    }

    // Failures are at most a few files out of ~1,000: safe to collect.
    val failures = parsed.collect { case Left(f) => f }.collect().toSeq
    val audio    = parsed.collect { case Right(w) => w }

    (audio, failures)
  }

  /** "<...>/data/raw/genres_original/blues/blues.00000.wav" -> ("blues.00000", "blues"). */
  private def trackIdAndLabel(path: String): (String, String) = {
    val parts   = path.split("[\\\\/]")
    val fileName = parts.last
    val label    = parts(parts.length - 2)
    val trackId  = fileName.stripSuffix(".wav")
    (trackId, label)
  }

  /**
   * Parses one WAV file's bytes into mono PCM samples in [-1, 1].
   * The RIFF header is read by searching for the "fmt " and "data" chunks
   * (never a fixed 44-byte offset), since real-world WAV files can carry
   * extra chunks before the audio data. Wrapped in Try by the caller.
   */
  def parseWav(bytes: Array[Byte], trackId: String, label: String): Try[WavAudio] = Try {
    val (sampleRate, channels, bitsPerSample, dataOffset, dataSize) = findFmtAndDataChunks(bytes)
    require(channels == AudioConfig.ExpectedChannels, s"expected mono audio, found $channels channel(s)")
    require(bitsPerSample == AudioConfig.ExpectedBitDepth, s"expected 16-bit PCM, found $bitsPerSample-bit")

    val samples = pcmToSamples(bytes, dataOffset, dataSize)
    require(samples.nonEmpty, "no PCM samples found in 'data' chunk")

    WavAudio(trackId, label, sampleRate, samples)
  }

  /**
   * Walks the RIFF chunk list from byte 12 onward, looking for "fmt " and
   * "data". Chunks are [4-byte id][4-byte little-endian size][body], and
   * the body is padded to an even number of bytes.
   * Returns (sampleRate, channels, bitsPerSample, dataChunkOffset, dataChunkSize).
   */
  private def findFmtAndDataChunks(bytes: Array[Byte]): (Int, Int, Int, Int, Int) = {
    require(bytes.length >= 12, "file is shorter than a minimal RIFF header")
    require(asciiAt(bytes, 0, 4) == "RIFF", "missing RIFF tag")
    require(asciiAt(bytes, 8, 4) == "WAVE", "missing WAVE tag")

    var offset        = 12
    var sampleRate    = -1
    var channels      = -1
    var bitsPerSample = -1
    var dataOffset    = -1
    var dataSize      = -1

    while (offset + 8 <= bytes.length && (sampleRate < 0 || dataOffset < 0)) {
      val chunkId    = asciiAt(bytes, offset, 4)
      val chunkSize  = readUInt32LE(bytes, offset + 4).toInt
      val bodyStart  = offset + 8

      chunkId match {
        case "fmt " =>
          channels      = readUInt16LE(bytes, bodyStart + 2)
          sampleRate    = readUInt32LE(bytes, bodyStart + 4).toInt
          bitsPerSample = readUInt16LE(bytes, bodyStart + 14)
        case "data" =>
          dataOffset = bodyStart
          dataSize   = chunkSize
        case _ => // ignore chunks we don't need (LIST, fact, id3, ...)
      }

      // Chunks are word-aligned: an odd-sized body is followed by one padding byte.
      offset = bodyStart + chunkSize + (chunkSize % 2)
    }

    require(sampleRate > 0, "'fmt ' chunk not found")
    require(dataOffset > 0, "'data' chunk not found")
    (sampleRate, channels, bitsPerSample, dataOffset, dataSize)
  }

  /** Converts the "data" chunk's 16-bit little-endian PCM bytes to doubles in [-1, 1]. */
  private def pcmToSamples(bytes: Array[Byte], dataOffset: Int, dataSize: Int): Array[Double] = {
    // A corrupt file can claim a data size larger than the bytes actually present;
    // clamp to what is available instead of reading past the end of the array.
    val available  = math.max(0, bytes.length - dataOffset)
    val safeSize   = math.min(math.max(dataSize, 0), available)
    val numSamples = safeSize / 2  // 2 bytes per 16-bit sample

    val samples = new Array[Double](numSamples)
    var i = 0
    while (i < numSamples) {
      samples(i) = readInt16LE(bytes, dataOffset + i * 2).toDouble / 32768.0
      i += 1
    }
    samples
  }

  private def asciiAt(bytes: Array[Byte], offset: Int, len: Int): String =
    new String(bytes, offset, len, "US-ASCII")

  private def readUInt16LE(bytes: Array[Byte], offset: Int): Int =
    (bytes(offset) & 0xff) | ((bytes(offset + 1) & 0xff) << 8)

  private def readUInt32LE(bytes: Array[Byte], offset: Int): Long =
    (bytes(offset) & 0xffL) |
    ((bytes(offset + 1) & 0xffL) << 8) |
    ((bytes(offset + 2) & 0xffL) << 16) |
    ((bytes(offset + 3) & 0xffL) << 24)

  /** Reads a signed 16-bit little-endian sample. */
  private def readInt16LE(bytes: Array[Byte], offset: Int): Short = {
    val lo = bytes(offset) & 0xff
    val hi = bytes(offset + 1) << 8  // Byte -> Int sign-extends; kept only for its low 16 bits below
    (hi | lo).toShort
  }

  // ==================================================================
  // Windowing
  // ==================================================================

  /**
   * Cuts a track into overlapping windows of AudioConfig.WindowSize samples,
   * hopping AudioConfig.WindowHop samples at a time. An incomplete final
   * window (shorter than WindowSize) is dropped rather than padded, so every
   * window carries the same amount of information.
   *
   * A full-length GTZAN track (661,794 samples) yields exactly 59 windows.
   */
  def windowTrack(audio: WavAudio): Seq[AudioWindow] = {
    val total = audio.samples.length
    val numWindows =
      if (total < AudioConfig.WindowSize) 0
      else (total - AudioConfig.WindowSize) / AudioConfig.WindowHop + 1

    (0 until numWindows).map { idx =>
      val start    = idx * AudioConfig.WindowHop
      val startSec = start.toDouble / audio.sampleRate
      val window   = audio.samples.slice(start, start + AudioConfig.WindowSize)
      AudioWindow(audio.trackId, audio.label, idx, startSec, audio.sampleRate, window)
    }
  }

  // ==================================================================
  // Framing: cuts each window into overlapping analysis frames.
  // Shared by time-domain features here and spectral/MFCC features later.
  // ==================================================================

  /**
   * Cuts a window into overlapping frames of AudioConfig.FrameSize samples,
   * hopping AudioConfig.FrameHop samples at a time. Same drop-incomplete-tail
   * rule as windowTrack, so every frame carries a full FrameSize samples.
   * A full window (22,050 samples) yields exactly 40 frames.
   */
  def frameSamples(samples: Array[Double]): Seq[Array[Double]] = {
    val total = samples.length
    val numFrames =
      if (total < AudioConfig.FrameSize) 0
      else (total - AudioConfig.FrameSize) / AudioConfig.FrameHop + 1

    (0 until numFrames).map { idx =>
      val start = idx * AudioConfig.FrameHop
      samples.slice(start, start + AudioConfig.FrameSize)
    }
  }

  // ==================================================================
  // Time-domain features (stage b): RMS and zero-crossing rate
  // ==================================================================

  /**
   * Extracts every feature for one window: frames the window, computes each
   * feature per frame, and reduces each to a mean/variance pair across the
   * frames (as required by CLAUDE.md). hann and melFilterbank are precomputed
   * and broadcast by the caller.
   *
   * The FFT magnitude spectrum is computed once per frame and reused for
   * both the spectral-shape features and the MFCCs, instead of running the
   * FFT twice.
   */
  def extractFeatures(window: AudioWindow, hann: Array[Double], melFilterbank: Array[Array[Double]]): WindowFeatures = {
    val frames = frameSamples(window.samples)

    val (rmsMean, rmsVar) = meanVar(frames.map(rms))
    val (zcrMean, zcrVar) = meanVar(frames.map(zeroCrossingRate))

    val magnitudesPerFrame = frames.map(f => magnitudeSpectrum(f, hann))
    val freqs = Array.tabulate(magnitudesPerFrame.head.length)(k => k.toDouble * window.sampleRate / AudioConfig.FrameSize)

    val spectralShape = magnitudesPerFrame.map(m => spectralFeaturesFromMagnitude(m, freqs))
    val (centroidMean, centroidVar)   = meanVar(spectralShape.map(_._1))
    val (bandwidthMean, bandwidthVar) = meanVar(spectralShape.map(_._2))
    val (rolloffMean, rolloffVar)     = meanVar(spectralShape.map(_._3))
    val (flatnessMean, flatnessVar)   = meanVar(spectralShape.map(_._4))

    val mfccPerFrame = magnitudesPerFrame.map(m => mfcc(powerSpectrum(m), melFilterbank, AudioConfig.NumMfcc))
    val mfccStats    = (0 until AudioConfig.NumMfcc).map(i => meanVar(mfccPerFrame.map(_(i))))

    WindowFeatures(
      trackId = window.trackId, label = window.label, windowIdx = window.windowIdx, startSec = window.startSec,
      rmsMean = rmsMean, rmsVar = rmsVar,
      zcrMean = zcrMean, zcrVar = zcrVar,
      spectralCentroidMean = centroidMean, spectralCentroidVar = centroidVar,
      spectralBandwidthMean = bandwidthMean, spectralBandwidthVar = bandwidthVar,
      rolloffMean = rolloffMean, rolloffVar = rolloffVar,
      flatnessMean = flatnessMean, flatnessVar = flatnessVar,
      mfcc1Mean = mfccStats(0)._1, mfcc1Var = mfccStats(0)._2,
      mfcc2Mean = mfccStats(1)._1, mfcc2Var = mfccStats(1)._2,
      mfcc3Mean = mfccStats(2)._1, mfcc3Var = mfccStats(2)._2,
      mfcc4Mean = mfccStats(3)._1, mfcc4Var = mfccStats(3)._2,
      mfcc5Mean = mfccStats(4)._1, mfcc5Var = mfccStats(4)._2,
      mfcc6Mean = mfccStats(5)._1, mfcc6Var = mfccStats(5)._2,
      mfcc7Mean = mfccStats(6)._1, mfcc7Var = mfccStats(6)._2,
      mfcc8Mean = mfccStats(7)._1, mfcc8Var = mfccStats(7)._2,
      mfcc9Mean = mfccStats(8)._1, mfcc9Var = mfccStats(8)._2,
      mfcc10Mean = mfccStats(9)._1, mfcc10Var = mfccStats(9)._2,
      mfcc11Mean = mfccStats(10)._1, mfcc11Var = mfccStats(10)._2,
      mfcc12Mean = mfccStats(11)._1, mfcc12Var = mfccStats(11)._2,
      mfcc13Mean = mfccStats(12)._1, mfcc13Var = mfccStats(12)._2
    )
  }

  /**
   * Root-mean-square energy of one frame. A silent (all-zero) frame gives
   * exactly 0, never NaN: the sum of squares is 0 and frame.length is
   * always positive, so no epsilon is needed here.
   */
  def rms(frame: Array[Double]): Double =
    math.sqrt(frame.map(x => x * x).sum / frame.length)

  /**
   * Fraction of adjacent sample pairs in a frame that cross zero (sign
   * change), using >= 0 as the sign test so an exact-zero sample does not
   * get double-counted as its own crossing.
   * For a pure sine at frequency f sampled at rate sr, this is ≈ 2f / sr.
   */
  def zeroCrossingRate(frame: Array[Double]): Double = {
    var crossings = 0
    var i = 1
    while (i < frame.length) {
      if ((frame(i) >= 0) != (frame(i - 1) >= 0)) crossings += 1
      i += 1
    }
    crossings.toDouble / frame.length
  }

  /** Mean and population variance of a sequence of per-frame values. */
  private def meanVar(values: Seq[Double]): (Double, Double) = {
    val n    = values.length
    val mean = values.sum / n
    val variance = values.map(v => math.pow(v - mean, 2)).sum / n
    (mean, variance)
  }

  // ==================================================================
  // Spectral features (stage c): centroid, bandwidth, roll-off, flatness
  // ==================================================================

  /**
   * Periodic Hann window of the given size: w[n] = 0.5 - 0.5*cos(2*pi*n/(N-1)).
   * Tapers each frame's edges to near zero before the FFT, which reduces
   * spectral leakage from cutting a continuous signal into finite frames.
   * Computed once by the caller and broadcast, never recomputed per frame.
   */
  private def computeHannWindow(size: Int): Array[Double] =
    Array.tabulate(size)(n => 0.5 - 0.5 * math.cos(2 * math.Pi * n / (size - 1)))

  /**
   * Magnitude spectrum of one frame: Hann-windows it, runs Breeze's FFT
   * (breeze.signal.fourierTr), and keeps only the non-negative-frequency
   * bins 0..N/2 (a real input's spectrum is symmetric, so the rest is
   * redundant). Bin k corresponds to frequency k * sampleRate / frameSize.
   */
  private def magnitudeSpectrum(frame: Array[Double], hann: Array[Double]): Array[Double] = {
    val windowed = Array.tabulate(frame.length)(i => frame(i) * hann(i))
    val spectrum = fourierTr(DenseVector(windowed))
    val halfLen  = frame.length / 2 + 1
    Array.tabulate(halfLen)(k => spectrum(k).abs)
  }

  /**
   * Centroid, bandwidth, roll-off (85%), and flatness of one frame, given
   * its already-computed magnitude spectrum (shared with MFCC extraction
   * so the FFT only runs once per frame).
   * A silent frame (magnitude ~ 0 everywhere) has no defined spectral
   * shape: rather than divide by ~0 and produce NaN/Infinity, it is
   * reported as (0, 0, 0, 0) explicitly.
   */
  private def spectralFeaturesFromMagnitude(magnitudes: Array[Double], freqs: Array[Double]): (Double, Double, Double, Double) = {
    val totalMag = magnitudes.sum

    if (totalMag < AudioConfig.Epsilon) {
      (0.0, 0.0, 0.0, 0.0)
    } else {
      val centroid  = spectralCentroid(freqs, magnitudes, totalMag)
      val bandwidth = spectralBandwidth(freqs, magnitudes, centroid, totalMag)
      val rolloff   = spectralRolloff(freqs, magnitudes, totalMag)
      val flatness  = spectralFlatness(magnitudes)
      (centroid, bandwidth, rolloff, flatness)
    }
  }

  /** Convenience wrapper used by the synthetic checks: FFT + spectral shape for one raw frame. */
  def spectralFeaturesForFrame(frame: Array[Double], sampleRate: Int, hann: Array[Double]): (Double, Double, Double, Double) = {
    val magnitudes = magnitudeSpectrum(frame, hann)
    val freqs      = Array.tabulate(magnitudes.length)(k => k.toDouble * sampleRate / frame.length)
    spectralFeaturesFromMagnitude(magnitudes, freqs)
  }

  /** Magnitude-weighted average frequency: "center of mass" of the spectrum, in Hz. */
  private def spectralCentroid(freqs: Array[Double], mags: Array[Double], totalMag: Double): Double =
    freqs.indices.map(k => freqs(k) * mags(k)).sum / totalMag

  /** Magnitude-weighted standard deviation of frequency around the centroid, in Hz. */
  private def spectralBandwidth(freqs: Array[Double], mags: Array[Double], centroid: Double, totalMag: Double): Double =
    math.sqrt(freqs.indices.map(k => math.pow(freqs(k) - centroid, 2) * mags(k)).sum / totalMag)

  /** Frequency below which AudioConfig.RolloffPercent of the frame's total magnitude lies. */
  private def spectralRolloff(freqs: Array[Double], mags: Array[Double], totalMag: Double): Double = {
    val threshold = AudioConfig.RolloffPercent * totalMag
    var cumulative = 0.0
    var k = 0
    while (k < mags.length && cumulative < threshold) {
      cumulative += mags(k)
      k += 1
    }
    freqs(math.min(math.max(k - 1, 0), freqs.length - 1))
  }

  /**
   * Wiener entropy: geometric mean / arithmetic mean of the power spectrum.
   * Near 1 for a flat, noise-like spectrum; near 0 for a peaky, tonal one.
   * Epsilon keeps log() and the final division safe for near-zero bins.
   */
  private def spectralFlatness(mags: Array[Double]): Double = {
    val power     = mags.map(m => m * m + AudioConfig.Epsilon)
    val geoMean   = math.exp(power.map(math.log).sum / power.length)
    val arithMean = power.sum / power.length
    geoMean / (arithMean + AudioConfig.Epsilon)
  }

  /** Power spectrum (magnitude^2) from an already-computed magnitude spectrum. */
  private def powerSpectrum(magnitudes: Array[Double]): Array[Double] =
    magnitudes.map(m => m * m)

  // ==================================================================
  // MFCC (stage d): mel filterbank, log compression, DCT-II
  // ==================================================================

  /**
   * Triangular mel filterbank (Slaney-style, area-normalized): numMel
   * filters spanning 0 Hz to Nyquist, evenly spaced on the mel scale so
   * they match how pitch is perceived (finer resolution at low frequencies,
   * coarser at high). Returns a numMel x (fftSize/2 + 1) matrix; row i
   * holds filter i's weight for every FFT bin. Computed once by the caller
   * and broadcast, never recomputed per frame.
   */
  private def computeMelFilterbank(sampleRate: Int, fftSize: Int, numMel: Int): Array[Array[Double]] = {
    def hzToMel(f: Double): Double = 2595.0 * math.log10(1.0 + f / 700.0)
    def melToHz(m: Double): Double = 700.0 * (math.pow(10.0, m / 2595.0) - 1.0)

    val numBins = fftSize / 2 + 1
    val nyquist = sampleRate / 2.0
    val melMin  = hzToMel(0.0)
    val melMax  = hzToMel(nyquist)

    // numMel + 2 points evenly spaced in mel, converted to Hz then to FFT bin indices;
    // each filter i is the triangle over (binPoints(i), binPoints(i+1), binPoints(i+2)).
    val melPoints = Array.tabulate(numMel + 2)(i => melMin + i * (melMax - melMin) / (numMel + 1))
    val hzPoints  = melPoints.map(melToHz)
    val binPoints = hzPoints.map(f => math.floor((fftSize + 1) * f / sampleRate).toInt)

    Array.tabulate(numMel) { i =>
      val left   = binPoints(i)
      val center = binPoints(i + 1)
      val right  = binPoints(i + 2)
      // Scales each filter so its area is comparable regardless of bandwidth
      // (wide filters at high mel indices would otherwise dominate).
      val enorm = 2.0 / (hzPoints(i + 2) - hzPoints(i))

      Array.tabulate(numBins) { bin =>
        if (bin < left || bin > right) 0.0
        else if (bin <= center) enorm * (bin - left).toDouble / math.max(center - left, 1)
        else enorm * (right - bin).toDouble / math.max(right - center, 1)
      }
    }
  }

  /** Mel-filterbank energies for one frame: each filter's weighted sum over the power spectrum. */
  private def melEnergies(power: Array[Double], filterbank: Array[Array[Double]]): Array[Double] =
    filterbank.map(filter => filter.indices.map(b => filter(b) * power(b)).sum)

  /**
   * MFCCs for one frame: log-compress the mel energies (epsilon avoids
   * log(0) for a silent frame, where every mel energy is exactly 0), then
   * apply an orthogonally-normalized DCT-II and keep the first numMfcc
   * coefficients (coefficient 0 is the overall log-energy level; the rest
   * describe the coarse shape of the log-mel spectrum).
   */
  private def mfcc(power: Array[Double], filterbank: Array[Array[Double]], numMfcc: Int): Array[Double] = {
    val logMel = melEnergies(power, filterbank).map(e => math.log(e + AudioConfig.Epsilon))
    dctII(logMel, numMfcc)
  }

  /** Orthogonally-normalized DCT-II (matches librosa/scipy's norm='ortho'), keeping the first numOut coefficients. */
  private def dctII(x: Array[Double], numOut: Int): Array[Double] = {
    val n = x.length
    Array.tabulate(numOut) { k =>
      val sum   = x.indices.map(i => x(i) * math.cos(math.Pi / n * (i + 0.5) * k)).sum
      val scale = if (k == 0) math.sqrt(1.0 / n) else math.sqrt(2.0 / n)
      sum * scale
    }
  }

  // ==================================================================
  // Synthetic correctness checks (run before touching real audio)
  // ==================================================================

  /** Small, fast checks on synthetic data, run automatically before every real job. */
  def runSyntheticChecks(): Unit = {
    println("=== Synthetic checks: windowing ===")

    // A full-length track (30 s at 22,050 Hz = 661,794 samples) must give exactly
    // 59 windows, indexed 0-58, with the last one starting at 29.0 s.
    val fullTrack = WavAudio("synthetic.full", "test", AudioConfig.ExpectedSampleRate, new Array[Double](661794))
    val fullWindows = windowTrack(fullTrack)
    require(fullWindows.size == 59, s"expected 59 windows for a full-length track, got ${fullWindows.size}")
    require(fullWindows.head.windowIdx == 0 && fullWindows.last.windowIdx == 58, "window_idx must run 0-58")
    require(
      math.abs(fullWindows.last.startSec - 29.0) < 1e-9,
      s"last window should start at 29.0s, got ${fullWindows.last.startSec}"
    )

    // A track shorter than one window must yield zero windows: no padding.
    val tooShort = WavAudio("synthetic.short", "test", AudioConfig.ExpectedSampleRate, new Array[Double](1000))
    require(windowTrack(tooShort).isEmpty, "a track shorter than one window must produce no windows")

    // A full window must frame into exactly 40 frames (2,048 samples, hop 512).
    val frames = frameSamples(new Array[Double](AudioConfig.WindowSize))
    require(frames.size == 40, s"expected 40 frames per window, got ${frames.size}")

    println("Synthetic windowing checks PASSED")

    println("\n=== Synthetic checks: time-domain features ===")

    // Silence must give RMS = 0 without raising an exception.
    val silence    = new Array[Double](AudioConfig.FrameSize)
    val silenceRms = rms(silence)
    require(silenceRms == 0.0, s"silence must give RMS = 0, got $silenceRms")

    // A pure 1,000 Hz sine at 22,050 Hz must give a ZCR close to 2 * 1000 / 22050.
    val sineFreq    = 1000.0
    val sampleRate  = AudioConfig.ExpectedSampleRate
    val sineFrame   = Array.tabulate(AudioConfig.FrameSize)(i => math.sin(2 * math.Pi * sineFreq * i / sampleRate))
    val expectedZcr = 2.0 * sineFreq / sampleRate
    val actualZcr   = zeroCrossingRate(sineFrame)
    require(
      math.abs(actualZcr - expectedZcr) < 0.1 * expectedZcr,
      s"sine ZCR should be close to $expectedZcr, got $actualZcr"
    )

    println(f"Silence RMS check: PASSED (rms = $silenceRms%.6f)")
    println(f"Sine ZCR check: PASSED (expected ~= $expectedZcr%.5f, got $actualZcr%.5f)")

    println("\n=== Synthetic checks: spectral features ===")

    val hann = computeHannWindow(AudioConfig.FrameSize)

    // Silence must give an all-zero spectral shape, not NaN/Infinity.
    val (silCentroid, silBandwidth, silRolloff, silFlatness) =
      spectralFeaturesForFrame(silence, sampleRate, hann)
    require(
      silCentroid == 0.0 && silBandwidth == 0.0 && silRolloff == 0.0 && silFlatness == 0.0,
      s"silence must give (0,0,0,0), got ($silCentroid, $silBandwidth, $silRolloff, $silFlatness)"
    )

    // A pure 1,000 Hz sine must give a spectral centroid close to 1,000 Hz:
    // nearly all of its (Hann-windowed) energy sits in the bin nearest 1,000 Hz.
    val (sineCentroid, _, _, _) = spectralFeaturesForFrame(sineFrame, sampleRate, hann)
    require(
      math.abs(sineCentroid - sineFreq) < 0.05 * sineFreq,
      s"sine spectral centroid should be close to $sineFreq Hz, got $sineCentroid Hz"
    )

    println("Silence spectral check: PASSED (centroid = bandwidth = rolloff = flatness = 0)")
    println(f"Sine centroid check: PASSED (expected ~= $sineFreq%.1f Hz, got $sineCentroid%.1f Hz)")

    println("\n=== Synthetic checks: MFCC ===")

    val melFilterbank = computeMelFilterbank(sampleRate, AudioConfig.FrameSize, AudioConfig.MelBands)

    // Silence: every mel band has exactly 0 energy, so every log-mel value is
    // the same constant (log(epsilon)). A DCT-II of a constant sequence puts
    // all its energy in coefficient 0; every higher coefficient is exactly 0
    // by the DCT basis's orthogonality to a constant signal.
    val silenceMfcc = mfcc(powerSpectrum(magnitudeSpectrum(silence, hann)), melFilterbank, AudioConfig.NumMfcc)
    val expectedMfcc0 = math.log(AudioConfig.Epsilon) * math.sqrt(AudioConfig.MelBands.toDouble)
    require(
      math.abs(silenceMfcc(0) - expectedMfcc0) < 1e-6,
      s"silence mfcc1 should be $expectedMfcc0, got ${silenceMfcc(0)}"
    )
    require(
      silenceMfcc.drop(1).forall(v => math.abs(v) < 1e-6),
      s"silence mfcc2..${AudioConfig.NumMfcc} should all be ~0, got ${silenceMfcc.drop(1).mkString(", ")}"
    )

    // Sine wave: MFCCs must be finite once real, non-uniform spectral energy
    // flows through the filterbank, log compression, and DCT.
    val sineMfcc = mfcc(powerSpectrum(magnitudeSpectrum(sineFrame, hann)), melFilterbank, AudioConfig.NumMfcc)
    require(sineMfcc.forall(v => !v.isNaN && !v.isInfinite), s"sine MFCCs must be finite, got ${sineMfcc.mkString(", ")}")

    println(f"Silence MFCC check: PASSED (mfcc1 = ${silenceMfcc(0)}%.4f, mfcc2..${AudioConfig.NumMfcc} ~= 0)")
    println("Sine MFCC finiteness check: PASSED")
  }
}
