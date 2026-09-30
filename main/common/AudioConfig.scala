package sonicspark.common

/**
 * Every tunable number used by audio feature extraction, in one place.
 * Nothing in AudioFeatures.scala should hard-code these values directly.
 *
 * Reference sample rate is GTZAN's own (22,050 Hz, mono, 16-bit PCM).
 * Windowing/frame sizes are expressed in samples so the math (e.g. windows
 * per track) is exact for files at that rate; extraction still reads the
 * real rate out of each file's header rather than assuming it.
 */
object AudioConfig {

  // ---------- Expected input format (GTZAN) ----------
  val ExpectedSampleRate = 22050
  val ExpectedChannels   = 1
  val ExpectedBitDepth   = 16

  // ---------- Windowing (stage a) ----------
  val WindowSize = 22050   // 1.0 s at 22,050 Hz
  val WindowHop  = 11025   // 0.5 s hop -> 50% overlap; incomplete tail windows are dropped

  // ---------- Framing inside each window (stage c/d, FFT analysis) ----------
  val FrameSize = 2048     // ~92.9 ms at 22,050 Hz
  val FrameHop  = 512      // 75% overlap between frames

  // ---------- Spectral features (stage c) ----------
  val RolloffPercent = 0.85  // fraction of total spectral energy below the roll-off frequency

  // ---------- MFCC (stage d) ----------
  val MelBands = 40   // mel filterbank size (librosa's default); MFCCs are computed from this
  val NumMfcc  = 13

  // ---------- Numerical safety ----------
  // Small constant added before divisions and logarithms so a zero-energy
  // frame (silence) produces 0, never NaN or -Infinity.
  val Epsilon = 1e-10
}
