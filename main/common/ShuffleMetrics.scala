package sonicspark.common

import org.apache.spark.SparkContext
import org.apache.spark.scheduler.{SparkListener, SparkListenerStageCompleted}

import java.util.concurrent.atomic.AtomicLong

/**
 * Measures total shuffle-write bytes produced by Spark while running a block of code,
 * by attaching a temporary SparkListener for the duration of the block.
 * Used by Phase 3's reflection section to compare reduceByKey vs. groupByKey.
 */
object ShuffleMetrics {

  /** Runs `block`, returning (its result, elapsed seconds, total shuffle-write bytes). */
  def timeAndShuffle[T](sc: SparkContext)(block: => T): (T, Double, Long) = {
    val bytesWritten = new AtomicLong(0)
    val listener = new SparkListener {
      override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
        bytesWritten.addAndGet(stageCompleted.stageInfo.taskMetrics.shuffleWriteMetrics.bytesWritten)
      }
    }

    sc.addSparkListener(listener)
    val start  = System.nanoTime()
    val result = block
    val seconds = (System.nanoTime() - start) / 1e9
    sc.removeSparkListener(listener)

    (result, seconds, bytesWritten.get())
  }
}
