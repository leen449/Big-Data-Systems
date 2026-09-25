package sonicspark.common

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths => JPaths}

/** One row of a before/after table in the report. */
case class Metric(stage: String, check: String, before: String, after: String, note: String = "")

/**
 * Collects stage metrics, prints them as a table, and saves them to outputs/stats/
 * so report tables come directly from the code.
 */
object Metrics {

  private val header = Seq("stage", "check", "before", "after", "note")

  /** Print metrics as a readable table in the console. */
  def show(metrics: Seq[Metric]): Unit = {
    val rows   = metrics.map(m => Seq(m.stage, m.check, m.before, m.after, m.note))
    val widths = header.indices.map(i => (header +: rows).map(_(i).length).max)

    def line(cells: Seq[String]): String =
      cells.zip(widths).map { case (c, w) => c.padTo(w, ' ') }.mkString("| ", " | ", " |")
    val sep = widths.map("-" * _).mkString("+-", "-+-", "-+")

    println(sep); println(line(header)); println(sep)
    rows.foreach(r => println(line(r)))
    println(sep)
  }

  /** Save metrics as a CSV file in outputs/stats/ and return its path. */
  def save(metrics: Seq[Metric], fileName: String): String = {
    val dir = JPaths.get(Paths.Stats)
    Files.createDirectories(dir)
    val file = dir.resolve(fileName)

    val lines = header.mkString(",") +: metrics.map { m =>
      Seq(m.stage, m.check, m.before, m.after, m.note).map(escape).mkString(",")
    }
    Files.write(file, lines.mkString("\n").getBytes(StandardCharsets.UTF_8))
    file.toString
  }
  /** Print any table (used for the feature ranking). */
  def showTable(header: Seq[String], rows: Seq[Seq[String]]): Unit = {
    val widths = header.indices.map(i => (header +: rows).map(_(i).length).max)
    def line(cells: Seq[String]): String =
      cells.zip(widths).map { case (c, w) => c.padTo(w, ' ') }.mkString("| ", " | ", " |")
    val sep = widths.map("-" * _).mkString("+-", "-+-", "-+")

    println(sep); println(line(header)); println(sep)
    rows.foreach(r => println(line(r)))
    println(sep)
  }

  /** Save any table as a CSV in outputs/stats/ (used for the feature ranking). */
  def saveTable(fileName: String, header: Seq[String], rows: Seq[Seq[String]]): String = {
    val dir = JPaths.get(Paths.Stats)
    Files.createDirectories(dir)
    val file  = dir.resolve(fileName)
    val lines = (header +: rows).map(_.map(escape).mkString(","))
    Files.write(file, lines.mkString("\n").getBytes(StandardCharsets.UTF_8))
    file.toString
  }
  
  // Wrap values containing commas or quotes so the CSV stays valid
  private def escape(s: String): String =
    if (s.exists(c => c == ',' || c == '"' || c == '\n')) "\"" + s.replace("\"", "\"\"") + "\""
    else s
}