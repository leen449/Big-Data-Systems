package sonicspark.common

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths => JPaths, StandardOpenOption}

/**
 * Prints a line to the console and appends the same line to results/rdd_output.txt,
 * so the saved report is exactly what was printed during the run (Phase 3).
 */
object Report {

  private val reportPath = JPaths.get(Paths.RddReport)

  /** Call once at the start of the program: creates a fresh, empty report file. */
  def start(): Unit = {
    Files.createDirectories(reportPath.getParent)
    Files.write(reportPath, Array.emptyByteArray)
  }

  /** Print one line and append it to the report file. */
  def log(line: String): Unit = {
    println(line)
    Files.write(reportPath, (line + System.lineSeparator()).getBytes(StandardCharsets.UTF_8), StandardOpenOption.APPEND)
  }

  /** A titled section banner, used to separate analyses in the report. */
  def section(title: String): Unit = {
    log("")
    log("=" * (title.length + 4))
    log(s"= $title =")
    log("=" * (title.length + 4))
  }

  /** Print (and save to the report) an aligned table with a header row. */
  def table(header: Seq[String], rows: Seq[Seq[String]]): Unit = {
    val widths = header.indices.map(i => (header +: rows).map(_(i).length).max)
    def line(cells: Seq[String]): String =
      cells.zip(widths).map { case (c, w) => c.padTo(w, ' ') }.mkString("| ", " | ", " |")
    val sep = widths.map("-" * _).mkString("+-", "-+-", "-+")

    log(sep); log(line(header)); log(sep)
    rows.foreach(r => log(line(r)))
    log(sep)
  }

  /** Save a small result table as its own CSV file under outputs/rdd/. */
  def saveCsv(fileName: String, header: Seq[String], rows: Seq[Seq[String]]): String = {
    val dir  = JPaths.get(Paths.RddOutputDir)
    Files.createDirectories(dir)
    val file = dir.resolve(fileName)
    val lines = (header +: rows).map(_.map(escape).mkString(","))
    Files.write(file, lines.mkString("\n").getBytes(StandardCharsets.UTF_8))
    file.toString
  }

  // Wrap values containing commas or quotes so the CSV stays valid
  private def escape(s: String): String =
    if (s.exists(c => c == ',' || c == '"' || c == '\n')) "\"" + s.replace("\"", "\"\"") + "\""
    else s
}
