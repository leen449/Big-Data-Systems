// =========================================================
// SonicSpark – build definition
// =========================================================

// ---------- Project info ----------
ThisBuild / version      := "0.1.0"
ThisBuild / scalaVersion := "2.12.18"   // must match Spark's Scala version

// ---------- Versions ----------
lazy val sparkVersion = "3.5.1"

// ---------- Project ----------
lazy val root = (project in file("."))
  .settings(
    name := "SonicSpark",

    // Spark libraries: core = RDDs, sql = DataFrames/SQL, mllib = machine learning
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core"  % sparkVersion,
      "org.apache.spark" %% "spark-sql"   % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion
    ),

    // Show helpful compiler warnings
    scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked"),

    // Run programs in a separate JVM so the options below are applied
    fork := true,
    outputStrategy := Some(StdoutOutput),

    // Memory limit for Spark (2 GB is enough for GTZAN; lower to 1g on small laptops)
    run / javaOptions += "-Xmx2g",

    // Required for Spark 3.5 to run on Java 17
    run / javaOptions ++= Seq(
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED"
    )
  )
