name := "My Project"
 
version := "1.0"

scalaVersion := "2.11.6"

//libraryDependencies += "org.scalacheck" %% "scalacheck" % "1.13.0" 

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"
libraryDependencies += "org.scalacheck" %% "scalacheck" % "1.12.2" % "test"

//resolvers += "Akka Snapshot Repository" at "http://repo.akka.io/snapshots/"

//libraryDependencies +=
 //"com.typesafe.akka" %% "akka-actor" % "2.4-SNAPSHOT"

libraryDependencies  ++= Seq(
  // other dependencies here
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-natives" % "0.12"
)

//libraryDependencies += "org.sameersingh.scalaplot" % "scalaplot" % "0.0.4"
libraryDependencies += "org.scalanlp" %% "breeze-viz" % "0.12" 
resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"


//resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  //"Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  //"Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
//)
