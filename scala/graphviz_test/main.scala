import graphviz._
import graphviz.io._

object App {

  case class MyClass(x: Int){
    def ---(b: MyClass):(Int, Int) = (x, b.x)
  }
  implicit def intToMyTrait(x:Int): MyClass = MyClass(x)

  def main(args: Array[String]) {
    println("Hi")
    println(1 --- 2)
    //val graph = createGraph("tmp", ("1", "2"), ("2", "3"))
    val graph = createGraph("tmp", 
      "1" :|("color" := "blue"):|("shape" := "point"),
      "2" :|("color" := "blue"):|("shape" := "point"),
      "3" :|("color" := "blue"):|("shape" := "point"),
      "4" :|("color" := "blue"):|("shape" := "point"),
      "5" :|("color" := "blue"):|("shape" := "point"),
      "6" :|("color" := "blue"):|("shape" := "point"),
      "7" :|("color" := "blue"):|("shape" := "point"),
      "8" :|("color" := "blue"):|("shape" := "point"),
      "9" :|("color" := "blue"):|("shape" := "point"),
      "10" :|("color" := "blue"):|("shape" := "point"),
      "1" --- "2" :| ("color" := "red"), 
      "2" --- "3",
      "3" --- "4",
      "4" --- "5",
      "5" --- "1" :| ("color" := "red"),

      "6" --- "7",
      "7" --- "8",
      "8" --- "9",
      "9" --- "10",
      "10" --- "6",

      "1" --- "6",
      "2" --- "7",
      "3" --- "8",
      "4" --- "9",
      "5" --- "10"
      )
    println(graph.render)
    saveGraph("tmp2.dot", "tmp2.png", graph)
  }
}
