import graphviz._
import sys.process._
import java.io._

package graphviz {
  package object io {
    def saveGraph(fname: String, pngFName: String, graph: Graph) = {
      val rendered = graph.render
      new PrintWriter(fname) { write(rendered); close }
      s"dot -Tpng ${fname} -o${pngFName}" !
      //println(s"echo \' ${rendered} \' | dot -Tpng -o${fname}")
      //(s"echo ${rendered}") #| (s"dot -Tpng -o${fname}") !
      //println("ls -al" #| "grep Foo" !)
      //println(Seq("echo", "1")  #| "dot -Tpng -otmp.png" !!)
    }
  }
}
