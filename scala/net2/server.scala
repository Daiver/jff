// Simple server
import java.net._
import java.io._
import scala.io._

object main {

    def main(args: Array[String]){
        val server = new ServerSocket(10000)
        while (true) {
            val s = server.accept()
            val in = new BufferedSource(s.getInputStream()).getLines()
            val out = new PrintStream(s.getOutputStream())
            val inText = in.next()
            out.println(inText)
            out.flush()
            s.close()
        }
    }
}
