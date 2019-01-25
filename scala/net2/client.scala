// Simple client
import java.net._
import java.io._
import scala.io._

object Client{

    def main(args: Array[String]) {
        val s = new Socket(InetAddress.getByName("localhost"), 10000)
        lazy val in = new BufferedSource(s.getInputStream()).getLines()
        val out = new PrintStream(s.getOutputStream())

        out.println("Hello, world")
        out.flush()
        println("Received: " + in.next())

        s.close()
    }
}
