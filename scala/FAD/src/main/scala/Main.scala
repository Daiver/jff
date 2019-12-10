//import scala.math.Numeric

object Main extends App {

  case class Fad(value: Float, der: Float)

  trait FadIsNumeric extends Numeric[Fad] {
    def plus(x: Fad, y: Fad): Fad = Fad(x.value + y.value, x.der + y.der)
    def minus(x: Fad, y: Fad): Fad = ???
    def times(x: Fad, y: Fad): Fad = Fad(x.value * y.value, x.value * y.der + x.der * y.value)
    def negate(x: Fad): Fad = ???
    def fromInt(x: Int): Fad = ???
    def parseString(str: String): Option[Fad] = ???
    def toInt(x: Fad): Int = ???
    def toLong(x: Fad): Long = ???
    def toFloat(x: Fad): Float = ???
    def toDouble(x: Fad): Double = ???
  }
  implicit object FadIsNumeric extends FadIsNumeric with Ordering[Fad] {
    override def compare(x: Fad, y: Fad): Int = ???
  }

  def square[T](x: T)(implicit num: Numeric[T]): T = {
    import num._
    x * x
  }
  val x = new Fad(3, 1)
  println(square(x))
  println("Hello World!")

}
