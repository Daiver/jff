//case class Fad()

case class Fad(value: Float, der: Float)
object Fad {

  trait FadIsNumeric extends Numeric[Fad] {
    def plus(x: Fad, y: Fad): Fad = Fad(x.value + y.value, x.der + y.der)

    def minus(x: Fad, y: Fad): Fad = Fad(x.value - y.value, x.der - y.der)

    def times(x: Fad, y: Fad): Fad = Fad(x.value * y.value, x.value * y.der + x.der * y.value)

    def negate(x: Fad): Fad = Fad(-x.value, -x.der)

    def fromInt(x: Int): Fad = Fad(x, 0)

    def parseString(str: String): Option[Fad] = ???

    def toInt(x: Fad): Int = ???

    def toLong(x: Fad): Long = ???

    def toFloat(x: Fad): Float = ???

    def toDouble(x: Fad): Double = ???
  }

  implicit object FadIsNumeric extends FadIsNumeric with Ordering[Fad] {
    override def compare(x: Fad, y: Fad): Int = ???
  }

}