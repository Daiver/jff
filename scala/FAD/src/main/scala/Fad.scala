//case class Fad()

case class Fad(value: Float, der: Float = 0.0f)
object Fad {

  trait FadIsFractional extends Fractional[Fad] {
    def plus(x: Fad, y: Fad): Fad = Fad(x.value + y.value, x.der + y.der)

    def minus(x: Fad, y: Fad): Fad = Fad(x.value - y.value, x.der - y.der)

    def times(x: Fad, y: Fad): Fad = Fad(x.value * y.value, x.value * y.der + x.der * y.value)

    def negate(x: Fad): Fad = Fad(-x.value, -x.der)

    def div(x: Fad, y: Fad): Fad =
      Fad(x.value / y.value, (y.value * x.der - x.value * y.der) / (y.value * y.value))

    def fromInt(x: Int): Fad = Fad(x, 0)

    def parseString(str: String): Option[Fad] = ???

    def toInt(x: Fad): Int = ???

    def toLong(x: Fad): Long = ???

    def toFloat(x: Fad): Float = ???

    def toDouble(x: Fad): Double = ???
  }

  implicit object FadIsFractional extends FadIsFractional with Ordering[Fad] {
    override def compare(x: Fad, y: Fad): Int = ???
  }
  implicit def floatToFad(x: Float): Fad = Fad(x, 0)
}
