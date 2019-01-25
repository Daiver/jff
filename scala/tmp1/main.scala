object App {
  trait FromIntConverter[T]{
	def apply(int: Int): T
  }

  def convertInt[T](int: Int)(implicit converter: FromIntConverter[T]): T = converter(int)

  implicit val IntToStringConverter = new FromIntConverter[String]{
	def apply(int: Int) = int.toString
  }

  implicit val IntToFloatConverter = new FromIntConverter[Float]{
	def apply(int: Int) = int.toFloat
  }

  implicit val IntToDoubleConverter = new FromIntConverter[Double]{
	def apply(int: Int) = int.toDouble
  }

  def main(args: Array[String]){
	//val result = convertInt[Float](5)
	val result = convertInt[Double](5)
	println(result.getClass)
  }
}
