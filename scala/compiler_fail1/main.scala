import scala.reflect.runtime.universe._

object App{

  def meth[T: TypeTag](value: T) = {
    if(typeOf[T] =:= typeOf[Double])
      println("Double")
    //if(typeOf[T] =:= typeOf[Int])
      //println("Int")
  }

  def main(args: Array[String]) = {}
}
