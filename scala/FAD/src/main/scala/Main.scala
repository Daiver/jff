class A(multiplier: Int){
  def f(x: Int): Int = multiplier * x
}

object Main extends App {
  def foo(): Int = 13
  def foo2(x: Int): Int = {
    x*x
  }
  val f = (x: Int) => x + 1
  val a = new A(7)
  println(f(13))
  println(foo())
  println(foo2(13))
  println(a.f(7))
  println("Hello World!")

}
