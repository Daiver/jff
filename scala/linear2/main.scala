import scala.io._
import breeze.linalg._

object MyProject{

  def linearLeastSquaresMethod(a:DenseMatrix[Double], b:DenseVector[Double]) = 
    breeze.linalg.pinv(a.t * a) * a.t * b

  def main(args:Array[String]){
    //breeze.linalg.
    println("Hi")
    val a = DenseMatrix((2.0, 2.0), (3.0, 4.0))
    val b = DenseVector.ones[Double](2)
    println(linearLeastSquaresMethod(a, b))
    //val m2 = DenseMatrix.ones[Double](5, 5)
    //val x = {1, 2, 3}
    //println(m1 * m2)

  }
}
