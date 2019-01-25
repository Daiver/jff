import scala.io._
import breeze.linalg._
import org.sameersingh.scalaplot.Implicits._
import org.sameersingh.scalaplot._
import math._

object MyProject{

  def vectorToArray[T](arr : DenseVector[T]): Seq[T] = 
    (0 until arr.length).map(arr(_))

  def linearLeastSquaresMethod(a:DenseMatrix[Double], b:DenseVector[Double]) = 
    breeze.linalg.pinv(a.t * a) * a.t * b

  def gradientDescent(
    gradFunc: DenseVector[Double] => DenseVector[Double],
    initialWeights: DenseVector[Double],
    nIter: Int,
    learningRate: Double) = {
      var weights = initialWeights
    for(iter <- (0 to nIter)) {
      //println(iter, weights)
      val delta = gradFunc(weights)
      //println("delta", delta)
      weights -= learningRate * delta
    }
    weights
  }

  def gradForLinearLeastSquares(
    dataset: DenseMatrix[Double],
    recalls: DenseVector[Double],
    weights: DenseVector[Double]) = {
    val nVars = weights.length
    val nData = dataset.rows
    val act  = dataset * weights - recalls
    dataset.t * act
  }

  def lineFromData(coeff : DenseVector[Double], data: DenseVector[Double]) = 
    (0 until data.length).map(coeff(0) * data(_) + coeff(1))

  def sigmoid(x: Double): Double = 1.0/(1 + Math.exp(-x))
  def gradSigmoid(x: Double): Double = sigmoid(x) * (1.0 - sigmoid(x))

  def deltaForOneNeuron(
    dataset: DenseMatrix[Double],
    recalls: DenseVector[Double],
    weights: DenseVector[Double]) = {
    val net   = dataset * weights
    val act   = net map sigmoid
    val diff  = act - recalls
    val grads = net map gradSigmoid
    diff :* grads
  }

  def gradForLog(
    dataset: DenseMatrix[Double],
    recalls: DenseVector[Double],
    weights: DenseVector[Double]) = {
    dataset.t * deltaForOneNeuron(dataset, recalls, weights)
  }

  def activateLog(
    weights: DenseVector[Double],
    dataset: DenseVector[Double]) = (dataset dot weights) map sigmoid

  def main(args:Array[String]){
    val data = DenseMatrix(
      (2.0, 1.0),
      (5.0, 1.0),
      (1.6, 1.0),
      (1.1, 1.0)
      )
    val values = DenseVector[Double](1.0, 1.0, 0.0, 0.0)
    val solution = gradientDescent(
        gradForLog(data, values, _),
        DenseVector.zeros[Double](2),
        50000,
        0.01
      )
    println(solution)
    println(values)
    //println(data map {activateLog(solution, _)})
    for(i <- (0 until data.rows))
      println(activateLog(solution, data(i, ::).t))
  }

  def main2(args:Array[String]){
    val x  = DenseVector(1.0, 2.0, 3.0, 4.0, 5, 6, 7, 8, 9, 10)
    val x1 = vectorToArray(x)
    val x2 = DenseMatrix((1.0, 1.0),
                         (2.0, 1.0),
                         (3.0, 1.0),
                         (4.0, 1.0),
                         (5.0, 1.0),
                         (6.0, 1.0),
                         (7.0, 1.0),
                         (8.0, 1.0),
                         (9.0, 1.0),
                         (10.0, 1.0)
                       )
    val y  = DenseVector(1.2, 1.6, 5.2, 6.4, 8.5, 9.0, 8.5, 6.0, 5.1, 1.0)
    val y1 = vectorToArray(y)
    val solution1 = linearLeastSquaresMethod(x2, y)
    //val solution = DenseVector[Double](2.34, -1.1)
    println("real solution", solution1)
    val solution = gradientDescent(
        gradForLinearLeastSquares(x2, y, _),
        DenseVector.zeros[Double](2),
        5000,
        0.00095
      )
    println("solution", solution)
    val y2 = lineFromData(solution, x)
    println("y1", y1)
    println("y2", y2)

   val rnd = new scala.util.Random(0)

   val data = xyChart(
     x1 -> Seq(
              Y(y1, style=XYPlotStyle.Points),
              Y(y2)))
  }
}
