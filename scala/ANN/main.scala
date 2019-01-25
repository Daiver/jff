import scala.io._
import breeze.linalg._
import org.sameersingh.scalaplot.Implicits._
import org.sameersingh.scalaplot._
import math._

object MyProject{

  def vectorToArray[T](arr : DenseVector[T]): Seq[T] = 
    (0 until arr.length).map(arr(_))

  def gradientDescent(
    gradFunc: DenseMatrix[Double] => DenseMatrix[Double],
    initialWeights: DenseMatrix[Double],
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

  def addBiasToDataset(x: DenseMatrix[Double]) = 
    DenseMatrix.horzcat(x, DenseMatrix.ones[Double](x.rows, 1))

  def optimizationForTwoLayer(
    weightsOutter: DenseMatrix[Double],
    weightsInner: DenseMatrix[Double],
    dataset: DenseMatrix[Double],
    recalls: DenseMatrix[Double],
    nIter: Int,
    learningRate: Double) = {
    //weightsOutter += weightsInner
    for(iter <- (0 to nIter)) {
      val innerAct  = activateLayer(weightsInner, dataset) //NxR R - inner layer size     
      val innerActB = addBiasToDataset(innerAct)
      val deltaOutter = deltaForOutLayerNeuron(innerActB, recalls, weightsOutter)
      val deltaInner = deltaForInnerLayer(deltaOutter, weightsOutter, dataset, weightsInner)
      weightsInner -= learningRate * (deltaInner.t) * dataset
      weightsOutter -= learningRate * (deltaOutter.t) * innerActB
    }
    (weightsOutter, weightsInner)
  }

  def sigmoid(x: Double): Double = 1.0/(1 + Math.exp(-x))
  def gradSigmoid(x: Double): Double = sigmoid(x) * (1.0 - sigmoid(x))

  def deltaForOutLayerNeuron(
    dataset: DenseMatrix[Double],
    recalls: DenseMatrix[Double],
    weights: DenseMatrix[Double]) = {
    val net   = dataset * weights.t
    val act   = net map sigmoid
    val diff  = act - recalls
    val grads = net map gradSigmoid
    diff :* grads
  }

  def deltaForInnerLayer(
    deltaFromPreviousLayer   : DenseMatrix[Double],
    weightsFromPreviousLayer : DenseMatrix[Double],
    dataset                  : DenseMatrix[Double],
    weights                  : DenseMatrix[Double]
    ) = {
    val tmp = (deltaFromPreviousLayer * weightsFromPreviousLayer)
    val tmp2 = tmp(::, 0 until weightsFromPreviousLayer.cols - 1)
    val net   = dataset * weights.t
    val grads = net map gradSigmoid
    //println("delta set shape", deltaFromPreviousLayer.rows, deltaFromPreviousLayer.cols)
    //println("tmp set shape", tmp.rows, tmp.cols)
    //println("grads set shape", grads.rows, grads.cols)
    tmp2 :* grads
  }

  def gradForSigmoid(
    dataset: DenseMatrix[Double],
    recalls: DenseMatrix[Double],
    weights: DenseMatrix[Double]) = {
    deltaForOutLayerNeuron(dataset, recalls, weights).t * dataset
  }

  def activateLayer(
    weights: DenseMatrix[Double],
    dataset: DenseMatrix[Double]) = (dataset * weights.t) map sigmoid

  def activateLayer(
    weights: DenseMatrix[Double],
    sample : DenseVector[Double]) = (sample.t * weights.t).t map sigmoid

  def activate(
    weightsInner:DenseMatrix[Double],
    weightsOutter:DenseMatrix[Double],
    sample : DenseVector[Double]
  ) = {
    val in = activateLayer(weightsInner, sample)
    val inB = DenseVector.vertcat(in, DenseVector(1.0))
    activateLayer(weightsOutter, inB)
  }

  def main(args:Array[String]){

    val data = DenseMatrix(
      (2.0, 1.0),
      (5.0, 1.0),
      (1.6, 1.0),
      (1.1, 1.0))
    val values = DenseMatrix(
      (1.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)
      )

    //println(addBiasToDataset(data))

/*    val solution = gradientDescent(*/
        //gradForSigmoid(data, values, _),
        //DenseMatrix.zeros[Double](2, 2),
        //500000,
        //0.05
      /*)*/

    val (solutionO, solutionI) = optimizationForTwoLayer(
        DenseMatrix.zeros[Double](2, 4),
        DenseMatrix.zeros[Double](3, 2),
        data,
        values,
        500000,
        0.1
      )
    println("Inner")
    println(solutionI)
    println("outter")
    println(solutionO)
    //println(activate(solutionI, solutionO, data(0, ::).t))
    //println(solution)
    println(values)
    //println(activateLayer(solution, data))
    for(i <- (0 until data.rows)){
      //println(activateLayer(solution, data(i, ::).t))
      println(activate(solutionI, solutionO, data(i, ::).t))
      println("----")
    }
  }

}
