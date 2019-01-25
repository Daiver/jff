
import breeze.linalg._

package object ml {

  case class SVM(direction: DenseVector[Double], intercept: Double) {

	def predict(sample: DenseVector[Double]): Int = 
	  if ((sample.dot(direction) + intercept) >= 0.0) 1 else -1

	def predict(samples: DenseMatrix[Double]): DenseVector[Int] = 
	  ((samples * direction + intercept) :>= 0.0) map {x => if (x) 1 else -1}

	def evaluate(samples: DenseMatrix[Double], labels: DenseVector[Int]): Int = 
	  (predict(samples) :== labels) map {x => if(x) 0 else 1} sum
	
  }

  def svmGradient(lambda: Double, weights: DenseVector[Double], sample: DenseVector[Double], label: Double): DenseVector[Double] = {
    val nFeatures = sample.length
    val wDirection = weights(0 until nFeatures)
    val wIntercept = weights(nFeatures)
    val activation = 1.0 - label * ((wDirection dot sample) + wIntercept)
    val grad = DenseVector.zeros[Double](nFeatures + 1)
    grad(0 until nFeatures) := lambda * 2.0 * wDirection
    grad(nFeatures) = 0.0
    if(activation >= 0.0){
      grad(0 until nFeatures) += - label * sample
      grad(nFeatures) = -label
    }
    grad
  }

  def svmError(weights: DenseVector[Double], data: DenseMatrix[Double], labels: DenseVector[Int]): Double = 
      SVM(weights(0 until weights.length - 1), weights(weights.length - 1)).evaluate(data, labels).toDouble

}
