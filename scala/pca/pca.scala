import scala.io._
import breeze.linalg._
import breeze.linalg.svd.SVD
import breeze.stats.mean
import math._

package ml.pca{

class PCA(nComponentsInit: Int){
  val nComponents          = nComponentsInit
  var means                = DenseVector.zeros[Float](0)
  var principialComponents = DenseMatrix.zeros[Float](0, 0)
  var eigenValues          = DenseVector.zeros[Float](0)

  def fit(x: DenseMatrix[Float]){
    val meansM = mean(x(::, *))
    means = meansM(0, ::).t
    val x1 = x(*, ::) - means

    val SVD(u, s, vt) = svd(x1)

    eigenValues  = DenseVector.zeros[Float](nComponents)
    principialComponents = DenseMatrix.zeros[Float](nComponents, vt.cols)
    for(i <- 0 until nComponents){
      eigenValues(i) = s(i)
      principialComponents(i, ::) := vt(i, ::)
    }
    eigenValues = eigenValues map {Math.sqrt(_).toFloat}
    /*(resValues map {Math.sqrt(_).toFloat}, resVectors, means(0, ::).t)*/
  }

  def transform(sample: DenseVector[Float]) = {
    principialComponents * (sample - means)
  }

  def transformInverse(sample: DenseVector[Float]) = {
    principialComponents.t * sample + means
  }

}

}
