import scala.util.Random
import breeze.linalg._

package object optimization {

  def SGD(
    errorFunc: (DenseVector[Double], DenseMatrix[Double], DenseVector[Int]) => Double,
    errorGrad: (DenseVector[Double], DenseVector[Double], Double) => DenseVector[Double],
      data: DenseMatrix[Double], 
      labels: DenseVector[Int], 
      wInit: DenseVector[Double], 
      nIters: Int, 
      batchSize: Int, 
      stepSize: Double,
      momentCoeff: Double
    ) = {
    assert(data.cols + 1 == wInit.length)
    assert(data.rows == labels.length)
    val nSamples    = data.rows
    val nFeatures   = data.cols
    var weights     = wInit
    var rand        = new Random(42)
    var bestWeights = weights
    var bestErr     = nSamples.toDouble
    var oldGrad     = DenseVector.zeros[Double](nFeatures + 1)

    for(iterInd <- 0 until nIters){
      val indices = util.Random.shuffle((0 until nSamples).toList)
      for(iterInd2 <- 0 until nSamples / batchSize){

        var grad : DenseVector[Double] = 
          (0 until batchSize).par.map( i => {
            val index  = indices(iterInd2 * batchSize + i)
            //val index  = rand.nextInt(nSamples)
            val sample = data(index, ::).t
            val label  = labels(index).toDouble
            errorGrad(weights, sample, label)
          }).par.reduce{_ + _}
        grad :/= batchSize.toDouble
        weights -= stepSize * (momentCoeff * oldGrad + grad)
        oldGrad = grad
      }
      val errCur = errorFunc(weights, data, labels)
      println("epoch: " + iterInd + " err: " + errCur)
      if(errCur < bestErr){
        bestErr = errCur
        bestWeights = weights
      }
    }

	bestWeights
  }
}
