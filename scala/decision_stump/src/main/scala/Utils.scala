
import breeze.linalg._

package ml {
  package object Utils {

    //TODO: make it generic
    def subsetRows(mat: DenseMatrix[Double], indices: Seq[Int]) = {
      val nIndices = indices.length
      val nCols    = mat.cols
      var res = DenseMatrix.zeros[Double](nIndices, nCols)
      for(i <- (0 until nIndices)){
        assert(indices(i) >= 0 && indices(i) < mat.rows)
        res(i, ::) := mat(indices(i), ::)
      }
      res
    }

    def subsetRows(vec: DenseVector[Int], indices: Seq[Int]) = {
      val nIndices = indices.length
      var res = DenseVector.zeros[Int](nIndices)
      for(i <- 0 until nIndices)
        res(i) = vec(indices(i))
      res
    }

    def freqDictFromLabels(values: DenseVector[Int]) = {
      var counts = DenseVector.zeros[Double](values.max + 1)
      for (value <- values){
        assert(value >= 0)
        counts(value) += 1
      }
      counts 
    }

    def freqDictFromLabelsWeighted(
        labels: DenseVector[Int],
        weights: DenseVector[Double],
        sampleIndices: Seq[Int],
        nClasses: Int) = {
      var counts = DenseVector.zeros[Double](nClasses)
      for (index <- sampleIndices){
        val label  = labels(index)
        val weight = weights(index)
        assert(label >= 0 && label < nClasses)
        counts(label) += weight
      }
      counts 
    }

    //TODO: rewrite it in stateless style
    def indicesForSplittedValues(dataset:DenseMatrix[Double], featInd: Int, threshold: Double) = 
      (0 until dataset.rows) partition (dataset(_, featInd) >= threshold)

    def majorClassFromFreqDict(freqsDict: DenseVector[Double]) = argmax(freqsDict)

    def majorClassifierError(freqsDict: DenseVector[Double]) = 1.0 - freqsDict.max / freqsDict.sum

    def sortedIndicesByDataset(
      dataset: DenseMatrix[Double], 
      featInd: Int,
      indices: Seq[Int]) = indices.sortBy(dataset(_, featInd))

  }
}
