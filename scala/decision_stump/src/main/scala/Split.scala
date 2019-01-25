
import breeze.linalg._
import ml.Utils._

package ml {
  package object DecisionSplit {

    //TODO: should be rewrited
    //looks over engineering but it is needed to make this method generic
    //also i dont't know way to do this in stateless style
    //Algo by Piotr Dollar. Translation from raw C by me
    def bestSplitForRealValuedDataset(
          dataset: DenseMatrix[Double], 
          weights: DenseVector[Double], 
          labels:  DenseVector[Int],
          sampleIndices: Seq[Int],
          nClasses: Int,
          featInd: Int
        ) = {
      assert(dataset.rows == labels.length && labels.length == weights.length)

      val nSamples = sampleIndices.length
      assert(nSamples > 1)

      val sortedIndices = sortedIndicesByDataset(dataset, featInd, sampleIndices)

      var leftFreqs    = freqDictFromLabelsWeighted(labels, weights, sampleIndices, nClasses)
      var rightFreqs   = DenseVector.zeros[Double](nClasses + 1)
      var leftWeights  = leftFreqs.sum
      var rightWeights = rightFreqs.sum

      var bestError     = majorClassifierError(leftFreqs)
      var bestThreshold = dataset(sortedIndices(0), featInd)

      //Last index is used to compute threshold
      for(indexOfIndex <- 0 until (nSamples - 1)){
        val index = sortedIndices(indexOfIndex)// nested indices lol
        val label = labels(index)
        val sampleWeight = weights(index)

        val eps = 1e-6
        val nextIndex = sortedIndices(indexOfIndex + 1)
        val curValue  = dataset(index, featInd)
        val nextValue = dataset(nextIndex, featInd)
        //assert(curValue >= 0.0)
        //assert(nextValue >= 0.0)
        val isValuesSame = (nextValue - curValue).abs < eps

        leftFreqs(label)  :-= sampleWeight
        rightFreqs(label) :+= sampleWeight
        leftWeights  -= sampleWeight
        rightWeights += sampleWeight
        val currentError = (leftWeights  * majorClassifierError(leftFreqs) + 
                            rightWeights * majorClassifierError(rightFreqs)) / (leftWeights + rightWeights)
        if(currentError < bestError && !isValuesSame){
          bestError = currentError
          bestThreshold = 0.5 * (curValue + nextValue)
        }
      }
      (bestError, bestThreshold)
    }

    def bestFeatureForRealValuedDataset(
      dataset: DenseMatrix[Double], 
      weights: DenseVector[Double], 
      labels:  DenseVector[Int],
      sampleIndices: Seq[Int],
      nClasses: Int) = {
      val nFeatures = dataset.cols

      var bestError = 1e20
      var bestFeatInd = -1
      var bestThreshold = 0.0

      for (featInd <- 0 until nFeatures){
        val (curErr, curThreshold) = bestSplitForRealValuedDataset(
          dataset,
          weights,
          labels,
          sampleIndices,
          nClasses, 
          featInd)

        if(curErr < bestError){
          bestError = curErr
          bestThreshold = curThreshold
          bestFeatInd = featInd
        }
      }
      (bestError, bestFeatInd, bestThreshold)
    }
  }
}
