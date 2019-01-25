import org.scalatest._

import breeze.linalg._

import ml.DecisionSplit._

class DecisionSplitSpec extends FlatSpec with Matchers {
   val eps = 1e-3

  "Split" should "splits dataset 01" in {
    val labels   = DenseVector(Array(0, 1, 0))
    val dataset  = DenseMatrix(Array(
                              1.0, 
                              2.0, 
                              0.0)).t.reshape(3, 1)
    val weights  = DenseVector.ones[Double](labels.length)
    val indices  = (0 until labels.length).toSeq
    val featInd  = 0
    val nClasses = labels.max + 1
    val (error, threshold) = bestSplitForRealValuedDataset(dataset, weights, labels, indices, nClasses, featInd)
    threshold should be (1.5 +- eps)
    error should be (0.0 +- eps)
  }

  it should "splits dataset 02" in {
    val labels   = DenseVector(Array(2, 1, 0))
    val dataset  = DenseMatrix(Array(
                              0.0,  
                              0.0,  
                              0.0)).t.reshape(3, 1)
    val weights  = DenseVector.ones[Double](labels.length)
    val indices  = (0 until labels.length).toSeq
    val featInd  = 0
    val nClasses = labels.max + 1
    val (error, threshold) = bestSplitForRealValuedDataset(dataset, weights, labels, indices, nClasses, featInd)
    threshold should be (0.0 +- eps)
  }

  it should "splits dataset 03" in {
    val labels   = DenseVector(Array(2, 1, 0))
    val dataset  = DenseMatrix(Array(
                              0.0, 3.0, 
                              0.0, 2.0, 
                              0.0, 0.0)).t.reshape(3, 2)
    val weights  = DenseVector.ones[Double](labels.length)
    val indices  = (0 until labels.length).toSeq
    val featInd  = 1
    val nClasses = labels.max + 1
    val (error, threshold) = bestSplitForRealValuedDataset(dataset, weights, labels, indices, nClasses, featInd)
    threshold should be (1.0 +- eps)
    error should be (1.0/3.0 +- eps)
  }

  "FindBestFeat" should "find best feature index" in {
    val labels   = DenseVector(Array(2, 1, 0))
    val dataset  = DenseMatrix(Array(
                              0.0, 3.0, 
                              0.0, 2.0, 
                              0.0, 0.0)).t.reshape(3, 2)
    val weights  = DenseVector.ones[Double](labels.length)
    val indices  = (0 until labels.length).toSeq
    val nClasses = labels.max + 1
    val (error, featInd, threshold) = bestFeatureForRealValuedDataset(dataset, weights, labels, indices, nClasses)

    error should be (1.0/3.0 +- eps)
    featInd should be (1)
    threshold should be (1.0 +- eps)
  }
}
