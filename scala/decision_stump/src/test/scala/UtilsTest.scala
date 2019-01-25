//import Tests._

import org.scalatest._
import org.scalatest.prop._

import breeze.linalg._
import breeze.numerics._
import ml.Utils._

class UtilsSpec extends FlatSpec with Matchers {

  val eps = 1e-3
  val labels1 = DenseVector(Array(1, 3, 4, 1, 2, 3, 2, 1))
  val freqDict1 = freqDictFromLabels(labels1)

  "Utils" should "subset" in {
    val mat = DenseMatrix(Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)).t.reshape(3, 2)
    val subsetted = subsetRows(mat, List(0, 2))
    subsetted.rows should be (2)
    subsetted.cols should be (2)
    val ans = DenseMatrix(Array(1.0, 2.0, 5.0, 6.0)).t.reshape(2, 2)
    val diff = abs(subsetted - ans)
    diff.sum should be (0.0 +- eps)
  }

  it should "major classify" in {
    val majorClass = majorClassFromFreqDict(freqDict1)
    majorClass should be (1)
  }

  it should "compute major classification error" in {
    val error = majorClassifierError(freqDict1)
    error should be (5.0/8.0 +- eps)
  }
  
  it should "sort indices by dataset and feature" in {
    val mat = DenseMatrix(Array(5.0, 2.0, 4.0, 1.0, 3.0, 6.0)).t.reshape(3, 2)
    val res = sortedIndicesByDataset(mat, 1, List(0, 1, 2))
    val ans = List(1, 0, 2)
    res should be (ans)
  }

  it should "compute weighted freq dict" in {
    val labels   = DenseVector(Array(1, 2, 3, 1))
    val weights  = DenseVector(Array(0.3, 10.0, 0.2, 0.5))
    val indices  = List(0, 2, 3)
    val nClasses = 4
    val freqDict = freqDictFromLabelsWeighted(labels, weights, indices, nClasses)
    val ans      = DenseVector(Array(0.0, 0.8, 0.0, 0.2))
    val diff     = abs(freqDict - ans)
    diff.sum should be (0.0 +- eps)
  }

}


