
import breeze.linalg._
import breeze.numerics._

import InputData._
import ml.Utils._
import ml.Classifier
import ml.DecisionStump._
import ml.DecisionSplit._
import InputData._

object App {

  def classifierError(clf: Classifier, testDataset: DenseMatrix[Double], testLabels: DenseVector[Int]) = {
    assert(testLabels.length == testDataset.rows)
    val predicted = clf.predict(testDataset)
    val diff = abs(predicted - testLabels) map {(x) => if(x > 0) 1 else 0}
    val nMissclassified = diff.sum
    nMissclassified.toDouble / testLabels.length.toDouble
  }

  def trainStump(
    dataset: DenseMatrix[Double],
    labels: DenseVector[Int]) = {
    assert(dataset.rows == labels.length)
    val nSamples = dataset.rows
    val weights = DenseVector.ones[Double](nSamples)
    val nClasses = labels.max + 1
    assert(nClasses == 10)
    val indices = 0 until nSamples
    val (err, feat, threshold) = bestFeatureForRealValuedDataset(dataset, weights, labels, indices, nClasses)
    val (rIndices, lIndices) = indicesForSplittedValues(dataset, feat, threshold)
    val lLabels = subsetRows(labels, lIndices)
    val rLabels = subsetRows(labels, rIndices)
    val lDict = freqDictFromLabels(lLabels)
    val rDict = freqDictFromLabels(rLabels)
    println(err, feat, threshold)
    Stump(NodeSplit(feat, threshold, NodeLeaf(lDict), NodeLeaf(rDict)))
  }

  def main(args: Array[String]){
    val mnistFolder = "/home/daiver/coding/data/mnist/"
    val trainImages = readMNISTImages (mnistFolder + "train-images-idx3-ubyte")
    val trainLabels = readMNISTLabels (mnistFolder + "train-labels-idx1-ubyte")
    val testImages  = readMNISTImages (mnistFolder + "t10k-images-idx3-ubyte")
    val testLabels  = readMNISTLabels (mnistFolder + "t10k-labels-idx1-ubyte")

    val clf = trainStump(trainImages, trainLabels)
    println("train error", classifierError(clf, trainImages, trainLabels))
    println("test error", classifierError(clf, testImages, testLabels))
  }
}
