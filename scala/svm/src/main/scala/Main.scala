
import scala.util.Random
import breeze.linalg._
import breeze.stats.distributions._
import ml._
import optimization._
import utils._

import InputData._

import breeze.plot._;

//Should be rewrited!
case class PairwiseClassifier(ind1: Int, ind2: Int, svm: SVM){
  def predict(sample: DenseVector[Double]): Int = {
    val svmResp = svm.predict(sample)
    if(svmResp == -1)
      ind1
    else
      ind2
  }
}

case class OneVsAllClassifier(classifiers: List[PairwiseClassifier], nClasses: Int) {

  def predictTable(sample: DenseVector[Double]) = {
    var table = DenseVector.zeros[Int](nClasses)
    for(clf <- classifiers){
      val res = clf.predict(sample)
      if(res != -1)
        table(res) += 1
    }
    table
  }

  def predict(sample: DenseVector[Double]): Int = predictTable(sample).argmax
  

  def predict(samples: DenseMatrix[Double]): DenseVector[Int] = {
    var res = DenseVector.zeros[Int](samples.rows)
    //Rewrite it
    for(i <- 0 until samples.rows){
      res(i) = predict(samples(i, ::).t)
    }
    res
  }

  def evaluate(samples: DenseMatrix[Double], labels: DenseVector[Int]): Int = 
    (predict(samples) :== labels) map {x => if(x) 0 else 1} sum
}

object App {

  def trainPairwiseClassifier(data: DenseMatrix[Double], labels: DenseVector[Int], ind1: Int, ind2: Int) = {
    val wInit = DenseVector.ones[Double](data.cols + 1) :/ data.cols.toDouble

    val data1   = subsetByLabel(data, labels, ind1)
    val data2   = subsetByLabel(data, labels, ind2)
    val labels1 = DenseVector.ones[Int](data1.rows) * -1
    val labels2 = DenseVector.ones[Int](data2.rows) 

    val dataTrain   = DenseMatrix.vertcat(data1, data2)
    val labelsTrain = DenseVector.vertcat(labels1, labels2)

    val batchSize = 550
    val momentCoeff = 1.9
    val weights = time { SGD(svmError, svmGradient(0.01, _, _, _), dataTrain, labelsTrain, wInit, 10, batchSize, 0.0001, momentCoeff) }
    val svm = SVM(weights(0 to weights.length - 2), weights(weights.length - 1))
    PairwiseClassifier(ind1, ind2, svm)
  }

  //More classifiers for the Classifier God!
  def trainOneVsAllClassifier(data: DenseMatrix[Double], labels: DenseVector[Int], nClasses: Int) = {
    var classifiers: List[PairwiseClassifier] = List()
    for(ind1 <- 0 until nClasses){
      for(ind2 <- ind1 until nClasses){
        if(ind1 != ind2){
          println("train " + ind1 + "-" + ind2)
          classifiers = classifiers :+ trainPairwiseClassifier(data, labels, ind1, ind2)
          println(classifiers.length)
        }
      }
    }
    OneVsAllClassifier(classifiers, nClasses)
  }

  def main(args: Array[String]) {
    println("Start!")

    val mnistFolder = "/home/daiver/coding/data/mnist/"
    val trainImages = readMNISTImages (mnistFolder + "train-images-idx3-ubyte")
    val trainLabels = readMNISTLabels (mnistFolder + "train-labels-idx1-ubyte")
    val testImages  = readMNISTImages (mnistFolder + "t10k-images-idx3-ubyte")
    val testLabels  = readMNISTLabels (mnistFolder + "t10k-labels-idx1-ubyte")

    val clf = trainOneVsAllClassifier(trainImages, trainLabels, 10)
    val errTrain = clf.evaluate(trainImages, trainLabels)
    val errTest  = clf.evaluate(testImages, testLabels)
    println("train error " + (100.0 * errTrain.toDouble / trainLabels.length.toDouble) + "%")
    println("test error " + (100.0 * errTest.toDouble / testLabels.length.toDouble) + "%")
    //println(clf.predictTable(testImages(0, ::).t))
    //println(testLabels(0))

    //val symbolToClassify = 1
/*    var sumOfErr = 0.0*/
    //for(symbolToClassify <- 0 to 9){
      //val data   = trainImages
      //val labels = trainLabels map {x => if(x == symbolToClassify) 1 else -1}

      //val nSamples = testImages.rows
      //val dataT = testImages
      //val labelsT = testLabels map {x => if(x == symbolToClassify) 1 else -1}

      //val wInit = DenseVector.ones[Double](data.cols + 1) :/ data.cols.toDouble

      //val batchSize = 512
      //val momentCoeff = 0.9
      //val weights = time { SGD(svmError, svmGradient(0.002, _, _, _), data, labels, wInit, 20, batchSize, 0.001, momentCoeff) }
      //val svm = SVM(weights(0 to weights.length - 2), weights(weights.length - 1))
      //val err = svm.evaluate(dataT, labelsT).toDouble / nSamples.toDouble
      //sumOfErr += err
      ////println(svm)
      //println("Num: " + symbolToClassify + " err: " + err + "\n")
    //}
    //println("Average err " + sumOfErr / 10.0)

    println("Finish!")
  }

}

