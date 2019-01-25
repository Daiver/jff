import collection.mutable.Stack
import org.scalatest._

import InputData._

class InputDataSpec extends FlatSpec with Matchers {

  val mnistFolder = "/home/daiver/coding/data/mnist/"

  "InputData" should "read train images" in {
    val trainImages = readMNISTImages (mnistFolder + "train-images-idx3-ubyte")
    trainImages.rows should be (60000)
    trainImages.cols should be (28 * 28)
  }

  it should "read test images" in {
    val testImages = readMNISTImages (mnistFolder + "t10k-images-idx3-ubyte")
    testImages.rows should be (10000)
    testImages.cols should be (28 * 28)
  }

  it should "read train labels" in {
    val trainLabels = readMNISTLabels (mnistFolder + "train-labels-idx1-ubyte")
    trainLabels.length should be (60000)
  }

  it should "read test labels" in {
    val testLabels = readMNISTLabels (mnistFolder + "t10k-labels-idx1-ubyte")
    testLabels.length should be (10000)
  }


}
