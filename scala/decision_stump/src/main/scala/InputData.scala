
import java.io._
import breeze.linalg._


package object InputData {
  def readMNISTImages(fname: String) = {
    //assert ( false ) //Not implemented yet
    var in = new DataInputStream(new BufferedInputStream(new FileInputStream(fname)))
    val magic = in.readInt
    if(magic != 2051){
      println(s"Magic not equal to 2051! $magic")
      System.exit(1)
    }
    val nImages = in.readInt
    val nRows   = in.readInt
    val nCols   = in.readInt
    //println(s"Num of images $countOfImages rows $countOfRows cols $countOfCols")

    var dataset = DenseMatrix.zeros[Double](nImages, nCols * nRows)
    for(sampleInd <- (0 until nImages))
      for(featInd <- (0 until (nCols * nRows)))
        dataset(sampleInd, featInd) = in.readUnsignedByte

    dataset
  }

  def readMNISTLabels(fname: String) = {
    var in = new DataInputStream(new BufferedInputStream(new FileInputStream(fname)))
    val magic = in.readInt
    if(magic != 2049){
      println(s"Magic not equal to 2049! $magic")
      System.exit(1)
    }
    val nLabels = in.readInt
    var labels = DenseVector.zeros[Int](nLabels)
    for(labelInd <- (0 until nLabels)){
      labels(labelInd) = in.readUnsignedByte
    }
    labels
  }

}
