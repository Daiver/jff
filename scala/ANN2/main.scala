import scala.io._
import java.io._
import breeze.linalg._
import org.sameersingh.scalaplot.Implicits._
import org.sameersingh.scalaplot._
import math._

object MyProject{

  def vectorToArray[T](arr : DenseVector[T]): Seq[T] = 
    (0 until arr.length).map(arr(_))

  def normalizeDataset(dataset: DenseMatrix[Float]) = {
    //val tmp = sum(dataset(::, *))
    //println(tmp)
    var dataset2 = dataset
    dataset2 :/= 255.0f
    //dataset2(*, ::) :/= tmp(0, ::).t
    dataset2
  }

  def addBiasToDataset(x: DenseMatrix[Float]) = 
    DenseMatrix.horzcat(x, DenseMatrix.ones[Float](x.rows, 1))

  def optimizationForTwoLayer(
    weightsOutterInit: DenseMatrix[Float],
    weightsInnerInit: DenseMatrix[Float],
    datasetInit: DenseMatrix[Float],
    recalls: DenseMatrix[Float],
    nIter: Int,
    learningRate: Float) = {
    var weightsOutter = weightsOutterInit
    var weightsInner = weightsInnerInit
    val dataset = addBiasToDataset(datasetInit)
    for(iter <- (0 to nIter)) {
      if(iter % 1 == 0){
        val ans = activate(weightsInner, weightsOutter, datasetInit)
        val diff = ans - recalls
        var err = 0.0
        for(i <- 0 until diff.rows)
          err += norm(diff(i, ::).t)

        println("Iter", iter, ":", err)
      }
      val innerAct  = activateLayer(weightsInner, dataset) //NxR R - inner layer size     
      val innerActB = addBiasToDataset(innerAct)
      //println("i", innerAct)
      val deltaOutter = deltaForOutLayerNeuron(innerActB, recalls, weightsOutter)
      val deltaInner = deltaForInnerLayer(deltaOutter, weightsOutter, dataset, weightsInner)
      weightsOutter -= learningRate * (deltaOutter.t) * innerActB
      weightsInner -= learningRate * (deltaInner.t) * dataset
    }
    (weightsOutter, weightsInner)
  }

  def sigmoid(x: Float): Float = 1.0f/(1.0f + Math.exp(-x)).toFloat
  def gradSigmoid(x: Float): Float = sigmoid(x) * (1.0f - sigmoid(x))

  def deltaForOutLayerNeuron(
    dataset: DenseMatrix[Float],
    recalls: DenseMatrix[Float],
    weights: DenseMatrix[Float]) = {
    val net   = dataset * weights.t
    val act   = net map sigmoid
    val diff  = act - recalls
    val grads = net map gradSigmoid
    diff :* grads
  }

  def deltaForInnerLayer(
    deltaFromPreviousLayer   : DenseMatrix[Float],
    weightsFromPreviousLayer : DenseMatrix[Float],
    dataset                  : DenseMatrix[Float],
    weights                  : DenseMatrix[Float]
    ) = {
    val w2 = weightsFromPreviousLayer(::, 0 to weightsFromPreviousLayer.cols - 2)
    val tmp = (deltaFromPreviousLayer * w2)
    //val tmp2 = tmp(::, 0 until weightsFromPreviousLayer.cols - 1)
    val net   = dataset * weights.t
    val grads = net map gradSigmoid
    tmp :* grads
  }

  def activateLayer(
    weights: DenseMatrix[Float],
    dataset: DenseMatrix[Float]) = (dataset * weights.t) map sigmoid

  def activateLayer(
    weights: DenseMatrix[Float],
    sample : DenseVector[Float]) = (sample.t * weights.t).t map sigmoid

  def activate(
    weightsInner:DenseMatrix[Float],
    weightsOutter:DenseMatrix[Float],
    sample : DenseVector[Float]
  ) = {
    val in = activateLayer(weightsInner, DenseVector.vertcat(sample, DenseVector(1.0f)))
    val inB = DenseVector.vertcat(in, DenseVector(1.0f))
    activateLayer(weightsOutter, inB)
  }

  def activate(
    weightsInner:DenseMatrix[Float],
    weightsOutter:DenseMatrix[Float],
    sample : DenseMatrix[Float]
  ) = {
    val in = activateLayer(weightsInner, addBiasToDataset(sample))
    val inB = addBiasToDataset(in)
    activateLayer(weightsOutter, inB)
  }

  def readData(fName: String) = {
    var lines = scala.io.Source.fromFile(fName).getLines
    val nVars = 13
    val nClasses = 3
    val nSamples = lines.length
    var dataset = DenseMatrix.zeros[Float](nSamples, nVars)
    var values  = DenseMatrix.zeros[Float](nSamples, nClasses)
    lines = scala.io.Source.fromFile(fName).getLines

    var i = 0
    for (line <- lines){
      var tokens = line split " " map {_.toFloat}
      for(j <- 0 until nVars)
        dataset(i, j) = tokens(j + 1)
      values(i, tokens(0).toInt - 1) = 1.0f
      i += 1
    }
    (dataset, values)
  }

  def readMNISTImages(fname: String) = {
    var in = new DataInputStream(new BufferedInputStream(new FileInputStream(fname)))
    val magic = in.readInt
    if(magic != 2051){
      println(s"Magic not equal to 2051! $magic")
      System.exit(1)
    }
    val countOfImages = in.readInt
    val countOfRows   = in.readInt
    val countOfCols   = in.readInt
    println(s"Num of images $countOfImages rows $countOfRows cols $countOfCols")

    var dataset = DenseMatrix.zeros[Float](countOfImages, countOfCols * countOfRows)
    for(sampleInd <- (0 until countOfImages))
      for(featInd <- (0 until (countOfCols * countOfRows)))
        dataset(sampleInd, featInd) = in.readByte

    dataset
  }

  def readMNISTLabels(fname: String) = {
    var in = new DataInputStream(new BufferedInputStream(new FileInputStream(fname)))
    val magic = in.readInt
    if(magic != 2049){
      println(s"Magic not equal to 2049! $magic")
      System.exit(1)
    }
    val countOfLabels = in.readInt
    println(s"Num of labels $countOfLabels")
    val countOfClasses = 10

    var values = DenseMatrix.zeros[Float](countOfLabels, countOfClasses)
    for(sampleInd <- (0 until countOfLabels))
      values(sampleInd, in.readByte) = 1.0f
    
    values
  }

  def main(args:Array[String]){
    val datasetRaw = readMNISTImages("/home/daiver/Downloads/train-images-idx3-ubyte")
    val dataset    = normalizeDataset(datasetRaw)
    val values     = readMNISTLabels("/home/daiver/Downloads/train-labels-idx1-ubyte")
    //println(dataset)

    val (solutionO, solutionI) = optimizationForTwoLayer(
        convert(DenseMatrix.rand(10, 301), Float) * 0.2f,
        convert(DenseMatrix.rand(300, 28 * 28 + 1), Float) * 0.2f,
        dataset,
        values,
        2000,
        0.0001f
      )
    var nErr = 0
    for(i <- (0 until dataset.rows)){
      //println(data(i, ::).t)
      val ans = activate(solutionI, solutionO, dataset(i, ::).t)
      val err = norm(ans - values(i, ::).t)
      if(err > 0.5)
        nErr += 1
      //println("err", err, ans, values(i, ::).t)
    }
    println("nErr", nErr)

  }

/*  def main3(args:Array[String]){*/
    //var (data, values) = readData("./wine_train")
    //val tmp = sum(data(::, *))
    ////data(::, *) :/= tmp
    //for(i <- 0 until data.rows)
      //data(i, ::) /= tmp(0, ::)
    //val (solutionO, solutionI) = optimizationForTwoLayer(
        //DenseMatrix.rand(3, 20)*0.5,
        //DenseMatrix.rand(19, 14)*0.5,
        ////DenseMatrix.ones[Float](3, 6),
        ////DenseMatrix.ones[Float](5, 14),
        //data,
        //values,
        //2000000,
        //0.02
      //)
    //println("Inner")
    //println(solutionI)
    //println("outter")
    //println(solutionO)

    //var (data2, values2) = readData("./wine_test")
    //val tmp2 = sum(data2(::, *))
    ////data(::, *) :/= tmp

    //for(i <- 0 until data2.rows)
      //data2(i, ::) /= tmp2(0, ::)

    //var nErr = 0
    //for(i <- (0 until data2.rows)){
      ////println(data(i, ::).t)
      //val ans = activate(solutionI, solutionO, data2(i, ::).t)
      //val err = norm(ans - values2(i, ::).t)
      //if(err > 0.5)
        //nErr += 1
      //println("err", err, ans, values2(i, ::).t)
    //}
    //println("nErr", nErr)
  //}

  //def main2(args:Array[String]){

    //val data = DenseMatrix(
      //(2.0/5, 6.0/9),
      //(5.0/5, 7.0/9),
      //(1.6/5, 8.0/9),
      //(1.1/5, 9.0/9))
    //val values = DenseMatrix(
      //(1.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)
      //)

    //val (solutionO, solutionI) = optimizationForTwoLayer(
        //DenseMatrix.rand(2, 4)*0.1,
        ////DenseMatrix.ones[Float](2, 4)*0.1,
        //DenseMatrix.rand(3, 3),
//[>        DenseMatrix(<]
          ////(0.1, 0.2, 0.7),
          ////(0.3, 0.05, 0.4),
          ////(0.0, 0.15, 0.8)
          //[>),<]
        ////DenseMatrix.ones[Float](3, 3)*0.1,
        //data,
        //values,
        //900,
        //1.1
      //)
    //println("Inner")
    //println(solutionI)
    //println("outter")
    //println(solutionO)
    
    //println(values)
    //for(i <- (0 until data.rows)){
      //println(activate(solutionI, solutionO, data(i, ::).t))
      //println("----")
    //}
  /*}*/

}
