import scala.io._
import scala.util._
import java.io._
import breeze.linalg._
import org.sameersingh.scalaplot.Implicits._
import org.sameersingh.scalaplot._
import org.sameersingh.scalaplot.Style.Color
import math._

object MyProject{

  def vectorToArray[T](arr : DenseVector[T]): Seq[T] = 
    (0 until arr.length).map(arr(_))

  def normalizeDataset(dataset: DenseMatrix[Double]) = {
    //val tmp = sum(dataset(::, *))
    //println(tmp)
    var dataset2 = dataset
    dataset2 :/= 255.0
    //dataset2(*, ::) :/= tmp(0, ::).t
    dataset2
  }

  def addBiasToDataset(x: DenseMatrix[Double]) = 
    DenseMatrix.horzcat(x, DenseMatrix.ones[Double](x.rows, 1))

  def simpleBatching(
    weightsOutterInit: DenseMatrix[Double],
    weightsInnerInit: DenseMatrix[Double],
    dataset: DenseMatrix[Double],
    recalls: DenseMatrix[Double],
    nIter: Int,
    learningRate: Double,
    batchSize: Int,
    nBatches: Int
  ) = {
    var weightsOutter = weightsOutterInit
    var weightsInner = weightsInnerInit
    //var dataset = datasetInit

    var rand = new Random()

    for (iter <- 0 until nBatches){
      var batchData   = DenseMatrix.zeros[Double](batchSize, dataset.cols)
      var batchValues = DenseMatrix.zeros[Double](batchSize, recalls.cols)
      for(i <- 0 until batchSize){
        val index = rand.nextInt(dataset.rows)
        batchData(i, ::) := dataset(index, ::)
        batchValues(i, ::) := recalls(index, ::)
      }
      val (weightsOutter2, weightsInner2) = optimizationForTwoLayer(
        weightsOutter, weightsInner, 
        batchData, batchValues, 
        nIter, learningRate)
      weightsOutter = weightsOutter2
      weightsInner  = weightsInner2

        val ans = activate(weightsInner, weightsOutter, dataset)
        val diff = ans - recalls
        var err = 0.0
        for(i <- 0 until diff.rows)
          err += norm(diff(i, ::).t)
        println(("OutterIter", iter, ":", err))
    }
    (weightsOutter, weightsInner)
  }

  def optimizationForTwoLayer(
    weightsOutterInit: DenseMatrix[Double],
    weightsInnerInit: DenseMatrix[Double],
    datasetInit: DenseMatrix[Double],
    recalls: DenseMatrix[Double],
    nIter: Int,
    learningRate: Double) = {
    var weightsOutter = weightsOutterInit
    var weightsInner = weightsInnerInit
    val dataset = addBiasToDataset(datasetInit)
    for(iter <- (0 to nIter)) {
      if(iter % 10 == 0){
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

  def sigmoid(x: Double): Double = 1.0/(1.0 + Math.exp(-x))
  def gradSigmoid(x: Double): Double = sigmoid(x) * (1.0 - sigmoid(x))
/*  def sigmoid(x: Double): Double = tanh(x)*/
  //def gradSigmoid(x: Double): Double = {
    //val denom = (cosh(2*x) + 1.0)
    //4 * cosh(x) * cosh(x) / (denom * denom)
  /*}*/

  def deltaForOutLayerNeuron(
    dataset: DenseMatrix[Double],
    recalls: DenseMatrix[Double],
    weights: DenseMatrix[Double]) = {
    val net   = dataset * weights.t
    val act   = net map sigmoid
    val diff  = act - recalls
    val grads = net map gradSigmoid
    diff :* grads
  }

  def deltaForInnerLayer(
    deltaFromPreviousLayer   : DenseMatrix[Double],
    weightsFromPreviousLayer : DenseMatrix[Double],
    dataset                  : DenseMatrix[Double],
    weights                  : DenseMatrix[Double]
    ) = {
    val w2 = weightsFromPreviousLayer(::, 0 to weightsFromPreviousLayer.cols - 2)
    val tmp = (deltaFromPreviousLayer * w2)
    //val tmp2 = tmp(::, 0 until weightsFromPreviousLayer.cols - 1)
    val net   = dataset * weights.t
    val grads = net map gradSigmoid
    tmp :* grads
  }

  def activateLayer(
    weights: DenseMatrix[Double],
    dataset: DenseMatrix[Double]) = (dataset * weights.t) map sigmoid

  def activateLayer(
    weights: DenseMatrix[Double],
    sample : DenseVector[Double]) = (sample.t * weights.t).t map sigmoid

  def activate(
    weightsInner:DenseMatrix[Double],
    weightsOutter:DenseMatrix[Double],
    sample : DenseVector[Double]
  ) = {
    val in = activateLayer(weightsInner, DenseVector.vertcat(sample, DenseVector(1.0)))
    val inB = DenseVector.vertcat(in, DenseVector(1.0))
    activateLayer(weightsOutter, inB)
  }

  def activate(
    weightsInner:DenseMatrix[Double],
    weightsOutter:DenseMatrix[Double],
    sample : DenseMatrix[Double]
  ) = {
    val in = activateLayer(weightsInner, addBiasToDataset(sample))
    val inB = addBiasToDataset(in)
    activateLayer(weightsOutter, inB)
  }

  def readData(fName: String) = {
    var lines    = scala.io.Source.fromFile(fName).getLines
    val nVars    = 13
    val nClasses = 3
    val nSamples = lines.length
    var dataset  = DenseMatrix.zeros[Double](nSamples, nVars)
    var values   = DenseMatrix.zeros[Double](nSamples, nClasses)
    lines = scala.io.Source.fromFile(fName).getLines

    values :-= 1.0

    var i = 0
    for (line <- lines){
      var tokens = line split " " map {_.toDouble}
      for(j <- 0 until nVars)
        dataset(i, j) = tokens(j + 1)
      values(i, tokens(0).toInt - 1) = 1.0
      i += 1
    }
    (dataset, values)
  }

  def readShroomsData(fname: String) = {
    var lines = scala.io.Source.fromFile(fname).getLines
    val nSamples = lines.length - 1
    val nVars = 112
    val nClasses = 1
    lines = scala.io.Source.fromFile(fname).getLines
    
    var dataset = DenseMatrix.zeros[Double](nSamples, nVars)
    var values  = DenseMatrix.zeros[Double](nSamples, nClasses)

    //values :-= 1.0

    //println(s"nSamples $nSamples $(dataset.cols) $(dataset.rows)")

    var i = 0
    for (line <- lines){
      var tokens = line split " " 
      val feats  = tokens.tail.map(_.split(":").head.toInt)
      if(feats.length != 0){
        for(j <- feats)
          dataset(i, j - 1) = 1.0
        values(i, 0) = tokens(0).toDouble - 1.0
        //values(i, tokens(0).toInt - 1) = 1.0
        i += 1
      }
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

    var dataset = DenseMatrix.zeros[Double](countOfImages, countOfCols * countOfRows)
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

    var values = DenseMatrix.zeros[Double](countOfLabels, countOfClasses)
    for(sampleInd <- (0 until countOfLabels))
      values(sampleInd, in.readByte) = 1.0
    
    values
  }

  def numericalGradient(
    dataset: DenseMatrix[Double],
    recalls: DenseMatrix[Double],
    weightsOutterInit: DenseMatrix[Double],
    weightsInnerInit: DenseMatrix[Double]) {
    var weightsOutter = weightsOutterInit
    var weightsInner = weightsInnerInit

    val eps = 0.001

    val diff1 = (activate(weightsInner, weightsOutter, dataset) - recalls) 
    val val1  = sum(diff1 :* diff1)
    
    var gradOutter = DenseMatrix.zeros[Double](weightsOutter.rows, weightsOutter.cols)
    var gradInner = DenseMatrix.zeros[Double](weightsInner.rows, weightsInner.cols)
    for(i <- 0 until gradInner.rows){
      for(j <- 0 until gradInner.cols){
        weightsInner(i, j) += eps
        val diff2 = (activate(weightsInner, weightsOutter, dataset) - recalls) 
        val val2  = sum(diff2 :* diff2)
        weightsInner(i, j) -= 2*eps
        val diff3 = (activate(weightsInner, weightsOutter, dataset) - recalls) 
        val val3  = sum(diff3 :* diff3)
        gradInner(i, j) = (val2 - val3)/(2*eps)
        weightsInner(i, j) += eps
      }
    }

    val datasetB = addBiasToDataset(dataset)
    val innerAct  = activateLayer(weightsInner, datasetB) //NxR R - inner layer size     
    val innerActB = addBiasToDataset(innerAct)
    val deltaOutter = deltaForOutLayerNeuron(innerActB, recalls, weightsOutter)
    val gradOutter2 = deltaOutter.t * innerActB * 2.0
    val deltaInner = deltaForInnerLayer(deltaOutter, weightsOutter, datasetB, weightsInner)
    val gradInner2 = deltaInner.t * datasetB * 2.0
    println("test")
    println(sum((gradInner - gradInner2):*(gradInner - gradInner2)))
    println(gradInner)
    println(gradInner2)
/*    println(sum((gradOutter - gradOutter2):*(gradOutter - gradOutter2)))*/
    //println(gradOutter)
    /*println(gradOutter2)*/
  }

  def main5(args:Array[String]){
    //var (data, values) = readData("./wine_train")
    var (data0, values0) = readShroomsData("/home/daiver/Downloads/shrooms.txt")
    val data = data0(0 to 5000, ::)
    val values = values0(0 to 5000, ::)
    //println(data)
    //val tmp = sum(data(::, *))
    //data(::, *) :/= tmp
    //for(i <- 0 until data.rows)
      //data(i, ::) /= tmp(0, ::)

/*      numericalGradient(*/
        //data,
        //values,
        //DenseMatrix.rand(2, 11) *0.6,
        //DenseMatrix.rand(10, 113)*0.3
        ////DenseMatrix.zeros[Double](2, 61),
        ////DenseMatrix.zeros[Double](60, 113)
      //)

      //return
/*    val (solutionO, solutionI) = simpleBatching(*/
        //DenseMatrix.rand(2, 161) *0.5,
        //DenseMatrix.rand(160, 113)*0.5,
        //data, values,
        //100,
        //0.5,
        //5, 
        //1000
      /*)*/
    val (solutionO, solutionI) = optimizationForTwoLayer(
        DenseMatrix.rand(1, 41) *0.12 - 0.06,
        DenseMatrix.rand(40, 113)*0.11 - 0.06,
        //DenseMatrix.rand(3, 10)*0.5,
        //DenseMatrix.rand(9, 14)*0.5,
        //DenseMatrix.rand(19, 14)*0.5,
        //DenseMatrix.ones[Double](3, 6),
        //DenseMatrix.ones[Double](5, 14),
        data,
        values,
        10000,
        //3000000,
        0.0003f
      )
    println("Inner")
    println(solutionI)
    println("outter")
    println(solutionO)

    //var (data2, values2) = readData("./wine_test")
    //var (data2, values2) = readShroomsData("/home/daiver/Downloads/shrooms.txt")
    //val tmp2 = sum(data2(::, *))
    val data2 = data0(5000 until data0.rows, ::)
    val values2 = values0(5000 until data0.rows, ::)
    //data(::, *) :/= tmp

    //for(i <- 0 until data2.rows)
      //data2(i, ::) /= tmp2(0, ::)

    var nErr = 0
    for(i <- (0 until data2.rows)){
      //println(data(i, ::).t)
      val ans = activate(solutionI, solutionO, data2(i, ::).t)
      val err = norm(ans - values2(i, ::).t)
      if(err > 0.5)
        nErr += 1
      if(i % 100 == 0)
        println("err", err, ans, values2(i, ::).t)
    }
    println("nErr", nErr)
  }

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
        ////DenseMatrix.ones[Double](2, 4)*0.1,
        //DenseMatrix.rand(3, 3),
//[>        DenseMatrix(<]
          ////(0.1, 0.2, 0.7),
          ////(0.3, 0.05, 0.4),
          ////(0.0, 0.15, 0.8)
          //[>),<]
        ////DenseMatrix.ones[Double](3, 3)*0.1,
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
  def main4(args:Array[String]){
    val datasetRaw = readMNISTImages("/home/daiver/Downloads/train-images-idx3-ubyte")
    val dataset    = normalizeDataset(datasetRaw)
    val values     = readMNISTLabels("/home/daiver/Downloads/train-labels-idx1-ubyte")
    //println(dataset)

    val (solutionO, solutionI) = optimizationForTwoLayer(
        convert(DenseMatrix.rand(10, 301), Double) * 0.2,
        convert(DenseMatrix.rand(300, 28 * 28 + 1), Double) * 0.2,
        dataset,
        values,
        2000,
        0.6
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

  def main(args:Array[String]){
    println("Start")
    val data = DenseMatrix(
      (0.0, 0.0),
      (5.0, 0.2),
      (1.0, 1.0),
      (1.0, 5.0),
      (2.0, 2.0),
      (3.0, 3.0),
      (4.0, 4.0),
      (5.0, 5.0),
      (4.0, 6.0),
      (3.0, 7.0),
      (2.0, 9.0),
      (6.0, 6.0),
      (7.0, 7.0),
      (7.0, 8.0),
      (8.0, 8.0),
      (8.0, 7.0),
      (8.0, 6.0),
      (9.0, 9.0),
      (9.0, 8.0),
      (9.0, 7.0),
      (5.0, 3.0),
      (6.0, 5.8),
      (6.0, 6.0),
      (5.0, 9.0),
      (2.5, 5.0),
      (0.5, 5.0)
      )
    val values = DenseMatrix(
      (0.0),
      (0.0),
      (0.0),
      (0.0),
      (0.0),
      (0.0),
      (0.0),
      (0.0),
      (0.0),
      (0.0),
      (0.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0),
      (1.0)
      )
    val(solutionO, solutionI) = optimizationForTwoLayer(
      DenseMatrix.rand(1, 7)*0.12,
      DenseMatrix.rand(6, 3)*0.12,
      data, values,
      555500,
      0.08
      )

    var nErr = 0
    for(i <- (0 until data.rows)){
      //println(data(i, ::).t)
      val ans = activate(solutionI, solutionO, data(i, ::).t)
      val err = norm(ans - values(i, ::).t)
      if(err > 0.5)
        nErr += 1
      println("err", err, ans, values(i, ::).t)
    }
    println("nErr", nErr)

    var zerosPoints = Seq[(Double, Double)]()
    var onesPoints = Seq[(Double, Double)]()
    for(i <- 0 until data.rows){
      if(Math.abs(values(i, 0)) < 0.0001){
        zerosPoints = zerosPoints :+ (data(i, 0), data(i, 1))
      }else{
        onesPoints = onesPoints :+ (data(i, 0), data(i, 1))
      }
    }
    //import org.sameersingh.scalaplot._

    var zerosDots = Seq[(Double, Double)]()
    var onesDots = Seq[(Double, Double)]()

    for(i <- 0.0 to 10.0 by 0.1){
      for(j <- 0.0 to 10.0 by 0.1){
        val sample = DenseVector(i, j)
        val ans = activate(solutionI, solutionO, sample)
        if(ans(0) < 0.5){
          zerosDots = zerosDots :+ (i, j)
        }else{
          onesDots = onesDots :+ (i, j)
        }
      }
    }

    println("L", (onesDots.length))
    println("L", (zerosDots.length))
    println(zerosDots(5))

    val chart = xyChart(List(
      XY(zerosPoints, style=XYPlotStyle.Points, color=Color.Red),
      XY(onesPoints, style=XYPlotStyle.Points, color=Color.Green),
      XY(zerosDots, style=XYPlotStyle.Dots, color=Color.Red)
      //XY(zerosDots, style=XYPlotStyle.Dots, color=Color.Green),
      //XY(onesDots, style=XYPlotStyle.Dots, color=Color.Green)
    ))
    output( PNG("./", "tmp.png"), chart)
  }
}
