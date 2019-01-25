import scala.io._
import breeze.linalg._
//import breeze.plot._
import org.sameersingh.scalaplot.Implicits._
import org.sameersingh.scalaplot._
import math._
//import org.sameersingh.scalaplot.GnuplotPlotter
//import org.sameersingh.scalaplot.Implicits.XYPlotStyle


object MyProject{

  def vectorToArray[T](arr : DenseVector[T]): Seq[T] = 
    (0 until arr.length).map(arr(_))

  def linearLeastSquaresMethod(a:DenseMatrix[Double], b:DenseVector[Double]) = 
    breeze.linalg.pinv(a.t * a) * a.t * b

  def gradientDescent(
    gradFunc: DenseVector[Double] => DenseVector[Double],
    initialWeights: DenseVector[Double],
    nIter: Int,
    learningRate: Double) = {
      var weights = initialWeights
    for(iter <- (0 to nIter)) {
      //println(iter, weights)
      val delta = gradFunc(weights)
      //println("delta", delta)
      weights -= learningRate * delta
    }
    weights
  }

  def gradForLinearLeastSquares(
    dataset: DenseMatrix[Double],
    recalls: DenseVector[Double],
    weights: DenseVector[Double]) = {
    val nVars = weights.length
    val nData = dataset.rows
    val act  = dataset * weights - recalls
    dataset.t * act
    //var grad = DenseVector.zeros[Double](weights.length)
    //val g = act * dataset 
/*    for(k <- (0 until nVars)){*/
//[>      for(i <- (0 until nData)){<]
        ////var x: Double  = act(i)
        ////grad(k) += x * dataset(i, k)
      //[>}<]
      ////grad(k) = act.t * dataset(::, k)
    /*}*/
    //grad  
  }

  def gradForThirdLeastSquares(
    dataset: DenseMatrix[Double],
    recalls: DenseVector[Double],
    weights: DenseVector[Double]) = {
    val nVars = weights.length
    val nData = dataset.rows
    var grad = DenseVector.zeros[Double](weights.length)
    for(k <- (0 until nVars)){
      for(i <- (0 until nData)){
        var x: Double  = 0
        for(j <- (0 until nVars)){
          x += weights(j) * Math.pow(dataset(i, 0), j)
        }
        grad(k) += (x - recalls(i)) * Math.pow(dataset(i, 0), k)
      }
    }
    grad  
  }
 

  def thirdFromData(coeff : DenseVector[Double], data: DenseVector[Double]) = 
    (0 until data.length).map(
      (x: Int) => {
      coeff(0) * Math.pow(data(x), 0) + 
      coeff(1) * Math.pow(data(x), 1) + 
      coeff(2) * Math.pow(data(x), 2) + 
      coeff(3) * Math.pow(data(x), 3) 
      }
    )

  def lineFromData(coeff : DenseVector[Double], data: DenseVector[Double]) = 
    (0 until data.length).map(coeff(0) * data(_) + coeff(1))

  def main(args:Array[String]){
    val x  = DenseVector(1.0, 2.0, 3.0, 4.0, 5, 6, 7, 8, 9, 10)
    val x1 = vectorToArray(x)
    val x2 = DenseMatrix((1.0, 1.0),
                         (2.0, 1.0),
                         (3.0, 1.0),
                         (4.0, 1.0),
                         (5.0, 1.0),
                         (6.0, 1.0),
                         (7.0, 1.0),
                         (8.0, 1.0),
                         (9.0, 1.0),
                         (10.0, 1.0)
                         )
    val y  = DenseVector(1.2, 1.6, 5.2, 6.4, 8.5, 9.0, 8.5, 6.0, 5.1, 1.0)
    val y1 = vectorToArray(y)
    val solution1 = linearLeastSquaresMethod(x2, y)
    //val solution = DenseVector[Double](2.34, -1.1)
    println("real solution", solution1)
    val solution = gradientDescent(
        //gradForThirdLeastSquares(x2, y, _),
        gradForLinearLeastSquares(x2, y, _),
        //-0.1 * DenseVector.ones[Double](4),
        DenseVector.zeros[Double](2),
        5000,
        0.00095
      )
    println("solution", solution)
    val y2 = lineFromData(solution, x)
    //val y2 = thirdFromData(solution, x)
    println("y1", y1)
    println("y2", y2)

    //val rnd = new scala.util.Random(42)
    //val xychart = xyChart(x1 ->Seq(
                            //Y(x1.map(_ + rnd.nextDouble - 0.5), style = XYPlotStyle.Dots),
                            //Y(x1, style=XYPlotStyle.Lines)
                          /*))*/

    //output(PNG("./", "tmp"), xychart)
    //output(GUI, xychart)
   //val x1 = 0.0 until 0.4 by 0.1
   //println(x1)
   val rnd = new scala.util.Random(0)

   val data = xyChart(
     x1 -> Seq(
              Y(y1, style=XYPlotStyle.Points),
              Y(y2)))

/*   val data = xyChart(*/
     //x1 -> Seq(
              //Y(x2),
              /*Y(x2.map(_ + rnd.nextDouble - 0.5), style = XYPlotStyle.Points)))*/
   //output(GUI, data)
   output(PNG("./", "scatter"), data)
/*   val plotter = new JFGraphPlotter(xyChart(data))*/
   /*plotter.writeToPdf("./", "name")*/

  }

  def main1(args:Array[String]){
      //val x = 0.0 until 5.0 * math.Pi by 0.1
      var b = DenseVector.ones[Double](9)
      b(3) = 5
      val x = (1 until 10).map(_.toDouble)
      val y1 = (1 until 10).map(j => math.pow(j, 1))
      val y2 = (1 until 100).map(j => math.pow(j, 2))
      val y3 = (1 until 100).map(j => math.pow(j, 3))
      val b1 = vectorToArray(b)
      //println(b.toArray[Double]())
/*      val tmp : Seq[Double] = (b).to[collection.immutable.Seq]*/
      //println(x.length)
      val xychart = xyChart(x ->(b1))
      //val xychart = xyChart(x ->(math.sin(_), math.cos(_)))
      output(GUI, xychart)
      //output(PNG("./", "test"), xyChart(x ->(math.sin(_), math.cos(_))))
/*    val f = Figure()*/
    //val p = f.subplot(0)
    //val x = linspace(0.0,1.0)
    //p += plot(x, x :^ 2.0)
    //p += plot(x, x :^ 3.0, '.')
    //p.xlabel = "x axis"
    //p.ylabel = "y axis"
    /*f.saveas("lines.png")*/
    //breeze.linalg.
/*    println("Hi")*/
    //val a = DenseMatrix((2.0, 2.0), (3.0, 4.0))
    //val b = DenseVector.ones[Double](2)
    /*println(linearLeastSquaresMethod(a, b))*/
    //val m2 = DenseMatrix.ones[Double](5, 5)
    //val x = {1, 2, 3}
    //println(m1 * m2)

  }
}
