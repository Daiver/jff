import scala.io._
import scala.util._
import java.io._
import java.awt.Image
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import javax.imageio._
import breeze.linalg._
import breeze.linalg.svd.SVD
import breeze.linalg.eig.Eig
import breeze.stats.mean
import org.sameersingh.scalaplot.Implicits._
import org.sameersingh.scalaplot._
import org.sameersingh.scalaplot.Style.Color
import math._

import ml.pca.PCA
import imageprocessing._

object MyProject {

  def vectorToArray[T](arr : DenseVector[T]): Seq[T] = 
    (0 until arr.length).map(arr(_))

  def loadATTFaceData(dirName: String) = {
    var dirNames = List[String]()
    for (file <- new File(dirName).listFiles) {
      dirNames = dirNames :+ (file + "/")
    }
    var fileNames = List[String]()
    for(dir <- dirNames){
      for (file <- new File(dir).listFiles) 
        fileNames = fileNames :+ (file.toString)
    }
    val nCols = 10304
    var data = DenseMatrix.zeros[Float](fileNames.length, nCols)
    for(i <- 0 until fileNames.length){
      val fileName = fileNames(i)
      var img = ImageIO.read(new File(fileName))
      data(i, ::).t := imageToVectorGrayScale(img)
    }
    data
  }

  def main(args: Array[String]){
    println("Hi")
    val data = loadATTFaceData("/home/daiver/Downloads/at_t_faces2")
    var pca = new PCA(1)
    pca.fit(data)
    val eigens = pca.eigenValues map ((x:Float) => {Math.sqrt(Math.sqrt(x)).toFloat})
    println(eigens)
/*    for(i <- 0 until 10){*/
      //val a = ei
      //val weight
    /*}*/
    var err = 0.0
    for(i <- 0 until data.rows){
      val sample = data(i, ::).t
      val trans  = pca.transform(sample)
      val res    = pca.transformInverse(trans)
      err += norm(res - sample)
      val img = vectorToBufferedImageGrayScale(res, 92, 112)
      var outputfile = new File("dump/" + i.toString + ".png")
      ImageIO.write(img, "png", outputfile)


      val img2 = vectorToBufferedImageGrayScale(sample, 92, 112)
      var outputfile2 = new File("dump/_" + i.toString + ".png")
      ImageIO.write(img2, "png", outputfile2)
    }
    println(err)

  }

}
