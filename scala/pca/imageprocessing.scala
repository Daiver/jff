import java.io._
import java.awt.Image
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import javax.imageio._

import breeze.linalg._

package object imageprocessing {
  def vectorToBufferedImageGrayScale(vec: DenseVector[Float], width: Int, height: Int) = {
    var img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    var data = Array.fill(vec.length){0}
    for(i <- 0 until vec.length){
      data(i) = 0x010101 * vec(i).toInt
    }
    img.setRGB(0, 0, width, height, data, 0, width)
    img
  }

  def imageToVectorGrayScale(img: BufferedImage) = {
    val pixels = (img.getRaster().getDataBuffer().asInstanceOf[DataBufferByte]).getData();
    var data = DenseVector.zeros[Float](pixels.length/3)
    for(j <- 0 until pixels.length/3)
      data(j) = pixels(3 * j + 0)
    data
  }

}
