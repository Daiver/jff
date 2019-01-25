
import breeze.linalg._

package object utils {

  def subsetByLabel(data: DenseMatrix[Double], labels: DenseVector[Int], label: Int): DenseMatrix[Double] = {
    val nRows = data.rows
    val nCols = data.cols
    assert(nRows == labels.length)
    val indices   = (0 until nRows) filter {x => labels(x) == label}
    val nIndices  = indices.length
    var resData   = DenseMatrix.zeros[Double](nIndices, nCols)
    for(i <- 0 until nIndices) {
      val index = indices(i)
      resData(i, ::) := data(index, ::)
    }
    resData
  }

  def time[R](block: => R): R = {  
      val t0 = System.nanoTime()
      val result = block    // call-by-name
      val t1 = System.nanoTime()
      println("Elapsed time: " + (t1 - t0) * 1e-9 + "s")
      result
  }


}
