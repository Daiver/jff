
import breeze.linalg._

package ml {
  abstract class Classifier {
    def predict(sample: DenseVector[Double]): Int
    def predict(samples: DenseMatrix[Double]): DenseVector[Int]
  }
}
