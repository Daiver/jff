import org.scalatest._

import breeze.linalg._

import ml.DecisionStump._

class DecisionStumpSpec extends FlatSpec with Matchers {
   val eps = 1e-3

  "Decision leaf" should "classify 01" in {
    val clf = Stump(NodeLeaf(DenseVector(Array(1.0, 0.0))))
    clf.predict(DenseVector(Array(0.0))) should be (0)
  }

  it should "classify 02" in {
    val leaf1 = NodeLeaf(DenseVector(Array(1.0, 0.0)))
    val leaf2 = NodeLeaf(DenseVector(Array(0.0, 1.0)))
    val clf   = Stump(NodeSplit(0, 0.4, leaf1, leaf2))
    clf.predict(DenseVector(Array(1.0))) should be (1)
    clf.predict(DenseVector(Array(0.2))) should be (0)
  }

}
