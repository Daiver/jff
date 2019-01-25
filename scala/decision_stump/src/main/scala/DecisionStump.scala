
import breeze.linalg._
import ml.Utils._
import ml.DecisionSplit._
import ml.Classifier

package ml {
  package object DecisionStump {

    abstract class Node { 
      def predict(sample: DenseVector[Double]): DenseVector[Double]
    }

    case class NodeSplit(
      val featInd:   Int,
      val threshold: Double,
      val left: Node,
      val right: Node
        ) extends Node {

      def predict(sample: DenseVector[Double]): DenseVector[Double] = 
        if(sample(featInd) >= threshold) 
          right.predict(sample)
        else 
          left.predict(sample)
      
    }

    case class NodeLeaf(val freqDict: DenseVector[Double]) extends Node {
      def predict(sample: DenseVector[Double]): DenseVector[Double] = freqDict
    }

    case class Stump(node: Node) extends Classifier {
      def predict(sample: DenseVector[Double]): Int = node.predict(sample).argmax
      def predict(samples: DenseMatrix[Double]): DenseVector[Int] = DenseVector(
        (0 until samples.rows).map((index) => node.predict(samples(index, ::).t).argmax).toArray)
    }



  }
}
