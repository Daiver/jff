package object graphviz {

  case class Attribute(key: String, value: String) {
    def render(): String = s"${key} = ${value}"
  }

  case class AttributePlaceholder(str: String){
    def := (other: AttributePlaceholder): Attribute = Attribute(str, other.str)
  }
  implicit def strToAttrPlaceholder(str: String):AttributePlaceholder = AttributePlaceholder(str)

  abstract class Statement {
    def render(): String
  }

  case class NodeStatement(
    name: String,
    attributes: Seq[Attribute]
    ) extends Statement {
      def :| (newAttr: Attribute): NodeStatement = NodeStatement(name, attributes :+ newAttr)
      override def render(): String = {
        val attrsRendered = attributes map {_.render} mkString ","
        s"${name}[${attrsRendered}];"
      }
  }
  implicit def strToNs(str: String) = NodeStatement(str, Seq())

  case class EdgeStatement(
    startNode:  String,
    finishNode: String,
    attributes: Seq[Attribute]
  ) extends Statement {

    def :| (newAttr: Attribute): EdgeStatement = EdgeStatement(startNode, finishNode, attributes :+ newAttr)

    override def render(): String = {
      val attrsRendered = attributes map {_.render} mkString ","
      s"${startNode} -- ${finishNode}[${attrsRendered}];"
    }
  }

  case class EdgeStatementPlaceholder(name: String) {
    def --- (other: EdgeStatementPlaceholder) = EdgeStatement(name, other.name, Seq())
  }
  implicit def stringToEsPlaceholder(str: String) = EdgeStatementPlaceholder(str)

  case class Graph(id: String, edges: Seq[Statement]){
    def render() = {
      val tmp = (edges map {x => s"\t${x.render}\n"} mkString "")
      s"graph ${id} {\n${tmp}}\n"
    }
  }

  def createGraph(id: String, edges: Statement*) = {
    Graph(id, edges)
  }

}
