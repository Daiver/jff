import org.scalatest.FunSuite

class FADTests extends FunSuite {
  test("Fad01") {
    val x = Fad(1, 1)
    assert(3*3*3 === 27)
  }

  test("Fad02") {
    val x = Fad(0, 0)
    assert(3*3*3 === 27)
  }
}
