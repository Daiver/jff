import org.scalatest.FunSuite
import Float._

class FADTests extends FunSuite {
  import Fad.FadIsFractional._

  def isAlmostEqual(a: Float, b: Float, eps: Float = 1e-9f): Boolean = (a - b).abs < eps
  def isAlmostEqualFad(a: Fad, b: Fad, eps: Float = 1e-9f): Boolean =
    isAlmostEqual(a.value, b.value, eps) && isAlmostEqual(a.der, b.der, eps)

  test("isAlmostEqual01") {
    assert(isAlmostEqual(5f, 5.01f, 0.1f))
    assert(!isAlmostEqual(5f, 5.2f, 0.1f))
  }

  test("isAlmostEqualFad01") {
    assert(isAlmostEqualFad(Fad(1, 2), Fad(1.01f, 1.96f), 0.1f))
    assert(!isAlmostEqualFad(Fad(0, 2), Fad(1.01f, 1.96f), 0.1f))
  }

  test("Fad01") {
    val x = Fad(1, 1)
    assert(x.value === 1.0f)
    assert(x.der === 1.0f)
  }

  test("Fad add 01") {
    val res1 = Fad(-1.0f) + Fad(2, 1)
    assert(isAlmostEqualFad(res1, Fad(1, 1)))
  }

  test("Fad add 02") {
    val res1 = Fad(2, 5) + Fad(1, 3)
    assert(isAlmostEqualFad(res1, Fad(3, 8)))
  }

  test("Fad minus 01") {
    val res1 = Fad(2, 5) - Fad(1, 3)
    assert(isAlmostEqualFad(res1, Fad(1, 2)))
  }

  test("Fad times 01") {
    val x = Fad(0, 0)
    val y: Fad = 2.0f
    val res1 = Fad(-1.0f) * x
    val res2 = Fad(-1.0f) * y
    assert(isAlmostEqualFad(res1, Fad(0, 0)))
    assert(isAlmostEqualFad(res2, Fad(-2, 0)))
  }

  test("Fad times 02") {
    val res1 = Fad(3.0f) * Fad(4, 5)
    val res2 = Fad(-2.0f, 1) * Fad(3, 1)
    assert(isAlmostEqualFad(res1, Fad(12, 15)))
    assert(isAlmostEqualFad(res2, Fad(-6, 1)))
  }

  test("Fad negate 01") {
    val res1 = -Fad(2, -5)
    assert(isAlmostEqualFad(res1, Fad(-2, 5)))
  }

  test("Fad division 01") {
    val x = Fad(5)
    val y = Fad(2)
    val res = x / y
    assert(isAlmostEqualFad(res, Fad(2.5f, 0)))
  }

  test("Fad division 02") {
    val x = Fad(5, 3)
    val y = Fad(2)
    val res = x / y
    assert(isAlmostEqualFad(res, Fad(2.5f, 1.5f)))
  }

  test("Fad division 03") {
    val x = Fad(5, 3)
    val y = Fad(2, 1)
    val res = x / y
    assert(isAlmostEqualFad(res, Fad(2.5f, 1.5f)))
  }
}
