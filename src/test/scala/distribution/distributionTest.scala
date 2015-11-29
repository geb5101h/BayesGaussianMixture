package org.apache.spark.mllib.stat.distribution

import org.scalatest._
import org.apache.spark.mllib.linalg.{ Vectors, Vector, Matrices, Matrix }

class distributionTest extends FunSuite {
  val alpha = Vectors.dense(1.0, 1.0, 1.0)
  val x = Vectors.dense(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
  val dirichlet = new Dirichlet(alpha)

  test("Testing density evaluation for dirichlet r.v.") {
    assert(math.abs(dirichlet.pdf(x) - 2.0) <= 1E-3)
  }
}