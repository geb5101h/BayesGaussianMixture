package org.apache.spark.mllib.stat.distribution

import org.scalatest._
import org.apache.spark.mllib.linalg.{ Vectors, Vector, Matrices, Matrix, DenseMatrix }

class distributionTest extends FunSuite {
  val alpha = Vectors.dense(1.0, 1.0, 1.0)
  val x = Vectors.dense(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
  val dirichlet = new Dirichlet(alpha)

  test("Testing density evaluation for dirichlet r.v.") {
    assert(math.abs(dirichlet.pdf(x) - 2.0) <= 1E-3)
  }

  val nu = 5.0
  val V = new DenseMatrix(2, 2, Array(1, .5, .5, 5))
  val wishart = new Wishart(nu, V)
  test("Testing density evaluation for Wishart r.v.") {
    assert(wishart.pdf(new DenseMatrix(2, 2, Array(1,0,0,1))).isInstanceOf[Double])
  }
}