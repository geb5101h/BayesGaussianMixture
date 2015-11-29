package org.apache.spark.mllib.stat.distribution

import org.apache.spark.mllib.linalg.{ Vectors, Vector, Matrices, Matrix }
import breeze.linalg.{ DenseVector => DBV, DenseMatrix => DBM, diag, max, eigSym, Vector => BV, Matrix => BM }

import breeze.linalg._
import breeze.math._
import breeze.numerics._

/**
 * This class provides basic functionality for the Wishart distribution.
 * The Wishart distribution is a matrix-valued distribution with
 * support over the cone of non-negative definite matrices.
 *
 * @param nu0 The degrees of freedom, integer
 * @param V A DxD positive definite matrix. The natural parameter for the
 *          exponential family
 */
class Wishart(
    nu0: Double,
    V: Matrix) extends Serializable {

  val D = V.numCols

  private val VBreeze = V.toBreeze.toDenseMatrix

  /* Some functionals of the parameters, lazily evaluated */

  private lazy val logDetVBreeze = logdet(VBreeze)._2

  private lazy val VInvBreeze = inv(VBreeze)

  private lazy val multLGTerm = Wishart.multLogGamma(D, nu0 / 2.0)

  /* The expectation of the log-Wishart random variable */
  lazy val expectationLog = {
    val offset = multLGTerm + D * math.log(2.0)
    offset - logDetVBreeze
  }

  /** Returns density  at given point, x */
  private def pdf(x: BM[Double]): Double = {
    math.exp(logpdf(x))
  }

  /** Returns the log-density  at given point, x */
  private def logpdf(x: BM[Double]): Double = {
    val xBreeze = x.toDenseMatrix
    val logDetXBreeze = logdet(xBreeze)._2

    val ret = ((nu0 - D - 1.0) / 2.0) * logDetXBreeze
    -(nu0 * D / 2.0) * math.log(2.0)
    +(nu0 / 2.0) * logDetVBreeze
    -trace(VBreeze * xBreeze) / 2.0
    -multLGTerm
    -math.log(math.Pi) * D * (D - 1.0) / 4.0

    ret
  }

  /** Returns density  at given point, x */
  def pdf(x: Matrix): Double = {
    logpdf(x.toBreeze)
  }

  /** Returns the log-density  at given point, x */
  def logpdf(x: Matrix): Double = {
    logpdf(x.toBreeze)
  }

}

object Wishart {
  /* Log of the un-normalized multivariate gamma function
   *
   */
  def multLogGamma(D: Int, N: Double): Double = {
    (1 to D).map(i => lgamma(N + (1.0 - i) / 2.0)).sum
  }
}
