package org.apache.spark.mllib.stat.distribution

import breeze.linalg.{ DenseVector => DBV, DenseMatrix => DBM, diag, max, eigSym, Vector => BV, Matrix => BM }
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{ Vectors, Vector, Matrices, Matrix }
//import org.apache.spark.mllib.stat.distribution.{ Wishart }

/**
 * This class provides basic functionality for the Normal-Wishart distribution.
 * The Normal-Wishart distribution is a distribution over the tuple (mu,Lambda),
 * where mu is a real-valued vector and Lambda is a positive-definite DxD matrix.
 * It is equivalent to the hierarchical model
 *      mu     ~ Normal(mu0,(lambda*Lambda)^-1)
 *      Lambda ~ Wishart(L,nu)
 *
 * We use the convention that L is the natural parameter for the exponential family
 *
 * @param mu The mean vector
 * @param lambda The scale parameter
 * @param L A DxD positive definite matrix, the natural parameter for the exponential family
 * @param nu Degrees of freedom parameter, integer
 */
class NormalWishart(
    val mu0: Vector,
    val lambda: Double,
    val L: Matrix,
    val nu: Double) extends Serializable {

  val D = L.numCols

  require(nu > D - 1, "The degrees of freedom is less than the dimension of W0")
  require(lambda > 0, "The scale parameter must be positive")

  private val mu0Breeze = mu0.toBreeze.toDenseVector
  private val LBreeze = L.toBreeze.toDenseMatrix

  private lazy val multLGTerm = Wishart.multLogGamma(D, nu / 2.0)

  private lazy val logDetLBreeze = logdet(LBreeze)._2

  private lazy val Linv = inv(LBreeze)

  /* The expectation of the log-Wishart random variable */
  lazy val expectationLogWishart = {
    val offset = multLGTerm + D * math.log(2.0)
    offset - logDetLBreeze
  }

  /* Evaluates the expectation of the quadratic form
   *     (x-mu)^T Lambda * (x-mu)
   *     over (mu,Lambda)
   */
  def quadraticForm(x: Vector): Double = { quadraticForm(x.toBreeze) }

  private def quadraticForm(x: BV[Double]): Double = {
    val xDenseMinusMu = x.toDenseVector - mu0Breeze.toDenseVector
    D / lambda + nu * (xDenseMinusMu dot (LBreeze \ xDenseMinusMu))
  }

  private def pdf(x: BV[Double], Sigma: BM[Double]): Double = {
    math.exp(logpdf(x, Sigma))
  }

  /** Returns the log-density  at given point, x */
  private def logpdf(x: BV[Double], Sigma: BM[Double]): Double = {
    val gaussSigma = Matrices.fromBreeze(inv(Sigma.toDenseMatrix) / lambda)
    val gaussian = new MultivariateGaussian(mu0, gaussSigma)
    val wishart = new Wishart(nu, L)
    gaussian.logpdf(x) + wishart.logpdf(Matrices.fromBreeze(Sigma))
  }

  def pdf(x: Vector, Sigma: Matrix): Double = {
    pdf(x.toBreeze, Sigma.toBreeze)
  }

  def logpdf(x: Vector, Sigma: Matrix): Double = {
    logpdf(x.toBreeze, Sigma.toBreeze)
  }
}