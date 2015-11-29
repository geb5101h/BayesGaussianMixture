package org.apache.spark.mllib.stat.distribution

import breeze.linalg.{Vector => BV, Matrix => BM}
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{Vector, Matrix}
//import org.apache.spark.mllib.stat.distribution.{MultivariateGaussian,Wishart}

/**
 * This class provides basic functionality for the mutlivariate Student's T distribution.
 * The MV Student's T distribution is a distribution over a real-valued vector
 *
 * @param mu The mean vector
 * @param W The precision matrix, DxD positive definite matrix
 * @param nu Degrees of freedom parameter, double
 */

class MultivariateStudentsT(
    mu: Vector,
    W: Matrix,
    nu: Double) extends Serializable {
  require(nu > 0, "Degrees of freedom must be positive")

  private val muBreeze = mu.toBreeze.toDenseVector
  private val WBreeze = W.toBreeze.toDenseMatrix

  val D = muBreeze.length

  private lazy val WBreezeInv = inv(WBreeze)

  private lazy val WLogDet = logdet(WBreeze)._2

  private def pdf(x: BV[Double]): Double = {
    math.exp(logpdf(x))
  }

  /** Returns the log-density  at given point, x */
  private def logpdf(x: BV[Double]): Double = {
    val normConstant = lgamma((nu + D) / 2.0) - lgamma(nu / 2.0)
    -D * 0.5 * math.log(math.Pi * nu)
    +0.5 * WLogDet
    val xDenseMinusMu = x.toDenseVector - muBreeze.toDenseVector
    normConstant - ((nu + D) / 2.0) * math.log(1.0 + (xDenseMinusMu.t * WBreeze * xDenseMinusMu) / nu)
  }

  def pdf(x: Vector): Double = {
    pdf(x.toBreeze)
  }

  def logpdf(x: Vector): Double = {
    logpdf(x.toBreeze)
  }
}