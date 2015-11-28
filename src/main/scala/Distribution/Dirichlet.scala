package org.apache.spark.mllib.stat.distribution

import breeze.linalg.{ DenseVector => DBV, DenseMatrix => DBM, diag, max, eigSym, Vector => BV }
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{ Vectors, Vector, Matrices, Matrix }
import org.apache.spark.mllib.util.MLUtils

/**
 * This class provides basic functionality for the Dirichlet distribution.
 * The Dirichlet distribution is a vector-valued distribution with
 * support over the probability simplex x_1+x_2+..+x_D = 1, x_i>=0 for each i
 * in 1,...,D.
 *
 * @param alpha The vector of intensity parameters
 */
class Dirichlet(
    val alpha: Vector) extends Serializable {

  require(alpha.toArray.map(x => x <= 0.0).contains(true), "Parameters must be strictly positive")

  private val alphaBreeze = alpha.toBreeze.toDenseVector

  val length = alphaBreeze.length

  /* Some functionals of alpha lazily evaluated */
  lazy private val sumAlpha = sum(alphaBreeze)

  lazy private val normConstant = lgamma(sumAlpha) - sum(lgamma(alphaBreeze))

  /* The expectation of the given Dirichlet random variable */
  lazy val expectation = {
    alphaBreeze / sumAlpha
  }

  /* The expectation of log of the given Dirichlet random variable */
  lazy val expectationLog = {
    val digammaVec = digamma(alphaBreeze)
    Vectors.fromBreeze(digammaVec - digamma(sumAlpha))
  }

  /** Returns density  at given point, x */
  private def pdf(x: BV[Double]): Double = {
    math.exp(logpdf(x))
  }

  /** Returns the log-density  at given point, x */
  private def logpdf(x: BV[Double]): Double = {
    val logX = log(x.toDenseVector)
    val ret = (alphaBreeze dot logX) + normConstant - sum(logX)
    ret
  }

  /** Returns density  at given point, x */
  def pdf(x: Vector): Double = {
    pdf(x.toBreeze)
  }

  /** Returns the log-density  at given point, x */
  def logpdf(x: Vector): Double = {
    logpdf(x.toBreeze)
  }
}