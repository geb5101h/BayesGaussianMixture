package org.apache.spark.mllib.clustering

import breeze.linalg.{ DenseVector => DBV, DenseMatrix => DBM, diag, max, eigSym, Vector => BV, Matrix => BM }
import org.apache.spark.mllib.util.{ MLUtils, Loader, Saveable }
import org.apache.spark.mllib.stat.distribution.{ MultivariateGaussian, Dirichlet, Wishart, NormalWishart, MultivariateStudentsT }
import org.apache.spark.mllib.linalg.{ Vectors, Vector, Matrices, Matrix }
import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.math._
import breeze.numerics._

/*
 * This class implements the variational Bayes Gaussian Mixture Model. In this model
 * the posterior distribution of the parameters is represented by a product of two
 * distributions:
 *    1. A Dirichlet distribution for the mixing weights
 *    2. A Normal-Wishart distribution for each cluster mean/precision
 * 
 * Furthermore, the predictive density is given by a mixture of multivariate 
 * Student's T distributions
 * 
 * @param dirichlet      The Dirichlet distribution for the mixing weights
 * @param normalWisharts The Array[NormalWishart] for mean/precisio pairs for
 * 	                     each cluster
 */
class VBGaussianMixtureModel(
    val dirichlet: Dirichlet,
    val normalWisharts: Array[NormalWishart]) extends Serializable { //with Saveable {

  def K: Int = dirichlet.length
  def D: Int = normalWisharts(0).D

  require(K == normalWisharts.length, "Number of mixture components is not consistent between dirichlet and normalWishart distributions")

  /*
   * Compute posterior probability at points
   */
  def predict(points: RDD[Vector]): RDD[Double] = {
    val sc = points.sparkContext
    val dirBC = sc.broadcast(dirichlet)
    val nwsBC = sc.broadcast(normalWisharts)
    points.map(point => computePredictiveDensity(point, dirBC.value, nwsBC.value))
  }

  def predict(point: Vector): Double = {
    computePredictiveDensity(point, dirichlet, normalWisharts)
  }

  /**
   * Maps given point to its most likely cluster index.
   */
  def predictLabel(points: RDD[Vector]): RDD[Int] = {
    points.map(x => predictLabel(x))
  }

  def predictLabel(point: Vector): Int = {
    val p = predictSoft(point)
    p.indexOf(p.max)
  }

  /**
   * Given the input vector, return the membership values to all mixture components.
   */
  def predictSoft(points: RDD[Vector]): RDD[Array[Double]] = {
    val sc = points.sparkContext
    val nwBC = sc.broadcast(normalWisharts)
    val dirBC = sc.broadcast(dirichlet)
    points.map(x => computeResponsibilities(x, nwBC.value, dirBC.value))
  }

  def predictSoft(point: Vector): Array[Double] = {
    computeResponsibilities(point, normalWisharts, dirichlet)
  }

  /**
   * Compute the partial assignments for each vector
   */
  private def computeResponsibilities(
    pt: Vector,
    normalWisharts: Array[NormalWishart],
    dirichlet: Dirichlet): Array[Double] = {
    val dirExpArray = dirichlet.expectationLog.toArray
    val rawResp = dirExpArray.zip(normalWisharts)
      .map {
        case (dir, nw) => {
          val logRawResp = dir + 0.5 * nw.expectationLogWishart
          -0.5 * nw.quadraticForm(pt)
          math.exp(logRawResp)
        }
      }
    val normConst = rawResp.sum
    rawResp.map(x => x / normConst)
  }

  /*
   * Compute predictive density at a point. Predictive density is a mixture of multivariate
   * Student's t distributions
   */
  private def computePredictiveDensity(x: Vector, dirichlet: Dirichlet, nws: Array[NormalWishart]): Double = {
    val alphaBreeze = dirichlet.alpha.toBreeze
    val alphaSum = sum(alphaBreeze)
    val weights = alphaBreeze / alphaSum

    val mvStudents = nws.map(nw => {
      val studentDF = nw.nu + 1.0 - nw.D
      val studentPrecision = Matrices.fromBreeze(studentDF / (1 + 1.0 / nw.beta) * inv(nw.L.toBreeze.toDenseMatrix))
      new MultivariateStudentsT(nw.mu0, studentPrecision, studentDF)
    })

    weights.toArray.zip(mvStudents)
      .map { case (w, d) => w * d.pdf(x) }
      .sum
  }
}