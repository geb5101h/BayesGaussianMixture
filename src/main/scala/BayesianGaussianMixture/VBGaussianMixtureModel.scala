package org.apache.spark.mllib.clustering

import breeze.linalg.{ DenseVector => DBV, DenseMatrix => DBM, diag, max, eigSym, Vector => BV, Matrix => BM }
import org.apache.spark.mllib.util.{ MLUtils, Loader, Saveable }
import org.apache.spark.mllib.stat.distribution.{ MultivariateGaussian, Dirichlet, Wishart, NormalWishart }
import org.apache.spark.mllib.linalg.{ Vectors, Vector, Matrices, Matrix }
import org.apache.spark.rdd.RDD

/*
 * This class implements the variational Bayes Gaussian Mixture Model. In this model
 * the posterior distribution of the parameters is represented by a product of two
 * distributions:
 *    1. A Dirichlet distribution for the mixing weights
 *    2. A Normal-Wishart distribution for each cluster mean/covariance
 * 
 * @param dirichlet      The Dirichlet distribution for the mixing weights
 * @param normalWisharts The Array[NormalWishart] for mean/cov pairs for
 * 	                     each cluster
 */
class VBGaussianMixtureModel(
    val dirichlet: Dirichlet,
    val normalWisharts: Array[NormalWishart]) extends Serializable { //with Saveable {

  def K: Int = dirichlet.length
  def D: Int = normalWisharts(0).D

  require(K == normalWisharts.length, "Number of mixture components is not consistent between dirichlet and normalWishart distributions")

  

  /**
   * Maps given point to its cluster index.
   */
  def predict(points: RDD[Vector]): RDD[Int] = {
    points.map(x => predict(x))
  }

  def predict(point: Vector): Int = {
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
  def computeResponsibilities(
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

}