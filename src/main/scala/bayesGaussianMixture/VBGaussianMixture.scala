package org.apache.spark.mllib.clustering

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.util.{ MLUtils, Loader, Saveable }
import org.apache.spark.mllib.linalg.{ BLAS, Vectors, Vector, Matrices, Matrix, DenseMatrix }
import breeze.linalg.{ DenseVector => DBV, DenseMatrix => DBM, diag, max, eigSym, Vector => BV, Matrix => BM }
import org.apache.spark.mllib.stat.distribution.{ Dirichlet, NormalWishart }

/**
 * This class estimates the Bayesian Gaussian Mixture Model. The model is
 * defined by a likelihood, which is a mixture of Gaussians, given by an array
 * of mean/variance parameters for each Gaussian component, and a vector of
 * mixing weights, as well as a prior distribution for the likelihood parameters.
 *
 * The hierarchical model is given by
 *     pi       ~ Dirichlet(alpha_0)
 *     Lambda_k ~ Wishart(W_0, v_0)
 *     mu_k     ~ Normal(mu_0, beta_0 * Lambda_k^-1)
 *     z_i      ~ Multinomial(1, pi)
 *     x_i      ~ Normal(mu_z_i, Lambda_z_i ^-1)
 * where i=1,...,N, N the number of data points and K the number of mixture components.
 * x_i is one observation, which is a real-valued vector of length D. z_i is the latent
 * cluster assignment vector for x_i.
 *
 * Since the posterior distribution for this model is not tractable, we employ the
 * variational Bayes approach of Bishop (2006). This optimizes a surrogate
 * function which is a lower bound on the true posterior. The resulting
 * optimization is called variational E-M, analogous to the regular GMM.
 *
 * The variational Bayes approach has been shown to have better computational stability
 * than the EM algorithm and is more resistant to overfitting the data, generally
 * outputting a parsimonious model, which obviates the need for cross-validation.
 *
 * @param k The number of independent Gaussians in the mixture model
 * @param convergenceTol The maximum change in log-likelihood at which convergence
 * is considered to have occurred.
 * @param maxIterations The maximum number of iterations to perform
 */

class VBGaussianMixture(
    private var k: Int,
    private var convergenceTol: Double,
    private var maxIterations: Int,
    private var seed: Long) extends Serializable {

  def run(data: RDD[Vector], gmmPrior: VBGMMPrior): VBGaussianMixtureModel = {
    val sc = data.sparkContext

    /*
     * Get hyperparameters
     */
    val alpha0 = gmmPrior.alpha0
    val beta0 = gmmPrior.beta0
    val mu0: Vector = gmmPrior.mu0
    val nu0 = gmmPrior.nu0
    val L0: Matrix = gmmPrior.L0
    val K = gmmPrior.K
    val D = L0.numCols

    val breezeData = data.map(_.toBreeze)

    var llh = Double.MinValue // current log-likelihood
    var llhp = 0.0 // previous log-likelihood
    var iter = 0
    /*
     * initialize posterior estimate
     */
    var dirichlet: Dirichlet = new Dirichlet(Vectors.dense(Array.fill(K)(alpha0)))
    var nw: Array[NormalWishart] = Array.fill(K)(new NormalWishart(mu0, beta0, L0, nu0))
    var gmmCurrent = new VBGaussianMixtureModel(dirichlet, nw)

    while (iter < maxIterations && math.abs(llh - llhp) > convergenceTol) {
      /*
       * Perform "E" step, updating
       * responsibilities and doing aggregations
       */
      val compute = sc.broadcast(ExpectationSum.add(gmmCurrent)_)

      // aggregate the cluster contribution for all sample points
      val sums = breezeData.aggregate(ExpectationSum.zero(K, D))(compute.value, _ += _)

      /*
       * Perform "M" step, updating
       * posterior params
       */
      println("sums weights: " + sums.weights.foldLeft("")((a,b)=>a+","+b))

      dirichlet = new Dirichlet(Vectors.dense(
        sums.weights
          .map(w => w + alpha0)))
      nw = {
        val betaNew = sums.weights.map(x => x + beta0)

        val muNew = (sums.weights).zip(sums.means)
          .map {
            case (w, m) =>
              (beta0 * mu0.toBreeze.toDenseVector + w * m.toDenseVector)
          }
          .zip(betaNew)
          .map { case (w, b) => Vectors.fromBreeze(w / b) }

        val nuNew = sums.weights.map(x => x + nu0)

        val Lnew = (sums.weights).zip(sums.sigmas)
          .map { case (w, s) => Matrices.fromBreeze(w * s.toDenseMatrix) }

        var i = 0
        var nwRet = new Array[NormalWishart](K)
        while (i < K) {
          nwRet(i) = new NormalWishart(muNew(i), betaNew(i), Lnew(i), nuNew(i))
          i += 1
        }
        nwRet
      }
      gmmCurrent = new VBGaussianMixtureModel(dirichlet, nw)
      println("Iteration number: " + iter)
      llhp = llh
      llh = sums.logLikelihood
      println("llhp " + llhp)
      println("llh " + llh)
      iter += 1

    }
    gmmCurrent
  }

  /* companion class to provide zero constructor for ExpectationSum
   * Adapted from Spark MLLIB GaussianMixtureModel.scala
   */
  private object ExpectationSum {
    def zero(k: Int, d: Int): ExpectationSum = {
      new ExpectationSum(0.0, Array.fill(k)(0.0),
        Array.fill(k)(BV.zeros(d)), Array.fill(k)(BM.zeros(d, d)))
    }

    // compute cluster contributions for each input point
    // (U, T) => U for aggregation
    def add(
      gmm: VBGaussianMixtureModel)(sums: ExpectationSum, x: BV[Double]): ExpectationSum = {
      val p = gmm.predictSoft(Vectors.fromBreeze(x))

      val pSum = p.sum
      sums.logLikelihood += math.log(pSum)
      var i = 0
      while (i < sums.k) {
        p(i) /= pSum
        sums.weights(i) += p(i)
        sums.means(i) += x * p(i)
        BLAS.syr(p(i), Vectors.fromBreeze(x),
          Matrices.fromBreeze(sums.sigmas(i)).asInstanceOf[DenseMatrix])
        i = i + 1
      }
      sums
    }
  }

  // Aggregation class for partial expectation results
  private class ExpectationSum(
      var logLikelihood: Double,
      val weights: Array[Double],
      val means: Array[BV[Double]],
      val sigmas: Array[BM[Double]]) extends Serializable {

    val k = weights.length

    def +=(x: ExpectationSum): ExpectationSum = {
      var i = 0
      while (i < k) {
        weights(i) += x.weights(i)
        means(i) += x.means(i)
        sigmas(i) += x.sigmas(i)
        i = i + 1
      }
      logLikelihood += x.logLikelihood
      this
    }
  }
}

/*
 * A class to encapsulate the hyperparameters
 * for the prior
 */
case class VBGMMPrior(
  val alpha0: Double,
  val beta0: Double,
  val mu0: Vector,
  val nu0: Double,
  val L0: Matrix,
  val K: Int)