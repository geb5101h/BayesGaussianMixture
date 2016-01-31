package bayesGaussianMixture

import org.scalatest._
import org.apache.spark.mllib.linalg.{ Vectors, Vector, Matrices, Matrix }
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.mllib.clustering.{ VBGaussianMixture, VBGMMPrior }

class VBGaussianMixtureTest extends FunSuite {

  val conf = new SparkConf()
    .setAppName("vbGMMTest")
    .setMaster("local[2]")
    .set("spark.executor.memory", "1g")
    .set("spark.driver.memory", "1g")

  val sc = new SparkContext(conf)

  val testData = sc.parallelize(List(
    Vectors.dense(1, 2, 1, 4),
    Vectors.dense(3, 4, 3, 5),
    Vectors.dense(6, 7, 6, 7),
    Vectors.dense(1, 1, 3, 3)))

  val vbPrior = VBGMMPrior(
    2.0,
    1.0,
    Vectors.dense(1, 1, 1, 1),
    5.0,
    Matrices.dense(4, 4, Array(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)),
    3)

  val vbGaussianMixture = new VBGaussianMixture(
    3,
    1e-5,
    100,
    12314)

  val vbGaussianMixtureModel = vbGaussianMixture.run(testData, vbPrior)

  test("Testing VB Gaussian mixture fit") {
    assert(1 == 1)
  }
}