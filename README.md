# BayesGaussianMixture
This package offers functionality for the variational Bayes Gaussian Mixture Model for density estimation and clustering 
in Apache Spark. It considers the following Bayesian hierarchical model:

![Latex of model](https://www.dropbox.com/s/npbm8fk3z1hwyts/BayesGaussianMixtureModel.gif?dl=1)

The package uses variational Bayes to produce a tractable posterior distribution over the parameters (Bishop, 2006). The posterior distribution
is estimated using the variational E-M algorithm. The fitted model can produce soft- and hard- cluster assignments, as well as
prediction using the posterior predictive density.
