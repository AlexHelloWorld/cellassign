% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/inference-tensorflow.R
\name{construct_gaussian_formulas}
\alias{construct_gaussian_formulas}
\title{construct cellassign formulas for optimization (batch-corrected & normalized expression matrix. Normal distribution hypothesis)}
\usage{
construct_gaussian_formulas(G, C, P, B, Y, shrinkage, min_delta,
  dirichlet_concentration, random_seed, tf, tfd)
}
\value{
A list of variables for EM optimization
}
\description{
construct cellassign formulas for optimization (batch-corrected & normalized expression matrix. Normal distribution hypothesis)
}
\keyword{internal}
