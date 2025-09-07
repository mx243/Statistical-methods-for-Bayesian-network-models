# S ~ Wishart(m, R, D), using parameterisation 
# P(S | m, R, D) = det(R) ^ (m + (D - 1) / 2) * det(S) ^ (m - 1) * exp(-trace(RS)) / ((2pi) ^ (D * (D - 1) / 4) * \prod_{j = 0}^{D - 1}Gamma(m + j / 2))
# where R is a D * D positive-definite symmetric matrix, m > 0.
# Conjugate to the scale parameter for Gaussian
# Conjugate prior for m: proGamma
#                 for R: Wishart
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

rng = np.random.default_rng()


def sample(m, R, D, n=1):
    df = 2 * m + D - 1
    scale = np.linalg.inv(2 * R)
    return sc.stats.wishart.rvs(df, scale, size=n)