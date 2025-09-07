# X ~ Gamma(m, r) (m > 0, r > 0), use parameterisation P(x | m, r) = r^m / Gamma(m) * x ^ (m - 1) * exp(-r * x)
# Conjugate to the scale for 1-D Gaussian and Poisson
# Conjugate priors for m: proGamma
#                  for r: Gamma
import numpy as np
from scipy import integrate
import scipy as sc
import matplotlib.pyplot as plt

rng = np.random.default_rng()


def logpdf(x, m, r):
    return m * np.log(r) + (m - 1) * np.log(x) - r * x - sc.special.gammaln(m)


def sample(m, r, n=None, left=0, right=np.inf): # Draw n Gamma samples on [left, right).
    m = np.array(m)
    r = np.array(r)
    if np.any(m <= 0) or np.any(r <= 0) or (m.shape != r.shape):
        raise ValueError('my.stat.distr.Gamma.sample: Invalid argument m or r.')

    left = np.maximum(left, 0)
    if left >= right:
        raise ValueError('my.stat.distr.Gamma.sample: Invalid sample range.')

    if (left == 0) and (right == np.inf):
        return rng.gamma(m, 1 / r, size=n)
    else:
        if m.size > 1: # ignore n in this case, return one sample for each pair of (m, r)
            u = rng.random(m.shape)
            prob_low = sc.special.gammainc(m, r * left)
            prob_high = sc.special.gammainc(m, r * right)
            return sc.special.gammaincinv(m, u * prob_high + (1 - u) * prob_low) / r
        else:
            u = rng.random(n)
            prob_low = sc.special.gammainc(m, r * left)
            prob_high = sc.special.gammainc(m, r * right)
            return sc.special.gammaincinv(m, u * prob_high + (1 - u) * prob_low) / r


def _test_sample(m, r, left, right): # test function for sample
    n = int(1e5)

    samples = sample(m, r, n, left, right)

    l = np.min(samples)
    r_ = np.max(samples)
    eps = (r_ - l) / 50

    const = integrate.quad(lambda x: np.exp(logpdf(x, m, r)), l, r_)[0]
    x_vec = np.arange(l, r_ + (r_ - l) / 1000, (r_ - l) / 1000)
    y_vec = np.exp(logpdf(x_vec, m, r)) / const
    
    f1, ax1 = plt.subplots()

    ax1.plot(x_vec, y_vec, 'g-')

    ax1.hist(samples, np.arange(l, r_ + eps, eps), density=True)