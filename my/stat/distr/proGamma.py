# The proGamma distributions are a family conjugate to the shape parameter m in the Gamma distribution.
# There are three common parameterisations of the Gamma parameter (m, r) when m varys:
# type 0: r the scale parameter is constant. The corresponding proGamma distribution on m is
#         P(m | a, b, D, typ = 0) \propto exp(-(a + D * b) * m) / (\prod_{j = 0}^{D - 1}\Gamma(m + j / 2)) ^ b (m > 0)
# type 1: r = (m - 1) * t for a fixed t, i.e. the mean of 1 / X where X ~ Gamma(m, r) stays constant as m (> 1) varies.
#         The corresponding proGamma distribution on m is
#         P(m | a, b, D, typ = 1) \propto exp(-(a + D * b) * m) * (m - 1) ^ (D * b * (m + (D - 1) / 2)) / (\prod_{j = 0}^{D - 1}\Gamma(m + j / 2)) ^ b (m > 1)
# type 2: r = m * t for a fixed t, i.e. the mean X where X ~ Gamma(m, r) stays constant as m (> 1) varies.
#         The corresponding proGamma distribution on m is
#         P(m | a, b, D, typ = 2) \propto exp(-(a + D * b) * m) * (m + (D - 1) / 2) ^ (D * b * (m + (D - 1) / 2)) / (\prod_{j = 0}^{D - 1}\Gamma(m + j / 2)) ^ b (m > 0)
import numpy as np
from scipy import integrate
import scipy as sc
import matplotlib.pyplot as plt
from .._core import ARS
import warnings

rng = np.random.default_rng()


def logpdf(m, a, b, D, typ): # Return the log density (up to a constant) of proGamma(a, b, D, typ) at m.
    m = np.array(m)
    tmp_mat = np.concatenate([np.arange(D)[:, np.newaxis] / 2] * m.size, axis=1) + m
    if typ == 0:
        if b <= 0:
            raise ValueError('my.stat.distr.proGamma.logpdf: Invalid argument b.')
        ans = - ((a / b + D) * m + np.sum(sc.special.gammaln(tmp_mat), axis=0)) * b
        ans[m <= 0] = -np.inf
    if typ == 1:
        if a <= 0 or b <= 0:
            raise ValueError('my.stat.distr.proGamma.logpdf: Invalid argument a or b.')
        ans = (- (a / b + D) * m + D * (m + (D - 1) / 2) * np.log(m - 1) - np.sum(sc.special.gammaln(tmp_mat), axis=0)) * b
        ans[m <= 1] = -np.inf
    if typ == 2:
        if a <= 0 or b <= 0:
            raise ValueError('my.stat.distr.proGamma.logpdf: Invalid argument a or b.')
        ans = (- (a / b + D) * m + D * (m + (D - 1) / 2) * np.log(m + (D - 1) / 2) - np.sum(sc.special.gammaln(tmp_mat), axis=0)) * b
        ans[m <= 0] = -np.inf
    if ans.size > 1:
        return ans
    else:
        return ans[0]


def grad_logpdf(m, a, b, D, typ): # Return the gradient of log(pdf) of proGamma(a, b, D, typ) at m.
    m = np.array(m)
    tmp_mat = np.concatenate([np.arange(D)[:, np.newaxis] / 2] * m.size, axis=1) + m
    if typ == 0:
        if b <= 0:
            raise ValueError('my.stat.distr.proGamma.grad_logpdf: Invalid argument b.')
        ans = - ((a / b + D) + np.sum(sc.special.psi(tmp_mat), axis=0)) * b
        ans[m <= 0] = np.inf
    if typ == 1:
        if a <= 0 or b <= 0:
            raise ValueError('my.stat.distr.proGamma.grad_logpdf: Invalid argument a or b.')
        ans = (- (a / b + D) + D * ((m + (D - 1) / 2) / (m - 1) + np.log(m - 1)) - np.sum(sc.special.psi(tmp_mat), axis=0)) * b
        ans[m <= 1] = np.inf
    if typ == 2:
        if a <= 0 or b <= 0:
            raise ValueError('my.stat.distr.proGamma.grad_logpdf: Invalid argument a or b.')
        ans = (- (a / b + D) + D * (np.log(m + (D - 1) / 2) + 1) - np.sum(sc.special.psi(tmp_mat), axis=0)) * b
        ans[m <= 0] = np.inf
    if ans.size > 1:
        return ans
    else:
        return ans[0]


def sample(a, b, D, typ=0, n=None, left=0, right=np.inf): # Draw n proGamma samples on [left, right) using ARS.
    logf = lambda m: logpdf(m, a, b, D, typ)
    dlogf_dx = lambda m: grad_logpdf(m, a, b, D, typ)

    if typ == 0:
        left = np.maximum(left, 0)
        if a / (D * b) < -20:
            warnings.warn('my.stat.distr.proGamma.sample: a < -20 * D * b leads to poor performance.')

        if not np.isinf(right):
            xs = np.arange(left, right, (right - left) / 10)[1 : ]
            return ARS(logf, dlogf_dx, xs, left, right, n)

        if dlogf_dx(left) < 0:
            return ARS(logf, dlogf_dx, np.array([2 * left, 3 * left, 4 * left]), left, right, n)

        tmp = np.maximum(np.exp(- a / (D * b)), 1) # where grad_logpdf is guaranteed to be negative
        while dlogf_dx(tmp / 2) < 0:
            tmp = tmp / 2
        return ARS(logf, dlogf_dx, np.append(np.arange(left, tmp, (tmp - left) / 10)[1 : ], [tmp, tmp * 2]), left, right, n)
    
    if typ == 1:
        left = np.maximum(left, 1)

        if not np.isinf(right):
            xs = np.arange(left, right, (right - left) / 10)[1 : ]
            return ARS(logf, dlogf_dx, xs, left, right, n)

        if dlogf_dx(left) < 0:
            return ARS(logf, dlogf_dx, np.array([2 * left, 3 * left, 4 * left]), left, right, n)

        tmp = (D + 3) * D * b / (2 * a) + 1 # where grad_logpdf is guaranteed to be negative
        while dlogf_dx(tmp / 2) < 0:
            tmp = tmp / 2
        return ARS(logf, dlogf_dx, np.append(np.arange(left, tmp, (tmp - left) / 10)[1 : ], [tmp, tmp * 2]), left, right, n)

    if typ == 2:
        left = np.maximum(left, 0)

        if not np.isinf(right):
            xs = np.arange(left, right, (right - left) / 10)[1 : ]
            return ARS(logf, dlogf_dx, xs, left, right, n)

        if dlogf_dx(left) < 0:
            return ARS(logf, dlogf_dx, np.array([2 * left, 3 * left, 4 * left]), left, right, n)

        tmp = (D + 1) * D * b / (2 * a) # where grad_logpdf is guaranteed to be negative
        while dlogf_dx(tmp / 2) < 0:
            tmp = tmp / 2
        return ARS(logf, dlogf_dx, np.append(np.arange(left, tmp, (tmp - left) / 10)[1 : ], [tmp, tmp * 2]), left, right, n)


def _test_sample(a, b, D, typ=0, left=0, right=np.inf): # test function for sample
    n = int(1e5)

    samples = sample(a, b, D, typ, n, left, right)

    l = np.min(samples)
    r_ = np.max(samples)
    eps = (r_ - l) / 50

    x_vec = np.arange(l, r_ + (r_ - l) / 1000, (r_ - l) / 1000)
    maxval = np.max(logpdf(x_vec, a, b, D, typ))
    const = integrate.quad(lambda x: np.exp(logpdf(x, a, b, D, typ) - maxval), l, r_)[0]
    y_vec = np.exp(logpdf(x_vec, a, b, D, typ) - maxval) / const
    
    f1, ax1 = plt.subplots()

    ax1.plot(x_vec, y_vec, 'g-')

    ax1.hist(samples, np.arange(l, r_ + eps, eps), density=True)