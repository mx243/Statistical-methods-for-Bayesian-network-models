# Discrete: P(X = n) = (1 - exp(-lambda)) * exp(-lambd * n), n >= 0, lambda > 0
# Continuous: P(X = x) = lambda * exp(-lambd * x), x >= 0, lambda > 0
import numpy as np
from .._core import sample_discrete
from scipy import integrate
import matplotlib.pyplot as plt

rng = np.random.default_rng()

def sample(lambda_, n=None, left=0, right=np.inf, is_discrete=True): # Draw n samples from Exp(lambda) on [left, right)
    if lambda_ <= 0:
        raise ValueError('my.stat.distr.Exponetial.sample: Invalid argument lambda.')

    left = np.maximum(left, 0)
    if left >= right:
        raise ValueError('my.stat.distr.Exponetial.sample: Invalid sample range.')

    if is_discrete:
        if right == np.inf:
            samples = np.floor(-np.log(1 - rng.random(n)) / lambda_) + left
            samples = samples.astype(int)
        else:
            samples = sample_discrete(n, logps=-lambda_ * np.arange(right - left)) + left
    else:
        if right == np.inf:
            samples = rng.exponential(scale=1 / lambda_, size=n) + left
        else:
            samples = -np.log(1 - rng.random(n) * (1 - np.exp(-lambda_ * (right - left)))) / lambda_ + left

    return samples


def _test_sample(lambda_, left, right, is_discrete=True): # test function for sample
    n = int(1e5)

    samples = sample(lambda_, n, left, right, is_discrete)

    l = np.min(samples)
    r = np.max(samples)
    eps = (r - l) / 50

    if is_discrete:
        const = np.sum(np.exp(- lambda_ * np.arange(l, r + 1)))
        x_vec = np.arange(l, r + 1)
        y_vec = np.exp(- lambda_ * x_vec) / const
    else:
        const = integrate.quad(lambda x: np.exp(- lambda_ * x), l, r)[0]
        x_vec = np.arange(l, r + (r - l) / 1000, (r - l) / 1000)
        y_vec = np.exp(- lambda_ * x_vec) / const
    
    f1, ax1 = plt.subplots()

    ax1.plot(x_vec, y_vec, 'g-')

    if is_discrete:
        ax1.hist(samples, np.arange(l - 0.5, r + 1.5), density=True)
    else:
        ax1.hist(samples, np.arange(l, r + eps, eps), density=True)