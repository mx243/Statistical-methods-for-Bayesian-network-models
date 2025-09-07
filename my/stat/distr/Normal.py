# X ~ Normal(mu, S), use parameterisation P(x | mu, S) = sqrt(det(S / (2 * pi))) * exp(- (1 / 2) * (x - mu)' * S * (x - mu)) (S is the inverse of the variance matrix)
# Conjugate to the mean for Gaussian
# Conjugate priors for mu: Gaussian
#                  for S: Wishart
import numpy as np
from scipy import integrate
import scipy as sc
import matplotlib.pyplot as plt
from .._core import forcechol, modchol

rng = np.random.default_rng()


def sample(mu, S, n=None, left=-np.inf, right=np.inf, use_modchol=False): # Draw n Normal samples on [left, right). 
    mu = np.array(mu)
    S = np.array(S)
    if mu.size == 1:
        if (left == -np.inf) and (right == np.inf):
            return rng.normal(size=n) / np.sqrt(S) + mu
        else:
            u = rng.random(size=n)
            c = np.sqrt(S / 2)
            prob_left = (1 + sc.special.erf((left - mu) * c)) / 2
            prob_right = (1 + sc.special.erf((right - mu) * c)) / 2
            return sc.special.erfinv(2 * (u * prob_right + (1 - u) * prob_left) - 1) / c + mu
    else: # Ignore left and right in this case
        if n is None: # In this case mu can be 2D where each row is a different mean.
            u = rng.standard_normal(mu.shape).T
            if use_modchol: # For extra numerical stability
                return np.linalg.solve(modchol(S).T, u).T + mu
            else:
                return np.linalg.solve(forcechol(S).T, u).T + mu
        else:
            u = rng.standard_normal([mu.size, n])
            if use_modchol: # For extra numerical stability
                return np.linalg.solve(modchol(S).T, u).T + mu
            else:
                return np.linalg.solve(forcechol(S).T, u).T + mu


def _test_sample(mu, S, left=-np.inf, right=np.inf, use_modchol=False):
    n = int(1e5)

    samples = sample(mu, S, n, left, right, use_modchol)

    l = np.min(samples)
    r = np.max(samples)
    eps = (r - l) / 50

    const = integrate.quad(lambda x: np.exp((- (1 / 2) * (x - mu) * S * (x - mu))), l, r)[0]
    x_vec = np.arange(l, r + (r - l) / 1000, (r - l) / 1000)
    y_vec = np.exp(- (1 / 2) * (x_vec - mu)* S * (x_vec - mu)) / const
    
    f1, ax1 = plt.subplots()

    ax1.plot(x_vec, y_vec, 'g-')

    ax1.hist(samples, np.arange(l, r + eps, eps), density=True)


def _test_3dsample(mu, S, use_modchol=False):
    n = int(1e7)

    #samples = sample(mu, S, n=n, use_modchol=use_modchol)

    samples2 = rng.multivariate_normal(mu, np.linalg.inv(S), size=n)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], marker='o')

    ax.scatter(samples2[:, 0], samples2[:, 1], samples2[:, 2], marker='^')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    '''