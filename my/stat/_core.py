import numpy as np
import scipy.integrate as integrate
import scipy as sc
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from time import time, sleep
from typing import Callable
from .._core import * 

rng = np.random.default_rng()


def modchol(M) -> np.ndarray: # Modified Cholesky decomposition (Gill-Murray-Wright)
    if not isinstance(M, np.ndarray):
        M = np.array(M)
    if np.any(np.iscomplex(M)): # Restrict to real case
        raise ValueError('my.stat.modchol: Input matrix is complex.') 
    elif M.ndim != 2:
        raise ValueError('my.stat.modchol: Input matrix is not 2D.') 
    N = M.shape[0]
    if N != M.shape[1]:
        raise ValueError('my.stat.modchol: Input matrix is not square.')

    if N == 1:
        if M > 0:
            return M
        else:
            return 1e-8
    
    M = (M + M.T) / 2 # For numberical stability

    delta = 1e-8
    gamma = np.max(np.abs(np.diag(M))) 
    xi = np.max(np.abs(M[np.logical_not(np.eye(N))]))

    beta = np.sqrt(np.max([gamma, xi / np.sqrt(N ** 2 - 1), delta])) # Upper boundary for off-diagonal elements of L.

    # LDL decomp
    D = np.full(N, np.nan)
    L = np.eye(N)

    for col in range(N):
        D[col] = np.max([np.abs(M[col, col] - np.sum(L[col, : col] ** 2 * D[ : col])), delta]) # Should be small when M is almost positive-definite.
        if col != N - 1:
            L[col + 1 : , col] = M[col + 1 : , col] - np.sum(D[np.newaxis, : col] * L[col : col + 1, : col] * L[col + 1 : , : col], axis=1)
            r = np.max(np.abs(L[col + 1 : , col])) / np.sqrt(D[col] * beta)
            if r > 1:
                D[col] = D[col] * r ** 2
            L[col + 1 : , col] = L[col + 1 : , col] / D[col]
    
    return L * np.sqrt(D[np.newaxis, :])


def forcechol(M) -> np.ndarray: # An easier and *10 faster version of modified Cholesky decomposition 
    if not isinstance(M, np.ndarray):
        M = np.array(M)
    if M.ndim != 2:
        raise ValueError('my.stat.forcechol: Input matrix is not 2D.') 
    N = M.shape[0]
    if N != M.shape[1]:
        raise ValueError('my.stat.forcechol: Input matrix is not square.')

    if N == 1:
        if M > 0:
            return M
        else:
            return 1e-8
    
    M = (M + M.T) / 2 # For numberical stability
    eps = np.min(np.abs(np.diag(M)))

    while 1:
        try:
            L = np.linalg.cholesky(M)
        except LinAlgError:
            M = M + eps * np.eye(N) # Force positive-definity
            eps = eps * 2
        else:
            break
    
    return L


def Eadd(a, b) -> np.ndarray: # Return log(exp(a) + exp(b)) without ever exponentiating a or b
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('my.stat.Eadd: Inputs are not the same size.')
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        raise ValueError('my.stat.Eadd: Input contains NaN.')
    
    s = np.full(a.shape, np.nan)

    ind1 = (a > b)
    ind2 = (b >= a)
    ind3 = np.logical_and((a == b), np.isinf(a))

    s[ind1] = a[ind1] + np.log(1 + np.exp(b[ind1] - a[ind1]))
    s[ind2] = b[ind2] + np.log(1 + np.exp(a[ind2] - b[ind2]))
    s[ind3] = a[ind3]

    return s


def Esum(a, axis, keepdims=False): # Return log(sum(exp(a), axis)) without ever exponentiating a
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if np.any(axis >= a.ndim) or np.any(axis < 0):
        raise ValueError('my.stat.Esum: Invalid axis.')
    elif np.any(np.isnan(a)):
        raise ValueError('my.stat.Esum: Input contains NaN.')

    max_val = np.max(a, axis, keepdims=True)

    return np.sum(max_val, axis, keepdims=keepdims) + np.log(np.sum(np.exp(a - max_val), axis, keepdims=keepdims))


def Edet(M, use_modchol=False): # Return log(det(M)) where M is a positive-definite symmetric matrix.
    if not isinstance(M, np.ndarray):
        M = np.array(M)
    if np.any(np.isnan(M)):
        raise ValueError('my.stat.Edet: Input contains NaN.')

    if use_modchol:
        return 2 * np.sum(np.log(np.diag(modchol(M)))) # When extra numerical stability is required.
    else:
        return 2 * np.sum(np.log(np.diag(forcechol(M))))


def Esub(a, b) -> np.ndarray: # Return log(exp(a) - exp(b)) without ever exponentiating a or b, requires a > b for every entry
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('my.stat.Esub: Inputs are not the same size.')
    if np.any(a <= b - 1e-10):
        raise ValueError('my.stat.Esub: Input a <= b.')
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        raise ValueError('my.stat.Esub: Input contains NaN.')

    s = np.full(a.shape, np.nan)

    ind1 = (a > b)
    ind2 = (a == b)
    ind3 = np.logical_and((a < b), (a > b - 1e-10))

    s[ind1] = a[ind1] + np.log(1 - np.exp(b[ind1] - a[ind1]))
    s[ind2] = -np.inf
    s[ind3] = b[ind3] + np.log(1 - np.exp(a[ind3] - b[ind3])) # Deal with numerical error.

    return s


def sample_discrete(n=None, ps=None, logps=None): # Return n samples from a discrete distribution. The probabilities can be given by themselves or their log (not necessarily normalised).
    if ((ps is not None) and np.any(np.isnan(ps))) or ((logps is not None) and np.any(np.isnan(logps))):
        raise ValueError('my.stat.sample_discrete: Input contains NaN.')

    if (logps is not None):
        if(not np.any(logps == np.inf)): 
            logps = logps - np.max(logps)
        ps = np.exp(logps) # Use logps if given.

    inf_ind = np.isinf(ps)
    ps_max = np.max(ps)

    if np.any(inf_ind):
        return rng.choice(np.nonzero(inf_ind)[0], size=n) # Sample randomly from points with infinite weight
    else:
        if ps_max <= 0:
            raise ValueError('my.stat.sample_discrete: Invalid ps or logps.')
        else:
            ps = ps / ps_max
            ps = ps / np.sum(ps)
            return rng.choice(ps.size, size=n, p=ps)


def _trim_xs(grads: np.ndarray, xs: np.ndarray, low, high): # Called in ARS
    if np.any(grads == np.inf):
        pos1 = np.nonzero(grads == np.inf)[0][-1] # Remove inf gradients on the left
        low = xs[pos1]
        grads = grads[pos1 + 1 : ]
        xs = xs[pos1 + 1 : ]
    
    if np.any(grads == -np.inf):
        pos2 = np.nonzero(grads == -np.inf)[0][0] # Remove -inf gradients on the right
        high = xs[pos2]
        grads = grads[ : pos2]
        xs = xs[ : pos2]

    current_pos = 0
    next_pos = 1
    ind = np.full(xs.shape, False)
    ind[0] = True
    ind[-1] = True

    while next_pos < xs.size - 1: # Remove xs where gradients are too close
        while (np.abs(grads[next_pos] - grads[current_pos]) < 1e-8) and (next_pos < xs.size - 1):
            next_pos = next_pos + 1
        current_pos = next_pos
        ind[current_pos] = True
        next_pos = current_pos + 1
    grads = grads[ind]
    xs = xs[ind]
            
    if (xs.size > 2) and (np.abs(grads[-1] - grads[-2]) < 1e-8):
        grads = np.delete(grads, -2)
        xs = np.delete(xs, -2)

    if (xs.size < 1) or ((grads[0] < 0) and (low == -np.inf)) or ((grads[-1] > 0) and (high == np.inf)):
        breakpoint()

    return grads, xs, low, high


def _get_bounds_from_xs(vals: np.ndarray, grads: np.ndarray, xs: np.ndarray, low, high): # Called in ARS
    gxs = grads * xs

    bounds = np.full(xs.size + 1, np.nan)
    bounds[0] = low
    bounds[-1] = high
    # Intersections of tangent lines.
    bounds[1 : -1] = ((vals[1 : ] - vals[0 : -1]) + (gxs[0 : -1] - gxs[1 : ])) / (grads[0 : -1] - grads[1 : ])
    if np.any(np.isnan(bounds)):
        breakpoint()

    # Make sure bounds are between xs (necessary due to numerical inaccuracy)
    bounds[1 : -1] = np.minimum(0.9999 * xs[1 : ] + 0.0001 * xs[0 : -1], np.maximum(0.9999 * xs[0 : -1] + 0.0001 * xs[1 : ], bounds[1 : -1]))

    return bounds


def _sample_piecewise_exponential(grads: np.ndarray, vals: np.ndarray, bounds: np.ndarray, xs: np.ndarray, n: int): # Draw n samples from a piecewise-exponential distribution.
    maxborderval = np.maximum(np.max(vals + grads * (bounds[ : -1] - xs)), vals[-1] + grads[-1] * (bounds[-1] - xs[-1]))
    vals = vals - maxborderval
    lb = bounds[ : -1]
    ub = bounds[1 : ]

    ind1 = (grads < 0)
    ind2 = (grads == 0)
    ind3 = (grads > 0)

    logps = np.full(grads.shape, np.nan) # Weights of the exponential segments
    logps[ind1] = (grads[ind1] * (lb[ind1] - xs[ind1]) + vals[ind1]) + np.log(1 - np.exp(grads[ind1] * (ub[ind1] - lb[ind1]))) - np.log(-grads[ind1])
    logps[ind2] = vals[ind2] + np.log(ub[ind2] - lb[ind2])
    logps[ind3] = (grads[ind3] * (ub[ind3] - xs[ind3]) + vals[ind3]) + np.log(1 - np.exp(grads[ind3] * (lb[ind3] - ub[ind3]))) - np.log(grads[ind3])

    # Pick up NaN caused by grad = Inf or -Inf
    logps[np.logical_or(grads == np.inf, grads == -np.inf)] = 0 
    if np.any(np.isnan(logps)):
        breakpoint()

    seg_samples = sample_discrete(logps=logps, n=n)
    u = rng.random(n)
    samples = np.full(n, np.nan)

    grad_seg = grads[seg_samples]
    lb_seg = lb[seg_samples]
    ub_seg = ub[seg_samples]

    ind1 = (grad_seg == 0)
    ind2 = np.logical_not(ind1)

    samples[ind1] = u[ind1] * ub_seg[ind1] + (1 - u[ind1]) * lb_seg[ind1]
    samples[ind2] = Eadd(np.log(u[ind2]) + grad_seg[ind2] * ub_seg[ind2], np.log(1 - u[ind2]) + grad_seg[ind2] * lb_seg[ind2]) / grad_seg[ind2]

    # Make sure samples are between corresponding upper and lower boundaries (necessary due to numerical inaccuracy)
    samples = np.minimum(ub_seg, np.maximum(lb_seg, samples))
    if np.any(np.isnan(samples)):
        breakpoint()

    return samples, seg_samples


def ARS(logf: Callable, dlogf_dx: Callable, xs: np.ndarray, low, high, n=None): # Use adaptive rejection sampling to draw n samples from a 1-D log-concave distribution
    # logf: f is a multiple of the pdf of the r.v.
    # [low, high]: range to sample from
    # xs: a strictly increasing array on (low, high), where dlogf_dx(xs[0]) > 0 if low = -Inf and dlogf_dx(xs[-1]) < 0 if high = Inf. These will be the points where the logf is tangent to its piecewise-linear upper-boundary.
    n_drawn = 0
    if n is None:
        n = 1
    samples_ans = np.full(n, np.nan)

    while n_drawn < n:
        grads = dlogf_dx(xs)
        if (xs.size < 1) or ((grads[0] < 0) and (low == -np.inf)) or ((grads[-1] > 0) and (high == np.inf)):
            raise ValueError('my.stat.ARS: Invalid input xs.')
        grads, xs, low, high = _trim_xs(grads, xs, low, high)

        vals = logf(xs)
        max_val = np.max(vals)
        vals = vals - max_val
        bounds = _get_bounds_from_xs(vals, grads, xs, low, high)

        samples, seg_samples = _sample_piecewise_exponential(grads, vals, bounds, xs, n - n_drawn) # Sample from piecewise-exponential proposal distribution.

        u = rng.random(n - n_drawn)
        logu = np.log(u)

        logf_at_samples = logf(samples) - max_val # Since vals have been reduced by max_val

        grads_at_samples = grads[seg_samples]
        xs_near_samples = xs[seg_samples]
        vals_at_xs_near_samples = vals[seg_samples]
        log_piecewise_exp_at_samples = grads_at_samples * (samples - xs_near_samples) + vals_at_xs_near_samples

        accept = (logu <= (logf_at_samples - log_piecewise_exp_at_samples)) # P_f / P_proposal is the probability of acceptance.
        samples_accepted = samples[accept]
        num_accept = np.sum(accept)
        samples_ans[n_drawn : np.minimum(n_drawn + num_accept, n)] = samples_accepted[ : np.minimum(num_accept, n - n_drawn)]

        n_drawn = n_drawn + num_accept
        xs = np.sort(np.concatenate((xs, samples)))

    if samples_ans.size > 1:
        return samples_ans
    else:
        return sample_ans[0]


def _test_ARS(logf: Callable, dlogf_dx: Callable, xs: np.ndarray, low, high): # test function for ARS
    n = int(1e5)

    samples = ARS(logf, dlogf_dx, xs, low, high, n)

    l = np.min(samples)
    r = np.max(samples)
    eps = (r - l) / 50

    const = integrate.quad(comp(np.exp, logf), l - eps, r + eps)[0]

    x_vec = np.arange(l, r + (r - l) / 1000, (r - l) / 1000)
    y_vec = np.exp(logf(x_vec)) / const

    f1, ax1 = plt.subplots()

    ax1.plot(x_vec, y_vec, 'g-')

    ax1.hist(samples, np.arange(l, r + eps, eps), density=True)

    samples2 = np.full(n, np.nan)

    for it in range(n):
        samples2[it] = ARS(logf, dlogf_dx, xs, low, high, 1)

    f2, ax2 = plt.subplots()

    ax2.plot(x_vec, y_vec, 'g-')

    ax2.hist(samples2, np.arange(l, r + eps, eps), density=True)
