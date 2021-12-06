"""
Nested stratified estimators.


"""

from collections import Counter
import itertools
import functools

import numpy as np
from numpy import random
from scipy import linalg

MAX_SIZE = 10**5
INT_TYPE = np.int32  # SIGNED INT, YOU DIM-WIT!!!
# int type for arrays of indices; to save some space we do not use uint64

pos_lambdas = list(range(1, 30, 2))  # 1, 3, 5, ...
lambdas_nested = [(i, -i) for i in pos_lambdas]
lambdas = list(itertools.chain(*lambdas_nested))  # 1, -1, 3, -3, ...

# moments of Unif[-1/2, 1/2]
unif_mom = {k: 1. / (2**k * (k + 1)) for k in range(2, 11, 2)}

def fact(k):
    """Factorial of k for k > 1.
    """
    return np.prod(np.arange(2, k + 1))

def vander_system(k):
    """Solution of the Vandermonde system that gives the coefficients of the
    nested strat estimates.
    """
    A = np.vander(lambdas[:k], increasing=True)
    b = np.zeros(k); b[0] = 1.
    return linalg.solve(A.T, b)

def cell_indices(k, d, m=0):
    """Indices of the cells.

    For d=2, k=10, m=0, generate (0, 0), ..., (0, 9), (1, 0), ..., (9, 9) which
    represent the k^d cells that partition [0,1]^d. We may also add m extra
    cells on both sides in each direction (m may be negative as well); so for
    m=-2, and again k=10, d=2:
        (2, 2), ..., (7, 7)

    Parameters
    ----------
    k   int
        size of partition is k^d
    d   int
        dimension
    m   int (default: 0)
        add m extra cells on both sides (in all the dimensions)

    Returns
    -------
    a (N,) or (N, d) (if d>1) numpy int array, where N=(k+2m)^d
    """
    one_dim = np.arange(-m, k + m)
    if d == 1:
        return one_dim
    mesh = np.meshgrid(*[one_dim for _ in range(d)])
    out = np.empty(((k + 2*m)**d, d), INT_TYPE)
    for i in range(d):
        out[:, i] = mesh[i].flatten()
    return out

def idx_it(k, d, m=0, max_size=MAX_SIZE):
    """Iterator that returns the centers in bunches of a certain size.
    """
    if d == 1:
        yield from idx_it_univariate(k, m, max_size)
    else:
        yield from idx_it_multivariate(k, d, m, max_size)

def idx_it_univariate(k, m, max_size):
    gs = k + 2 * m  # grid size
    l, r = divmod(gs, max_size)
    if l == 0:
        yield cell_indices(k, 1, m=m)
    else:
        left = -m
        for i in range(l):
            yield np.arange(left, left + max_size)
            left += max_size
        if r > 0:
            yield np.arange(left, left + r)

def idx_it_multivariate(k, d, m, max_size):
    """Iterator that yields bunch of indices (to avoid memory overflow).
    """
    gs = k + 2 * m
    s = int(np.floor(np.log(max_size) / np.log(gs)))
    if d <= s:
        yield cell_indices(k, d)
    else:
        idx = np.empty((gs**s, d), INT_TYPE)
        if s == 1:
            idx[:, 0] = cell_indices(k, s, m=m)
        else:
            idx[:, :s] = cell_indices(k, s, m=m)
        idx2 = cell_indices(k, d - s, m=m)
        for i in range(idx2.shape[0]):
            idx[:, s:] = idx2[i]
            yield idx

def centers(idx, k):
    """Centers of the cells indexed by idx.
    """
    return (idx + 0.5) / k

def which_layer(idx, k):
    if idx.ndim == 1:
        return np.minimum(idx, k - 1 - idx)
    else:
        low = np.min(idx, axis=1)
        high = np.min(k - 1 - idx, axis=1)
        return np.minimum(low, high)

def unif(c, k):
    return random.uniform(-0.5 / k, 0.5 / k, c.shape)

# Vanishing functions
#####################

def zero_padding(func):
    """Decorator that extends (to 0) the definition of function outside of [0,1]^d.
    """
    @functools.wraps(func)
    def zf(u):
        N = u.shape[0]
        out = np.zeros(N) # 0 outside [0, 1]^d
        inside = np.logical_and(u < 1., u > 0)
        if inside.ndim > 1:
            inside = np.all(inside, axis=1)
        out[inside] = func(u[inside])
        neval = inside.sum()
        return out, neval
    return zf

def partial_sums(k, d, order=1, phi=None):
    """Computes sums of phi(c + lambda[i] * U) for i=1, ..., l
    """
    psums = np.zeros(order)
    nevals = np.zeros(order, np.uint64)  # uint64: nb evals may be large
    m = (order - 1) // 2
    for idx in idx_it(k, d, m=m):
        c = centers(idx, k)
        u = unif(c, k)
        for i in range(order):
            phin, ne = phi(c + lambdas[i] * u)
            psums[i] += phin.sum()
            nevals[i] += ne
    return psums, nevals

def vanish_estimates(k=10, d=2, order=1, phi=None):
    """Compute all estimates up to a given order for a vanishing function.
    """
    phi = zero_padding(phi)
    psums, nevals = partial_sums(k, d, order=order, phi=phi)
    N = k**d
    pmeans = psums / N
    ests = np.empty(order)
    for i in range(order):
        ests[i] = np.dot(vander_system(i + 1), pmeans[:(i + 1)])
    return ests, np.cumsum(nevals)

# General (non-vanishing) functions
###################################

def numdx2(phi, x, i, h):
    if x.ndim == 1:
        N = x.shape[0]
        hv = h
    else:
        N, d = x.shape
        v = np.eye(d)[i]
        hv = h * v
    dx2 = (phi(x + hv) + phi(x - hv) - 2. * phi(x)) / h**2
    nevals = 4 * N
    return dx2, nevals

def numdxdy(phi, x, i, j, h):
    N, d = x.shape
    ident = np.eye(d)
    v = ident[i] + ident[j]
    w = ident[i] - ident[j]
    dxdy = (phi(x + h * v) + phi(x - h * v) - phi(x + h * w) - phi(x - h *w)
           ) / (4 * h**2)
    nevals = 4 * N
    return dxdy, nevals

def order2_correct(c, u, k, d, phi, deriv):
    if deriv is None:
        h = 0.3 / k  # TODO
        if d == 1:
            H, nevals = numdx2(phi, c, 0, h)
        else:
            nevals = 0
            N, d = c.shape
            H = np.zeros((N, d, d))
            for i in range(d):
                H[:, i, i], ne = numdx2(phi, c, i, h)
                nevals += ne
                for j in range(i):
                    H[:, i, j], ne = numdxdy(phi, c, i, j, h)
                    nevals += ne
    else:
        H = deriv(2, c)
        nevals = c.shape[0] * ((d * (d + 1)) // 2)
    if d == 1:
        trace, quad = H, H * u**2
    else:
        quad, trace = 0., 0.
        for i in range(d):
            trace += H[:, i, i]
            quad += H[:, i, i] * u[:, i]**2
            for j in range(i):
                quad += 2. * H[:, i, j] * u[:, i] * u[:, j]
    expect_quad = trace * (unif_mom[2] / k**2)
    correct = -0.5 * (quad - expect_quad)
    return correct, nevals

def mult(ct, d):
    return fact(d) // np.prod([fact(v) for v in ct.values()])

def order4_correct(c, u, k, d, phi, deriv):
    N = c.shape[0]
    if deriv is None:
        raise NotImplementedError
    D = deriv(4, c)
    if d == 1:
        order4_term = D * u**4
        expect_order4 = D * unif_mom[4] / k**4
        nevals = N
    else:
        order4_term = 0.
        expect_order4 = 0.
        nevals = 0
        for i in range(d):
            for j in range(i+1):
                for k in range(j+1):
                    for l in range(k+1):
                        ct = collections.Counter([i, j, k, l])
                        uijkl = u[i, :] * u[j, :] * u[k, :] * u[l, :]
                        mu = mult(ct, d)
                        order4_term += mu * D[:, i, j, k, l] * uijkl
                        nevals += N
                        if i==j==k==l:
                            expect_order4 += (D[:, i, i, i, i] 
                                              * unif_mom[4] / k**4)
                        if all([v % 2 == 0 for v in ct.values()]):
                            expect_order4 += (mu * D[:, i, j, k, l] *
                                                   unif_mom[2]**2 / k**4)
    correct = (expect_order4 - order4_term) / (2*3*4)
    return correct, nevals

def control_variate(i, c, u, k, d, phi, deriv):
    if i == 2:
        return order2_correct(c, u, k, d, phi, deriv)
    elif i==4:
        return order4_correct(c, u, k, d, phi, deriv)
    else:
        raise ValueError('not implemented')

def local_est(c, k, d, order, phi, deriv):
    """unbiased estimate based on derivatives (Zc in my notes)
    """
    u = unif(c, k)
    est =  0.5 * (phi(c + u) + phi(c - u))
    nevals = 2 * c.shape[0]
    for i in range(2, order, 2):
        cv, ne = control_variate(i, c, u, k, d, phi, deriv)
        est += cv; nevals += ne
    return est, nevals

def estimate(k, d, order=1, phi=None, deriv=None):
    inner, outer = 0., 0.
    alphas = vander_system(order)
    m = (order - 1) // 2
    nevals = 0
    for idx in idx_it(k, d):
        c = centers(idx, k)
        lay = which_layer(idx, k)
        c1 = c[lay >= m] if m > 0 else c
        u = unif(c1, k)
        for a, l in zip(alphas, lambdas):
            inner += np.sum(a * phi(c1 + l * u))
        nevals += order * c1.shape[0]
        if order >= 3:
            outer_layers = lay < 2 * m
            idx2 = idx[outer_layers]
            c2 = c[outer_layers]
            lay2 = lay[outer_layers]
            coeffs = np.ones(c2.shape[0])
            coeffs[lay2 >= m] -= (alphas[0] + alphas[1])
            for i in range(2, order):
                l = lambdas[i]; a = alphas[i]
                mi = (abs(l) - 1) // 2
                for incr in cell_indices(1, d, m=mi):
                    coeffs[which_layer(idx2 + incr, k) >= m] -= a / (abs(l)**d)
            le, nev = local_est(c2, k, d, order, phi, deriv)
            outer += np.sum(coeffs * le)
            nevals += nev
    est = (outer + inner) / k**d
    return {'est': est, 'inner_term': inner, 'outer_term': outer, 
            'nevals': nevals}

def phi(u):
    return np.sum(np.exp(u), axis=1)

def deriv(k, u):
    N, d = u.shape
    if k==2:
        H = np.zeros((N, d, d))
        for i in range(d):
            H[:, i, i] = np.exp(u[:, i])
        return H
    elif k==4:
        D = np.zeros((N, d, d, d, d))
        for i in range(d):
            D[:, i, i, i, i] = np.exp(u[:, i])
        return D

if __name__ == '__main__':
    out = estimate(100, 2, order=6, phi=phi, deriv=None)
