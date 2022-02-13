"""
Nested stratified estimators.


"""

from collections import Counter
import itertools
import functools

from findiff import FinDiff
import numpy as np
from numpy import random
from scipy import linalg
from scipy.special import comb

import numdiff

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

def order2_correct(c, u, k, d, phi, deriv):
    if deriv is None:
        h = 0.4 / k
        if d == 1:
            H, nevals = numdiff.dx2(phi, c, h)
        else:
            H, nevals = numdiff.hessian(phi, c, h)
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
        h = 0.2 / k  # TODO
        if d == 1:
            D, nevals = numdiff.dx4(phi, c, h)
        else:
            D, nevals = numdiff.deriv4(phi, c, h)
    else:
        D = deriv(4, c)
        nevals = N * comb(d + 3, 4, exact=True)
        # based on bijection i<=j<=k<=l <=> i<j+1<k+2<l+3
    if d == 1:
        order4_term = D * u**4
        expect_order4 = D * unif_mom[4] / k**4
    else:
        order4_term = 0.
        expect_order4 = 0.
        nevals = 0
        for g in range(d):
            for h in range(g+1):
                for i in range(h+1):
                    for j in range(i+1):
                        ct = Counter([g, h, i, j])
                        uijkl = u[:, g] * u[:, h] * u[:, i] * u[:, j]
                        mu = mult(ct, 4)
                        order4_term += mu * D[:, g, h, i, j] * uijkl
                        # nevals += N
                        if g==h==i==j:
                            expect_order4 += (D[:, i, i, i, i] 
                                              * unif_mom[4] / k**4)
                        if all([v % 2 == 0 for v in ct.values()]):
                            expect_order4 += (mu * D[:, g, h, i, j] *
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

def deprecated_estimate(k, d, order=1, phi=None, deriv=None):
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

def estimate(k, d, order=1, phi=None):
    N = k ** d
    h = 1. / k
    hh = 0.5 * h # half h
    gr = np.linspace(hh, 1. - hh, k)
    lg = [gr] * d
    mg = np.meshgrid(*lg, indexing='ij')
    c = np.vstack([a.flatten() for a in mg]).T
    u =  h * np.random.rand(N, d) - hh
    est = np.mean(phi(c + u))
    if order == 1:
        return est
    est = 0.5 * (est + np.mean(phi(c - u)))
    if order == 2:
        return est
    fmg = np.reshape(phi(c), tuple([k] * d))
    cv2 = 0.
    a = order - 2
    for i in range(d):
        op = FinDiff(i, h, 2, acc=a)
        dx2 = op(fmg).flatten()
        cv2 += np.mean(dx2 * (u[:, i]**2 - h**2 * unif_mom[2]))
        for j in range(i):
            op = FinDiff((i, h), (j, h), acc=a)
            dxdy = op(fmg).flatten()
            cv2 += 2. * np.mean(dxdy * u[:, i] * u[:, j])
    est = est - 0.5 * cv2
    if order == 4:
        return est
    cv4 = 0.
    a = order - 4
    for i in range(d):
        op = FinDiff(i, h, 4, acc=a)
        dx4 = op(fmg).flatten()
        cv4 += np.mean(dx4 * (u[:, i]**4 - h**4 * unif_mom[4]))
        for j in range(i):
            op = FinDiff((i, h, 3), (j, h, 1), acc=a)
            dx3dy = op(fmg).flatten()
            cv4 += 4. * np.mean(dx3dy * u[:, i]**3 * u[:, j])
            op = FinDiff((i, h, 1), (j, h, 3), acc=a)
            dxdy3 = op(fmg).flatten()
            cv4 += 4. * np.mean(dxdy3 * u[:, i] * u[:, j]**3)
            op = FinDiff((i, h, 2), (j, h, 2), acc=a)
            dx2dy2 = op(fmg).flatten()
            cv4 += 6. * np.mean(dx2dy2 * (u[:, i]**2 * u[:, j]**2 - h**4 * unif_mom[2]**2))
            for k in range(j):
                op = FinDiff((i, h, 2), (j, h, 1), (k, h, 1), acc=a)
                dx2dydz = op(fmg).flatten()
                cv4 += 12. * np.mean(dx2dydz * u[:, i]**2 * u[:, j] * u[:, k])
                op = FinDiff((i, h, 1), (j, h, 2), (k, h, 1), acc=a)
                dxdy2dz = op(fmg).flatten()
                cv4 += 12. * np.mean(dxdy2dz * u[:, i] * u[:, j]**2 * u[:, k])
                op = FinDiff((i, h, 1), (j, h, 1), (k, h, 2), acc=a)
                dxdydz2 = op(fmg).flatten()
                cv4 += 12. * np.mean(dxdydz2 * u[:, i] * u[:, j] * u[:, k]**2)
                for l in range(k):
                    op = FinDiff((i, h), (j, h), (k, h), (l, h), acc=a)
                    dxdydzdt = op(fmg).flatten()
                    cv4 += 24. * np.mean(dxdydzdt * u[:, i] * u[:, j] * u[:, k] * u[:, l])
    est = est - cv4 / 24.
    return est

def estimate_with_nevals(k, d, order=1, phi=None):
    est = estimate(k, d, order=order, phi=phi)
    nevals = k**d * min(order, 3)
    return {'est': est, 'nevals': nevals}

def diff_cv(ns, fmg, u, h, a):
    # TODO
    lns = [(i, h, k) for i, k in ns]
    op = FinDiff(*lns, acc=a)
    deriv = op(fmg).flatten()
    for i, k in ns:
        deriv *= u[:, i]**k
    return np.mean(deriv)


if __name__ == '__main__':
    out = estimate(100, 2, order=6, phi=phi, deriv=None)
