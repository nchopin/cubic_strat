"""
Nested stratified estimators.


"""

from collections import Counter, defaultdict
import functools
import itertools as itt
import time

from findiff import FinDiff
import numpy as np
from numpy import random
from scipy import linalg
from scipy.special import comb

from cubic_strat import numdiff

class StratError(Exception):
    pass

class TooSmallkError(StratError):
    pass

MAX_SIZE = 10**5
INT_TYPE = np.int32  # SIGNED INT, YOU DIM-WIT!!!
# int type for arrays of indices; to save some space we do not use uint64

pos_lambdas = list(range(1, 30, 2))  # 1, 3, 5, ...
lambdas_nested = [(i, -i) for i in pos_lambdas]
lambdas = list(itt.chain(*lambdas_nested))  # 1, -1, 3, -3, ...

# moments of Unif[-1/2, 1/2]
unif_mom = {k: 1. / (2**k * (k + 1)) for k in range(2, 11, 2)}

def timer(func):
    """Decorator that adds cpu time to the output.

    Function must return a dict.
    """
    def timed_func(*args, **kwargs):
        starting_time = time.perf_counter()
        out = func(*args, **kwargs)
        out['cpu'] = time.perf_counter() - starting_time
        return out

    return timed_func

def cycles(*l):
    """List of all the circular shifts of a given list.

    > cycles(1, 2, 3)
    [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    """
    return [list(np.roll(l, k)) for k in range(len(l))]

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
        nevals = inside.sum()
        return out, nevals
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

@timer
def vanish_estimates(k, d, order=1, phi=None):
    """Compute all estimates up to a given order for a vanishing function.
    """
    phi = zero_padding(phi)
    psums, nevals = partial_sums(k, d, order=order, phi=phi)
    N = k**d
    pmeans = psums / N
    ests = np.empty(order)
    for i in range(order):
        ests[i] = np.dot(vander_system(i + 1), pmeans[:(i + 1)])
    return {'estimates': ests, 'nevals': np.cumsum(nevals)}

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
        raise NotImplementedError()

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

################################
### New (non-vanishing) estimate

class cv_generator:
    def __init__(self, u, fc, h):
        self.u = u
        self.fc = fc
        self.h = h

    def cv(self, *ns, acc=2):
        lns = [(i, self.h, k) for i, k in ns]
        op = FinDiff(*lns, acc=acc)
        deriv = op(self.fc).flatten()
        pu, avg = 1., 1.
        all_even = all(k % 2 == 0 for i, k in ns)
        for i, k in ns:
            pu = pu * self.u[:, i]**k
            if all_even:
                avg = avg * self.h**k * unif_mom[k]
        if not(all_even):
            avg = 0.
        return np.mean(deriv * (pu - avg))

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
    if k < (3 * order) // 2 - 1:
        raise TooSmallkError('k must be at least (3/2) * order - 1')
    if order % 2 != 0:
        raise NotImplementedError('Order must be even (or 1)')
    fmg = np.reshape(phi(c), tuple([k] * d))
    cvgen =  cv_generator(u, fmg, h)
    cvs = defaultdict(float)  # default to 0. 
    a = order - 2
    for i in range(d):
        cvs[2] += cvgen.cv((i, 2), acc=a)
        for j in range(i):
            cvs[2] += 2. * cvgen.cv((i, 1), (j, 1), acc=a)
    est = est - 0.5 * cvs[2]
    if order == 4:
        return est
    a = order - 4
    for i in range(d):
        cvs[4] += cvgen.cv((i, 4), acc=a)
        for j in range(i):
            cvs[4] += 4. * cvgen.cv((i, 3), (j, 1), acc=a)
            cvs[4] += 4. * cvgen.cv((i, 1), (j, 3), acc=a)
            cvs[4] += 6. * cvgen.cv((i, 2), (j, 2), acc=a)
            for k in range(j):
                for s, t, u in cycles(i, j, k):
                    cvs[4] += 12. * cvgen.cv((s, 2), (t, 1), (u, 1), acc=a)
                for l in range(k):
                    cvs[4] += 24. * cvgen.cv((i, 1), (j, 1), (k, 1), (l, 1),
                                             acc=a)
    est = est - cvs[4] / 24.
    if order == 6:
        return est
    a = order - 6
    for i in range(d):
        cvs[6] += cvgen.cv((i, 6), acc=a)
        for j in range(i):
            for s, t in cycles(i, j):
                cvs[6] += 6. * cvgen.cv((s, 5), (t, 1), acc=a)
                cvs[6] += 15. * cvgen.cv((s, 4), (t, 2), acc=a)
            cvs[6] += 20. * cvgen.cv((i, 3), (j, 3), acc=a)
            for k in range(j):
                for s, t, u in cycles(i, j, k):
                    cvs[6] += 30. * cvgen.cv((s, 4), (t, 1), (u, 1))
                for s, t, u in itt.permutations([i, j, k]):
                    cvs[6] += 60. * cvgen.cv((s, 3), (t, 2), (u, 1))
                cvs[6] += 90. * cvgen.cv((i, 2), (j, 2), (k, 2), acc=a)
                for l in range(k):
                    for s, t, u, v in cycles(i, j, k, l):
                        cvs[6] += 120. * cvgen.cv((s, 3), (t, 1), (u, 1), 
                                                  (v, 1), acc=a)
                    for s, t, u, v in [(i, j, k, l), (i, k, j, l), (i, l, k, j),
                                       (j, k, i, l), (j, l, i, k), (k, l, i, j)]:
                        cvs[6] += 180. * cvgen.cv((s, 2), (t, 2), (u, 1), (v, 1))
                    for m in range(l):
                        for s, t, u, v, w in cycles(i, j, k, l, m):
                            cvs[6] += 360. * cvgen.cv((s, 2), (t, 1), (u, 1), (v, 1),
                                                      (w, 1), acc=a)
                        for n in range(m):
                            cvs[6] += fact(6) * cvgen.cv((i, 1), (j, 1), (k, 1), 
                                                         (l, 1), (m, 1), (n, 1),
                                                        acc=a)
    est = est - cvs[6] / fact(6)
    if order == 8:
        return est
    a = order - 8
    for i in range(d):
        cvs[8] += cvgen.cv((i, 8), acc=a)
        for j in range(i):
            for s, t in cycles(i, j):
                cvs[8] += 8. * cvgen.cv((s, 7), (t, 1), acc=a)
                cvs[8] += 28. * cvgen.cv((s, 6), (t, 2), acc=a)
                cvs[8] += 56. * cvgen.cv((s, 5), (t, 3), acc=a)
            cvs[8] += 70. * cvgen.cv((i, 4), (j, 4), acc=a)
            for k in range(j):
                raise NotImplementedError('Order 10 not implemented for dim>2')
    est = est - cvs[8] / fact(8)
    if order == 10:
        return est
    else:
        raise NotImplementedError('Orders above 10 not implemented')

@timer
def estimate_with_nevals(k, d, order=1, phi=None):
    try:
        est = estimate(k, d, order=order, phi=phi)
    except TooSmallkError:
        est = None
    nevals = k**d * min(order, 3)
    return {'est': est, 'nevals': nevals}


if __name__ == '__main__':
    out = estimate(100, 2, order=6, phi=phi, deriv=None)
