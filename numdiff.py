"""
Numerical derivatives with a twist:

* works for arrays;
* compute all derivatives of order 2, 4 simultaneously;
* accuracy is O(h^2).

"""

import numpy as np

def dx2(f, x, h):
    der = (f(x + h) + f(x - h) - 2. * f(x)) / h**2
    nevals = 3 * x.shape[0]
    return der, nevals

def hessian(f, x, h):
    def dv2(v):
        return (f(x + v) + f(x - v) - tfx) / h**2
    N, d = x.shape
    hess = np.empty((N, d, d))
    tfx = 2. * f(x)
    iden = np.eye(d)
    for i in range(d):
        v = h * iden[i, :]
        hess[:, i, i] = dv2(v)
    for i in range(d):
        for j in range(i):  # j<i
            v = h * (iden[i, :] + iden[j, :])
            hess[:, i, j] = 0.5 * (dv2(v) -hess[:, i, i] - hess[:, j, j])
            hess[:, j, i] = hess[:, i, j]
    nevals = N * (1 +  d * (d + 1))
    return hess, nevals

co = np.ones((3, 2))
co[1, 0] = 2.
co[2, 1] = 2. 
Ainv = np.array([[10., -1., -1.], 
                 [-6., 1., 0.5],
                 [-6., 0.5, 1.]]) / 12.

def dx4(f, x, h):
    deriv = (f(x + 2. * h) + f(x - 2. * h) - 4. * f(x + h) - 4. * f(x - h)
             + 6. * f(x)) / h**4
    nevals = 5 * x.shape[0]
    return deriv, nevals

def deriv4(f, x, h):
    def dv4(v):
        hv = h * v
        return (1. / h**4) * (f(x + 2. * hv) + f(x - 2. * hv) 
                              - 4. * f(x + hv) - 4. * f(x - hv) + sfx)
    N, d = x.shape
    if d != 2:
        raise ValueError('deriv4: only dim=2 implemented')
    D = np.empty((N, d, d, d, d))
    sfx = 6. * f(x)
    nevals = N
    iden = np.eye(d)
    for i in range(d):
        D[:, i, i, i, i] = dv4(iden[i, :])
        nevals += 4 * N
    for i in range(d): 
        for j in range(i):
            dv = np.empty((N, 3))
            for k in range(3):
                v = co[k, 0] * iden[i, :] + co[k, 1] * iden[j, :]
                dv[:, k] = (dv4(v) - co[k, 0]**4 * D[:, i, i, i, i] 
                            - co[k, 1]**4 * D[:, j, j, j, j])
                nevals += 4 * N
                w = dv @ Ainv.T
                D[:, i, i, j, j] = w[:, 0]
                D[:, i, i, i, j] = w[:, 1]
                D[:, i, j, j, j] = w[:, 2]
    # TODO other derivatives 
    return D, nevals
