import numpy as np

ident = 'dick2D'
d = 2
title_plot = r'$\varphi(x, y) = y  e^{xy}$'
mat_file_name = 'Mathieu/f4_d2_a%i.csv'
true_val = np.exp(1.) - 2.
orders = [1, 2, 3, 4]
min_neval = 20
max_neval = 10**6
nks = 10
nreps = 50
deriv_methods = ['exact']

def xpy(u):
    return np.prod(u, axis=u.ndim - 1)

def phi(u):
    return u[:, 1] * np.exp(xpy(u))

def deriv(k, u):
    if k==2:
        H = np.empty((u.shape[0], 2, 2))
        xy = xpy(u)
        exy = np.exp(xy)
        H[:, 0, 0] = u[:, 1]**3 * exy
        H[:, 0, 1] = u[:, 1] * (2. + xy) * exy
        H[:, 1, 0] = H[:, 0, 1]
        H[:, 1, 1] = u[:, 0] * (2. + xy) * exy
        return H
    elif k==4:
        D = np.empty((u.shape[0], 2, 2, 2, 2))
        x, y = u[:, 0], u[:, 1]
        x2 = x * x
        y2 = y * y
        xy = xpy(u)
        exy = np.exp(xy)
        D[:, 0, 0, 0, 0] = exy * y**5
        dx3dy = exy * (x * y**4 + 4. * y**3)
        D[:, 0, 0, 0, 1] = dx3dy
        D[:, 0, 0, 1, 0] = dx3dy 
        D[:, 0, 1, 0, 0] = dx3dy
        D[:, 1, 0, 0, 0] = dx3dy 
        dx2dy2 = exy * (x2 * y**3 + 6. * x * y2 + 6. * y)
        D[:, 0, 0, 1, 1] = dx2dy2
        D[:, 0, 1, 0, 1] = dx2dy2
        D[:, 1, 0, 0, 1] = dx2dy2
        D[:, 0, 1, 1, 0] = dx2dy2
        D[:, 1, 0, 1, 0] = dx2dy2
        D[:, 1, 1, 0, 0] = dx2dy2
        dxdy3 = exy * x * (x2 * y2 + 6. * xy + 6.)
        D[:, 1, 1, 1, 0] = dx3dy
        D[:, 1, 1, 0, 1] = dx3dy 
        D[:, 1, 0, 1, 1] = dx3dy
        D[:, 0, 1, 1, 1] = dx3dy 
        D[:, 1, 1, 1, 1] = exy * x**3 * (xy + 4.)
        return D
    else:
        raise ValueError('not implemented')
