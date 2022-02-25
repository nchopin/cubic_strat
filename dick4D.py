import numpy as np

ident = 'dick4D'
d = 4
title_plot = r'$\varphi(x, y, z, t) = y z^2 t^3  e^{xyzt}$'
mat_folder = 'simuDick/d4'
true_val = np.exp(1.) - (8. / 3.)
orders = [1, 2, 4, 6, 8]
min_neval = 10
max_neval = 10**7
nks = 20
nreps = 50

def phi(u):
    return u[:, 1] * u[:, 2]**2 * u[:, 3]**3 * np.exp(np.prod(u, axis=-1))

