import numpy as np
from scipy.special import factorial

ident = 'dick6D'
d = 6
title_plot = r'$\varphi(x, y, z, t, u, v) = y z^2 t^3 u^4 v^5 e^{xyztuv}$'
mat_folder = 'simuDick/d6'
true_val = np.exp(1.) - np.sum(1. / factorial(np.arange(6)))
orders = [1, 2, 4, 6, 8]
min_neval = 100
max_neval = 2 * 10**7
nks = 20
nreps = 50

def phi(u):
    pu = 1.
    for i in range(1, d):
        pu = pu * u[:, i]**i
    return pu * np.exp(np.prod(u, axis=-1))

