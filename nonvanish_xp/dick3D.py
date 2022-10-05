import numpy as np

ident = 'dick3D'
d = 3
title_plot = r'$f_3(u) = u_2 u_3^2 e^{u_1 u_2 u_3}$'
true_val = np.exp(1.) - (5. / 2.)
orders = [1, 2, 4, 6, 8]
min_neval = 10
max_neval = 10**7
nks = 20
nreps = 50

def phi(u):
    return u[:, 1] * u[:, 2]**2 * np.exp(np.prod(u, axis=-1))

