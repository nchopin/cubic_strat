import numpy as np

ident = 'dick4D'
d = 4
title_plot = r'$\varphi(x, y, z, t) = t  e^{xyzt}$'
# mat_file_name = 'Mathieu/f4_d2_a%i.csv'
true_val = np.exp(1.) - (8. / 3.)
orders = [1, 2, 3, 4, 5, 6]
min_neval = 20
max_neval = 10**7
nks = 10
nreps = 500

def phi(u):
    return u[:, 1] * u[:, 2]**2 * u[:, 3]**3 * np.exp(np.prod(u, axis=-1))

