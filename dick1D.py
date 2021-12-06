import numpy as np

ident = 'dick1D'
d = 1
title_plot = r'$\varphi(x) = x e^x$'
mat_file_name = 'Mathieu/f3_d1_a%i.csv'
true_val = 1.
min_neval = 10
max_neval = 5 * 10**5
orders = [1, 2, 3, 4, 5, 6]
nreps = 500
nks = 30
save_file = 'results/dick1D.pkl'
deriv_methods = ['exact', 'num']

def phi(u):
    return u * np.exp(u)

def deriv(k, u):
    return np.exp(u) * (u + k)
