import numpy as np

ident = 'dick1D'
d = 1
title_plot = r'$f_1(u)=ue^u$'
mat_folder = 'simuDick/d1/'
true_val = 1.
min_neval = 9
max_neval = 10**6
orders = [1, 2, 4, 6, 8, 10]
nreps = 50
nks = 20
save_file = 'results/dick1D.pkl'
deriv_methods = ['exact', 'num']

def phi(u):
    return u * np.exp(u)

def deriv(k, u):
    return np.exp(u) * (u + k)
