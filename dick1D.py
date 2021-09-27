ident = 'dick1D'
d = 1
title_plot = r'$\varphi(x) = x e^x$'
mat_file_name = 'Mathieu/f3_d1_a%i.csv'
true_val = 1.
min_neval = 100
max_neval = 10**6
orders = list(range(1, 9))
nreps = 5
nks = 10
save_file = 'results/dick1D.pkl'

def phi(u):
    return u * np.exp(u)

def deriv(k, u):
    return np.exp(u) * (u + k)
