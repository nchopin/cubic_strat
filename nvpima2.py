import pima_common as pim

d = 2
tau = 1.
scale_prop = 1.5
min_neval = 10**2
max_neval = 10**6  # TODO
orders = [1, 2, 4, 6, 8, 10]
nks = 10
nreps = 50
ident = 'nvpima%i-tau%.1f-scale%.1f' % (d, tau, scale_prop)
title_plot = r'Pima $s=%i$' % d

def phi(u):
    return pim.phi(u, tau=tau, scale_prop=scale_prop)

