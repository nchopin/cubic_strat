import pima_common as pim

d = 6
tau = 1.
scale_prop = 1.5
min_neval = 10**3
max_neval = 10**10
max_order = 8
nks = 10
nreps = 50
ident = 'pima%i-tau%.1f-scale%.1f' % (d, tau, scale_prop)

def phi(u):
    return pim.phi(u, tau=tau, scale_prop=scale_prop)

