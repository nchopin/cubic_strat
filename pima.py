import pickle
import numpy as np
from scipy import stats

from particles import datasets as dts

dataset = dts.Pima()
d = 4
preds = dataset.data[:, :d]
scale_prior = 5.
scale_prop = 1.5
tau = 1.

min_neval = 10**3
max_neval = 5. * 10**8
max_order = 10
nks = 10
nreps = 50
ident = 'pima%i-tau%.1f-scale%.1f' % (d, tau, scale_prop)

with open('results/pima_mean_cov.pkl', 'rb') as f:
    dts = pickle.load(f)
    dt = dts[d]
    mu = dt['mu']
    Cu = dt['Cu'] / scale_prop

def logpost(beta):
    loglik = np.sum( - np.log(1. + np.exp(-np.dot(preds, beta))))
    logprior = -(0.5 / scale_prior**2) * np.sum(beta**2)
    return (logprior + loglik)

def psi(u, t):
    umu = u * (1. - u)
    tm1 = 2. * u - 1
    umutau = umu**t
    z = tm1 / umutau
    jac = 2. / umutau + t * tm1**2 / umu**(t + 1)
    ljac = np.sum(np.log(jac), axis=1)
    return z, ljac

def phi(u):
    N, p = u.shape
    z, ljac = psi(u, tau)
    x = mu + z @ Cu
    cst = 0.5 * np.sum(np.log(np.diag(dt['Cu']))) 
    lq = ljac - cst
    lp = np.empty(N)
    for n in range(N):
        lp[n] = logpost(x[n, :])
    lw = lp - lq - dt['maxlp']
    return np.exp(lw)
