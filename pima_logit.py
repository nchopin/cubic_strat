import pickle
import numpy as np
from scipy import stats

from particles import datasets as dts

dataset = dts.Pima()
d = 2
preds = dataset.data[:, :d]
scale_prior = 5.
scale_logit = 1.0 # TODO

min_neval = 100
max_neval = 10**9
max_order = 6
nks = 10
nreps = 50
ident = 'pima%i_scale%.2f' % (d, scale_logit)

with open('results/pima_mean_cov.pkl', 'rb') as f:
    dts = pickle.load(f)
    dt = dts[d]

def logpost(beta):
    loglik = np.sum( - np.log(1. + np.exp(-np.dot(preds, beta))))
    logprior = -(0.5 / scale_prior**2) * np.sum(beta**2)
    return (logprior + loglik)

def phi(u):
    z = stats.logistic.ppf(u)
    x = dt['mu'] + z @ dt['Cu'] / scale_logit
    cst = 0.5 * np.sum(np.log(np.diag(dt['Cu']))) - d * np.log(scale_logit)
    lq = np.sum(stats.logistic.logpdf(z), axis=1) - cst
    N, p = u.shape
    lp = np.empty(N)
    for n in range(N):
        lp[n] = logpost(x[n, :])
    lw = lp - lq - dt['maxlw']
    return np.exp(lw)
