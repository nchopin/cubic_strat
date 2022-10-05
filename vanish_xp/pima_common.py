import pickle
import numpy as np
from scipy import stats

from particles import datasets as dts

dataset = dts.Pima()
preds = dataset.data
scale_prior = 5.


with open('results/pima_mean_cov.pkl', 'rb') as f:
    dts = pickle.load(f)

def logpost(beta):
    dimbeta = beta.shape[-1]
    lin = np.dot(preds[:, :dimbeta], beta)
    loglik = np.sum( - np.log(1. + np.exp(-lin)))
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

def phi(u, tau=1., scale_prop=1.5):
    N, d = u.shape
    z, ljac = psi(u, tau)
    mu = dts[d]['mu']
    Cu = dts[d]['Cu'] / scale_prop
    x = mu + z @ Cu
    cst = 0.5 * np.sum(np.log(np.diag(Cu)))
    lq = ljac - cst
    lp = np.empty(N)
    for n in range(N):
        lp[n] = logpost(x[n, :])
    lw = lp - lq - dts[d]['maxlp']
    return np.exp(lw)
