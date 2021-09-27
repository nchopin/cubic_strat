"""
Computes some intermediate results for Pima indian examples. 


"""


import jax.numpy as jnp
import jax.scipy.optimize as opt
import numpy as np
import pickle
from scipy import stats
from scipy import linalg
from sklearn.linear_model import LogisticRegression

import particles.datasets as dts
from particles import resampling as rs

dataset = dts.Pima()
preds = dataset.data
npreds, dim_max = preds.shape
scale_prior = 5  # as in Chopin & Ridgway
scale_logit = 1.4  
# it's ok to use a different scale later; here it is used only to compute an
# approx bound for the log-weights (to avoid overflow when computing the phi
# function)
N = 100

def minuslogpost(beta, dim):
    loglik = jnp.sum( - jnp.log(1. + jnp.exp(-jnp.dot(preds[:, :dim], beta))))
    logprior = -(0.5 / scale_prior**2) * jnp.sum(beta**2)
    return -(logprior + loglik)

lap = {}
for dim in range(2, dim_max + 1):
    beta0 = jnp.zeros(dim)
    rez = opt.minimize(minuslogpost, beta0, args=(dim,),
                       method='BFGS', options={'maxiter': 50})
    mu = rez.x
    Sigma = rez.hess_inv
    Cu = linalg.cholesky(Sigma, lower=False)
    z = stats.logistic.rvs(size=(N, dim))
    cst = 0.5 * np.sum(np.log(np.diag(Cu))) - np.log(scale_logit)  # TODO 
    lq = np.sum(stats.logistic.logpdf(z), axis=1) - cst
    beta = mu + z @ Cu / scale_logit
    lp = np.empty(N)
    for n in range(N):
        lp[n] = - minuslogpost(beta[n, :], dim)
    lw = lp - lq
    maxlw = lw.max()
    lap[dim] = {'mu': mu, 'Cu': Cu, 'maxlw': maxlw}
    print('dim %i: ESS=%.2f' % (dim, rs.essl(lw)))

with open('results/pima_mean_cov.pkl', 'wb') as f:
    pickle.dump(lap, f)
