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
    maxlp = - minuslogpost(mu, dim)
    Sigma = rez.hess_inv
    Cu = linalg.cholesky(Sigma, lower=False)
    lap[dim] = {'mu': mu, 'Cu': Cu, 'maxlp': maxlp}
    print(lap[dim])

with open('results/pima_mean_cov.pkl', 'wb') as f:
    pickle.dump(lap, f)
