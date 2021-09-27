import jax.numpy as jnp
import jax.scipy.optimize as opt
import numpy as np
import pickle
from scipy import stats
from scipy import linalg
from sklearn.linear_model import LogisticRegression

import particles.datasets as dts

dataset = dts.Pima()
dim = 3
x = dataset.data[:, :dim]
scale_prior = 10.

def minuslogpost(beta):
    loglik = jnp.sum( - jnp.log(1. + jnp.exp(-jnp.dot(x,beta))))
    logprior = -(0.5 / scale_prior**2) * jnp.sum(beta**2)
    return -(logprior + loglik)

beta0 = jnp.zeros(dim)
rez = opt.minimize(minuslogpost, beta0, method='BFGS', options={'maxiter': 50})
mu = rez.x
Sigma = rez.hess_inv
Ct = linalg.cholesky(Sigma, lower=False)

# check against sk-learn
# lr = LogisticRegression(C=scale_prior**2, max_iter=500)
# rd = dataset.raw_data
# rx = rd[:, :-1]
# rx = 0.5 * (rx - rx.mean(axis=0)) / rx.std(axis=0)
# ry = rd[:, -1]
# lr.fit(rx, ry)

# check importance sampling
# scale_logit = 3.1416 / np.sqrt(3)
scale_logit = 1.4
N = 10**3
z = stats.logistic.rvs(size=(N, dim))
cst = 0.5 * np.sum(np.log(np.diag(Ct))) - np.log(scale_logit)  # TODO 
lq = np.sum(stats.logistic.logpdf(z), axis=1) - cst
beta = mu + z @ Ct / scale_logit
lp = np.empty(N)
for n in range(N):
    lp[n] = - minuslogpost(beta[n, :])

lw = lp - lq
maxlw = lw.max()

dmp = {'mu': mu, 'Ct': Ct, 'maxlw': maxlw, 'scale_logit': scale_logit}
with open('intermediate_results/pima_mean_cov.pkl', 'wb') as f:
    pickle.dump(dmp, f)
