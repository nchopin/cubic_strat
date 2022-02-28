import numpy as np
import pandas as pd

from particles.utils import multiplexer
import strat 

import nvpima2 as pb

results = []
for order in pb.orders:
    # compute range of k such that nr of evaluations is between two bounds 
    lkmin = (np.log(pb.min_neval / min(3, order))) / pb.d
    ko = 2 if order <= 2 else (3 * order // 2 - 1)
    lkmin = max(lkmin, np.log(ko))  # ko = min value of k allowed by method
    lkmax = (np.log(pb.max_neval / min(3, order))) / pb.d
    lkrange = np.linspace(lkmin, lkmax, pb.nks)
    karr = np.unique(np.round(np.exp(lkrange))).astype('int')
    ks = list(karr)
    print('order: %i, k=%r' % (order, ks))
    rez = multiplexer(f=strat.estimate_with_nevals, d=pb.d, k=ks, 
                      phi=pb.phi, order=[order], nruns=pb.nreps, nprocs=0)
    results += rez

df = pd.DataFrame(results)
if hasattr(pb, 'true_val'):
    df['rel-mse'] = (df['est'] / pb.true_val - 1.)**2
df.to_pickle('results/%s.pkl' % pb.ident)
