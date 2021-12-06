import numpy as np
import pandas as pd

from particles.utils import multiplexer
import strat 

import dick1D as pb

all_deriv_meth = {'exact': None, 'num': pb.deriv}
der = {k: all_deriv_meth[k] for k in pb.deriv_methods}

results = []
for order in pb.orders:
    # compute range of k such that nr of evaluations is between two bounds 
    lkmin = (np.log(pb.min_neval / order)) / pb.d
    lkmin = max(lkmin, np.log(2.)) # 1 not allowed
    lkmax = (np.log(pb.max_neval / order)) / pb.d
    lkrange = np.linspace(lkmin, lkmax, pb.nks)
    karr = np.unique(np.round(np.exp(lkrange))).astype('int')
    ks = list(karr)
    print('order: %i, k=%r' % (order, ks))
    rez = multiplexer(f=strat.estimate, d=pb.d, k=ks, phi=pb.phi, 
                      order=[order], deriv=der,
                      nruns=pb.nreps, nprocs=1)
    results += rez

df = pd.DataFrame(results)
if hasattr(pb, 'true_val'):
    df['rmse'] = (df['est'] / pb.true_val - 1.)**2
df.to_pickle('results/%s.pkl' % pb.ident)
