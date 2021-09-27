import numpy as np
import pandas as pd

from particles.utils import multiplexer
import strat 

import dick4D as pb

results = []
for order in pb.orders:
    # compute range of k such that nr of evaluations is between two bounds 
    lkmin = (np.log(pb.min_neval / order)) / pb.d
    lkmin = max(lkmin, np.log(2.)) # 1 not allowed
    lkmax = (np.log(pb.max_neval / order)) / pb.d
    lkrange = np.linspace(lkmin, lkmax, pb.nks)
    karr = np.unique(np.round(np.exp(lkrange))).astype('int')
    ks = list(karr)
    if order >= 3 and hasattr(pb, 'deriv'):
        der = {'num': None, 'exact': pb.deriv}
    else:  # if order=1, 2 no need to compute derivatives
        der = {'exact': None}
    print('order: %i, k=%r' % (order, ks))
    rez = multiplexer(f=strat.estimate, d=pb.d, k=ks, phi=pb.phi, 
                      order=[order], deriv=der,
                      nruns=pb.nreps, nprocs=0)
    results += rez

df = pd.DataFrame(results)
df.to_pickle('results/%s.pkl' % pb.ident)

