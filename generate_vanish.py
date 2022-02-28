import numpy as np
import pandas as pd

from particles.utils import multiplexer
import strat

# set considered problem here
import pima2 as pb

# compute range of values for k
lkmin = (np.log(pb.min_neval / pb.max_order)) / pb.d
lkmin = max(lkmin, np.log(2.)) # 1 not allowed
lkmax = (np.log(pb.max_neval / pb.max_order)) / pb.d
lkrange = np.linspace(lkmin, lkmax, pb.nks)
karr = np.unique(np.round(np.exp(lkrange))).astype('int')
ks = list(karr)
print('k=%r' % ks)

mp_results = multiplexer(f=strat.vanish_estimates, k=ks, d=pb.d, phi=pb.phi,
                         order=pb.max_order, nruns=pb.nreps, nprocs=0)

results = []
for r in mp_results:
    for i in range(pb.max_order):
        dr = {'est': r['estimates'][i], 'order': i + 1, 
              'nevals': r['nevals'][i], 'k': r['k'], 'cpu': r['cpu']}
        results.append(dr)

df = pd.DataFrame(results)
if hasattr(pb, 'true_val'):
    df['rel-mse'] = (df['est'] / pb.true_val - 1.)**2
df.to_pickle('results/%s.pkl' % pb.ident)
