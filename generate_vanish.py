import numpy as np
import pandas as pd

from particles.utils import multiplexer
import strat

# import logit as pb  # TODO
# import beta2D as pb
import pima as pb

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
    ests, nevals = r['output']
    for i in range(pb.max_order):
        dr = {'est': ests[i], 'order': i + 1, 'neval': nevals[i], 'k': r['k']}
        results.append(dr)

df = pd.DataFrame(results)
df.to_pickle('results/%s.pkl' % pb.ident)

