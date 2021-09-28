from matplotlib import pyplot as plt
import pandas as pd

ident = 'pima2_scale1.00'

df = pd.read_pickle('results/%s.pkl' % ident)
if 'rmse' in df:
    dfm = df.groupby(['k', 'order']).mean().reset_index()
    key = 'mse'
else:
    dfm = df.groupby(['k', 'order']).mean().reset_index()
    dfv = df.groupby(['k', 'order']).var().reset_index()
    grand_mean = dfm['est'].mean()
    dfm['var'] = dfv['est'] / grand_mean**2
    key = 'var'

# plots
#######
plt.style.use('ggplot')
plt.figure()
min_order, max_order = dfm['order'].min(), dfm['order'].max()
for o in range(min_order, max_order + 1):
    dfo = dfm[dfm['order']==o]
    plt.plot(dfo['neval'], dfo[key], label=o)
plt.title(ident)
plt.xlabel('nr evaluations')
plt.ylabel(key)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('plots/%s_var_vs_N.pdf' % ident)
plt.show()
