from matplotlib import pyplot as plt
import pandas as pd

import dick2D as pb

df = pd.read_pickle('results/%s.pkl' % pb.ident)
if 'rmse' in df:
    dfm = df.groupby(['k', 'order']).mean().reset_index()
    key = 'rmse'
else:
    dfm = df.groupby(['k', 'order']).mean().reset_index()
    dfv = df.groupby(['k', 'order']).var().reset_index()
    grand_mean = dfm['est'].mean()
    dfm['var'] = dfv['est'] / grand_mean**2
    key = 'var'

machine_eps = 5.e-16
dfm = dfm[dfm[key] > machine_eps**2]

dfds = {}
if hasattr(pb, 'mat_folder'):
    for a in [1, 2, 3]:
        with open('%s/Dick_alpha%i.text' %(pb.mat_folder, a), 'r') as f:
            dfd = pd.read_csv(f, sep=' ', names=['k', 'mse'], header=0)
            dfd['N'] = 2**dfd['k']
            dfd = dfd[dfd['N'] > pb.min_neval]
            dfds[a] = dfd

# plots
#######
plt.style.use('ggplot')
# color cycle of ggplot
# colors = [None,
#           '#1f77b4',
#           '#ff7f0e',
#           '#2ca02c',
#           '#d62728',
#           '#9467bd',
#           '#8c564b',
#           '#e377c2',
#           '#7f7f7f',
#           '#bcbd22',
#           '#17becf']
colors = [None, 'r', 'b', 'm', 'k', 'c', 'y', 'g', 'pink', 'gray', 'orange']

fig, ax = plt.subplots()
min_order, max_order = dfm['order'].min(), dfm['order'].max()
for order in range(min_order, max_order + 1):
    col = colors[order]
    dfo = dfm[(dfm['order'] == order)]
    if not(dfo.empty):
        label = 'order=%i' % order
        ax.plot(dfo['nevals'], dfo['rmse'], alpha=0.7, lw=4,
                color=colors[order], label=label)
plt.title(pb.title_plot)
plt.xlabel('nr evaluations')
plt.ylabel(key)
plt.xscale('log')
plt.yscale('log')
for a, dfd in dfds.items():
    plt.plot(dfd['N'], dfd['mse'], ':', color=colors[a], 
             label='Dick alpha=%i' % a)

plt.legend()
plt.savefig('plots/%s_var_vs_N.pdf' % pb.ident)
plt.show()
