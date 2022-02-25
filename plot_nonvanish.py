from matplotlib import pyplot as plt
import pandas as pd

import nvpima4 as pb

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

machine_eps = 1.2e-15
dfm = dfm[dfm[key] > machine_eps**2]

dfds = {}
if hasattr(pb, 'mat_folder'):
    for a in [1, 2, 3]:
        with open('%s/Dick_alpha%i.txt' %(pb.mat_folder, a), 'r') as f:
            dfd = pd.read_csv(f, sep=' ', names=['k', 'mse'], header=0)
            dfd['N'] = 2**dfd['k']
            dfd = dfd[dfd['N'] >= dfm['nevals'].min()]
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
lines = []
for order in range(min_order, max_order + 1):
    col = colors[order]
    dfo = dfm[(dfm['order'] == order)]
    if not(dfo.empty):
        line, = ax.plot(dfo['nevals'], dfo[key], alpha=0.7, lw=4,
                        color=colors[order])
        lines.append(line)
        k = dfo['nevals'].argmax()
        x = dfo['nevals'].to_numpy()[k] * 1.2
        y = dfo[key].to_numpy()[k]
        ax.text(x, y, '%i' % order, va='top', ma='left', color=col)
plt.title(pb.title_plot)
plt.xlabel('nr evaluations')
plt.ylabel(key)
plt.xscale('log')
plt.yscale('log')
lines_d = []
for a, dfd in dfds.items():
    line, = plt.plot(dfd['N'], dfd['mse'], ':', color=colors[a])
    lines_d.append(line)
    k = dfd['N'].argmax()
    x = dfd['N'].to_numpy()[k] * 1.2
    y = dfd['mse'].to_numpy()[k]
    ax.text(x, y, '%i' % a, va='top', ma='left', color=colors[a])

if dfds:
    plt.legend([lines[0], lines_d[0]], ['strat', 'Dick'], loc=3)
plt.savefig('plots/%s_var_vs_N.pdf' % pb.ident)
plt.show()
