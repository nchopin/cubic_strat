from matplotlib import pyplot as plt
import pandas as pd

import dick1D as pb


df = pd.read_pickle('results/%s.pkl' % pb.ident)
if 'rmse' in df:
    dfm = df.groupby(['k', 'order', 'deriv']).mean().reset_index()
    key = 'rmse'
else:
    dfm = df.groupby(['k', 'order', 'deriv']).mean().reset_index()
    dfv = df.groupby(['k', 'order', 'deriv']).var().reset_index()
    grand_mean = dfm['est'].mean()
    dfm['var'] = dfv['est'] / grand_mean**2
    key = 'var'

machine_eps = 2.2e-16
dfm = dfm[dfm[key] > machine_eps**2]

if hasattr(pb, 'mat_file_name'):
    dick_alphas = [1, 2, 3]
    dfs_mat = {a: pd.read_csv(pb.mat_file_name % a) for a in dick_alphas}
    for a, dfa in dfs_mat.items():
        dfa['var'] = dfa['var'] / pb.true_val**2 
        dfs_mat[a] = dfa[dfa['var'] > machine_eps**2]
    # TODO document
else:
    dfs_mat = None

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
fmts = {'exact': '-', 'num': '--', 'dick': ':'}

fig, ax = plt.subplots()
min_order, max_order = dfm['order'].min(), dfm['order'].max()
for order in range(min_order, max_order + 1):
    col = colors[order]
    for der in ['exact', 'num']:
        dfo = dfm[(dfm['order'] == order) & (dfm['deriv'] == der)]
        if not(dfo.empty):
            label = 'order=%i' % order
            if der == 'num':
                label += ' (num)'
            ax.plot(dfo['nevals'], dfo['rmse'], fmts[der], alpha=0.7, lw=4,
                    color=colors[order], label=label)
plt.title(pb.title_plot)
plt.xlabel('nr evaluations')
plt.ylabel(key)
plt.xscale('log')
plt.yscale('log')
if dfs_mat:
    for a in dick_alphas:
        df = dfs_mat[a]
        plt.plot(df['N'], df['var'], fmts['dick'], color=colors[a], 
                 label='dick alpha=%i' % a)

plt.legend()
plt.savefig('plots/%s_var_vs_N.pdf' % pb.ident)
plt.show()
