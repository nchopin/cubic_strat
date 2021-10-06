from matplotlib import pyplot as plt
import pandas as pd

d = 2
tau = 1.
scale = 1.5
ident = 'pima'

file_name = '%s%i-tau%.1f-scale%.1f' % (ident, d, tau, scale)
df = pd.read_pickle('results/%s.pkl' % file_name)
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
for o in range(min_order, max_order + 1):
    col = colors[o]
    dfo = dfm[dfm['order']==o]
    ax.plot(dfo['neval'], dfo[key], col, lw=3, alpha=0.8)
    k = dfo['neval'].argmax()
    x = dfo['neval'].to_numpy()[k] * 1.2
    y = dfo[key].to_numpy()[k]
    ax.text(x, y, '%i' % o, va='top', ma='left', color=col)
plt.title(r'%s $s=%i$' % (ident.capitalize(), d))
plt.xlabel('nr evaluations')
plt.ylabel(key)
plt.xscale('log')
plt.yscale('log')
# plt.legend()
plt.savefig('plots/%s_var_vs_N.pdf' % file_name)
plt.show()
