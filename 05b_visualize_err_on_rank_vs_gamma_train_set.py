
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("models_rank_prec_fixed_01XY.pkl")

df = df.groupby(['Gamma', 'Rank']).max(numeric_only=True).unstack(level=0).reset_index()
df = df.iloc[:-15,:] # drop last ten lines
df = df.iloc[::-1] #reverse lines order


column_titles = [(    'Rank',        ''),
            ('MaxError', '0.25133'),
            ('MaxError', '0.79477'),
            ('MaxError',  '2.5133'),
            ('MaxError', '7.9477'),
            ('MaxError',  '25.133'),
            ('MaxError', '79.477'),
            ('MaxError',  '251.33')]


df = df.reindex(columns=column_titles)

gammas = [g[1] for g in df.columns[1:]]
ranks = np.array(df['Rank'].tolist())

opt = df.MaxError.to_numpy()
print(opt.min(axis=0))

#delta = 5e-3
#opt[opt > 1e-3+delta] = np.nan
#opt_rank = len(ranks) - 3 - np.array(opt.notna()[::-1].idxmax().tolist()[1:])

opt_rank = [np.argwhere(a <= max(5e-3,a.min())).max() for a in np.array(opt).T ]

#opt_rank = [np.argwhere(np.diff(a <= max(4e-3,a.min()))).max() for a in np.array(opt).T ]

print(opt_rank)
opt_rank =  np.array(opt_rank)

#opt_rank = [29, 26, 27,  7, 20, 25, 19]
#opt_rank = [29, 26, 27,  11, 20, 25, 19]


prc = df.MaxError # drop first column

from matplotlib.colors import LogNorm

#plt.imshow(prc)

fig = plt.figure(figsize = [8,8]) #4 x 3

ax = fig.add_subplot()

#cp = ax.contourf(X, Y, Z)
#pos = ax.matshow(prc, norm=LogNorm(),  aspect=.2, interpolation = 'nearest') #cmap = 'rainbow',
pos = ax.matshow(prc, aspect=.2, interpolation = 'nearest')

fig.colorbar(pos, shrink=.75) #orientation = 'horizontal'
ax.set_title('Precision of identified models (train)', fontdict={'fontsize': 16, 'fontweight': 'bold'})
ax.set_xlabel(r'Dumping rate $\gamma$', fontdict={'fontsize': 14, 'fontweight': 'light'})
ax.set_ylabel('Rank', fontdict={'fontsize': 14, 'fontweight': 'light'})

gamma_ticks = range(7)
gamma_labels = gammas

rank_ticks = np.array([3, 5, 7, 10, 20, 30]) - 3
rank_ticks = np.array(len(ranks) - 1 - rank_ticks)

rank_labels = [ranks[r] for r in rank_ticks]

ax.set_xticks(gamma_ticks)
ax.set_xticklabels(gamma_labels)
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks(rank_ticks)
ax.set_yticklabels(rank_labels)

ax.plot(opt_rank, "*-r", label = 'optimum rank', )
ax.legend()
#ax.set_ylim([0,35])

fig.savefig('prec_of_rank_vs_gamma_best.png', transparent=True)