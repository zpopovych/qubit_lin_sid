import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/home/zah/PycharmProjects/Kurt2021/2021JUN6')
print('Current directory', os.getcwd())

#df = pd.read_pickle("models_rank_prec_fixed.pkl")
#df = pd.read_pickle("models_100_ranks_prec_fixed.pkl")
df = pd.read_pickle("models_100_rank_prec.pkl")


df = df.groupby(['Gamma', 'Rank']).max(numeric_only=True).unstack(level=0).reset_index()
#df = df.iloc[:-15,:] # drop last ten lines
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
#print(opt.min(axis=0))

#delta = 5e-3
#opt[opt > 1e-3+delta] = np.nan
#opt_rank = len(ranks) - 3 - np.array(opt.notna()[::-1].idxmax().tolist()[1:])

opt_rank1 = [np.argwhere(a <= max(1e-1,a.min())).max() for a in np.array(opt).T ]
opt_rank2 = [np.argwhere(a <= max(1e-2,a.min())).max() for a in np.array(opt).T ]




#opt_rank3_5 = [np.argwhere(a <= max(5e-3,a.min())).max() for a in np.array(opt).T ]

def rank_finder(th, a):
    indx = np.argwhere(a <= th)
    return np.nan if len(indx)==0 else indx.max()

opt_rank2 =  [rank_finder(1e-2, a) for a in np.array(opt).T ]

opt_rank3 = [rank_finder(1e-3, a) for a in np.array(opt).T ]

opt_rank3[6] = -1000

opt_rank3_5 = [rank_finder(5e-3, a) for a in np.array(opt).T ]

opt_rank4 = [np.argwhere(a <= max(1e-4,a.min())).max() for a in np.array(opt).T ]
opt_rank5 = [np.argwhere(a <= max(1e-5,a.min())).max() for a in np.array(opt).T ]

#opt_rank = [np.argwhere(np.diff(a <= max(4e-3,a.min()))).max() for a in np.array(opt).T ]

#print(opt_rank)
#opt_rank =  np.array(opt_rank)

#opt_rank = [29, 26, 27,  7, 20, 25, 19]
#opt_rank = [29, 26, 27,  11, 20, 25, 19]


prc = df.MaxError # drop first column

prc[prc>1] = np.nan

from matplotlib.colors import LogNorm

#plt.imshow(prc)

fig = plt.figure(figsize = [5,10]) #4 x 3

ax = fig.add_subplot()

#cp = ax.contourf(X, Y, Z)
pos = ax.matshow(prc, norm=LogNorm(),  aspect=.2, interpolation = 'nearest') #cmap = 'rainbow',
#pos = ax.matshow(prc, aspect=.2, interpolation = 'nearest')

fig.colorbar(pos, shrink=.75) #orientation = 'horizontal'
ax.set_title('Error of original \n identified models  ', fontdict={'fontsize': 16, 'fontweight': 'bold'})
ax.set_xlabel(r'Dumping rate $\gamma$', fontdict={'fontsize': 14, 'fontweight': 'light'})
ax.set_ylabel('Rank', fontdict={'fontsize': 14, 'fontweight': 'light'})

gamma_ticks = range(7)
gamma_labels = gammas

rank_ticks = np.array([3, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90]) - 3
rank_ticks = np.array(len(ranks) - 1 - rank_ticks)

rank_labels = [ranks[r] for r in rank_ticks]

ax.set_xticks(gamma_ticks)
ax.set_xticklabels(gamma_labels)
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks(rank_ticks)
ax.set_yticklabels(rank_labels)

from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator

def inter(y):
    #X_Y_Spline = make_interp_spline(range(7), y, k=3)
    X_Y_Spline = PchipInterpolator(range(7), y)
    X = np.linspace(0, 6, 500)
    Y = X_Y_Spline(X)
    return Y

x = np.linspace(0, 6, 500)

ax.plot(opt_rank2, "*-", c='yellow', label = r'$10^{-2}$')
ax.plot(opt_rank3_5, "*-", c='orange', label = r'$5 \cdot 10^{-3}$')
ax.plot(opt_rank3, "*-r", label = r'$10^{-3}$')

print('1e-3', 99-np.array(opt_rank3))
print('5e-3', 99-np.array(opt_rank3_5))
print('5e-3', 99-np.array(opt_rank2))

#ax.set_ylim(96,45)
ax.set_ylim(96,3)

#ax.plot(x, inter(opt_rank2), "-", c='yellow', label = r'$10^{-2}$')
#ax.plot(x, inter(opt_rank3_5), "-", c='orange',  label = r'$5 \cdot 10^{-3}$')
#ax.plot(x, inter(opt_rank3), "-r", label = r'$10^{-3}$')
#ax.plot(opt_rank4, "*-r", label = r'$10^{-4}$')
#ax.plot(opt_rank5, "*-r", label = r'$10^{-4}$')


ax.legend()
#ax.set_ylim([0,35])

fig.savefig('prec_of_rank_vs_gamma_best.png', transparent=True)