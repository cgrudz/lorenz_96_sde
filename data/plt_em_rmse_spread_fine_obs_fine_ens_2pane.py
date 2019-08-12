import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
from matplotlib import rc
#import ipdb
import glob


method = 'em'

def process_data(fnames, method):
    rmse = np.zeros([5, 5])
    spread = np.zeros([5, 5])
    
    for i in range(5):
        for j in range(5):
            f = open(fnames[i*5 + j], 'rb')
            tmp = pickle.load(f)
            f.close()


            ana_stat = tmp[method + '_ana_stat']

            rmse[4 - i, j] = np.mean(ana_stat[0,:])
            spread[4 - i, j] = np.mean(ana_stat[1,:])

    return [rmse, spread]

fnames = sorted(glob.glob('./ens_bias/fine_ensemble_step/fine_obs_step/ens_bias_diff*'))
[rmse, spread] = process_data(fnames, method)
[rk_fine_rmse, rk_fine_spread] = process_data(fnames, 'ty')

rmse = rmse - rk_fine_rmse
spread = spread / rk_fine_spread


fig = plt.figure(figsize=(9,4))
ax3 = fig.add_axes([.365, .23, .04, .6])
ax2 = fig.add_axes([.835, .23, .04, .6])
ax1 = fig.add_axes([.570, .23, .26, .6])
ax0 = fig.add_axes([.100, .23, .26, .6])


color_rmse = sns.diverging_palette(145, 280, s=85, l=25, n=80)
color_spread = sns.diverging_palette(220, 68, n=80)
#color_spread = sns.cubehelix_palette(n_colors=80, start=1.90)

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

norm_1 = MidpointNormalize(midpoint=0)
norm_2 = MidpointNormalize(midpoint=1)



sns.heatmap(rmse, linewidth=0.5, ax=ax0, cbar_ax=ax3, vmax=0.0075, vmin=0, cmap=color_rmse, norm=norm_1)
sns.heatmap(spread, linewidth=0.5, ax=ax1, cbar_ax=ax2, vmax=1.030, vmin=1.0, cmap=color_spread,norm=norm_2)


ax2.tick_params(
        labelsize=20)

ax1.tick_params(
        labelsize=20,
        labelleft=False,
        left=False)

ax0.tick_params(
        labelsize=20)

ax3.tick_params(
        labelsize=20,
        labelleft=False,
        labelright=True,
        right=True,
        left=False)
ax2.tick_params(
        labelsize=20,
        labelleft=False,
        labelright=True,
        right=True,
        left=False)


ax1.set_xticklabels(['0.1', '', '0.5', '', '1.0'])
ax0.set_xticklabels(['0.1', '', '0.5', '', '1.0'])
ax1.set_yticklabels(['0.1', '', '0.5', '', '1.0'][::-1])
ax0.set_yticklabels(['0.1', '', '0.5', '', '1.0'][::-1])

plt.figtext(.2525, .90, 'RMSE difference', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.7225, .90, 'Spread ratio', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.025, .52, r'Diffusion level $s$', horizontalalignment='center', verticalalignment='center', fontsize=24, rotation='90')
plt.figtext(.50, .04, r'Observational error variance $r$', horizontalalignment='center', verticalalignment='center', fontsize=24)


plt.show()
