import numpy as  np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
import pickle
#import ipdb
import glob
from scipy.stats import linregress

fig = plt.figure(figsize=(16,9))
ax5 = fig.add_axes([.767, .075, .17, .7])
ax4 = fig.add_axes([.591, .075, .17, .7])
ax3 = fig.add_axes([.415, .075, .17, .7])
ax2 = fig.add_axes([.239, .075, .17, .7])
ax1 = fig.add_axes([.063, .075, .17, .7])

ax1.tick_params(
        axis='x',
        labelsize='36',
        pad=10)

ax1.tick_params(
        axis='y',
        labelsize='36',
        pad=2)

ax2.tick_params(
        axis='y',
        labelleft=False)

ax2.tick_params(
        axis='x',
        labelsize='36',
        pad=10)

ax3.tick_params(
        axis='y',
        labelleft=False)

ax4.tick_params(
        axis='y',
        labelleft=False)

ax3.tick_params(
        axis='x',
        labelsize='36',
        pad=10)

ax4.tick_params(
        axis='x',
        labelsize='36',
        pad=10)

ax5.tick_params(
        axis='x',
        labelsize='36',
        pad=10)

ax5.tick_params(
        axis='y',
        labelleft=False,
        labelright=True,
        labelsize='36',
        pad=1)


axs = [ax1, ax2, ax3, ax4, ax5]

color_list = ['#80cdc1', '#a6611a', '#dfc27d']
symbol_list = ['>', 'o', 'v']
diff_list = [0.1, 0.25, 0.50, 0.75, 1.0]
lines_list = []

for j in range(len(diff_list)):
    # we will loop over the diffusion coefficients of interest
    fnames = glob.glob('./rate_of_convergence/strong_diff_' + str(diff_list[j]).zfill(2) + '*')
    fnames = sorted(fnames)

    ax = axs[j]


    for k in range(len(fnames)):
        f = open(fnames[k], 'rb')
        tmp = pickle.load(f)
        f.close()

        params = tmp['params']
        slope = params[0]
        intercept = params[1]
        x = tmp['x']
        y = tmp['y']

        def func(x):
            return 10**(slope * np.log(x) / np.log(10) + intercept)

        l = ax.scatter(x, y, marker=symbol_list[k], s=400, color=color_list[k])
        lines_list.append(l)

        ax.plot(x, func(x), color=color_list[k], linewidth=4)
        


for i in range(5):
    ax = axs[i]
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xbound([.00030,.40])
    ax.set_ybound([.5*.1**5,1.45])
    ax.set_xticks([0.001,0.1])

leg = [lines_list[0], lines_list[1], lines_list[2]]
lab = ['Euler-Maruyama', 'Runge-Kutta', 'Taylor']

fig.legend(leg, lab, loc='upper center', ncol=3, fontsize=35)
fig.text(0.148, .80, 'S=' + str(diff_list[0]).zfill(2), ha='center', fontsize=35)
fig.text(0.324, .80, 'S=' + str(diff_list[1]).zfill(2), ha='center', fontsize=35)
fig.text(0.5, .80, 'S=' + str(diff_list[2]).zfill(2), ha='center', fontsize=35)
fig.text(0.676, .80, 'S=' + str(diff_list[3]).zfill(2), ha='center', fontsize=35)
fig.text(0.852, .80, 'S=' + str(diff_list[4]).zfill(2), ha='center', fontsize=35)

plt.show()

