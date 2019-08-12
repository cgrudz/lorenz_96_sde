import numpy as  np
from matplotlib import pyplot as plt
#import glob
import matplotlib as mpl
mpl.style.use('classic')
#from matplotlib import rcParams
#rcParams['text.usetex'] = True
import pickle
#import ipdb
import glob
import scipy.stats


h = 0.01
fig = plt.figure(figsize=(12,16))

ax11 = fig.add_axes([.06, .24, .176, .20])
ax12 = fig.add_axes([.236, .24, .176, .20])
ax13 = fig.add_axes([.412, .24, .176, .20])
ax14 = fig.add_axes([.588, .24, .176, .20])
ax15 = fig.add_axes([.764, .24, .176, .20])
ax16 = fig.add_axes([.06, .04, .176, .20])
ax17 = fig.add_axes([.236, .04, .176, .20])
ax18 = fig.add_axes([.412, .04, .176, .20])
ax19 = fig.add_axes([.588, .04, .176, .20])
ax20 = fig.add_axes([.764, .04, .176, .20])


ax1 = fig.add_axes([.06, .67, .176, .20])
ax2 = fig.add_axes([.236, .67, .176, .20])
ax3 = fig.add_axes([.412, .67, .176, .20])
ax4 = fig.add_axes([.588, .67, .176, .20])
ax5 = fig.add_axes([.764, .67, .176, .20])
ax6 = fig.add_axes([.06, .47, .176, .20])
ax7 = fig.add_axes([.236, .47, .176, .20])
ax8 = fig.add_axes([.412, .47, .176, .20])
ax9 = fig.add_axes([.588, .47, .176, .20])
ax10 = fig.add_axes([.764, .47, .176, .20])

ax_list = fig.get_axes()


# set the tick parameters for the left most blocks
ax1.tick_params(
        axis='y',
        labelsize='30')
ax1.tick_params(
        axis='x',
        labelbottom=False)

ax6.tick_params(
        axis='y',
        labelsize='30')
ax6.tick_params(
        axis='x',
        labelbottom=False)

# set the tick parameters for the right most blocks
ax5.tick_params(
        axis='y',
        labelsize='30',
        labelleft=False,
        labelright=True)

ax5.tick_params(
        axis='x',
        labelbottom=False)

ax10.tick_params(
        axis='y',
        labelsize='30',
        labelleft=False,
        labelright=True)

ax10.tick_params(
        axis='x',
        labelbottom=False)


# set tick parameters for the remaining bottom row
ax7.tick_params(
        axis='y',
        labelleft=False)

ax7.tick_params(
        axis='x',
        labelbottom=False)

ax8.tick_params(
        axis='y',
        labelleft=False)

ax8.tick_params(
        axis='x',
        labelbottom=False)

ax9.tick_params(
        axis='y',
        labelleft=False)

ax9.tick_params(
        axis='x',
        labelbottom=False)

# set tick parameters for the interior boxes
ax2.tick_params(
        labelleft=False,
        labelbottom=False)

ax3.tick_params(
        labelleft=False,
        labelbottom=False)

ax4.tick_params(
        labelleft=False,
        labelbottom=False)

# set the tick parameters for the left most blocks
ax11.tick_params(
        axis='y',
        labelsize='30')
ax11.tick_params(
        axis='x',
        labelbottom=False)

ax16.tick_params(
        axis='y',
        labelsize='30')
ax16.tick_params(
        axis='x',
        labelsize='30')

# set the tick parameters for the right most blocks
ax15.tick_params(
        axis='y',
        labelsize='30',
        labelleft=False,
        labelright=True)

ax15.tick_params(
        axis='x',
        labelbottom=False)

ax20.tick_params(
        axis='y',
        labelsize='30',
        labelleft=False,
        labelright=True)

ax20.tick_params(
        axis='x',
        labelsize='30')


# set tick parameters for the remaining bottom row
ax17.tick_params(
        axis='y',
        labelleft=False)

ax17.tick_params(
        axis='x',
        labelsize='30')

ax18.tick_params(
        axis='y',
        labelleft=False)

ax18.tick_params(
        axis='x',
        labelsize='30')

ax19.tick_params(
        axis='y',
        labelleft=False)

ax19.tick_params(
        axis='x',
        labelsize='30')

# set tick parameters for the interior boxes
ax12.tick_params(
        labelleft=False,
        labelbottom=False)

ax13.tick_params(
        labelleft=False,
        labelbottom=False)

ax14.tick_params(
        labelleft=False,
        labelbottom=False)



diff = [0.1, 0.25, 0.5, 0.75, 1.0]


f = open('./ens_sum_stats/ens_mean_' + str(h).zfill(3) + '.txt', 'rb')
data = pickle.load(f)
f.close()

y_max = []


[e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max] = data['diff_' + str(diff[0]).zfill(2)]
y_max.append(np.max([e_max, r_max]))

l1 = ax1.errorbar(np.arange(1,2001) * .01, e_med, yerr=e_h, ecolor='#80cdc1', color='#328175', elinewidth=3, lw=3)
ax1.plot(np.arange(1,2001) * 0.01, e_min, color='#328175', linestyle='--', linewidth=3)
ax1.plot(np.arange(1,2001) * 0.01, e_max, color='#328175', linestyle='--', linewidth=3)

l2 = ax6.errorbar(np.arange(1,2001) * .01, r_med, yerr=r_h, ecolor='#e3994f', color='#9a5918', elinewidth=3, lw=3)
ax6.plot(np.arange(1,2001) * 0.01, r_min, color='#9a5918', linestyle='--', linewidth=3)
ax6.plot(np.arange(1,2001) * 0.01, r_max, color='#9a5918', linestyle='--', linewidth=3)

[e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max] = data['diff_' + str(diff[1]).zfill(2)]
y_max.append(np.max([e_max, r_max]))


ax2.errorbar(np.arange(1,2001) * .01, e_med, yerr=e_h, ecolor='#80cdc1', color='#328175', elinewidth=3, lw=3)
ax2.plot(np.arange(1,2001) * 0.01, e_min, color='#328175', linestyle='--', linewidth=3)
ax2.plot(np.arange(1,2001) * 0.01, e_max, color='#328175', linestyle='--', linewidth=3)

ax7.errorbar(np.arange(1,2001) * .01, r_med, yerr=r_h, ecolor='#e3994f', color='#9a5918', elinewidth=3, lw=3)
ax7.plot(np.arange(1,2001) * 0.01, r_min, color='#9a5918', linestyle='--', linewidth=3)
ax7.plot(np.arange(1,2001) * 0.01, r_max, color='#9a5918', linestyle='--', linewidth=3)

[e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max] = data['diff_' + str(diff[2]).zfill(2)]
y_max.append(np.max([e_max, r_max]))

ax3.errorbar(np.arange(1,2001) * .01, e_med, yerr=e_h, ecolor='#80cdc1', color='#328175', elinewidth=3, lw=3)
ax3.plot(np.arange(1,2001) * 0.01, e_min, color='#328175', linestyle='--', linewidth=3)
ax3.plot(np.arange(1,2001) * 0.01, e_max, color='#328175', linestyle='--', linewidth=3)

ax8.errorbar(np.arange(1,2001) * .01, r_med, yerr=r_h, ecolor='#e3994f', color='#9a5918', elinewidth=3, lw=3)
ax8.plot(np.arange(1,2001) * 0.01, r_min, color='#9a5918', linestyle='--', linewidth=3)
ax8.plot(np.arange(1,2001) * 0.01, r_max, color='#9a5918', linestyle='--', linewidth=3)

[e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max] = data['diff_' + str(diff[3]).zfill(2)]
y_max.append(np.max([e_max, r_max]))

ax4.errorbar(np.arange(1,2001) * .01, e_med, yerr=e_h, ecolor='#80cdc1', color='#328175', elinewidth=3, lw=3)
ax4.plot(np.arange(1,2001) * 0.01, e_min, color='#328175', linestyle='--', linewidth=3)
ax4.plot(np.arange(1,2001) * 0.01, e_max, color='#328175', linestyle='--', linewidth=3)

ax9.errorbar(np.arange(1,2001) * .01, r_med, yerr=r_h, ecolor='#e3994f', color='#9a5918', elinewidth=3, lw=3)
ax9.plot(np.arange(1,2001) * 0.01, r_min, color='#9a5918', linestyle='--', linewidth=3)
ax9.plot(np.arange(1,2001) * 0.01, r_max, color='#9a5918', linestyle='--', linewidth=3)

[e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max] = data['diff_' + str(diff[4]).zfill(2)]
y_max.append(np.max([e_max, r_max]))


ax5.errorbar(np.arange(1,2001) * .01, e_med, yerr=e_h, ecolor='#80cdc1', color='#328175', elinewidth=3, lw=3)
ax5.plot(np.arange(1,2001) * 0.01, e_min, color='#328175', linestyle='--', linewidth=3)
ax5.plot(np.arange(1,2001) * 0.01, e_max, color='#328175', linestyle='--', linewidth=3)

ax10.errorbar(np.arange(1,2001) * .01, r_med, yerr=r_h, ecolor='#e3994f', color='#9a5918', elinewidth=3, lw=3)
ax10.plot(np.arange(1,2001) * 0.01, r_min, color='#9a5918', linestyle='--', linewidth=3)
ax10.plot(np.arange(1,2001) * 0.01, r_max, color='#9a5918', linestyle='--', linewidth=3)

fig.text(0.148, .89, 'S=0.1', ha='center', va='center', fontsize=30)
fig.text(0.324, .89, 'S=0.25', ha='center', va='center', fontsize=30)
fig.text(0.500, .89, 'S=0.5', ha='center', va='center', fontsize=30)
fig.text(0.676, .89, 'S=0.75', ha='center', va='center', fontsize=30)
fig.text(0.852, .89, 'S=1.0', ha='center', va='center', fontsize=30)
lab = ['Euler-Maruyama', 'Runge-Kutta']
fig.legend([l1,l2], lab, loc='upper center', ncol=2, fontsize=30)

y_max = np.max(y_max)

for i in range(10, 20):
    ax = ax_list[i]
    ax.set_ylim([0,y_max])
    ax.set_xlim([0, 10.1])
    ax.set_xticks(np.arange(1,10, 2))
    ax.set_yticks(np.arange(0,4) * 1.0 + .5 )


f = open('./ens_sum_stats/ens_spread_' + str(h).zfill(3) + '.txt', 'rb')
data = pickle.load(f)
f.close()


y_max = []
y_min = []


[e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max] = data['diff_' + str(diff[0]).zfill(2)]
y_max.append(np.max([e_max, r_max]))
y_min.append(np.min([e_min, r_min]))

ax11.errorbar(np.arange(1,2001) * .01, e_med, yerr=e_h, ecolor='#80cdc1', color='#328175', elinewidth=3, lw=3)
ax11.plot(np.arange(1,2001) * 0.01, e_min, color='#328175', linestyle='--', linewidth=3)
ax11.plot(np.arange(1,2001) * 0.01, e_max, color='#328175', linestyle='--', linewidth=3)

ax16.errorbar(np.arange(1,2001) * .01, r_med, yerr=r_h, ecolor='#e3994f', color='#9a5918', elinewidth=3, lw=3)
ax16.plot(np.arange(1,2001) * 0.01, r_min, color='#9a5918', linestyle='--', linewidth=3)
ax16.plot(np.arange(1,2001) * 0.01, r_max, color='#9a5918', linestyle='--', linewidth=3)

[e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max] = data['diff_' + str(diff[1]).zfill(2)]
y_max.append(np.max([e_max, r_max]))
y_min.append(np.min([e_min, r_min]))


ax12.errorbar(np.arange(1,2001) * .01, e_med, yerr=e_h, ecolor='#80cdc1', color='#328175', elinewidth=3, lw=3)
ax12.plot(np.arange(1,2001) * 0.01, e_min, color='#328175', linestyle='--', linewidth=3)
ax12.plot(np.arange(1,2001) * 0.01, e_max, color='#328175', linestyle='--', linewidth=3)

ax17.errorbar(np.arange(1,2001) * .01, r_med, yerr=r_h, ecolor='#e3994f', color='#9a5918', elinewidth=3, lw=3)
ax17.plot(np.arange(1,2001) * 0.01, r_min, color='#9a5918', linestyle='--', linewidth=3)
ax17.plot(np.arange(1,2001) * 0.01, r_max, color='#9a5918', linestyle='--', linewidth=3)

[e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max] = data['diff_' + str(diff[2]).zfill(2)]
y_max.append(np.max([e_max, r_max]))
y_min.append(np.min([e_min, r_min]))

ax13.errorbar(np.arange(1,2001) * .01, e_med, yerr=e_h, ecolor='#80cdc1', color='#328175', elinewidth=3, lw=3)
ax13.plot(np.arange(1,2001) * 0.01, e_min, color='#328175', linestyle='--', linewidth=3)
ax13.plot(np.arange(1,2001) * 0.01, e_max, color='#328175', linestyle='--', linewidth=3)

ax18.errorbar(np.arange(1,2001) * .01, r_med, yerr=r_h, ecolor='#e3994f', color='#9a5918', elinewidth=3, lw=3)
ax18.plot(np.arange(1,2001) * 0.01, r_min, color='#9a5918', linestyle='--', linewidth=3)
ax18.plot(np.arange(1,2001) * 0.01, r_max, color='#9a5918', linestyle='--', linewidth=3)

[e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max] = data['diff_' + str(diff[3]).zfill(2)]
y_max.append(np.max([e_max, r_max]))
y_min.append(np.min([e_min, r_min]))

ax14.errorbar(np.arange(1,2001) * .01, e_med, yerr=e_h, ecolor='#80cdc1', color='#328175', elinewidth=3, lw=3)
ax14.plot(np.arange(1,2001) * 0.01, e_min, color='#328175', linestyle='--', linewidth=3)
ax14.plot(np.arange(1,2001) * 0.01, e_max, color='#328175', linestyle='--', linewidth=3)

ax19.errorbar(np.arange(1,2001) * .01, r_med, yerr=r_h, ecolor='#e3994f', color='#9a5918', elinewidth=3, lw=3)
ax19.plot(np.arange(1,2001) * 0.01, r_min, color='#9a5918', linestyle='--', linewidth=3)
ax19.plot(np.arange(1,2001) * 0.01, r_max, color='#9a5918', linestyle='--', linewidth=3)

[e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max] = data['diff_' + str(diff[4]).zfill(2)]
y_max.append(np.max([e_max, r_max]))
y_min.append(np.min([e_min, r_min]))


ax15.errorbar(np.arange(1,2001) * .01, e_med, yerr=e_h, ecolor='#80cdc1', color='#328175', elinewidth=3, lw=3)
ax15.plot(np.arange(1,2001) * 0.01, e_min, color='#328175', linestyle='--', linewidth=3)
ax15.plot(np.arange(1,2001) * 0.01, e_max, color='#328175', linestyle='--', linewidth=3)

ax20.errorbar(np.arange(1,2001) * .01, r_med, yerr=r_h, ecolor='#e3994f', color='#9a5918', elinewidth=3, lw=3)
ax20.plot(np.arange(1,2001) * 0.01, r_min, color='#9a5918', linestyle='--', linewidth=3)
ax20.plot(np.arange(1,2001) * 0.01, r_max, color='#9a5918', linestyle='--', linewidth=3)

y_max = np.max(y_max)
y_min = np.min(y_min)

for i in range(10):
    ax = ax_list[i]
    ax.set_ylim([.6,2.1])
    ax.set_xlim([0, 10.1])
    ax.set_xticks(np.arange(1,10, 2))
    ax.set_yticks(np.arange(0.6,2.0,0.4))

plt.show()

