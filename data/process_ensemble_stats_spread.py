import numpy as  np
import pickle
import ipdb
import glob
import scipy.stats


h = 0.01

diff = [0.1, 0.25, 0.5, 0.75, 1.0]


def five_num_summary(data, axis):
    med = np.median(data, axis=axis)
    q1 = np.percentile(data, 10, interpolation='midpoint', axis=axis)
    q3 = np.percentile(data, 90, interpolation='midpoint', axis=axis)

    h = np.array([q1, q3])
    h = np.abs(h - med)
    
    min_sum = np.min(data, axis=axis)
    max_sum = np.max(data, axis=axis)

    return med, h, min_sum, max_sum

def rms_deviation(base, alt):

    dev = base - alt
    return np.sqrt(np.mean(dev*dev,axis=0))


def process_data(diff, h):

    diff = str(diff).zfill(2)
    fnames_comparison = glob.glob('./ens_stats/*h_' + str(h).zfill(2) + '*_diffusion_' + diff + '*')
    fnames_comparison = sorted(fnames_comparison)

    fnames_benchmark = glob.glob('./ens_stats/*h_' + str(0.001).zfill(2) + '*_diffusion_' + diff + '*')
    fnames_benchmark = sorted(fnames_benchmark)

    ens_s_deviation = np.zeros([2000, 2, len(fnames_benchmark)])
    
    for i in range(len(fnames_comparison)):
        name_comparison = fnames_comparison[i]
        f = open(name_comparison, 'rb')
        tmp_comparison = pickle.load(f)
        f.close()

        name_benchmark = fnames_benchmark[i]
        f = open(name_benchmark, 'rb')
        tmp_benchmark = pickle.load(f)
        f.close()
        
        init_cond_comparison = int(name_comparison[-10:-4])
        init_cond_benchmark = int(name_benchmark[-10:-4])

        seed_comparison = int(name_comparison[37:42])
        seed_benchmark = int(name_benchmark[37:42])

        if (init_cond_comparison - init_cond_benchmark) != 0:
            print('ERROR initial condition')
        
        if (seed_comparison - seed_benchmark) !=0:
            print('ERROR seed')


        e_spread = tmp_comparison['e_spread']
        r_spread = tmp_comparison['r_spread']
        t_spread = tmp_benchmark['t_spread']

        ens_s_deviation[:, 0, i] = e_spread / t_spread
        ens_s_deviation[:, 1, i] = r_spread / t_spread
 
    
    e_med, e_h, e_min, e_max = five_num_summary(np.squeeze(ens_s_deviation[:,0,:]), 1)
    r_med, r_h, r_min, r_max = five_num_summary(np.squeeze(ens_s_deviation[:,1,:]), 1)
    
    return [e_med, e_h, e_min, e_max, r_med, r_h, r_min, r_max]

data = {}

y_max = []

lab = 'diff_' + str(diff[0]).zfill(2)
data[lab] =  process_data(diff[0], h)

lab = 'diff_' + str(diff[1]).zfill(2)
data[lab] = process_data(diff[1], h)

lab = 'diff_' + str(diff[2]).zfill(2)
data[lab] = process_data(diff[2], h)

lab = 'diff_' + str(diff[3]).zfill(2)
data[lab] = process_data(diff[3], h)

lab = 'diff_' + str(diff[4]).zfill(2)
data[lab] = process_data(diff[4], h)

f = open('./ens_sum_stats/ens_spread_' + str(h).zfill(3) + '.txt', 'wb')
pickle.dump(data, f)
f.close()
