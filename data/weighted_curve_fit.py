import numpy as np
import scipy as sp
from scipy import stats
import glob
from scipy.optimize import curve_fit
import pickle
import ipdb


diff_list = [.1, .25, .5, .75, 1.0]
conv_list = ['weak', 'strong']
plt_range = [.001, .01, .1]

def order_conv_line(x, a, b):
    return a * x + b


for j in range(len(diff_list)):
    for convergence in conv_list:
        # we will loop over the diffusion coefficients of interest
        fnames = glob.glob('./weak_strong_conv/weak_strong_convergence_beta_03_nanl_0100_diff_' + str(diff_list[j]).zfill(2) + '*')
        fnames = sorted(fnames)

        ty = np.zeros([3, len(fnames)])
        rk = np.zeros([3, len(fnames)])
        em = np.zeros([3, len(fnames)])

        for i in range(len(fnames)):
            name = fnames[i]
            f = open(name, 'rb')
            tmp = pickle.load(f)
            f.close()

            ty[:, i] = tmp['ty_' + convergence]
            em[:, i] = tmp['em_' + convergence]
            rk[:, i] = tmp['rk_' + convergence]

        ty_mean = np.mean(ty, axis=1)
        ty_vari = np.mean( (ty.transpose()- ty_mean)**2, axis=0)
        em_mean = np.mean(em, axis=1)
        em_vari = np.mean( (em.transpose()- em_mean)**2, axis=0)
        rk_mean = np.mean(rk, axis=1)
        rk_vari = np.mean( (rk.transpose()- rk_mean)**2, axis=0)

        scheme_names = ['ty', 'em', 'rk']

        err_means = [ty_mean, em_mean, rk_mean]
        err_varis = [ty_vari, em_vari, rk_vari]

        for i in range(3):
            err = err_means[i][::-1]
            varis = err_varis[i][::-1]

            
            Y = np.log(err) / np.log(10)
            X = np.log(plt_range) / np.log(10)

            [popt, pcov] = curve_fit(order_conv_line, X, Y, sigma=np.sqrt(varis))
            save_data = {'params':popt, 'uncertainty':pcov, 'x':plt_range, 'y':err}

            out_name ='./rate_of_convergence/' + convergence + '_diff_' + str(diff_list[j]).zfill(2) + '_' + scheme_names[i] + '.txt'
            f = open(out_name, 'wb')
            pickle.dump(save_data, f)
            f.close

