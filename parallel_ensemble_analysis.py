from ipyparallel import Client
import sys
import numpy as np
import glob
import pickle

########################################################################################################################
# set up parallel client

rc = Client()
dview = rc[:]

with dview.sync_imports():
    from ensemble_statistics_analysis import experiment

fnames = glob.glob('./data/tay_obs_seed_000_sys_dim_10_analint_001*')
fnames = sorted(fnames)

exps = []

for i in range(len(fnames)):
    f = open(fnames[i], 'rb')
    tmp = pickle.load(f)
    f.close()

    params = tmp['params']
    x_init = tmp['tobs']

    for j in range(100):
        args = []
        args.append(np.squeeze(x_init[:, j]))
        args.append(j)

        for k in params:
            args.append(k)

        exps.append(args)

# run the experiments given the parameters and write to text files, in parallel over the initializations

completed = dview.map_sync(experiment, exps)

print(completed)

sys.exit()


########################################################################################################################
