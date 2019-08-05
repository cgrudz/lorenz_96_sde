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
    from weak_strong_convergence import experiment

fnames = glob.glob('./data/tay_obs_seed_000_sys_dim_10_analint_001_diffusion_*')
fnames = sorted(fnames)
exps = []

for i in range(len(fnames)):
    f = open(fnames[i], 'rb')
    tmp = pickle.load(f)
    f.close()

    params = tmp['params']
    x_init = tmp['tobs']
    for j in range(1,10):
        for p in [1]:
            args = []
            args.append(x_init[:,j])
            args.append(j)
            args.append(params[1])
            args.append(p)
            args.append(params[-1])
            exps.append(args)

# run the experiments given the parameters and write to text files, in parallel over the initializations

completed = dview.map_sync(experiment, exps)

print(completed)

sys.exit()


########################################################################################################################
