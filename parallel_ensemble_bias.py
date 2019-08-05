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
    from vary_ensemble_integration import exp

fnames = glob.glob('../data/obs_trajs/fine_coarse_obs/h_001/*')
fnames = sorted(fnames)
exps = []
obs_un = [0.1,0.25, 0.5,0.75, 1.0]

for i in range(len(fnames)):
    for j in range(5):
        f = open(fnames[i], 'rb')
        tmp = pickle.load(f)
        f.close()

        tobs = tmp['tobs']
        params = tmp['params']
        args = [tobs, params[2], params[1], obs_un[j], params[3], j]
        exps.append(args)

# run the experiments given the parameters and write to text files, in parallel over the initializations

completed = dview.map_sync(exp, exps)

print(completed)

sys.exit()


########################################################################################################################
