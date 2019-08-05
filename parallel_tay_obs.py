from ipyparallel import Client
import sys
import numpy as np

########################################################################################################################
# set up parallel client

rc = Client()
dview = rc[:]

with dview.sync_imports():
    from generate_tay_sde_obs import experiment

exps = []
seed = 0
diffusion = [0.1, 0.25, 0.5, 0.75, 1.0] 
tanl = [0.1, 0.25, 0.5, 1] 

for i in diffusion:
    for j in tanl:
        exps.append([seed, i, j])

# run the experiments given the parameters and write to text files, in parallel over the initializations

completed = dview.map_sync(experiment, exps)

print(completed)

sys.exit()


########################################################################################################################
