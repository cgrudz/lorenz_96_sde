from l96 import l96_2tay_sde
from l96 import alpha
from l96 import rho
import numpy as np
import pickle
import copy
import ipdb

def experiment(args):
    """This experiment will spin a "truth" trajectory of the stochastic l96 and store a series of analysis points

    This is a function of the ensemble number which initializes the random seed.  This will pickle the associated
    trajectories for processing in data assimilation experiments."""

    ####################################################################################################################
    [seed, diffusion, analint] = args

    # static parameters
    f = 8

    # model dimension
    sys_dim = 10
    
    # fourier truncation
    p = 1

    # time step
    h = .0005

    # number of observations
    nanl = 100100

    # spin onto random attractor in continous time
    spin = 5000

    # number of discrete forecast steps
    fore_steps = int(analint / h)

    # define the initialization of the model
    np.random.seed(seed)
    xt = np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * sys_dim)

    # static parameters based on fourier approxmimation truncation
    RHO = rho(p)
    ALPHA = alpha(p)

    # spin is the length of the spin period in the continuous time variable
    for i in range(int(spin / h)):
        #ipdb.set_trace()
        # recursively integrate one step forward
        xt = l96_2tay_sde(xt, h, [f, diffusion, p, RHO, ALPHA])

    # generate the full length of the truth tajectory which we assimilate
    tobs = np.zeros([sys_dim, nanl])

    for i in range(nanl):
        # integrate until the next observation time
        for j in range(fore_steps):
            # integrate one step forward
            xt = l96_2tay_sde(xt, h, [f, diffusion, p, RHO, ALPHA])

        tobs[:, i] = xt

    params = [seed, diffusion, analint, h, f]
    data = {'tobs': tobs, 'params': params}
    fname = './data/tay_obs_seed_' + str(seed).zfill(3) + '_sys_dim_' + str(sys_dim).zfill(2) + '_analint_' + \
            str(analint).zfill(3) + '_diffusion_' + str(diffusion).zfill(3) + '.txt'
    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()

    return args

########################################################################################################################

#print(experiment([0, 1.0, 0.1]))
