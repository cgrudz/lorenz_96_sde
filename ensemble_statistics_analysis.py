from l96 import alpha
from l96 import rho
from l96 import l96
from l96 import l96_jacobian
import numpy as np
import pickle
#import time
import ipdb

########################################################################################################################
# Euler-Murayama path

def em_step_path(x, xi, h, args):
    """This will propagate the state x one step forward by euler-murayama

    step size is h and the weiner process is assumed to have a scalar diffusion coefficient"""

    # unpack the arguments for the integration step
    [f, diffusion] = args

    # rescale the standard normal to variance h
    W = xi * np.sqrt(h)

    # step forward by interval h
    x_step = x +  h * l96(x, f) + diffusion * W

    return x_step


########################################################################################################################
# Stochastic Runge-Kutta, 4 step
# This is the four step runge kutta scheme for stratonovich calculus, described in Hansen and Penland 2005
# The rule has strong convergence order 1

def rk_step_path(x, xi, h, args):
    """One step of integration rule for l96 4 step stratonovich runge kutta

    Here it is assumed that the Brownian motion is given a priori, and we wish to reconstruc the path"""

    # unpack the arguments
    [f, diffusion] = args

    # rescale the standard normal to variance h
    W = xi * np.sqrt(h)

    # Define the four terms of the RK scheme recursively
    k1 = l96(x, f) * h + diffusion * W
    k2 = l96(x + .5 * k1, f) * h + diffusion * W
    k3 = l96(x + .5 * k2, f) * h + diffusion * W
    k4 = l96(x + k3, f) * h + diffusion * W

    return x + (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)

########################################################################################################################
# non-linear L96 Runge Kutta vectorized for ensembles


def l96_rk4_step(x, h, f):

    # calculate the evolution of x one step forward via RK-4
    k_x_1 = l96(x, f)
    k_x_2 = l96(x + k_x_1 * (h / 2.0), f)
    k_x_3 = l96(x + k_x_2 * (h / 2.0), f)
    k_x_4 = l96(x + k_x_3 * h, f)

    x_step = x + (h / 6.0) * (k_x_1 + 2 * k_x_2 + 2 * k_x_3 + k_x_4)

    return x_step


########################################################################################################################
# 2nd order strong taylor SDE step
# This method is derived from page 359, NUMERICAL SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS, KLOEDEN & PLATEN;
# this uses the approximate statonovich integrals defined on page 202
# this depends on rho and alpha as above


def ty_step_path(x, xi, h, args):
    """One step of integration rule for l96 second order taylor rule

    Here it is assumed that the Brownian motion is given a priori, and we wish to reconstruct it using this
    discretization scheme.  We will, however, generate an independent Brownian bridge process."""
    # Infer system dimension
    sys_dim = len(x)

    # unpack the args for the integration step
    # note that a and b are computed directly via the brownian bridge process, up to a truncation of b.  This is
    # performed outside of the integration step for this conceptual only simulation
    [alpha, rho, p, f, diffusion] = args

    # draw standard normal samples used to define the brownian bridge process
    rndm = np.random.standard_normal([sys_dim, 2*p + 2])
    mu = rndm[:, 0]
    phi = rndm[:, 1]

    zeta = rndm[:, 2: p+2]
    eta = rndm[:, p+2:]
 
    ### define the auxiliary functions of random fourier coefficients, a and b

    # denominators for the a series
    tmp = np.tile(1 / np.arange(1, p+1), [sys_dim, 1])

    # vector of sums defining a terms
    a = -2 * np.sqrt(h * rho) * mu - np.sqrt(2*h) * np.sum(zeta * tmp, axis=1) / np.pi

    # denominators for the b series
    tmp = np.tile(1 / np.arange(1, p+1)**2, [sys_dim, 1])

    # vector of sums defining b terms
    b = np.sqrt(h * alpha) * phi + np.sqrt(h / (2 * np.pi**2) ) * np.sum(eta * tmp, axis=1)


    # Compute the deterministic dxdt and the jacobian equations
    dx = l96(x, f)
    Jac_x = l96_jacobian(x)

    # vector of first order Stratonovich integrals
    J_pdelta = (h / 2) * (np.sqrt(h) * xi + a)

    def Psi(l, j):
        # psi will be a generic function of the indicies l and j, we will define psi plus and psi minus via this
        psi = h**2 * xi[l] * xi[j] / 3 + h**(1.5) * (xi[l] * a[j] + xi[j] * a[l]) / 4 + h * a[l] * a[j] / 2 \
              - h**(1.5) * (xi[l] * b[j] + xi[j] * b[l]) / (2 * np.pi)
        return psi

    # we define the approximations of the second order Stratonovich integral
    psi_plus = np.array([Psi((i-1) % sys_dim, (i+1) % sys_dim) for i in range(sys_dim)])
    psi_minus = np.array([Psi((i-2) % sys_dim, (i-1) % sys_dim) for i in range(sys_dim)])

    # the final vectorized step forward is given as
    x_step = x + dx * h + h**2 * .5 * Jac_x @ dx    # deterministic taylor step 
    x_step += diffusion * np.sqrt(h) * xi           # stochastic euler step
    x_step += diffusion * Jac_x @ J_pdelta          # stochastic first order taylor step
    x_step += diffusion**2 * (psi_plus - psi_minus) # stochastic second order taylor step

    return x_step


########################################################################################################################


def exp(args):
    """This experiment will generate initial conditions on the random attractor by iid sampling a long trajectory

    This is a function of the ensemble number which initializes the random seed, the diffusion, and the truncation p.
    We generate samples on the attractor after a long spin with the second order taylor method.  These samples are
    forward states of the trajectory, sampled at even intervals of tanl along the trajectory."""

    ####################################################################################################################
    #t_0 = time.time()
    # load the parameters from the arguments 
    [x_init, i, seed, diff, h, f] = args
    sys_dim = len(x_init)


    # number of ensemble members generated from the initial condition
    N_ens = 100

    # time at which we compute an analysis of the ensemble in continuous time
    tanl = .01

    # the number of analyses we produce of the forward ensemble
    nanl = 2000

    # fourier truncation
    p = 1
    
    # static parameters based on fourier truncation
    RHO = rho(p)
    ALPHA = alpha(p)

    # set the storage for the ensemble means
    t_mean = np.zeros([sys_dim, nanl])
    e_mean = np.zeros([sys_dim, nanl])
    r_mean = np.zeros([sys_dim, nanl])
    a_mean = np.zeros([sys_dim, nanl])

    # set the storage for the spread of ensembles
    t_spread = np.zeros([nanl])
    e_spread = np.zeros([nanl])
    r_spread = np.zeros([nanl])
    a_spread = np.zeros([nanl])
    
    # we copy the initial condition into N_ens copies to forward propagate
    X_t_ens = np.tile(x_init, (N_ens, 1))
    X_e_ens = np.tile(x_init, (N_ens, 1))
    X_r_ens = np.tile(x_init, (N_ens, 1))
    X_a_ens = np.tile(x_init, (N_ens, 1))

    # set random seed for the same ensemble noise processes
    np.random.seed(seed)

    # for each forward time when we analyze the ensemble
    for j in range(nanl):
        #looping over the ensemble member
        for k in range(N_ens):
            # integrate until the next sample time
            for l in range(int(tanl/h)):
                # generate the weiner process over the interval at a fine discretization
                xi = np.random.standard_normal([sys_dim, int(round(tanl / 0.001))])

                # then compute the brownian motion a the current step size, re-normalized to unit variance
                tmp = np.zeros([sys_dim, int(round(tanl / h))])
                for m in range(int(round(tanl / h ))):
                    tmp[:, m] = np.sum(xi[:, m * int(h / 0.001) : (m + 1) * int(h / 0.001)], axis=1) / np.sqrt(h / 0.001)
                
                # reset xi to be the Brownian path as generated by the finer discretization, normalized to have each component
                # drawn from a normal of unit variance
                xi = tmp


                # recursivley integrating one step forward via second order taylor, EM and RK schemes
                # note that the same weiner process is utilized for each integration scheme
                X_t_ens[k, :] = ty_step_path(X_t_ens[k, :], np.squeeze(xi[:, l]), h, [ALPHA, RHO, p, f, diff])
                X_e_ens[k, :] = em_step_path(X_e_ens[k, :], np.squeeze(xi[:, l]), h, [f, diff])
                X_r_ens[k, :] = rk_step_path(X_r_ens[k, :], np.squeeze(xi[:, l]), h, [f, diff])
                X_a_ens[k, :] = l96_rk4_step(X_r_ens[k, :], h, f)
  
            # make a final perturbation by the same Brownian process all at the end instead, for the ad hoc method
            ipdb.set_trace()
            X_a_ens[k, :] = X_a_ens[k, :] + diff * np.sum(xi * h, axis=1)
  
        ### then produce statistics of the ensemble at the analysis time
        
        # the ensemble mean for each method
        t_mean[:, j] = np.mean(X_t_ens, axis=0)
        e_mean[:, j] = np.mean(X_e_ens, axis=0)
        r_mean[:, j] = np.mean(X_r_ens, axis=0)
        a_mean[:, j] = np.mean(X_a_ens, axis=0)

	# we compute the spread as in whitaker & louge 98 by the standard deviation of the mean square deviation of the ensemble
        t_spread[j] = np.sqrt( ( 1 / (N_ens - 1) ) * np.sum(np.mean( (np.squeeze(t_mean[:, j]) - X_t_ens)**2, axis=1)))
        e_spread[j] = np.sqrt( ( 1 / (N_ens - 1) ) * np.sum(np.mean( (np.squeeze(e_mean[:, j]) - X_e_ens)**2, axis=1)))
        r_spread[j] = np.sqrt( ( 1 / (N_ens - 1) ) * np.sum(np.mean( (np.squeeze(r_mean[:, j]) - X_r_ens)**2, axis=1)))
        a_spread[j] = np.sqrt( ( 1 / (N_ens - 1) ) * np.sum(np.mean( (np.squeeze(a_mean[:, j]) - X_a_ens)**2, axis=1)))

    data = {
            'e_mean': e_mean, 'e_spread': e_spread, 
            'r_mean': r_mean, 'r_spread': r_spread, 
            't_mean': t_mean, 't_spread': t_spread, 
            'a_mean': a_mean, 'a_spread': a_spread 
            }
    
    fname = './data/ensemble_stats/' \
            'ensemble_statistics_h_' + str(h).zfill(3) + '_sys_dim_' + str(sys_dim).zfill(2) + '_tanl_' + \
            str(tanl).zfill(3) + '_diffusion_' + str(diff).zfill(3) + \
            '_init_con_' + str(i).zfill(6) + '.txt'
    
    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()
    #print(time.time() - t_0)
    return i

########################################################################################################################


# Code below used for a single run, for debugging purposes

#f = open('../data/obs_trajs/fine_coarse_obs/h_001/tay_obs_seed_000_sys_dim_10_analint_0.1_diffusion_0.1_h_0.001.txt', 'rb')
#tmp = pickle.load(f)
#f.close()
#
#tobs = tmp['tobs']
#params = tmp['params']
#
#args = [tobs[:, 0], 0, params[0], params[1], 0.01, params[4]]
#
#print(exp(args))
#
