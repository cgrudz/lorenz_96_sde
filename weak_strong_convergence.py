from l96 import l96
from l96 import l96_jacobian
from l96 import alpha
from l96 import rho
import numpy as np
import pickle
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
# 2nd order strong taylor SDE step
# This method is derived from page 359, NUMERICAL SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS, KLOEDEN & PLATEN;
# this uses the approximate statonovich integrals defined on page 202
# this depends on rho and alpha as above


def ty_step_path(x, xi, h, args):
    """One step of integration rule for l96 second order taylor rule

    Here it is assumed that the Brownian motion is given a priori, and we wish to reconstruct it using this
    discretization scheme"""
    # Infer system dimension
    sys_dim = len(x)

    # unpack the args for the integration step
    # note that a and b are computed directly via the brownian bridge process, up to a truncation of b.  This is
    # performed outside of the integration step for this conceptual only simulation
    [a, b, p, f, diffusion] = args

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
# auxiliary function to compute the fourier coefficients of the brownian bridge

def compute_a_b(w, fine_delta, coarse_delta, p):
    # the array w is the data over which we compute the brownian bridge
    # Delta is the time step of the fine scale Brownian process
    # p is the order of truncation of the Brownian bridge process
    [sys_dim, n_steps] = np.shape(w)

    # compute the cumulative brownian paths 
    w = np.cumsum(w, axis=1)

    # and the weighted vector of the final value, with respect to tau / coarse_delta
    W_tau_delta = np.tile( np.linspace( 1 / n_steps, 1, n_steps), [sys_dim, 1]).transpose() * np.squeeze(w[:, -1])

    # the brownian bridge is computed at each point
    bridge = (w.transpose() - W_tau_delta).transpose()

    # compute a directly from the zeroth order fourier coefficient, via the definition of the right reimann sum
    a = (2 / coarse_delta) * np.sum( bridge * fine_delta, axis=1)
    
    # we compute the b by the sin fourier components, up to the truncation at p
    b = np.zeros(sys_dim)

    for r in range(1, p+1):
        # define the matrix of the sin coefficients for the discretized brownian bridge
        sin_cof = np.sin( 2 * np.pi * r * np.linspace( 1 / n_steps, 1, n_steps) )
        sin_cof = np.tile(sin_cof, [sys_dim, 1])

        b += (1 / r) * (2 / coarse_delta) * np.sum( bridge * sin_cof * fine_delta, axis=1)

    return a, b


########################################################################################################################

def experiment(args):
    """This experiment will test strong convergence of the EM and order 2.0 Taylor scheme.

    We will initialize a fine realization of the EM scheme with discretization step of 2.5X10^(-7) as the basis for
    comparison.  We run the EM and Taylor at coarser discretizations to test at what order they match the above"""
    ####################################################################################################################
    
    # we load the arguments for the experiment, including intiial condition, diffusion coefficient, forcing f, and 
    # inital seed
    [x_init, sample_num, diff, p, f] = args
    sys_dim = len(x_init)
 
    # number of ensemble members generated from the initial condition
    N_ens = 2

    # T is the time horizon, this should be order .5 or less
    T = 0.1

    # gamma is the control parameter for how fine we take the fine representation
    gamma = 7

    # fine scale integration step
    Delta = T**gamma

    # beta controls the how fine the coarse simulation is --- note that gamma must be greater than 2 X beta
    beta = 3

    # coarse integration step sizes
    dt = T**np.arange(1, beta + 1)

    # storage for comparison between simulations
    ty = np.zeros([beta, N_ens, sys_dim])
    em = np.zeros([beta, N_ens, sys_dim])
    rk = np.zeros([beta, N_ens, sys_dim])
    
    truth = np.ones([N_ens, sys_dim])

    for N in range(N_ens):
        # reset the seed over the ensemble number for independent noise sequences from the same initial condition
        # seed = N
        #np.random.seed(seed)
       
        # we compute the integration recursively in the state x, reinitialize with the same state but different
        # noise realizations
        x = x_init
        W = np.zeros([sys_dim, int(round(T / T**gamma))])

        for i in range(int(round(T / T**gamma))):
            # step forward the reference path
            w = np.random.standard_normal(sys_dim)
            x = em_step_path(x, w, Delta, [f, diff])
            W[:, i] = w 
            
        # we store the true solution only at the final time when we make the analysis of the absolute difference
        truth[N, :]  =  x

        # then with respect to the N-th ensemble, noise process
        for i in range(beta):
            # cycle through the coarse grained discretizations
            h = dt[i]

            # define the number of discretization points in the coarse scale, over forecast time 0.1
            nanl = int(round(T / h))

            # define the coarse grained brownian increments, transformed to standard normal
            t_w = int(len(W[0, :]) / nanl)

            # initialize the euler murayama and taylor schemes
            x_em = x_init
            x_ty = x_init
            x_rk = x_init
            

            for k in range(nanl):
                # define the brownian increments compute the fourier coefficients of the brownian bridge
                xi = np.sum( W[:, k * t_w : (k + 1) * t_w], axis=1 ) / np.sqrt(t_w)
                a, b = compute_a_b(W[:, k * t_w: (k + 1) * t_w] * np.sqrt(Delta), Delta, h, p)
                
                # iterate the coarse grain solution forward
                x_em = em_step_path(x_em, xi, h, [f, diff])
                x_ty = ty_step_path(x_ty, xi, h, [a, b, p, f, diff])
                x_rk = rk_step_path(x_rk, xi, h, [f, diff])
            
            # store the final time step
            em[i, N, :] = x_em
            ty[i, N, :] = x_ty
            rk[i, N, :] = x_rk
        
    # compute the means of the difference over the ensemble of noise realizations
    em_weak = np.mean(em - truth, axis=1)
    ty_weak = np.mean(ty - truth, axis=1)
    rk_weak = np.mean(rk - truth, axis=1)
    
    # and compute the root mean square difference
    ty_weak = np.sqrt( np.sum(ty_weak*ty_weak, axis=1) / sys_dim)
    em_weak = np.sqrt( np.sum(em_weak*em_weak, axis=1) / sys_dim)
    rk_weak = np.sqrt( np.sum(rk_weak*rk_weak, axis=1) / sys_dim)

    # take the root mean square difference in euclidean norm between the em and the fine solution
    em_strong = em - truth
    em_strong = np.sqrt( np.sum(em_strong*em_strong, axis=2) / sys_dim )
    
    ty_strong = ty - truth
    ty_strong = np.sqrt( np.sum(ty_strong*ty_strong, axis=2) / sys_dim )

    rk_strong = rk - truth
    rk_strong = np.sqrt( np.sum(rk_strong*rk_strong, axis=2) / sys_dim )
    
    # and average this over the ensemble of different noise realizations
    em_strong = np.mean(em_strong, axis=1)
    ty_strong = np.mean(ty_strong, axis=1)
    rk_strong = np.mean(rk_strong, axis=1)

    data = {'em_strong': em_strong, 'em_weak': em_weak, 'ty_strong': ty_strong, 'ty_weak':ty_weak,
            'rk_strong': rk_strong, 'rk_weak': rk_weak}
    
    fname = './data/weak_strong_convergence' + \
            '_beta_' + str(beta).zfill(2) + '_nanl_' + \
            str(N_ens).zfill(4) + '_diff_' + str(diff).zfill(2) + \
            '_sample_' + str(sample_num).zfill(4) + '.txt'
    
    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()

    return i

########################################################################################################################

#
#f = open('../data/obs_trajs/tay_obs_seed_000_sys_dim_10_analint_0.1_diffusion_0.5.txt', 'rb')
#tmp = pickle.load(f)
#f.close()
#
#params = tmp['params']
#x_init = tmp['tobs']
#args = []
#args.append(x_init[:,0])
#args.append(0)
#args.append(params[1])
#args.append(1)
#args.append(params[-1])
#
#print(experiment(args))
#
