import numpy as np
import pickle
import glob
import copy
from l96 import l96
from l96 import l96_jacobian
#import ipdb

########################################################################################################################
# Non-linear model vectorized for ensembles


def l96V(x, f):
    """"This describes the derivative for the non-linear Lorenz 96 Model of arbitrary dimension n.

    This will take the state vector x of shape sys_dim X ens_dim and return the equation for dxdt"""

    # shift minus and plus indices
    x_m_2 = np.concatenate([x[-2:, :], x[:-2, :]])
    x_m_1 = np.concatenate([x[-1:, :], x[:-1, :]])
    x_p_1 = np.concatenate([x[1:,:], np.reshape(x[0,:], [1, len(x[0, :])])], axis=0)

    dxdt = (x_p_1-x_m_2)*x_m_1 - x + f

    return dxdt


########################################################################################################################
# Euler-Murayama path

def em_step_path(x, xi, h, args):
    """This will propagate the ensemble state vector one step forward by euler-maruyama

    Step size is h and the weiner process is assumed to have a scalar diffusion coefficient. The realization of the 
    Brownian motion must be supplied as a standard normal variable xi, that is to be used across methods."""
    
    # unpack the arguments for the integration step
    [f, diffusion] = args

    # rescale the standard normal to variance h
    W = xi * np.sqrt(h)

    # step forward by interval h
    x_step = x +  h * l96V(x, f) + diffusion * W

    return x_step


########################################################################################################################
# Stochastic Runge-Kutta, 4 step
# This is the four step runge kutta scheme for stratonovich calculus, described in Hansen and Penland 2005
# The rule has strong convergence order 1

def rk_step_path(x, xi, h, args):
    """One step of integration rule for l96 4 step stratonovich runge kutta

    Here it is assumed that the Brownian motion is given a priori, and we wish to reconstruct the path.  The value xi
    is a standard normal vector, pre-generated to be used across the different methods."""
    # unpack the arguments
    [f, diffusion] = args

    # rescale the standard normal to variance h
    W = xi * np.sqrt(h)

    # Define the four terms of the RK scheme recursively
    k1 = l96V(x, f) * h + diffusion * W
    k2 = l96V(x + .5 * k1, f) * h + diffusion * W
    k3 = l96V(x + .5 * k2, f) * h + diffusion * W
    k4 = l96V(x + k3, f) * h + diffusion * W

    return x + (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)

########################################################################################################################
# non-linear L96 Runge Kutta vectorized for ensembles


def l96_rk4_stepV(x, h, f):

    # calculate the evolution of x one step forward via RK-4
    k_x_1 = l96V(x, f)
    k_x_2 = l96V(x + k_x_1 * (h / 2.0), f)
    k_x_3 = l96V(x + k_x_2 * (h / 2.0), f)
    k_x_4 = l96V(x + k_x_3 * h, f)

    x_step = x + (h / 6.0) * (k_x_1 + 2 * k_x_2 + 2 * k_x_3 + k_x_4)

    return x_step
########################################################################################################################
# auxiliary functions for the 2nd order taylor expansion
# these need to be computed once, only as a function of the order of truncation of the fourier series, p

def rho(p):
        return 1/12 - .5 * np.pi**(-2) * np.sum(1 / np.arange(1, p+1)**2)

def alpha(p):
        return (np.pi**2) / 180 - .5 * np.pi**(-2) * np.sum(1 / np.arange(1, p+1)**4)


########################################################################################################################
# 2nd order strong taylor SDE step
# This method is derived from page 359, NUMERICAL SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS, KLOEDEN & PLATEN;
# this uses the approximate statonovich integrals defined on page 202
# this depends on rho and alpha as above


def l96_2tay_sde(x, h, args):
    """One step of integration rule for l96 second order taylor rule

    Note that the discretization error depends loosely on p.  rho and
    alpha are to be computed by the auxiliary functions, depending only on p, and supplied for all steps.  xi is a 
    standard normal vector to be used across all methods."""

    # Infer system dimension
    sys_dim = len(x)
    # unpack the args for the integration step
    [f, diffusion, p, RHO, ALPHA, xi] = args

    # Compute the deterministic dxdt and the jacobian equations
    dx = l96(x, f)
    Jac_x = l96_jacobian(x)

    ### random variables
    
    # Vectors xi, mu, phi are sys_dim X 1 vectors of iid standard normal variables, 
    # zeta and eta are sys_dim X p matrices of iid standard normal variables. Functional relationships describe each
    # variable W_j as the transformation of xi_j to be of variace given by the length of the time step h. The functions
    # of random Fourier coefficients a_i, b_i are given in terms mu/ eta and phi/zeta respectively.
    
    # draw standard normal samples
    rndm = np.random.standard_normal([sys_dim, 2*p + 2])
    mu = rndm[:, 0]
    phi = rndm[:, 1]

    zeta = rndm[:, 2: p+2]
    eta = rndm[:, p+2:]
    
    ### define the auxiliary functions of random fourier coefficients, a and b
    
    # denominators for the a series
    tmp = np.tile(1 / np.arange(1, p+1), [sys_dim, 1])

    # vector of sums defining a terms
    a = -2 * np.sqrt(h * RHO) * mu - np.sqrt(2*h) * np.sum(zeta * tmp, axis=1) / np.pi
    
    # denominators for the b series
    tmp = np.tile(1 / np.arange(1, p+1)**2, [sys_dim, 1]) 

    # vector of sums defining b terms
    b = np.sqrt(h * ALPHA) * phi + np.sqrt(h / (2 * np.pi**2) ) * np.sum(eta * tmp, axis=1)

    # vector of first order Stratonovich integrals
    J_pdelta = (h / 2) * (np.sqrt(h) * xi + a)
    
    ### auxiliary functions for higher order stratonovich integrals ###
    def Psi(l, j):
        # psi will be a generic function of the indicies l and j, we will define psi plus and psi minus via this
        psi = h**2 * xi[l] * xi[j] / 3 + h * a[l] * a[j] / 2 + h**(1.5) * (xi[l] * a[j] + xi[j] * a[l]) / 4 \
              - h**(1.5) * (xi[l] * b[j] + xi[j] * b[l]) / (2 * np.pi) 
        return psi

    # we define the approximations of the second order Stratonovich integral
    psi_plus = np.array([Psi((i-1) % sys_dim, (i+1) % sys_dim) for i in range(sys_dim)])
    psi_minus = np.array([Psi((i-2) % sys_dim, (i-1) % sys_dim) for i in range(sys_dim)])

    # the final vectorized step forward is given as
    x_step = x + dx * h + h**2 * .5 * Jac_x @ dx   # deterministic taylor step 
    x_step += diffusion * np.sqrt(h) * xi           # stochastic euler step
    x_step += + diffusion * Jac_x @ J_pdelta        # stochastic first order taylor step
    x_step += diffusion**2 * (psi_plus - psi_minus) # stochastic second order taylor step

    return x_step

########################################################################################################################

def analyze_ensemble(ens, truth):
    """This will compute the ensemble RMSE as compared with the true twin, and the spread.

    Here we will compute the RMSE and the geometric average of the eigenvalues"""
    
    # infer the shapes
    [sys_dim, N_ens] = np.shape(ens)
    
    # compute the ensemble mean
    mean = np.mean(ens, axis=1)
   
    # compute the RMSE of the ensemble mean
    rmse = np.sqrt( np.mean( (truth - mean)**2 ) )
    
    # compute the anomalies
    A_t = (ens.transpose() - mean) / np.sqrt(N_ens - 1)

    # and the ensemble covariances
    S = A_t.transpose() @ A_t

    # then the eigenvalues
    lam = np.linalg.eigvalsh(S)

    # we compute the spread as in whitaker & louge 98 by the standard deviation of the mean square deviation of the ensemble
    spread = np.sqrt( ( 1 / (N_ens - 1) ) * np.sum(np.mean( (mean - ens.transpose())**2, axis=1)))

    return [rmse, spread]

########################################################################################################################
# Stochastic EnKF analysis step

def enkf_stoch_analysis(ens, obs_perts, obs_cov):

    """This is a function to perform a vanilla stochastic EnKF analysis step

    this takes an ensemble, a matrix of perturbed observations and the ensemble estimated observational uncertainty, 
    thereafter performing the analysis"""
    # first infer the ensemble dimension and the system dimension 
    [sys_dim, N_ens] = np.shape(ens)

    # we compute the ensemble mean and normalized anomalies
    X_mean = np.mean(ens, axis=1)
    
    A_t = (ens.transpose() - X_mean) / np.sqrt(N_ens - 1)

    # and the ensemble covariances
    S = A_t.transpose() @ A_t

    # we compute the ensemble based gain and the analysis ensemble
    K_gain = S @ np.linalg.inv( S + obs_cov)
    ens = ens + K_gain @ (obs_perts - ens)

    return ens


##########################################################################################################################

def exp(args):

    # we unpack parameters used for the integration run
    [tru_seq, tanl, diff, obs_un, obs_h, seed] = args

    # set system paramters
    sys_dim = 10
    h = 0.001
    f = 8
    params = [f, diff]
    RHO = rho(1)
    ALPHA = alpha(1)


    # set filter parameters
    obs_dim = 10
    nanl = 25000
    burn = 5000
    N_ens = 100
    tanl_steps = int(tanl / h)

    # generate the initial condition for all filters
    X_em =  np.random.multivariate_normal(tru_seq[:, 0], np.eye(sys_dim) * obs_un, size=N_ens).transpose()
    X_rk = copy.copy(X_em)
    X_ah = copy.copy(X_em)
    X_ty = copy.copy(X_em)

    # create storage for the forecast and analysis statistics
    em_for_stat = np.zeros([2, nanl])
    em_ana_stat = np.zeros([2, nanl])

    rk_for_stat = np.zeros([2, nanl])
    rk_ana_stat = np.zeros([2, nanl])
    
    ah_for_stat = np.zeros([2, nanl])
    ah_ana_stat = np.zeros([2, nanl])

    ty_for_stat = np.zeros([2, nanl])
    ty_ana_stat = np.zeros([2, nanl])
    
    # generate the observation sequence
    tru_seq = tru_seq[:, 1: burn + nanl +1]
    obs_seq = tru_seq.transpose() + np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * obs_un, size=(burn + nanl))
    obs_seq = obs_seq.transpose()


    for i in range(nanl + burn):
        # we loop over the analysis cycles
    
        # generate the brownian process over the length of the observation interval
        W = np.random.standard_normal([sys_dim, N_ens, tanl_steps])

        for j in range(tanl_steps):
            # we take tanl_steps forward to the next observation time
            
            # first choosing the noise matrix to be used by all ensembles
            xi = np.squeeze(W[:,:,j])

            # propagate each of the ensembles forward
            X_em = em_step_path(X_em, xi, h, params)
            X_rk = rk_step_path(X_rk, xi, h, params)
            X_ah = l96_rk4_stepV(X_ah, h, f)
            
            for k in range(N_ens):
                # we compute the ensemble propagation in a non-vectorized format
                args = [f, diff, 1, RHO, ALPHA, np.squeeze(xi[:,k])] 
                X_ty[:, k] = l96_2tay_sde(np.squeeze(X_ty[:, k]), h, args)

        # make a final perturbation by the same Brownian process all at the end instead, for the ad hoc method
        X_ah = X_ah + diff * np.sum(W * h, axis=2)

        if i >= burn:
            # forecast RMSE and spread calculated
            em_for_stat[:, i - burn] = analyze_ensemble(X_em, tru_seq[:, i])
            rk_for_stat[:, i - burn] = analyze_ensemble(X_rk, tru_seq[:, i])
            ah_for_stat[:, i - burn] = analyze_ensemble(X_ah, tru_seq[:, i])
            ty_for_stat[:, i - burn] = analyze_ensemble(X_ty, tru_seq[:, i])

        # we use the perturbed observation (stochastic EnKF) so that we will want to generate the same perturbed observations
        # over each ensemble (though different accross samples)
        obs_pert = np.sqrt(obs_un) *  np.random.standard_normal([sys_dim, N_ens])
        obs_pert = (obs_pert.transpose() - np.mean(obs_pert, axis=1)).transpose()
        obs_cov = (obs_pert @ obs_pert.transpose()) / (N_ens - 1)

        # after computing the empirical observation error covariance, and the mean zero perturbations, we add these to the
        # original observation
        obs_pert = (obs_seq[:, i] + obs_pert.transpose()).transpose()

        # perform a kalman filtering step
        X_em =  enkf_stoch_analysis(X_em, obs_pert, obs_cov)
        X_rk =  enkf_stoch_analysis(X_rk, obs_pert, obs_cov)
        X_ah =  enkf_stoch_analysis(X_ah, obs_pert, obs_cov)
        X_ty =  enkf_stoch_analysis(X_ty, obs_pert, obs_cov)

        if i >= burn:
            # analysis RMSE and spread calculated
            em_ana_stat[:, i - burn] = analyze_ensemble(X_em, tru_seq[:, i])
            rk_ana_stat[:, i - burn] = analyze_ensemble(X_rk, tru_seq[:, i])
            ah_ana_stat[:, i - burn] = analyze_ensemble(X_ah, tru_seq[:, i])
            ty_ana_stat[:, i - burn] = analyze_ensemble(X_ty, tru_seq[:, i])


    data = {
            'em_for_stat': em_for_stat, 'em_ana_stat': em_ana_stat,
            'rk_for_stat': rk_for_stat, 'rk_ana_stat': rk_ana_stat,
            'ah_for_stat': ah_for_stat, 'ah_ana_stat': ah_ana_stat,
            'ty_for_stat': ty_for_stat, 'ty_ana_stat': ty_ana_stat
            }

    fname = './data/ens_bias_data_final/ens_bias_diff_' + str(diff).zfill(2) + '_tanl_' + str(tanl).zfill(2) + '_obs_un_' + str(obs_un).zfill(2) + \
            '_seed_' + str(seed).zfill(2) + \
            '_nens_' + str(N_ens).zfill(4) + '_nanl_' + str(nanl).zfill(3) + '_h_' + str(h).zfill(3) + '_obs_h_' + str(obs_h).zfill(3) +  '.txt'

    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()


    return args

########################################################################################################################

#f = open('../data/obs_trajs/fine_coarse_obs/h_001/tay_obs_seed_000_sys_dim_10_analint_0.1_diffusion_0.1_h_0.001.txt', 'rb')
#tmp = pickle.load(f)
#f.close()
#
#tobs = tmp['tobs']
#params = tmp['params']
#
#args = [tobs, params[2], params[1], .25, params[3], params[0]] 
#
#print(exp(args))
