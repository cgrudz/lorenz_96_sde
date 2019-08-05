# Module containing integration schemes for Lorenz 96
import numpy as np
#import ipdb

########################################################################################################################
# Non-linear model


def l96(x, f):
    """"This computes the time derivative for the non-linear deterministic Lorenz 96 Model of arbitrary dimension n."""

    # shift minus and plus indices
    x_m_2 = np.concatenate([x[-2:], x[:-2]])
    x_m_1 = np.concatenate([x[-1:], x[:-1]])
    x_p_1 = np.append(x[1:], x[0])

    dxdt = (x_p_1-x_m_2)*x_m_1 - x + f

    return dxdt

########################################################################################################################
# Jacobian


def l96_jacobian(x):
    """"This computes the Jacobian of the Lorenz 96, for arbitrary dimension, equation about the point x."""

    x_dim = len(x)

    dxF = np.zeros([x_dim, x_dim])

    for i in range(x_dim):
        i_m_2 = np.mod(i - 2, x_dim)
        i_m_1 = np.mod(i - 1, x_dim)
        i_p_1 = np.mod(i + 1, x_dim)

        dxF[i, i_m_2] = -x[i_m_1]
        dxF[i, i_m_1] = x[i_p_1] - x[i_m_2]
        dxF[i, i] = -1.0
        dxF[i, i_p_1] = x[i_m_1]

    return dxF

########################################################################################################################
# non-linear L96 Runge Kutta


def l96_rk4_step(x, h, f):
    """Fourth order Runge-Kutta method for step size h"""

    # calculate the evolution of x one step forward via RK-4
    k_x_1 = l96(x, h, f)
    k_x_2 = l96(x + k_x_1 * (h / 2.0), f)
    k_x_3 = l96(x + k_x_2 * (h / 2.0), f)
    k_x_4 = l96(x + k_x_3 * h, f)

    x_step = x + (h / 6.0) * (k_x_1 + 2 * k_x_2 + 2 * k_x_3 + k_x_4)

    return x_step

########################################################################################################################
# non-linear L96 2nd order Taylor Method


def l96_2tay_step(x, h, f):
    """Second order Taylor method for step size h"""

    # calculate the evolution of x one step forward via the second order Taylor expansion

    # first derivative
    dx = l96(x, f)

    # second order taylor expansion
    x_step = x + dx * h + .5 * l96_jacobian(x) @ dx * h**2

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

    Note that the discretization error depends loosely on p.  The upper bound on the error
    of the approximate stratonovich integrals is order h**2*rho, which depends on the order of truncation p.  rho and
    alpha are to be computed by the auxiliary functions, depending only on p, and supplied for all steps."""

    # Infer system dimension
    sys_dim = len(x)

    # unpack the args for the integration step
    [f, diffusion, p, rho, alpha] = args

    # Compute the deterministic dxdt and the jacobian equations
    dx = l96(x, f)
    Jac_x = l96_jacobian(x)

    ### random variables
    
    # Vectors xi, mu, phi are sys_dim X 1 vectors of iid standard normal variables, 
    # zeta and eta are sys_dim X p matrices of iid standard normal variables. Functional relationships describe each
    # variable W_j as the transformation of xi_j to be of variace given by the length of the time step h. The functions
    # of random Fourier coefficients a_i, b_i are given in terms mu/ eta and phi/zeta respectively.
    
    # draw standard normal samples
    rndm = np.random.standard_normal([sys_dim, 2*p + 3])
    xi = rndm[:, 0]
    
    mu = rndm[:, 1]
    phi = rndm[:, 2]

    zeta = rndm[:, 3: p+3]
    eta = rndm[:, p+3:]
    
    ### define the auxiliary functions of random fourier coefficients, a and b
    
    # denominators for the a series
    tmp = np.tile(1 / np.arange(1, p+1), [sys_dim, 1])

    # vector of sums defining a terms
    a = -2 * np.sqrt(h * rho) * mu - np.sqrt(2*h) * np.sum(zeta * tmp, axis=1) / np.pi
    
    # denominators for the b series
    tmp = np.tile(1 / np.arange(1, p+1)**2, [sys_dim, 1]) 

    # vector of sums defining b terms
    b = np.sqrt(h * alpha) * phi + np.sqrt(h / (2 * np.pi**2) ) * np.sum(eta * tmp, axis=1)

    # vector of first order Stratonovich integrals
    J_pdelta = (h / 2) * (np.sqrt(h) * xi + a)

    
    ### auxiliary functions for higher order stratonovich integrals ###

    # the triple stratonovich integral reduces in the lorenz 96 equation to a simple sum of the auxiliary functions, we
    # define these terms here abstractly so that we may efficiently compute the terms
    def C(l, j):
        #ipdb.set_trace()
        C = np.zeros([p, p])
        # we will define the coefficient as a sum of matrix entries where r and k do not agree --- we compute this by a
        # set difference
        indx = set(range(1, p+1))

        for r in range(1, p+1):
            # vals are all values not equal to r
            vals = indx.difference([r])
            for k in vals:
                # and for row r, we define all columns to be given by the following, inexing starting at zero
                C[r-1, k-1] = (r / (r**2 - k**2)) * ((1/k) * zeta[l, r-1] * zeta[j, k-1] + (1/r) * eta[l, r-1] * eta[j, k-1] )

        # we return the sum of all values scaled by -1/2pi^2
        return .5 * np.pi**(-2) * np.sum(C)

    def Psi(l, j):
        # psi will be a generic function of the indicies l and j, we will define psi plus and psi minus via this
        psi = h**2 * xi[l] * xi[j] / 3 + h * a[l] * a[j] / 2 + h**(1.5) * (xi[l] * a[j] + xi[j] * a[l]) / 4 \
              - h**(1.5) * (xi[l] * b[j] + xi[j] * b[l]) / (2 * np.pi) - h**2 * (C(l,j) + C(j,l))
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
# Euler-Murayama step

def l96_em_sde(x, h, args):
    """This will propagate the state x one step forward by euler-murayama

    step size is h and the weiner process is assumed to have a scalar diffusion coefficient"""
    
    # unpack the arguments for the integration step
    [f, diffusion] = args

    # infer dimension and draw realizations of the wiener process
    sys_dim = len(x)
    W =  np.sqrt(h) * np.random.standard_normal([sys_dim])

    # step forward by interval h
    x_step = x +  h * l96(x, f) + diffusion * W

    return x_step


########################################################################################################################
# Step the tangent linear model


def l96_step_TLM(x, Y, h, nonlinear_step, args):
    """"This function describes the step forward of the tangent linear model for Lorenz 96 via RK-4

    Input x is for the non-linear model evolution, while Y is the matrix of perturbations, h is defined to be the
    time step of the TLM.  This returns the forward non-linear evolution and the forward TLM evolution as
    [x_next,Y_next]"""

    h_mid = h/2

    # calculate the evolution of x to the midpoint
    x_mid = nonlinear_step(x, h_mid, args)

    # calculate x to the next time step
    x_next = nonlinear_step(x_mid, h_mid, args)

    k_y_1 = l96_jacobian(x).dot(Y)
    k_y_2 = l96_jacobian(x_mid).dot(Y + k_y_1 * (h / 2.0))
    k_y_3 = l96_jacobian(x_mid).dot(Y + k_y_2 * (h / 2.0))
    k_y_4 = l96_jacobian(x_next).dot(Y + k_y_3 * h)

    Y_next = Y + (h / 6.0) * (k_y_1 + 2 * k_y_2 + 2 * k_y_3 + k_y_4)

    return [x_next, Y_next]


########################################################################################################################
