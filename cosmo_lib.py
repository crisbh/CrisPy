import numpy as np
from scipy.integrate import simps


def Hubble(z, H0, Omega_m0, conformal=False):
    """
    Return Hubble parameter in units of H0 (typically km/s/Mpc)
    """
    #    H0/= 3.08568E19 # in 1/s
    H = H0 * np.sqrt(Omega_m0 * (1.0 + z) ** 3 + (1.0 - Omega_m0))
    if conformal:
        H /= 1 + z
    return H


def Hubble_horizon(z, H0, Omega_m0):
    """
    Return Hubble horizon size in units of Mpc
    """
    c = 3.0e5  # speed of light in km/s
    H = H0 * np.sqrt(Omega_m0 * (1.0 + z) ** 3 + (1.0 - Omega_m0))
    return c / H


def D_a(a, H0, Omega_m0):
    """
    Returns linear growth factor at a given scale factor
    """
    a_grid = np.linspace(1, a, 50)
    H = H0 * np.sqrt(Omega_m0 / a_grid**3 + (1 - Omega_m0))
    Omega_ma = Omega_m0 / a_grid**3 * (H0 / H) ** 2
    D_int = (Omega_ma**0.545 - 1.0) / a_grid
    D_exp = simps(D_int, dx=a_grid[1] - a_grid[0])
    D_a = a * np.exp(D_exp)
    return D_a


def f_lin(a, H0, Omega_m0):
    """
    Returns linear growth rate using a growth index parametrisation
    """
    H = H0 * np.sqrt(Omega_m0 / a**3 + (1 - Omega_m0))
    Omega_ma = Omega_m0 / a**3 * (H0 / H) ** 2
    f_growth = Omega_ma**0.545
    return f_growth


def delta_lin_tophat_EdS(delta_i, k_pert, aini, aexp, x=[], y=[], z=[]):
    """
        Returns the linear density perturbation of the quasi-spherical
        tophat model at a target aexp, assuming an EdS (dust) Universe.
        Notice that x, y, z in 0,1
    """
    delta_ini = delta_i/3. * (np.sin(k_pert * x) + np.sin(k_pert * y) + np.sin(k_pert * z)) 
    delta_lin = delta_ini * aexp / aini
    return delta_lin
