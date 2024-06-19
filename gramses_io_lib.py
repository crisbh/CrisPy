import os

import numpy as np
import pandas as pd

"""
Useful functions to load output GRAMSES grav data.
To be used in combination with the readgrav script.
"""


def read_snapshot_info(filename_snapshot, debug=False):
    """
    Look for the info.txt file associated to the snapshot
    """
    root_dir_snapshot = os.path.dirname(os.path.abspath(filename_snapshot))
    snapshot_index = root_dir_snapshot[-3:]
    filename_snapshot_info = root_dir_snapshot + "/info_00" + snapshot_index + ".txt"

    # Load the relevant data from the info file (rest is Hilbert ordering)
    snapshot_info_file = pd.read_csv(
        filename_snapshot_info,
        delim_whitespace=True,
        header=None,
        names=["col1", "col2", "col3"],
    )
    snapshot_info_file = snapshot_info_file[:17]

    print(f"Loaded info file {filename_snapshot_info}")

    if debug:
        print("Contents of file:", snapshot_info_file)

    # Function to find third column values based on first column
    def lookup_info(first_col_entry, dataframe):
        df = dataframe
        value: dataframe  # fix diagnostics Quickfix list
        value = df[df["col1"] == first_col_entry]["col3"].values
        if debug:
            print(f"for {first_col_entry} found value = {value} ")
        if len(value) == 0 or pd.isna(value[0]):
            print(f"Value in col3= {value}. Will try using col2 instead")
            value = df[df["col1"] == first_col_entry]["col2"].values
            print(f"Value in col2= {value}")
            if not (len(value) == 0 or pd.isna(value[0])):
                print("col2 looks OK. Using that one...")

        return value

    # Getting info from info.txt file
    nstep_coarse_info = int(lookup_info("nstep_coarse=", snapshot_info_file))
    boxL_info = float(lookup_info("boxlen", snapshot_info_file))
    levelmin_info = int(lookup_info("levelmin", snapshot_info_file))
    levelmax_info = int(lookup_info("levelmax", snapshot_info_file))
    ncpu_info = int(lookup_info("ncpu", snapshot_info_file))
    aexp_info = float(lookup_info("aexp", snapshot_info_file))
    H0_info = float(lookup_info("H0", snapshot_info_file))
    Omegam0_info = float(lookup_info("omega_m", snapshot_info_file))

    assert (
        levelmin_info == levelmax_info
    ), "Error: I expect levelmin=levelmax when dealing with readgrav data. Please check"

    print("parameters found in the info file:")
    print(f"nstep_coarse = {nstep_coarse_info}")
    print(f"boxL = {boxL_info} [Mpc/h]")
    print(f"levelmin = {levelmin_info}")
    print(f"ncpu = {ncpu_info}")
    print(f"aexp = {aexp_info}")
    print(f"H0   = {H0_info} [km/s/Mpc]")
    print(f"Omega_m0 = {Omegam0_info}")

    print("Done reading info file.")
    print("-----------------------")

    return [
        nstep_coarse_info,
        boxL_info,
        levelmin_info,
        ncpu_info,
        aexp_info,
        H0_info,
        Omegam0_info,
    ]


def load_readgrav_data(filename_snapshot, box_size, levelmin, aexp):
    """
    Read GRAMSES simulation data.
    In particular, read the grav-type files that have been converted from Fortran format
    to python format using the readgrav.py script. As such, knowledge about which was stored
    into which array is required to use this function.

    returns:
            - for scalar fields, returns 3D array.
            - for vector fields, returns (3,3D) array, where first index are x,y,z components.

    """

    grid_size = int(2**levelmin)

    # Read file translated from fortran format to python by the readgrav_gr.py code
    data_file = np.load(filename_snapshot)
    grav_raw = np.array(data_file)
    print(f"Loaded {filename_snapshot}")
    print(f"Shape of data array: {grav_raw.shape}")

    rho = grav_raw[:, 0]
    grad_b = np.array([grav_raw[:, 1], grav_raw[:, 2], grav_raw[:, 3]])
    gr_pot = np.array(
        [
            grav_raw[:, 4],
            grav_raw[:, 5],
            grav_raw[:, 6],
            grav_raw[:, 7],
            grav_raw[:, 8],
            grav_raw[:, 9],
        ]
    )
    gr_mat = np.array(
        [
            grav_raw[:, 10],
            grav_raw[:, 11],
            grav_raw[:, 12],
            grav_raw[:, 13],
            grav_raw[:, 14],
        ]
    )
    x = np.array(grav_raw[:, 15])
    y = np.array(grav_raw[:, 16])
    z = np.array(grav_raw[:, 17])

    # Construct 3D box of the various fields
    print(f"Mapping into 3D arrays with grid size {grid_size}...")

    # Initialise boxes
    rho_box = np.zeros((grid_size, grid_size, grid_size))
    s0_box = np.zeros((grid_size, grid_size, grid_size))
    si_box = np.zeros((3, grid_size, grid_size, grid_size))
    Psi_box = np.zeros((grid_size, grid_size, grid_size))
    Phi_box = np.zeros((grid_size, grid_size, grid_size))
    Bi_box = np.zeros((3, grid_size, grid_size, grid_size))
    Bvi_box = np.zeros((3, grid_size, grid_size, grid_size))
    di_b_box = np.zeros((3, grid_size, grid_size, grid_size))

    # Coordinates/indices of the grid
    x_box, y_box, z_box = (grid_size * np.array([x, y, z])).astype(int)

    # Density field
    rho_box[(x_box, y_box, z_box)] = rho
    s0_box[(x_box, y_box, z_box)] = gr_mat[0]

    # Comoving momentum field
    for i, imat in enumerate([1, 2, 3]):
        # print(i, imat)
        si_box[(i, x_box, y_box, z_box)] = gr_mat[imat]

    # Shift vector components
    for i, pot in enumerate([2, 3, 4]):
        # print(f'i={i} and pot={pot}, gr_pot =', gr_pot[pot])
        Bi_box[(i, x_box, y_box, z_box)] = gr_pot[pot]

    # grad(b)
    for i in range(0, 3):
        di_b_box[(i, x_box, y_box, z_box)] = grad_b[i]
        # print(f'i={i} and di_b_box =', grad_b[i])

    # Assemble the Vector part of the shift vector (i.e. no scalar mode)
    Bvi_box = Bi_box - 4.0 * di_b_box
    # Bvi_box = Bi_box

    # Conversion from internal code units to physical units -- do not modify!
    ctilde = 3e3 / box_size
    norm_Bi = 1.0 / aexp**2 / ctilde
    norm_si = 2.0 / (ctilde * box_size) ** 2 / aexp**4

    Bvi_box *= norm_Bi
    si_box *= norm_si

    # Conformal factor perturbation (fully nonlinear)
    Psi = gr_pot[0] / (aexp * ctilde) ** 2 / 2.0

    # Lapse function perturbation (fully nonlinear)
    lapse = 1.0 + gr_pot[1] / (aexp * ctilde) ** 2 / (1.0 - Psi)
    Phi = 0.5 * (lapse**2 - 1.0)

    Phi_box[(x_box, y_box, z_box)] = Phi
    Psi_box[(x_box, y_box, z_box)] = Psi

    print("Done mapping into arrays.")
    print("-------------------------")

    return [s0_box, si_box, Phi_box, Psi_box, Bvi_box, x_box, y_box, z_box]


def a_exp(z):
    a = 1.0 / (1.0 + z)
    return a


def redshift(a):
    z = 1.0 / a - 1.0
    return z


def r2g_fac(z, box_size):
    """
    Only used for gr_pot outputs transformed with r2g.
    (not for standard particle-velocity data)
    """
    r2g_vel_fac = box_size * 100 / (a_exp(z)) ** 1.5
    return r2g_vel_fac
