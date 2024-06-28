"""
   the program is run as
     python3 program input_file output_dir
"""

import glob
import os

import numpy as np

box_size = 500.0  # Mpc/h
mesh_size = 2000

# Specify input files and output directory
input_file = "/cosma7/data/dp004/dc-barr3/DTFE_FORGE_out/momentum/DTFE_FORGE_test_snapdir_026_grid_2000_momentum_x.npy"
root_out_dir = "/cosma7/data/dp004/dc-barr3/DTFE_FORGE_out/momentum/"
# input_file   = sys.argv[1]
# root_out_dir = sys.argv[2]

"""
 Function to calculate the curl of a vector field V=(V_x,V_y,V_z) via finite difference
 array_x, array_y, array_z represent the components of V
 The V array should be in the format of a numpy array of dimension (grid_len,grid_len,grid_len).
 Here:
 grid_len (int)   is the 1D size of the grid where V is defined
 box_len  (float) is the box size in physical (e.g. Mpc/h) units
 The function will return the (x,y,z)-components of the curl.
"""


def curl_array_3D(array_x, array_y, array_z, grid_len, box_len):
    Vx1_z = np.zeros((grid_len, grid_len, grid_len))
    Vx1_y = np.zeros((grid_len, grid_len, grid_len))
    Vy1_z = np.zeros((grid_len, grid_len, grid_len))
    Vy1_x = np.zeros((grid_len, grid_len, grid_len))
    Vz1_y = np.zeros((grid_len, grid_len, grid_len))
    Vz1_x = np.zeros((grid_len, grid_len, grid_len))
    Vx2_z = np.zeros((grid_len, grid_len, grid_len))
    Vx2_y = np.zeros((grid_len, grid_len, grid_len))
    Vy2_z = np.zeros((grid_len, grid_len, grid_len))
    Vy2_x = np.zeros((grid_len, grid_len, grid_len))
    Vz2_y = np.zeros((grid_len, grid_len, grid_len))
    Vz2_x = np.zeros((grid_len, grid_len, grid_len))

    # Define left and right nodes along directions for FD:
    Vx1_z = np.roll(array_x, 1, axis=2)
    Vx1_y = np.roll(array_x, 1, axis=1)
    Vy1_z = np.roll(array_y, 1, axis=2)
    Vy1_x = np.roll(array_y, 1, axis=0)
    Vz1_y = np.roll(array_z, 1, axis=1)
    Vz1_x = np.roll(array_z, 1, axis=0)

    Vx2_z = np.roll(array_x, -1, axis=2)
    Vx2_y = np.roll(array_x, -1, axis=1)
    Vy2_z = np.roll(array_y, -1, axis=2)
    Vy2_x = np.roll(array_y, -1, axis=0)
    Vz2_y = np.roll(array_z, -1, axis=1)
    Vz2_x = np.roll(array_z, -1, axis=0)

    # Calculate FD using shifted arrays
    dx = box_len / float(grid_len)
    curl_V_x = ((Vz1_y - Vz2_y) - (Vy1_z - Vy2_z)) / (2 * dx)
    curl_V_y = ((Vx1_z - Vx2_z) - (Vz1_x - Vz2_x)) / (2 * dx)
    curl_V_z = ((Vy1_x - Vy2_x) - (Vx1_y - Vx2_y)) / (2 * dx)
    return [curl_V_x, curl_V_y, curl_V_z]


for file in glob.glob(input_file):

    absfilename = file
    tempname = os.path.basename(absfilename)
    filename = os.path.splitext(tempname)[0]

    print("Starting Calculation for (x,y,z) components of: " + filename[:-2])

    # Read files with momentum components
    mom_x = np.load(absfilename[:-5] + "x" + absfilename[len(absfilename) - 4 :])
    mom_y = np.load(absfilename[:-5] + "y" + absfilename[len(absfilename) - 4 :])
    mom_z = np.load(absfilename[:-5] + "z" + absfilename[len(absfilename) - 4 :])

    curl_mom = curl_array_3D(mom_x, mom_y, mom_z, mesh_size, box_size)

    # Save FD-Curl result
    np.save(root_out_dir + "FD_curl_" + filename[:-2], curl_mom)
    print("Calculation finished for file: " + filename[:-2])

print("Curl of 3-vector field done")
