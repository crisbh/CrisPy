#!/usr/bin/env python

# This script read data from grav_ files and save to .npy
# Fortran unformatted files seem to be composed with several blocks of different datatype.
# Each block looks like this:
# |(int) Size of DATA || (dtype) DATA ||(int) Size of DATA |

import numpy as np
import struct
from struct import *
import time
import sys
import os


programName = sys.argv[0].rsplit('/')[-1]
programOptionsDesc = programName + ' ' + ' '.join(sys.argv[1:])
help = """
Use this to rewrite the output of the linear density reconstruction into a single file storing only the relevant information:
usage:  %s   path_to_folder_where_output_is_stored     number_of_files     level_min

where:
    path_to_folder_where_output_is_stored -- needs to end with "/"
    number_of_files -- gives the number of file
    level_min -- gives the grid level used to calculate the number of grid cells along each direction = 2 ** level_min

The output is written in the folder as the input files and will be named: "grav_*****.out"
""" % ( programName )

if len(sys.argv) not in [4]:
    print(help)
    sys.exit( 1 )


#t1 = time.clock()

#simulation details
levelmin  = int(sys.argv[3]) # 2^levelmin = grid size^1/3
levelmax  = levelmin
twotondim = 8
numGrid   = 2**levelmin
dX        = 1. / numGrid     # grid spacing

#files to be read
numfiles = int(sys.argv[2]) # number of files in output_00002
path     = sys.argv[1]      # reconstruction code output
fileroot = 'grav_'+sys.argv[1][-6:-1]+'.out'
outfile  = fileroot


def readIndices( pmax0, N, ncache ):
    '''Function that returns the number of bytes corresponding to each value saved in the Ramses files.'''
    pmin = pmax0 + 8*N-4 + (N-1)*8*ncache
    pmax = pmin + ncache*8
    return pmin, pmax


# read the data and save it in the output array
# store: phi,f1,f2,f3,gr5,gr6,gr7,gr8,gr9,gr10,rho,gr_m1,gr_m2,gr_m3,gr_m4,x,y,z
data  = np.empty( (numGrid,numGrid,numGrid,18), np.float32 )
Ngrid = 0
for i in range( 1, numfiles+1 ):
#    t2 = time.clock()
    infile = fileroot + str(i).zfill(5)
    with open( path+infile,'rb') as f:
        content = f.read()
    print( 'Reading "'+ path+infile +'"  : ' ),
    pmin = 0
    pmax = 48

    ############ 'header info'############
    info = unpack('i'*3*4,content[pmin:pmax])
    ncpu,ndim,nlevelmax,nboundary = info[1],info[4],info[7],info[10]
    #~ print  ncpu,ndim,nlevelmax,nboundary

    ############ levels ############
    for ilevel in range(levelmin,levelmax+1):
        ############ nboundary+ncpu ############
        for ibound in range( 1, nboundary+ncpu+1 ):
            pmin0 = pmax
            pmax0 = pmin0 + 4*3*2
            info = unpack('i'*3*2,content[pmin0:pmax0])
            currlevel, ncache = info[1], info[4]
            Ngrid += ncache*twotondim
            #~ print Ngrid, ilevel, pmin0, pmax0, currlevel, ncache

            if ncache == 0:
                pmax = pmax0
                continue

            ############ twotondim ############
            for ind in range(1,twotondim+1):
                j=0

                ############ get the coordinates: x, y, z ############
                index = {}
#                for N in ([1,2,3]):
                for N in ([16,17,18]):
                    pmin, pmax = readIndices( pmax0, N, ncache )
                    info = unpack( 'd'*ncache, content[pmin:pmax] )
                    index[N] = ( np.array(info) /dX ).astype(np.int)   # gives the index in the output array

                ############ read everything: x,y,z, q1,q2,q3, rho  ############
#                for N in ([1,2,3,9,10,11,12]):
                for N in ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]):
                    pmin, pmax = readIndices( pmax0, N, ncache )
                    info = unpack( 'd'*ncache, content[pmin:pmax] )
#                    data[ index[1], index[2], index[3], j ] = info
                    data[ index[16], index[17], index[18], j ] = info
                    j = j+1
                pmax0=pmax+4
            pmax=pmax0
    f.close()
#    print ( str(time.clock()-t2)+' s')

print( 'There were '+ str(Ngrid) +' total entries for a grid of size '+ str(numGrid) +'^3, which corresponds to %.2f%% entries.' % (Ngrid*100./numGrid**3) )


# write the data to file
#t2 = time.clock()
print( 'Saving to output file "'+path+outfile+'"  : ' ),
data.shape = -1, data.shape[-1]
np.save( path+outfile, data )
#print( str(time.clock()-t2)+' s' )


print('Done')
#print( 'Total time : '+str(time.clock()-t1)+' s, N = %d'%( numGrid**3 ) )
sys.exit()
