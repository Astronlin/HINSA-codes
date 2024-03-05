## cut out a 3D cube along specific axes.
## First version: 04/24/2021 lingrui llrw@163.com
## Last update: 04/25/2021 lingrui llrw@163.com
## Write out data and wcs information

import sys
import numpy as np
import warnings
import astropy.units as u
from spectral_cube import SpectralCube
import os
from astropy.io import fits

cubefile     =    str(sys.argv[1])
centerRA     =    str(sys.argv[2])
centerDEC    =    str(sys.argv[3])
sizeRA       =    str(sys.argv[4])
sizeDEC      =    str(sys.argv[5])
coord        =    str(sys.argv[6])
vl           =    eval(sys.argv[7])
vu           =    eval(sys.argv[8])

cube         =    SpectralCube.read(cubefile)
crtf_str     =    'centerbox[[{},{}],[{},{}]],coord={}'.format(centerRA,centerDEC,sizeRA,sizeDEC,coord)
subcube      =    cube.subcube_from_crtfregion(crtf_str)

subcube_slab = subcube.spectral_slab(vl*u.km/u.s,vu*u.km/u.s)
    
subcube_slab = subcube_slab.with_spectral_unit(u.km/u.s) # unit conversion


info_out = '_cubecut_{}_{}_{}_{}_{}_{}'.format(centerRA,centerDEC,sizeRA,sizeDEC,vl,vu)
fileout = cubefile[:-5]+info_out+'.fits'
sys.stdout.write ( "Writing results in %s\n" % fileout)
if os.path.exists (fileout):
    os.unlink (fileout)
hduout = fits.PrimaryHDU(np.array(subcube_slab.unmasked_data[:]),header=subcube_slab.header)
hduout.writeto(fileout)


