import matplotlib.pyplot as plt
from scipy.constants import *
import numpy as np
import os
from   astropy.io import fits
import astropy.units as u
import sys
from   astropy.io import ascii


filename = 'Dec+2150_06_05_arcdrift-M01_W_0008.fits'
spec     = fits.open(filename)


from astropy.table import Table
t = Table.read(filename)


spec[0].header
spec[1].header

spec[1].data.columns
# output:
#ColDefs(
#    name = 'OBSNUM'; format = '1K'
#    name = 'SCAN'; format = '1K'
#    name = 'OBSTYPE'; format = '16A'
#    name = 'QUALITY'; format = '1L'
#    name = 'UTOBS'; format = '1D'
#    name = 'DATE-OBS'; format = '24A'
#    name = 'OBJ_RA'; format = '1D'
#    name = 'OBJ_DEC'; format = '1D'
#    name = 'OFF_RA'; format = '1D'
#    name = 'OFF_DEC'; format = '1D'
#    name = 'TSYS'; format = '1D'
#    name = 'EXPOSURE'; format = '1D'
#    name = 'NCHAN'; format = '1K'
#    name = 'FREQ'; format = '1D'
#    name = 'CHAN_BW'; format = '1D'
#    name = 'BEAM_EFF'; format = '1D'
#    name = 'PRESSURE'; format = '1D'
#    name = 'TAMBIENT'; format = '1D'
#    name = 'WINDSPD'; format = '1D'
#    name = 'WINDDIR'; format = '1D'
#    name = 'DATA'; format = '262144E'; dim = '(4,65536)'
#)

spec[1].data['DATA']
spec[1].data['DATA'].shape # output: (2048, 65536, 4)

spec[1].data[:]['DATA'][0]

plt.plot(np.arange(65536),spec[1].data[:]['DATA'][0,:,1])
plt.show()
plt.close()
