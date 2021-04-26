### First version: 04/26/2021 jzhou@smail.nju.edu.cn
### Last update: 04/26/2021 jzhou@smail.nju.edu.cn
### HINSA identification

typical_size=3
typical_velocity_width=10

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.utils import data
from spectral_cube import SpectralCube
from astropy.wcs import WCS
from reproject import reproject_interp

info=np.loadtxt('sun_17_eq',dtype=str)[50]
print(info)
hdul=fits.open(info[3])
cube = SpectralCube.read(hdul)
print(hdul[0].header)
pixelsize=abs(hdul[0].header['CDELT1'])


# In[2]:


center=info[1:3].astype(float)
center[0]+=0.1
center[1]+=0.1
lat_range = [center[1]-0.3, center[1]+0.3] * u.deg
lon_range = [center[0]-0.3, center[0]+0.3] * u.deg
sub_cube = cube.subcube(xlo=lon_range[0], xhi=lon_range[1], ylo=lat_range[0], yhi=lat_range[1])
sub_cube_slab = sub_cube.spectral_slab((float(info[4])-30) *u.km / u.s,(float(info[4])+20) *u.km / u.s)
print(sub_cube_slab)
vel=np.linspace(-82905.835,-82905.835+184.0307099*272,273)


# In[3]:


from scipy.optimize import curve_fit
def gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def triplegauss(x,a0,x0,sigma0,a1,x1,sigma1,a2,x2,sigma2):
    return gaussian(x,a0,x0,sigma0)+gaussian(x,a1,x1,sigma1)+gaussian(x,a2,x2,sigma2)
p0                   = ([50 , -60000 , 300   , 40   , -50000  , 300   ,60   ,-40000    , 300])   # peak , mean , width in Lsun/Hz , GHz , GHz
param_bounds         = ([20 , -65000 ,   0   , 0    ,  -52500 , 0     ,   0 ,  -43000  ,    0],
                        [100, -55000 , 10000 , 100  ,-42500   , 10000 ,100  ,-40000    , 10000])
shape=np.shape(sub_cube_slab)
sub_cube_slab[:,5,9].quicklook()
result=np.zeros([9,shape[1],shape[2]])
gauss_1st=np.zeros(shape)

for i in range(1):
    for j in range(1):
        print(i,j)
        spec=sub_cube_slab[:,i,j]
        popt,pcov = curve_fit(triplegauss,vel, spec,p0=p0, bounds=param_bounds)
        result[:,i,j]=popt
        gauss_1st[:,i,j]=gaussian(vel,result[0,i,j],result[1,i,j],result[2,i,j])
print(result)

print("popt",popt)
#print("pcov",pcov)
def show_fit(spec):
    popt,pcov = curve_fit(triplegauss,vel, spec,p0=p0, bounds=param_bounds)
    plt.clf()
    fig, ax_f = plt.subplots(1, sharex=True, sharey=False)
    ax_f.plot(vel, spec, linewidth=1,  label=r'data')
    ax_f.plot(vel, triplegauss(vel,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8]),color='red',linewidth=2,label='Overall fit',alpha=0.6)
    ax_f.plot(vel, gaussian(vel,popt[0],popt[1],popt[2]),color='red',linewidth=0.8,label='Individual fit',alpha=0.6)
    ax_f.plot(vel, gaussian(vel,popt[3],popt[4],popt[5]),color='red',linewidth=0.8,alpha=0.6)
    ax_f.plot(vel, gaussian(vel,popt[6],popt[7],popt[8]),color='red',linewidth=0.8,alpha=0.6)


# In[4]:


plt.imshow(result[7,:,:])
plt.colorbar()


# In[5]:


plt.imshow(gauss_1st[260,:,:])
plt.colorbar()


# In[6]:


chan=163
sub_cube_slab[chan,:,:].quicklook()
print(vel[chan]/1000)
#show_fit(sub_cube_slab[:,17,17])


# In[7]:


sub_cube_slab[:,32,28].quicklook()
param_bounds         = ([20  , -65000    ,   0    , 0    ,  -52500 , 0     ,   0 ,   -43000,      0],
                        [100, -55000 , 10000 , 100  ,-42500, 10000,100  ,-40000, 10000])
show_fit(sub_cube_slab[:,18,18])


# In[8]:


#initial substract
center_pixel=[int((shape[1]-1)/2),int((shape[2]-1)/2)]
r=typical_size/60/pixelsize
print(r)
def get_region(center,r):
    position=[]
    background=[]
    for i in range(int(np.floor(center[0]-r)),int(np.floor(center[0]+r))+1):
        for j in range(int(np.floor(center[1]-r)),int(np.floor(center[1]+r))+1):
            if(np.sqrt((i-center[0])**2+(j-center[1])**2)<=r):
                position.append([i,j])
            if((np.sqrt((i-center[0])**2+(j-center[1])**2)>r)*(np.sqrt((i-center[0])**2+(j-center[1])**2)<1.5*r)):
                background.append([i,j])
    return position,background
region,bkg=get_region(center_pixel,r)
spectrum=np.zeros([shape[0],len(region)])
for i in range(len(region)):
    spectrum[:,i]=sub_cube_slab[:,region[i][0],region[i][1]]
target_spec=np.mean(spectrum,axis=1)

spectrum_bkg=np.zeros([shape[0],len(bkg)])
for i in range(len(bkg)):
    spectrum_bkg[:,i]=sub_cube_slab[:,bkg[i][0],bkg[i][1]]
bkg_spec=np.mean(spectrum_bkg,axis=1)
print(region)
plt.plot(vel,target_spec)
plt.show()
plt.plot(vel,bkg_spec)
plt.show()
plt.plot(vel,target_spec-bkg_spec)


# In[9]:


model_image=(sub_cube_slab[153,:,:]+sub_cube_slab[173,:,:])/2
import os
os.system('rm *model*.fits')
os.system('rm *residual*.fits')
fits.writeto('model_origin.fits',np.array(model_image))
index=[]
data=[]
model_data=[]
import copy
cube_model=np.array(sub_cube_slab)
for chan in range(153,173):
    target_data=sub_cube_slab[chan,:,:]
    for i in range(shape[1]):
        for j in range(shape[2]):
            if([i,j] not in region):
                index.append([i,j])
                data.append(target_data[i,j])
                model_data.append(model_image[i,j].value)
    param_bounds=([-50],[50])
    a=np.mean(np.array(data)-np.array(model_data))
    cube_model[chan,:,:]=model_image+a
plt.imshow(cube_model[160,:,:])
origin_sub=np.array(sub_cube_slab)
residual=origin_sub-cube_model


# In[10]:


plt.plot(vel,residual[:,17,17])
plt.imshow(residual[167,:,:])
hdul[0].data=residual
fits.writeto('residual.fits',residual)


# In[11]:



fits.writeto('model.fits',cube_model)


# In[12]:


from scipy.interpolate import RegularGridInterpolator
from numpy import linspace, zeros, array
ra = linspace(283.758333 ,284.358334,37)
dec = linspace(4.958339,5.558340,37)
v = linspace(-81985.681,-31929.328,273)

T = np.array(sub_cube_slab)

for point in region:
    T[153:173,point[0],point[1]]=np.nan
os.system('rm nontarget.fits')
fits.writeto('nontarget.fits',T)
fn = RegularGridInterpolator((v,ra,dec), T,method='nearest')

pts = array([[284,5,284],[-55000,5.1,284]])
#fn(pts)
cube_fit=np.zeros([173-153,3])
for point in region:
    cube_fit[:,0]=v[153:173]
    cube_fit[:,1],cube_fit[:,2]=ra[point[0]],dec[point[1]]
    T[153:173,point[0],point[1]]=fn(array(cube_fit))
os.system('rm model_interp.fits')
fits.writeto('model_interp.fits',T)


# In[13]:


from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
kernel = Gaussian2DKernel(x_stddev=1)
T = np.array(sub_cube_slab)
os.system('rm origin.fits')
fits.writeto('origin.fits',T)
for point in region:
    T[:,point[0],point[1]]=np.nan
for chan in range(273):
    T[chan,:,:]=interpolate_replace_nans(T[chan,:,:],kernel)
os.system('rm *con*.fits')
fits.writeto('model_convo.fits',T)
residual_con=origin_sub-T
fits.writeto('residual_con.fits',residual_con)


# In[14]:


plt.imshow(T[155,:,:])
print(region)


# In[ ]:




