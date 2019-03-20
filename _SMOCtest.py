# Luc IJspeert
# smoc test file
##
import PyQt5
import matplotlib.pyplot as plt

import astropy.io.fits as fits

# test file
fits_image_filename = fits.util.get_testdata_filepath('test0.fits')
# open the file
hdul = fits.open(fits_image_filename)
# show info
hdul.info()
# close the file
hdul.close()

## some ways to manipulate th header
#cleaner way to open file (closes it automagically)
with fits.open(fits_image_filename) as hdul:
    # show info
    hdul.info()
    print(hdul[0].header['DATE'],'\n')
    print(hdul[0].header[7],'\n')
    hdr = hdul[0].header
    hdr['targname'] = 'NGC121-a'    # just adds the keyword to the list if it doesn't exist
    hdr[27] = 99                    # just adds the value to this position
    hdr['targname'] = ('NGC121-a', 'the observation target')    #adds description
    print(hdr.comments['targname'],'\n')
    hdr.set('observer', 'Edwin Hubble') # diff way of updating/creating
    # history/comment cards are not updated, always added
    hdr['history'] = 'I updated this file 2/26/09'
    hdr['comment'] = 'Edwin Hubble really knew his stuff'
    hdr['comment'] = 'I like using HST observations'
    print(hdr['comment'],'\n')
    # to change them, use indexing
    hdr['history'][0] = 'I updated this file on 2/27/09'
    hdr['comment'][1] = 'I like using JWST observations'
    print(repr(hdr) ,'\n')        # a more better way to read it than just print(hdr)
    # print only part
    print(repr(hdr[:3]))
    print(list(hdr.keys())) 

## open specified path
with fits.open('c:\\Users\\Luc\\Documents\\MasterStage\\baolab-0.94.1e\\dk555.fits') as hdul:
    # show info
    hdul.info() 
    print(hdul[0].header)
    print(hdul[0].data)
    
## image data
with fits.open(fits_image_filename) as hdul:
    # show info
    hdul.info()
    data = hdul[1].data         #data = hdul['sci', 1].data
    print(data.shape, data.dtype.name)
    print(data[1, 4])       # pixel value at x=5, y=2
    print(data[30:40, 10:20])
    data[30:40, 10:20] = 999
    print(data[30:40, 10:20])


with fits.open(fits_image_filename) as hdul:
    photflam = hdul[1].header['photflam']   	   # 'inverse sensitivity' (=0)
    exptime = hdr['exptime']                        # (hdr = hdul[0].header)
    data = data*photflam                            # counts to flux
    
## Update mode, flush() or close() will save changes
with fits.open(fits_image_filename, mode='update') as hdul:
    # Change something in hdul.
    hdul.flush()        # changes are written back to original.fits (not needed in 'with' open-update mode, close() achieves the same)

## or just write (changes) to file
with fits.open(fits_image_filename) as hdul:
    # writes current (in memory) data to fits
    #hdul.writeto('newtable.fits')           # wrote to C:\Users\Luc
    hdul.writeto('c:\\Users\\Luc\\Documents\\MasterStage\\newtable.fits')
    
## create new fits
import numpy as np

n = np.arange(100.0)            # a simple sequence of floats from 0.0 to (99.9)->99.
hdu = fits.PrimaryHDU(n)        # create a PrimaryHDU object to encapsulate the data
hdul = fits.HDUList([hdu])      # the hdulist to contain the PrimaryHDU
hdul.writeto('c:\\Users\\Luc\\Documents\\MasterStage\\AstroImSim\\new1.fits')       # write it to a new file
# or, shortcut for the last two steps:
hdu.writeto('c:\\Users\\Luc\\Documents\\MasterStage\\AstroImSim\\new2.fits')

print(hdul.info())
print(repr(hdul[0].header))
print(hdul[0].data)

# making an extension HDU
hdr = fits.Header()
hdr['OBSERVER'] = 'Edwin Hubble'
hdr['COMMENT'] = "Here's some commentary about this FITS file."
primary_hdu = fits.PrimaryHDU(header=hdr)
hdu = fits.ImageHDU(n)
hdul = fits.HDUList([primary_hdu, hdu])
hdul.writeto('c:\\Users\\Luc\\Documents\\MasterStage\\AstroImSim\\new3.fits')

# or just append one
hdul.append(hdu)
hdul.writeto('c:\\Users\\Luc\\Documents\\MasterStage\\AstroImSim\\new4.fits')

print(hdul.info())
print(repr(hdul[0].header))
print(hdul[2].data)

## convenience functions, not to be used in scripts
hdr = fits.getheader(fits_image_filename, 0)  # get primary HDU's header
flt = fits.getval(fits_image_2_filename, 'filter', 0)   # get primary HDU's value for 'filter'
# The function getdata() gets the data of an HDU. (can include header)
data, hdr = getdata(fits_image_filename, 1, header=True)

fits.writeto('out.fits', data, hdr)     # writes to new file
fits.append('out.fits', data, hdr)      # adds to existing or new file

fits.update(filename, dat, header=hdr, ext=5)  # update the 5th extension

# get a difference report of ext 2 of inA and inB, ignoring HISTORY and COMMMENT keywords
fits.printdiff('inA.fits', 'inB.fits', ext=2, ignore_keywords=('HISTORY','COMMENT'))    
 
 
## plot images
import matplotlib.pyplot as plt

from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)    # use nice plot parameters  

from astropy.utils.data import get_pkg_data_filename    
# get the example image
image_file = get_pkg_data_filename('tutorials/FITS-images/HorseHead.fits')    
    
# quick and dirty way
image_data = fits.getdata(image_file, ext=0)    
    
# display the image
plt.figure()
plt.imshow(image_data, cmap='gray')
plt.colorbar()
plt.show()









## distribution tests
import distributions as dists

scale = 1
maxval = 1000
minval = 20
power = -2.0
n = 10000

dist1 = dists.Schuster(power, scale, n)
#dist2 = maxval - ((maxval-minval)*np.random.power(power, n) + minval)
dist3 = []
maxval = max(dist1)
for i in range(n):
    dist3.append(dists.PowerLaw(minval, maxval, power, n))

plt.hist(np.log10(dist1), bins=20, histtype='step', label='Schuster')
#plt.hist(dist2, bins=20, histtype='step')
plt.hist(np.log10(dist3), bins=20, histtype='step', label='PowLaw')
plt.legend()
plt.show()
 
## plotting of dists
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ObjectGen as og

objects = og.Ellipsoid(10000, 'Exponential', axes=[1,5,3])

fig, ax = plt.subplots()
ax.scatter(objects[:,0], objects[:,1], marker='.', linewidths=0.0, alpha=0.5)
axis_size = max(max(objects[:,0]), max(objects[:,1]), max(objects[:,2]))
ax.set_xlim(-axis_size, axis_size) 
ax.set_ylim(-axis_size, axis_size)
ax.set(aspect='equal', adjustable='datalim')            # set equal aspect ratio and adjust datalimits to achieve (instead of axis size) 
plt.show()    
## 3d
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.scatter(objects[:,0], objects[:,1], objects[:,2], marker='.', linewidths=0.0)
ax.set_xlim3d(-axis_size, axis_size) 
ax.set_ylim3d(-axis_size, axis_size)
ax.set_zlim3d(-axis_size, axis_size)
ax.set(aspect='equal', adjustable='box')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#ax.auto_scale_xyz
plt.show()

    
## [intermezzo] plotting using new process
import _thread
import multiprocessing as mtp

import visualizer as viz

#_thread.start_new_thread(theplot, ())      # doesn't work for plotting

p = mtp.Process(target=viz.Objects2D, args=(objects,))
p.start()
#p.join()
#p.terminate()

# [NOTE] plt.show(block=False) does the same thing; actually, plotting doesn't block the program at all...
## plotting 2d arrays
array_to_disp =  np.random.randint(100, size=[100,100])

fig, ax = plt.subplots()
ax.matshow(array_to_disp)
plt.show()

## or 2d histogram
from matplotlib.colors import LogNorm
from matplotlib import cm

x_arr = np.random.randn(100000)
y_arr = np.random.randn(100000)

#fig, ax = plt.subplots()
plt.hist2d(x_arr, y_arr, bins=40, norm=LogNorm(), cmap=cm.afmhot)
plt.colorbar()
plt.show()

## colours!
viz.Objects3D(objects, colour=np.arange(2000, 20000, (20000-2000)/10000))       # old, doesn't work anymore

## convert T_eff to T_col
# Teff    Tcol
# 5780    5900
# 9500    15000

plt.plot([5780, 9500], [5900, 15000], linewidth=5)

#line:
def line(T_eff):
    return 455/186*T_eff - 766250/93
def line2(T_c):
    return 186/455*T_c + 306500/91    
# parabola, assuming the point (0,0) is included
def parabola(T_eff):
    return 0.0001500500329963205*T_eff**2 + 0.1534720549560082*T_eff
# two line segments, assuming the point (0,0) is included 
def duoline(T_eff):
    return (T_eff > 5780)*(455/186*T_eff - 766250/93) + (T_eff <= 5780)*295/289*T_eff
def duoline2(T_c):
    return (T_c > 5780)*(186/455*T_c - 306500/91) + (T_c<= 5780)*289/295*T_c

plt.plot(np.arange(1000, 15000, 100), line(np.arange(1000, 15000, 100)))
plt.plot(np.arange(1000, 15000, 100), parabola(np.arange(1000, 15000, 100)))
plt.plot(np.arange(1000, 15000, 100), duoline(np.arange(1000, 15000, 100)))
plt.xlim(-1000, 20000)
plt.ylim(-5000, 40000)
plt.xlabel('T_eff')
plt.ylabel('T_col')
plt.show()

## usefull constants and formulas
https://www.astro.princeton.edu/~gk/A403/constants.pdf
https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/magnitudes.pdf
https://en.wikipedia.org/wiki/Photometric_system
## even better colouring [NOTE] functions have been moved to NotUsed
import conversions as conv
# temps: 2500 - 40000
# edit, can go from 1000 to 40000
input_temp = np.arange(1000, 40000, 10, dtype=float)
mag = np.linspace(1.63, -0.35, len(input_temp))

T_c_from_eff = conv.Temperature(input_temp, convert='T_eff')
T_eff_from_c = conv.Temperature(input_temp, convert='T_c')
T_c_from_M = conv.Temperature(mag, convert='mag1')
T_BB_from_M = conv.Temperature(mag, convert='mag2')
T_eff_from_M = conv.Temperature(mag, convert='mag3')

colours = conv.TemperatureToRGB(input_temp).transpose()
colours2 = conv.TemperatureToRGB(T_c_from_eff).transpose()
colours3 = conv.TemperatureToRGB(T_eff_from_c).transpose()
colours4 = conv.TemperatureToRGB(T_c_from_M).transpose()
colours5 = conv.TemperatureToRGB(T_BB_from_M).transpose()
colours6 = conv.TemperatureToRGB(T_eff_from_M).transpose()

fig, ax1 = plt.subplots()

ax1.scatter(input_temp, np.ones(len(input_temp)), marker='|', s=2000, c=colours)
ax1.scatter(T_c_from_eff, np.zeros(len(input_temp))+0.9, marker='|', s=2000, c=colours2)
ax1.scatter(T_eff_from_c, np.zeros(len(input_temp))+0.8, marker='|', s=2000, c=colours3)
ax1.scatter(T_c_from_M, np.zeros(len(input_temp))+0.7, marker='|', s=2000, c=colours4)
ax1.scatter(T_BB_from_M , np.zeros(len(input_temp))+0.6, marker='|', s=2000, c=colours5)
ax1.scatter(T_eff_from_M, np.zeros(len(input_temp))+0.5, marker='|', s=2000, c=colours6)
ax1.set_xlabel('T_eff')
ax1.set_title('Colour from effective temperature')
plt.show()

## Output temps from same input temp
fig, ax1 = plt.subplots()
ax1.scatter(input_temp, input_temp, marker='.', s=10, c=colours)
ax1.scatter(input_temp, T_c_from_eff, marker='+', s=20, c=colours2)
ax1.scatter(input_temp, T_eff_from_c, marker='x', s=20, c=colours3)
ax1.set_xlabel('input_temp')
ax1.set_ylabel('temperature')
ax1.set_title('Output temps from same input temp')
plt.show()
#outcome of plots: just leave the conversion to RGB as is.
##Output temps from same input magnitude
fig, ax1 = plt.subplots()
ax1.scatter(mag, T_c_from_M, marker='.', s=10, c=colours4)
ax1.scatter(mag, T_BB_from_M, marker='+', s=20, c=colours5)
ax1.scatter(mag, T_eff_from_M, marker='x', s=20, c=colours6)
ax1.set_xlabel('mag B-V')
ax1.set_ylabel('temperature')
ax1.set_title('Output temps from same input magnitude')
plt.show()
# outcome of plots: T_eff is best fit by 8540/([B-V] + 0.865)
## Picture of objects with colours!
# cd documents/documenten_radboud_university/masterstage/astroimsim
import numpy as np
import matplotlib.pyplot as plt
import ObjectGen as obg
import visualizer as viz
import conversions as conv

temps = np.arange(1000, 15000, (15000-1000)/10000, dtype=float)

objects = obg.Ellipsoid(10000, 'Exponential', axes=[1,3,2])

viz.Objects3D(objects, colour='temperature', T_eff=temps)



## IMF
def IMF(M, M_L=0.08, M_U=100.):
    # normalized to 1 Msun
    C_U = 0.74*0.35/(0.5**(-0.35) - M_U**(-0.35))
    C_L = 2*C_U         # 0.26*-0.65/(M_L**(0.65) - 0.5**(0.65)) [only worked for M_L=0.15]
    return (M >= 0.5)*M**(-2.35)*C_U + (M < 0.5)*M**(-1.35)*C_L
    
def N1_IMF(M, M_L=0.15, M_U=100.):
    # normalized to 1 Msun
    return M**(-2.35)*0.35/(M_L**(-0.35) - M_U**(-0.35))
    
def N2_IMF(M, M_L=0.15, M_U=100.):
    # normalized to 1 probability
    return M**(-2.35)*1.35/(M_L**(-1.35) - M_U**(-1.35))
    
# worked out IMF, normalized to 1 probability 
def IMFprob(M, M_L=0.08, M_U=150, M_mid=0.5):
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = 1/(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))
    C_U = C_L*M_mid
    
    return (M < M_mid)*C_L*M**(-1.35) + (M >= M_mid)*C_U*M**(-2.35)
    # if (M < 0.5):
    #     return C_L*M**(-1.35)
    # else:
    #     return C_U*M**(-2.35)


masses = np.arange(0.08, 100, 0.01)

fig, ax1 = plt.subplots()
ax1.plot(np.log10(masses), np.log10(IMF(masses)), label='IMF')
ax1.plot(np.log10(masses), np.log10(N1_IMF(masses)), label='N1_IMF')
ax1.plot(np.log10(masses), np.log10(N2_IMF(masses)), label='N2_IMF')
ax1.plot(np.log10(masses), np.log10(IMFprob(masses)), label='IMFprob')
ax1.set_xlabel('log(M) (Msun)')
ax1.set_ylabel('log(N) relative number')
ax1.set_title('Output of the IMF')
plt.legend()
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(masses, np.log10(N1_IMF(masses)))
ax1.plot(masses, np.log10(N2_IMF(masses)))
ax1.plot(masses, np.log10(IMFprob(masses)))
ax1.set_xlabel('M (Msun)')
ax1.set_ylabel('log(N) relative number')
ax1.set_title('Output of the IMF')
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(np.log10(masses), N1_IMF(masses))
ax1.plot(np.log10(masses), N2_IMF(masses))
ax1.plot(np.log10(masses), IMFprob(masses))
ax1.set_xlabel('log(M) (Msun)')
ax1.set_ylabel('N relative number')
ax1.set_title('Output of the IMF')
plt.show()

## inverse IMF (tries)
def inverseIMF(f):
    return 0.08*(f)**(-1/2.35)
    
def inverseIMF1(f, M_U=150, M_L=0.15):
    return (f + M_U**(-2.35))**(-1/2.35) - 1 + M_L + M_U**(-2.35)
    
def inverseIMF2(f):
    return (2*2**(7/20) + 25*5**(7/20)/(2*2**(7/20))*f)**(-1/1.35)
    
random = np.random.rand(1000000) + 0.00001
masses = inverseIMF1(random)
masses2 = inverseIMF2(random)

plt.hist(np.log10(masses[masses > 10]), bins=50)
plt.show(block=False)
print(max(masses), min(masses))
print(max(masses2), min(masses2))

## inverse IMF
def invIMF(n=1, M_L=0.08, M_U=150, M_mid=0.5):
    IMF_L = IMFprob(M_L)
    IMF_U = IMFprob(M_U)            # note: value is lower than IMF_L
    IMF_mid = IMFprob(M_mid)
    IMF_dist = (IMF_L - IMF_U)*np.random.rand(n) + IMF_U
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = 1/(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))
    C_U = C_L*M_mid
    return (IMF_dist > IMF_mid)*(IMF_dist/C_L)**(-1/1.35) + (IMF_dist <= IMF_mid)*(IMF_dist/C_U)**(-1/2.35)
    
def invIMF2(f, M_L=0.08, M_U=150, M_mid=0.5):
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = 1/(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))
    C_U = C_L*M_mid
    return (IMF_dist > IMF_mid)*(IMF_dist/C_L)**(-1/1.35) + (IMF_dist <= IMF_mid)*(IMF_dist/C_U)**(-1/2.35)

M_L=0.08 
M_U=150
M_mid=0.5
IMF_L = IMFprob(M_L)
IMF_U = IMFprob(M_U)            # note: value is lower than IMF_L
IMF_mid = IMFprob(M_mid)
IMF_dist = (IMF_L - IMF_U)*np.logspace(-6.5, 0, 100) + IMF_U

masses = np.arange(0.08, 100, 0.01)
mass_dist = invIMF(1000000)
number, bins = np.histogram(mass_dist, density=True, bins='auto')

fig, ax1 = plt.subplots()
ax1.plot(np.log10(bins[:-1]), np.log10(number))
ax1.plot(np.log10(invIMF2(IMF_dist)), np.log10(IMF_dist), '+-')     # confirms the inverse is right
ax1.plot(np.log10(masses), np.log10(IMFprob(masses)))
# ax1.plot(np.log10(mass_dist), np.log10(number))
ax1.set_xlabel('log(M) (Msun)')
ax1.set_ylabel('log(N) relative number')
ax1.set_title('Output of the IMF')
plt.show()

## The correct dist. from pdf to cdf to cdf^-1
# all previous 'inverted IMF functions were wrong, since one needs to invert the cmf not pdf
# The only ones that may matter/are correct are: 
# IMFprob (forward pdf), invIMF2 (inverted pdf), CIMF (cdf) and invCIMF (inverted cdf)

def CIMF(M, M_L=0.08, M_U=150, M_mid=0.5):
    """The cumulative IMF"""
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = 1/(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))
    N_a = C_L/0.35*(M_L**(-0.35) - M**(-0.35))
    N_b = C_L*(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M**(-1.35))
    return (M < M_mid)*N_a + (M >= M_mid)*N_b

def invCIMF(n=1, M_L=0.08, M_U=150, M_mid=0.5):
    """The inverted cumulative IMF"""
    N_dist = np.random.rand(n)
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = 1/(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))
    # the mid value in the CDF
    N_mid = C_L/0.35*(M_L**(-0.35) - M_mid**(-0.35))
    # the inverted CDF
    M_a = (M_L**(-0.35) - 0.35*N_dist/C_L)**(-1/0.35)
    M_b = ((1.35/0.35*M_L**(-0.35) - M_mid**(-0.35)/0.35 - 1.35*N_dist/C_L)/M_mid)**(-1/1.35)
    return (N_dist < N_mid)*M_a + (N_dist >= N_mid)*M_b


masses = np.arange(0.08, 150, 0.01)
mass_dist = invCIMF(1000000)
number, bins = np.histogram(mass_dist, density=True, bins='auto')

fig, ax1 = plt.subplots()
# fig: probability IMF vs cumulative IMF vs inverted cumul IMF 
ax1.plot(np.log10(bins[:-1]), np.log10(number), label='inv cdf')
ax1.plot(np.log10(masses), np.log10(IMFprob(masses)), label='pdf')
ax1.plot(np.log10(masses), np.log10(CIMF(masses)), label='cdf')
ax1.set_xlabel('log(M) (Msun)')
ax1.set_ylabel('log(N) relative number')
ax1.set_title('Output of the inverted cumulative IMF')
plt.legend()
plt.show()

## total M to N_obj

def invCIMF2(N_dist, M_L=0.08, M_U=150, M_mid=0.5):
    """The inverted cumulative IMF"""
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = 1/(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))
    # the mid value in the CDF
    N_mid = C_L/0.35*(M_L**(-0.35) - M_mid**(-0.35))
    # the inverted CDF
    M_a = (M_L**(-0.35) - 0.35*N_dist/C_L)**(-1/0.35)
    M_b = ((1.35/0.35*M_L**(-0.35) - M_mid**(-0.35)/0.35 - 1.35*N_dist/C_L)/M_mid)**(-1/1.35)
    return (N_dist < N_mid)*M_a + (N_dist >= N_mid)*M_b

N_obj = 10**6
M_tot = np.sum(invCIMF2(np.linspace(0, 1, N_obj)))
#M_tot = np.sum(invCIMF(N_obj))
N_est = conv.MtotToNobj(M_tot, M_L=0.08, M_U=150, M_mid=0.5)
M_est = np.sum(invCIMF(n=int(N_est)))
print('Number of objects is {0:1.2e}% off of given input number'.format((N_est - N_obj)/N_obj))
print('Mass of objects is {0:1.2e}% off of given input mass'.format((M_est - M_tot)/M_tot))






## ObjGen and HRD, CMD
import numpy as np
import matplotlib.pyplot as plt
import ObjectGen as obg
import visualizer as vis
import formulas as form
import distributions as dist
import conversions as conv
import ObjectGen as obg
import AstObjectClass as aoc

def BVmagToTemp(BV):
    """[Experimental, use at own risk (onlky works (somewhat) up to ~10kK)]"""
    return 4600*(1/(0.92*(BV) + 1.7) + 1/(0.92*(BV) + 0.62))

M_obj = obg.ObjectMasses(N_obj=10000, M_tot=0, mass=[0.15, 0.5, 150])[0]
M_obj_act, logL_obj, logTe_obj, mag_obj = obg.IsochroneProps(M_obj, 8.2, 0.019)
M_obj_act2, logL_obj2, logTe_obj2, mag_obj2 = obg.IsochroneProps(M_obj, 6.8, 0.004)

logTe_obj = np.append(logTe_obj, logTe_obj2)
logL_obj = np.append(logL_obj, logL_obj2)

mag_names = np.array(['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K'])
mag1_obj = np.append(mag_obj[np.where(mag_names == 'R')], mag_obj2[np.where(mag_names == 'R')])
mag2_obj = np.append(mag_obj[np.where(mag_names == 'I')], mag_obj2[np.where(mag_names == 'I')])

magBV_obj = mag1_obj - mag2_obj

vis.HRD(10**logTe_obj, logL_obj)
vis.CMD(magBV_obj, mag2_obj)
# vis.HRD(BVmagToTemp(magBV_obj), logL_obj)

##
objects = obg.Ellipsoid(20000, 'Exponential', axes=[1,2,1.5])
viz.Objects3D(objects, colour='temperature', T_eff=10**logTe_obj)

## wvl/temp to RGB
import numpy as np
import matplotlib.pyplot as plt
import conversions as conv

xs = np.arange(380, 780, 0.01)
ys = np.zeros_like(xs)
input_temp = np.arange(1000, 40000, 10, dtype=float)
ytemp = np.zeros_like(input_temp)

fig, ax = plt.subplots()
ax.scatter(xs, ys, marker='|', s=2000, c=conv.WavelengthToRGB(xs).transpose())
ax.set_xlabel('Wavelength (nm)')
plt.show()

fig, ax = plt.subplots()
ax.scatter(input_temp, ytemp, marker='|', s=2000, c=conv.TemperatureToRGB(input_temp).transpose())
ax.set_xlabel('Temperature (K)')
plt.show()

## simple obj, dist=0
ast_obj = aoc.AstObject(struct='ellipsoid', N_obj=0, M_tot=1000, age=[6.7, 8, 7.5], rel_num=[1], Z=[0.019, 0.004, 0.019], distance=0)

vis.HRD(10**ast_obj.log_Te, ast_obj.log_L)
vis.CMD(ast_obj.abs_mag[1] - ast_obj.abs_mag[2], ast_obj.abs_mag[2])
vis.CMD(ast_obj.app_mag[1] - ast_obj.app_mag[2], ast_obj.app_mag[2])

## dist=10, dist=1000
ast_obj = aoc.AstObject(struct='ellipsoid', N_obj=0, M_tot=1000, age=[6.7, 8, 7.5], rel_num=[1], Z=[0.019, 0.004, 0.019], distance=10)

vis.HRD(10**ast_obj.log_Te, ast_obj.log_L)
vis.CMD(ast_obj.abs_mag[1] - ast_obj.abs_mag[2], ast_obj.abs_mag[2])
vis.CMD(ast_obj.app_mag[1] - ast_obj.app_mag[2], ast_obj.app_mag[2])

ast_obj = aoc.AstObject(struct='ellipsoid', N_obj=0, M_tot=1000, age=[6.7, 8, 7.5], rel_num=[1], Z=[0.019, 0.004, 0.019], distance=100)

vis.HRD(10**ast_obj.log_Te, ast_obj.log_L)
vis.CMD(ast_obj.abs_mag[1] - ast_obj.abs_mag[2], ast_obj.abs_mag[2])
vis.CMD(ast_obj.app_mag[1] - ast_obj.app_mag[2], ast_obj.app_mag[2])

## relative number
ast_obj = aoc.AstObject(struct='ellipsoid', N_obj=10000, M_tot=0, age=[9, 8, 6.7], rel_num=[5, 3, 1], Z=[0.019, 0.008, 0.004], distance=100)

vis.HRD(10**ast_obj.log_Te, ast_obj.log_L)

print(ast_obj.TotalLuminosity())
print(ast_obj.HalfLumRadius())
print(form.Distance(ast_obj.objects))



## power law test

N = 10000
list = []
pow1 = -3.0
pow2 = -3.0

for i in range(N):
    list.append(dist.PowerLaw(n=1, power=pow1, min=1e-6, max=10))

arr = dist.PowLaw(n=10000, power=pow2, min=1e-6, max=10)

hist1, bins1 = np.histogram(list, bins='auto')
hist2, bins2 = np.histogram(arr, bins='auto')

fig, ax = plt.subplots()
ax.step(np.log10(bins1[:-1]), np.log10(hist1), label='some powlaw, pow={0}'.format(pow1))
ax.step(np.log10(bins2[:-1]), np.log10(hist2), label='my powlaw, power={0}'.format(pow2))
ax.plot([-6, (-6 - 3/pow1)], [3.14, 0.14], label='slope {0}'.format(pow1))
ax.plot([-6, (-6 - 3/(pow2-1))], [3.14, 0.14], label='slope {0}'.format(pow2-1))
ax.legend()
plt.show()

# the other power law has a slope of 1 less than wanted. Not integrated!

## Pearson VII
scale = 1.0
xs = np.logspace(-2, 2, 100)
ys = 3*xs**2/scale**2*(1 + xs**2/scale**2)**(-2.5)

arr = dist.PearsonVII_r(n=1000000, s=scale)

hist, bins = np.histogram(arr, bins='auto', density=True)

fig, ax = plt.subplots()
# ax.step(bins[:-1], hist, label='pearson VII generator')
# ax.plot(xs, ys, label='pearson VII theoretical')
# log
ax.step(np.log10(bins[:-1]), np.log10(hist), label='pearson VII generator')
ax.plot(np.log10(xs), np.log10(ys), label='pearson VII theoretical')
ax.legend()
plt.show()

## exp (interpolated)
def Exponential_r(r, s=1.0):
    """pdf of radial exponential distribution."""
    rs = r/s
    return rs**2/(2*s)*np.exp(-rs)

def Nexponential_r(r, s=1.0):
    """cdf of radial exponential distribution."""
    rs = r/s
    return 1 - (rs**2/2 + rs + 1)*np.exp(-rs)
    
rvals = np.logspace(-2, 2, 1000)
Nvals_exp = Nexponential_r(rvals)
r_inter_exp = np.interp(np.random.rand(1000000), Nvals_exp, rvals)

hist, bins = np.histogram(r_inter_exp, bins='auto', density=True)

fig, ax = plt.subplots()
ax.step(bins[:-1], hist, label='Exp interp')
ax.plot(rvals, Exponential_r(rvals), label='Exp')
# log
# ax.step(np.log10(bins1[:-1]), np.log10(hist1), label='Exp interp')
# ax.plot(np.log10(rvals), np.log10(Exponential_r(rvals)), label='Exp')
ax.legend()
plt.show()

## norm (interpolated)
def Normal_r(r, s=1.0):
    """pdf of radial normal distribution."""
    rs2 = (r/s)**2
    return 4/(np.sqrt(np.pi)*s)*rs2*np.exp(-rs2)
    
def Nnormal_r(r, s=1.0):
    """cdf of radial normal distribution."""
    rs = r/s
    return sps.erf(rs) - 2/np.sqrt(np.pi)*rs*np.exp(-rs**2)
    
rvals = np.logspace(-2, 1, 1000)
Nvals_norm = Nnormal_r(rvals)
r_inter_norm = np.interp(np.random.rand(1000000), Nvals_norm, rvals)

hist, bins = np.histogram(r_inter_norm, bins='auto', density=True)

fig, ax = plt.subplots()
ax.step(bins[:-1], hist, label='Normal interp')
ax.plot(rvals, Normal_r(rvals), label='Normal')
# log
# ax.step(np.log10(bins[:-1]), np.log10(hist), label='Normal interp')
# ax.plot(np.log10(rvals), np.log10(Normal_r(rvals)), label='Normal')
ax.legend()
plt.show()

## Cauchy (interpolated)
def SquaredCauchy_r(r, s=1.0):
    """cdf of radial squared Cauchy distribution."""
    rs2 = (r/s)**2
    return 4/(np.pi*s)*rs2*(1 + rs2)**(-2)

def Nsquaredcauchy_r(r, s=1.0):
    """cdf of radial squared Cauchy distribution."""
    rs = r/s
    return 2/np.pi*(np.arctan(rs) - rs/(1 + rs**2))
    
rvals = np.logspace(-2, 2, 1000)
Nvals = Nsquaredcauchy_r(rvals)
r_inter = np.interp(np.random.rand(1000000), Nvals, rvals)

hist, bins = np.histogram(r_inter, bins='auto', density=True)

fig, ax = plt.subplots()
ax.step(bins[:-1], hist, label='Cauchy interp')
ax.plot(rvals, SquaredCauchy_r(rvals), label='Cauchy')
# log
# ax.step(np.log10(bins[:-1]), np.log10(hist), label='Cauchy interp')
# ax.plot(np.log10(rvals), np.log10(SquaredCauchy_r(rvals)), label='Cauchy')
ax.legend()
plt.show()

## Pearson (interpolated)
def PearsonVII_r(r, s=1.0):
    """pdf of radial Pearson type VII distribution."""
    rs2 = (r/s)**2
    return 3/s*rs2*(1 + rs2)**(-2.5)
    
def NpearsonVII_r(r, s=1.0):
    """cdf of radial Pearson type VII distribution."""
    rs = r/s
    return rs**3/(1 + rs**2)**(3/2)

rvals = np.logspace(-2, 2, 1000)
Nvals = NpearsonVII_r(rvals)
r_inter = np.interp(np.random.rand(1000000), Nvals, rvals)

hist, bins = np.histogram(r_inter, bins='auto', density=True)

fig, ax = plt.subplots()
ax.step(bins[:-1], hist, label='Pearson interp')
ax.plot(rvals, PearsonVII_r(rvals), label='Pearson')
# log
# ax.step(np.log10(bins[:-1]), np.log10(hist), label='Pearson interp')
# ax.plot(np.log10(rvals), np.log10(PearsonVII_r(rvals)), label='Pearson')
ax.legend()
plt.show()

## speed test inverse vs interp sampling
import time

many = 10**6

t1 = time.time()
r_inv = dist.PearsonVII_r(n=many)
t2 = time.time()
rvals = np.logspace(-3, 4, 1000)
Nvals = NpearsonVII_r(rvals)
r_inter = np.interp(np.random.rand(many), Nvals, rvals)
t3 = time.time()

print('Time for inverse: {0}, time for interp: {1}'.format(t2 - t1, t3 - t2))


hist, bins = np.histogram(r_inv, bins='auto', density=True)
hist2, bins2 = np.histogram(r_inter, bins='auto', density=True)

fig, ax = plt.subplots()
ax.step(bins[:-1], hist, label='Pearson inverse')
ax.step(bins2[:-1], hist2, label='Pearson interp')
ax.legend()
plt.show()

# aaaaand interp is faster!

## dist vs radial dist
# take care to remove the '_r' appending code
objects = obg.Ellipsoid(100000, dist_type='Exponential')
objects2 = obg.Ellipsoid(100000, dist_type='Exponential_r')
# objects = obg.Ellipsoid(100000, dist_type='Normal')
# objects2 = obg.Ellipsoid(100000, dist_type='Normal_r')

# objects = obg.Ellipsoid(100000, dist_type='SquaredCauchy_r')

# objects2 = obg.Ellipsoid(100000, dist_type='PearsonVII_r')

vis.Objects2D(objects)
vis.Objects2D(objects2)

vis.Objects3D(objects)
vis.Objects3D(objects2)


## Tests on radii (Cartesian, cylindrical, spherical)
import numpy as np
import matplotlib.pyplot as plt

import formulas as form
import conversions as conv
import distributions as dist

def GenRadii(N_obj=1, dist_type='Exponential_r', scale=1.0):
    r_dist = eval('dist.' + dist_type)(n=N_obj, s=scale)                                            # the radial distribution
    phi_dist = dist.AnglePhi(N_obj)                                                                 # dist for angle with x axis
    theta_dist = dist.AngleTheta(N_obj)                                                             # dist for angle with z axis
    
    xyz = conv.SpherToCart(r_dist, theta_dist, phi_dist).transpose()
    
    radii = form.Distance2D(xyz)
    radii2 = form.Distance(xyz)
    
    return radii, radii2
    
ref = -10*np.log(1-np.random.rand(10**6))   # exp dist

rvals = np.logspace(-3, 4, 1000) 
Nvals = 1 - np.exp(-rvals/10)*(1 + rvals/10)
ref2 = np.interp(np.random.rand(10**6), Nvals, rvals)

    
# the 2d and 3d radii
rads, rads2 = GenRadii(10**6, scale=10)

hist, bins = np.histogram(rads, bins=np.logspace(-0.5, 3.5, 50), density=True)
hist2, bins2 = np.histogram(rads2, bins=np.logspace(-0.5, 3.5, 50), density=True)
# hist_ref, bins_ref = np.histogram(dist.Exponential(10**6, s=10), bins=np.logspace(-0.5, 3.5, 50), density=True)
hist_ref, bins_ref = np.histogram(ref, bins=np.logspace(-0.5, 3.5, 50), density=True)
hist_ref2, bins_ref2 = np.histogram(ref2, bins=np.logspace(-0.5, 3.5, 50), density=True)


fig, ax = plt.subplots()
# ax.step(np.log10(bins[:-1]), np.log10(hist/bins[:-1]))
ax.plot(np.log10(bins[:-1]), np.log10(hist/bins[:-1]/np.max(hist/bins[:-1])), label='2D radii')
# ax.step(np.log10(bins2[:-1]), np.log10(hist2/bins2[:-1]**2))
ax.plot(np.log10(bins2[:-1]), np.log10(hist2/bins2[:-1]**2/np.max(hist2/bins2[:-1]**2)), label='3D radii')
# ax.step(np.log10(bins_ref[:-1]), np.log10(hist_ref))
ax.plot(np.log10(bins_ref[:-1]), np.log10(hist_ref/np.max(hist_ref)), label='Cartesian exp')
# ax.step(np.log10(bins_ref[:-1]), np.log10(hist_ref))
ax.plot(np.log10(bins_ref2[:-1]), np.log10(hist_ref2/bins[:-1]/np.max(hist_ref2/bins[:-1])), label='cylindrical radial exp')
plt.legend()
plt.show()

## King Globular
def KingGlobular_r(r, s=1.0, R=None):
    """pdf of radial King distribution for Globular clusters."""
    if (R is None):
        R = 30*s                                                                                  # typical globular cluster has R/s ~ 30
    rs2 = (r/s)**2
    Rs = R/s
    Rs2 = (R/s)**2
    C = (Rs**3/3/(1 + Rs2) + np.log(Rs + (1 + Rs2)**(1/2))/(1 + Rs2)**(1/2) - np.arctan(Rs))**-1
    return C/s*rs2*(1/(1 + rs2)**(1/2) - 1/(1 + Rs2)**(1/2))**2
    
def N_KingGlobular_r(r, s=1.0, R=None):
    """cdf of radial King distribution for Globular clusters."""
    if (R is None):
        R = 30*s                                                                                  # typical globular cluster has R/s ~ 30
    rs = r/s
    rs2 = (r/s)**2
    Rs = R/s
    Rs2 = (R/s)**2
    C = (Rs**3/3/(1 + Rs2) + np.log(Rs + (1 + Rs2)**(1/2))/(1 + Rs2)**(1/2) - np.arctan(Rs))**-1
    C2 = 1/(1 + Rs2)**(1/2)
    return C*(rs - np.arctan(rs) - C2*rs*(1 + rs2)**(1/2) + C2*np.log(rs + (1 + rs2)**(1/2)) + C2**2*rs**3/3)

s = 2
R = 100
rvals = np.logspace(-2, np.log10(R), 1000)
Nvals = N_KingGlobular_r(rvals, s, R)
r_inter = np.interp(np.random.rand(1000000), Nvals, rvals)

hist, bins = np.histogram(r_inter, bins='auto', density=True)

fig, ax = plt.subplots()
# ax.step(bins[:-1], hist, label='King interp')
# ax.plot(rvals, KingGlobular_r(rvals, s, R), label='King')
# log
# ax.step(np.log10(bins[:-1]), np.log10(hist), label='King interp')
# ax.plot(np.log10(rvals), np.log10(KingGlobular_r(rvals, s, R)), label='King')
# log cartesian
ax.step(np.log10(bins[:-1]), np.log10(hist/bins[:-1]**2), label='King interp')
ax.plot(np.log10(rvals), np.log10(KingGlobular_r(rvals, s, R)/rvals**2), label='King')
ax.legend()
plt.show()


## compare dists
N = 10**7
sc = 10
hist_ref, bins_ref = np.histogram(dist.Exponential_r(N, s=sc), bins=np.logspace(-0.5, 3.5, 50), density=True)
hist_ref2, bins_ref2 = np.histogram(dist.Normal_r(N, s=sc), bins=np.logspace(-0.5, 3.5, 50), density=True)
hist_ref3, bins_ref3 = np.histogram(dist.SquaredCauchy_r(N, s=sc), bins=np.logspace(-0.5, 3.5, 50), density=True)
hist_ref4, bins_ref4 = np.histogram(dist.PearsonVII_r(N, s=sc), bins=np.logspace(-0.5, 3.5, 50), density=True)
hist_ref5, bins_ref5 = np.histogram(dist.KingGlobular_r(N, s=sc), bins=np.logspace(-0.5, 3.5, 50), density=True)

fig, ax = plt.subplots()
ax.plot(np.log10(bins_ref[:-1]), np.log10(hist_ref/bins_ref[:-1]**2), label='Exponential')
ax.plot(np.log10(bins_ref2[:-1]), np.log10(hist_ref2/bins_ref2[:-1]**2), label='Normal')
ax.plot(np.log10(bins_ref3[:-1]), np.log10(hist_ref3/bins_ref3[:-1]**2), label='SquaredCauchy')
ax.plot(np.log10(bins_ref4[:-1]), np.log10(hist_ref4/bins_ref4[:-1]**2), label='PearsonVII')
ax.plot(np.log10(bins_ref5[:-1]), np.log10(hist_ref5/bins_ref5[:-1]**2), label='King')
ax.set_title('Distributions, N={0:1.1e}, s={1}, R=30*s'.format(N, sc))
ax.legend()
plt.show()

## King plot
hist_ref1, bins_ref1 = np.histogram(dist.KingGlobular_r(N, s=5), bins=np.logspace(-0.5, 3.5, 50), density=True)
hist_ref2, bins_ref2 = np.histogram(dist.KingGlobular_r(N, s=10), bins=np.logspace(-0.5, 3.5, 50), density=True)
hist_ref3, bins_ref3 = np.histogram(dist.KingGlobular_r(N, s=15), bins=np.logspace(-0.5, 3.5, 50), density=True)

fig, ax = plt.subplots()
ax.plot(np.log10(bins_ref1[:-1]), np.log10(hist_ref1/bins_ref1[:-1]**2), label='5')
ax.plot(np.log10(bins_ref2[:-1]), np.log10(hist_ref2/bins_ref2[:-1]**2), label='10')
ax.plot(np.log10(bins_ref3[:-1]), np.log10(hist_ref3/bins_ref3[:-1]**2), label='15')
ax.set_title('King, s=var, R=30*s'.format(N, sc))
ax.legend()
plt.show()


## cumul test
rvals = np.logspace(-2, 4, 1000)
N = 10**5

s_exp = 500
s_norm = 1500
s_cau = 300
s_pea = 1100
s_king = 1500

fig, ax = plt.subplots()
ax.plot(rvals, dist.cdf_Exponential_r(rvals, s=s_exp), label='Exp, s={0}'.format(s_exp), c='darkgreen', lw=0.7)
ax.plot(rvals, dist.cdf_Normal_r(rvals, s=s_norm), label='Normal, s={0}'.format(s_norm), c='blue', lw=0.7)
ax.plot(rvals, dist.cdf_Squaredcauchy_r(rvals, s=s_cau), label='Cauchy, s={0}'.format(s_cau), c='purple', lw=0.7)
ax.plot(rvals, dist.cdf_PearsonVII_r(rvals, s=s_pea), label='Pearson, s={0}'.format(s_pea), c='c', lw=0.7)
ax.plot(rvals, dist.cdf_KingGlobular_r(rvals, s=s_king, R=3800), label='King, s={0}'.format(s_king), c='black', lw=0.7)

dist1 = dist.Exponential_r(N, s=s_exp)
dist2 = dist.Normal_r(N, s=s_norm)
dist3 = dist.SquaredCauchy_r(N, s=s_cau)
dist4 = dist.PearsonVII_r(N, s=s_pea)
dist5 = dist.KingGlobular_r(N, s=s_king, R=3800)

ax.plot(np.sort(dist1), np.cumsum(dist1)/np.sum(dist1), label='cumsum Exp, s={0}'.format(s_exp), c='darkgreen', lw=0.7)
ax.plot(np.sort(dist2), np.cumsum(dist2)/np.sum(dist2), label='cumsum Normal, s={0}'.format(s_norm), c='blue', lw=0.7)
ax.plot(np.sort(dist3), np.cumsum(dist3)/np.sum(dist3), label='cumsum Cauchy, s={0}'.format(s_cau), c='purple', lw=0.7)
ax.plot(np.sort(dist4), np.cumsum(dist4)/np.sum(dist4), label='cumsum Pearson, s={0}'.format(s_pea), c='c', lw=0.7)
ax.plot(np.sort(dist5), np.cumsum(dist5)/np.sum(dist5), label='cumsum King, s={0}'.format(s_king), c='black', lw=0.7)

ax.set_xlim(0, 3000)
ax.set_ylim(0, 1)
ax.legend()
ax.set_title('Cumulative radial distribution')
plt.show()


## half something radius
astobj = aoc.AstObject(N_obj=10000, age=[8], Z=[0.019])
hlr = astobj.HalfLumRadius()
hmr = astobj.HalfMassRadius()
hlr2 = astobj.HalfLumRadius(spher=False)
hmr2 = astobj.HalfMassRadius(spher=False)
radii = astobj.ObjRadii()

print('sph: HMR={0:2.3f}, HLR={1:2.3f} \ncyl: HMR={6:2.3f}, HLR={7:2.3f} \n  min= {2:2.3f}, max = {3:2.3f} \n  mean={4:2.3f}, median={5:2.3f}'
                        .format(hmr, hlr, np.min(radii), np.max(radii), np.mean(radii), np.median(radii), hmr2, hlr2))

fig, ax = plt.subplots()
ax.scatter(astobj.objects[:, 0], astobj.objects[:, 1], marker='.', linewidths=0.0, alpha=0.5, c='grey')

xs = np.cos(np.arange(0, 2*np.pi + np.pi/16, np.pi/16))
ys = np.sin(np.arange(0, 2*np.pi + np.pi/16, np.pi/16))
ax.plot(hmr*xs, hmr*ys, c='darkviolet', label='spherical hmr')
ax.plot(hmr2*xs, hmr2*ys, c='blue', label='cylindrical hmr')
ax.plot(hlr*xs, hlr*ys, c='m', label='spherical hlr')
ax.plot(hlr2*xs, hlr2*ys, c='darkgreen', label='cylindrical hlr')

axis_size = max(max(astobj.objects[:,0]), max(astobj.objects[:,1]))
ax.set_xlim(-axis_size, axis_size) 
ax.set_ylim(-axis_size, axis_size)
ax.set(aspect='equal', adjustable='datalim')
plt.legend()
plt.show() 


## simCADO
src = sim.source.cluster()
sim.run(src, filename='data\my_first_sim.fits')

fh.ShowInfo('my_first_sim.fits')
fh.PrintHdr('my_first_sim.fits')
fh.PrintData('my_first_sim.fits')
fh.PlotFits('my_first_sim.fits')
##
sim.run(src, filename='data\my_first_sim.fits', OBS_EXPTIME=600, OBS_NDIT=6)

fh.PlotFits('my_first_sim.fits')

# OBS_EXPTIME [in seconds] sets the length of a single exposure. The default setting is for a 60s exposure
# OBS_NDIT sets how many exposures are taken. The default is 1.

## making a source
astobj = aoc.AstObject(N_obj=3, age=[8], Z=[0.019], distance=10**6)

# coords in arcsec! - fov 53 arcsec (wide) or 16 (zoom)
# available filters: B, BrGamma, CH4_169, CH4_227, Fell_166, H, H2_212, H2O_204, Hcont_158, I, J, K, Ks, NH3_153, PaBeta, R, U, V, Y, z
#                 so [U, B, V, R, I, J, H, K] are all there

x_as = np.arctan(astobj.objects[0]/astobj.distance)*648000/np.pi            # assumed to be in pc
y_as = np.arctan(astobj.objects[1]/astobj.distance)*648000/np.pi            # *648000/np.pi is rad to as

src = sim.source.stars(mags=astobj.app_mag[2]-22, x=x_as, y=y_as, filter_name='V', spec_types='M0IV')
# mode=['wide', 'zoom'], detector_layout=['small', 'centre', 'full']
sim.run(src, filename='data\my_first_sim.fits', mode='wide', detector_layout='centre', OBS_EXPTIME=600)

fh.PlotFits('my_first_sim.fits')


print('\nB-V to spectral type')
print(np.arange(-0.4, 2.4, 0.002))
print(np.unique(sim.source.BV_to_spec_type(np.arange(-0.4, 2.4, 0.002))))

##
astobj = aoc.AstObject(N_obj=3, age=[8], Z=[0.019], distance=10**6)

x_as = np.arctan(astobj.objects[0]/astobj.distance)*648000/np.pi            # assumed to be in pc
y_as = np.arctan(astobj.objects[1]/astobj.distance)*648000/np.pi            # *648000/np.pi is rad to as

src = sim.source.stars(mags=np.clip(astobj.app_mag[2]-18, 10, 30), x=x_as, y=y_as, filter_name='V', spec_types='F1V')
for i in range(100):
    astobj = aoc.AstObject(N_obj=3, age=[8], Z=[0.019], distance=10**6)
    x_as = np.arctan(astobj.objects[0]/astobj.distance)*648000/np.pi
    y_as = np.arctan(astobj.objects[1]/astobj.distance)*648000/np.pi
    src += sim.source.stars(mags=np.clip(astobj.app_mag[2]-18, 10, 30), x=x_as, y=y_as, filter_name='V', spec_types='F1V')
##

sim.run(src, filename='data\my_first_sim.fits', mode='zoom', detector_layout='centre', OBS_EXPTIME=60)

fh.PlotFits('my_first_sim.fits')









## loading/saving

hdr= '\n'.join(["line1", "line2", "line3"])
arr = np.ones([100,10])

np.savetxt('test.txt', arr, header=hdr, delimiter='         ')
# https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.savetxt.html
# numpy.loadtxt  numpy.fromstring  numpy.fromfile  numpy.load

import pickle

class Company(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

with open('company_data.pkl', 'wb') as output:
    company1 = Company('banana', 40)
    pickle.dump(company1, output, -1)

    company2 = Company('spam', 42)
    pickle.dump(company2, output, -1)

del company1
del company2

with open('company_data.pkl', 'rb') as input:
    company2 = pickle.load(input)
    print(company2.name) # -> spam
    print(company2.value)  # -> 42
    #order is what matters here
    company1 = pickle.load(input)
    print(company1.name)  # -> banana
    print(company1.value)  # -> 40

    # company2 = pickle.load(input)
    # print(company2.name) # -> spam
    # print(company2.value)  # -> 42
    
##
astobj = aoc.AstObject(N_obj=1000, age=[8], Z=[0.019], distance=10**6)

aoc.AstObject.SaveTo(astobj, 'astobj1-test.pkl')        # or astobj.SaveTo('astobj1-test.pkl')

astobj2 = aoc.AstObject.LoadFrom('astobj1-test.pkl')

## calling from Pyzo shell
import sys
import subprocess
subprocess.call([sys.executable, 'MCopdr1.py', '-N 1000', '-alpha', '1'])

subprocess.call([sys.executable, 'constructor.py', '-N 1000', '-rdistpar', '1', '-rdist', 'Exponential'])

## default args
def get_default_args(func):
    sig = inspect.signature(func)
    # {k: v.default for k, v in sig.parameters.items() if k is not 'n'}
    return {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}



## ** radiusprojection **
# 
#
##
import numpy as np
import matplotlib.pyplot as plt

import formulas as form
import visualizer as vis
import conversions as conv

def AnglePhi(n=1):
    """Uniformly chosen angle(s) between 0 and 2 pi."""
    return 2*np.pi*np.random.rand(int(n)) 
    
def AngleTheta(n=1):
    """Angle(s) between 0 and pi chosen from a sinus distribution."""
    return np.arccos(2*np.random.rand(int(n)) - 1)
    
def Radius(n=1, power=2):
    """Uniform radius"""
    k = power
    return Power(n, k)

def Power(n, k, x_max=1):
    """Positive power law distribution with power k. Draws n numbers."""
    N_dist = np.random.rand(int(n))
    return N_dist**(1/(k + 1))*x_max
    
def GenSphereProj(n, r_dist=Radius):
    """Generates the spherical dist and projects it."""
    r = r_dist(n)
    theta = AngleTheta(n)
    phi = AnglePhi(n)
    
    xyz = conv.SpherToCart(r, theta, phi).transpose()
    
    #vis.Objects2D(xyz)
    
    radii = form.Distance2D(xyz)
    return radii, r, theta

## uniform sphere
n = 10**7
rad1, r1, t1 = GenSphereProj(n)
r_ref = np.logspace(-2, np.log10(1), n/100)

hist, bins = np.histogram(rad1, bins='auto', density=True)
hist2, bins2 = np.histogram(r1, bins='auto', density=True)


fig, ax = plt.subplots()
ax.plot(r_ref, 4*np.sqrt(1 - r_ref), label='sqrt(1 - r)')
ax.plot(r_ref, 3*np.sqrt(1 - r_ref**2), label='sqrt(1 - r**2)')
ax.step(bins[1:-1], hist[1:]/bins[1:-1], label='rho')
ax.step(bins2[1:-1], hist2[1:]/bins2[1:-1]**2, label='r')

# ax.plot(np.log10(r_ref), np.log10(4*np.sqrt(1 - r_ref)), label='sqrt(1 - r)')
# ax.plot(np.log10(r_ref), np.log10(3*np.sqrt(1 - r_ref**2)), label='sqrt(1 - r**2)')
# ax.step(np.log10(bins[1:-1]), np.log10(hist[1:]/bins[1:-1]), label='rho')
# ax.step(np.log10(bins2[1:-1]), np.log10(hist2[1:]/bins2[1:-1]**2), label='r')
ax.set_title('Uniform sphere')
ax.set_xlabel('log(r)')
ax.set_ylabel('log(density)')
ax.legend()
plt.show()
    
##    
def f_r(rho, R):
    root = np.sqrt(R**2 - rho**2)
    term1 = rho*R*root
    term2 = rho**3/2*np.log((R + root)/(R - root))
    return term1 + term2
  
R = 1  
n = 10**7
rad1, r1, t1 = GenSphereProj(n)
r_ref = np.logspace(-2, np.log10(1), n/100)

hist, bins = np.histogram(rad1, bins='auto', density=True)
hist2, bins2 = np.histogram(r1, bins='auto', density=True)

fig, ax = plt.subplots()
ax.plot(r_ref, 3*f_r(r_ref, R)/r_ref, label='sqrt(1 - r)')
ax.plot(r_ref, 3*np.sqrt(1 - r_ref**2), label='sqrt(1 - r**2)')
ax.step(bins[1:-1], hist[1:]/bins[1:-1], label='rho')
ax.step(bins2[1:-1], hist2[1:]/bins2[1:-1]**2, label='r')

# ax.plot(np.log10(r_ref), np.log10(4*np.sqrt(1 - r_ref)), label='sqrt(1 - r)')
# ax.plot(np.log10(r_ref), np.log10(3*np.sqrt(1 - r_ref**2)), label='sqrt(1 - r**2)')
# ax.step(np.log10(bins[1:-1]), np.log10(hist[1:]/bins[1:-1]), label='rho')
# ax.step(np.log10(bins2[1:-1]), np.log10(hist2[1:]/bins2[1:-1]**2), label='r')
ax.set_title('Uniform sphere')
ax.set_xlabel('log(r)')
ax.set_ylabel('log(density)')
ax.legend()
plt.show()  
    
##
rad1, r1, t1 = GenSphereProj(10**5)

theta = np.arange(0, np.pi, 0.01)

fig, ax = plt.subplots()
ax.scatter(np.cos(t1), rad1)
ax.plot(np.cos(theta), np.sqrt(1-np.cos(theta)**2), c='r')
ax.set_title('Uniform sphere')
ax.set_xlabel('cos(theta)')
ax.set_ylabel('rho')
plt.show() 
    
    
    
## normal
import distributions as dist

rad1, r1, t1 = GenSphereProj(10**5, dist.Normal_r)
    
r_ref = np.logspace(-2, np.log10(3), 10**3)

hist, bins = np.histogram(rad1, bins='auto', density=True)
hist2, bins2 = np.histogram(r1, bins='auto', density=True)

fig, ax = plt.subplots()
ax.plot(np.log10(r_ref), np.log10(dist.pdf_Normal(r_ref)/r_ref**2), label='sqrt(1 - r**2)')
ax.step(np.log10(bins[1:-1]), np.log10(hist[1:]/bins[1:-1]), label='rho')
ax.step(np.log10(bins2[1:-1]), np.log10(hist2[1:]/bins2[1:-1]**2), label='r')
ax.set_title('Uniform sphere')
ax.set_xlabel('log(r)')
ax.set_ylabel('log(density)')
ax.legend()
plt.show()      
    
## Cauchy
rad1, r1, t1 = GenSphereProj(10**5, dist.SquaredCauchy_r) 
    
r_ref = np.logspace(-2, np.log10(10), 10**3)

hist, bins = np.histogram(rad1, bins='auto', density=True)
hist2, bins2 = np.histogram(r1, bins='auto', density=True)

fig, ax = plt.subplots()
ax.plot(np.log10(r_ref), np.log10(dist.pdf_SquaredCauchy(r_ref)/r_ref**2), label='ref')
ax.step(np.log10(bins[1:-1]), np.log10(hist[1:]/bins[1:-1]), label='rho')
ax.step(np.log10(bins2[1:-1]), np.log10(hist2[1:]/bins2[1:-1]**2), label='r')
ax.set_title('Uniform sphere')
ax.set_xlabel('log(r)')
ax.set_ylabel('log(density)')
ax.legend()
plt.show()
    
    
    
    
    
    
    
    





## 2D distributions - King
import matplotlib.pyplot as plt

def pdf_KingGlobular_rho(r, s=1.0, R=None):
    """pdf of 2D radial King distribution for Globular clusters."""
    if (R is None):
        R = 30*s                                                                                    # typical globular cluster has R/s ~ 30
    rs2 = (r/s)**2
    Rs2 = (R/s)**2
    C = (Rs2/2/(1 + Rs2) + np.log(1 + Rs2)/2 - 2)**(-1)
    return C/s*r/s*(1/(1 + rs2)**(1/2) - 1/(1 + Rs2)**(1/2))**2
    
def cdf_KingGlobular_rho(r, s=1.0, R=None):
    """cdf of 2D radial King distribution for Globular clusters."""
    if (R is None):
        R = 30*s                                                                                    # typical globular cluster has R/s ~ 30
    r = np.clip(r, 0, R)                                                                            # make sure r doesn't go above R (cdf is wrong there)
    rs2 = (r/s)**2
    Rs2 = (R/s)**2
    C = (Rs2/2/(1 + Rs2) + np.log(1 + Rs2)/2 - 2)**(-1)
    C2 = 1/(1 + Rs2)**(1/2)
    return C*(np.log(1 + rs2)/2 - 2*C2*(1 + rs2)**(1/2) + C2**2*rs2/2)
    
def KingGlobular_rho(n=1, s=1.0, R=None):
    """2D Radial King distribution for globular clusters with scale height s and outter radius R. Draws n numbers."""
    if (R is None):
        R = 30*s                                                                                    # typical globular cluster has R/s ~ 30
    rvals = np.logspace(-2, np.log10(R), 1000)
    Nvals = cdf_KingGlobular_rho(rvals, s, R)
    return np.interp(np.random.rand(int(n)), Nvals, rvals)
    
s = 2
R = 100

r_inter = KingGlobular_rho(n=1e6, s=s, R=R)
rvals = np.logspace(-2, np.log10(R), 1000)

hist, bins = np.histogram(r_inter, bins='auto', density=True)

fig, ax = plt.subplots()
ax.step(np.log10(bins[:-1]), np.log10(hist/bins[:-1]), label='King interp')
ax.plot(np.log10(rvals), np.log10(pdf_KingGlobular_rho(rvals, s, R)/rvals), label='King')
ax.legend()
plt.show()

## conversion between King 3D and 2D
def cdf_KingGlobular_r(r, s=1.0, R=None):
    """cdf of radial King distribution for Globular clusters."""
    if (R is None):
        R = 30*s                                                                                    # typical globular cluster has R/s ~ 30
    r = np.clip(r, 0, R)                                                                            # make sure r doesn't go above R (cdf is wrong there)
    rs = r/s
    rs2 = (r/s)**2
    Rs = R/s
    Rs2 = (R/s)**2
    C2 = 1/(1 + Rs2)**(1/2)
    C = (Rs**3/3*C2**2 + np.log(Rs + 1/C2)*C2 - np.arctan(Rs))**-1
    return C*(rs - np.arctan(rs) - C2*rs*(1 + rs2)**(1/2) + C2*np.log(rs + (1 + rs2)**(1/2)) + C2**2*rs**3/3)
    
def KingGlobular_r(n=1, s=1.0, R=None):
    """Radial King distribution for globular clusters with scale height s and outter radius R. Draws n numbers."""
    if (R is None):
        R = 30*s                                                                                    # typical globular cluster has R/s ~ 30
    rvals = np.logspace(-2, np.log10(R), 1000)
    Nvals = cdf_KingGlobular_r(rvals, s, R)
    return np.interp(np.random.rand(int(n)), Nvals, rvals)
    
def pdf_KingGlobular_r(r, s=1.0, R=None):
    """pdf of radial King distribution for Globular clusters."""
    if (R is None):
        R = 30*s                                                                                    # typical globular cluster has R/s ~ 30
    rs2 = (r/s)**2
    Rs = R/s
    Rs2 = (R/s)**2
    C = (Rs**3/3/(1 + Rs2) + np.log(Rs + (1 + Rs2)**(1/2))/(1 + Rs2)**(1/2) - np.arctan(Rs))**(-1)
    return C/s*rs2*(1/(1 + rs2)**(1/2) - 1/(1 + Rs2)**(1/2))**2

def AnglePhi(n=1):
    """Uniformly chosen angle(s) between 0 and 2 pi."""
    return 2*np.pi*np.random.rand(int(n)) 
    
def AngleTheta(n=1):
    """Angle(s) between 0 and pi chosen from a sinus distribution."""
    return np.arccos(2*np.random.rand(int(n)) - 1)
    
def PolToCart(r, theta):
    """Converts polar coords to Cartesian. Optimized for arrays."""
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.array([x, y])
    
def SpherToCart(r, theta, phi):
    """Converts spherical coords to Cartesian. Optimized for arrays."""
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.array([x, y, z])
    
def Distance2D(points):
    return (points[0]**2 + points[1]**2)**(1/2)

def CalcRadii(rads):
    phi = AnglePhi(len(rads))
    theta = AngleTheta(len(rads))
    xyz = SpherToCart(rads, theta, phi)
    return Distance2D(xyz)
    
def CalcRadii2D(rads):
    phi = AnglePhi(len(rads))
    xy = PolToCart(rads, phi)
    return Distance2D(xy)

s1 = 25
s2 = 40
R = 500
n = 10**6
rvals = np.logspace(0, np.log10(R-10), n)
kingr = KingGlobular_r(n, s1, R)
kingrho = KingGlobular_rho(n, s1, R)

hist, bins = np.histogram(CalcRadii(rvals), bins='auto', density=True)
hist2, bins2 = np.histogram(rvals, bins='auto', density=True)

hist3, bins3 = np.histogram(CalcRadii(kingr), bins='auto', density=True)                            # the malefactor
hist4, bins4 = np.histogram(kingr, bins='auto', density=True)
hist5, bins5 = np.histogram(CalcRadii2D(kingrho), bins='auto', density=True)


fig, ax = plt.subplots()
# ax.plot(np.log10(bins[:-1]), np.log10(hist/bins[:-1]), label='r vals projected')
# ax.plot(np.log10(bins2[:-1]), np.log10(hist2/bins2[:-1]), label='r vals')

ax.plot(np.log10(rvals), np.log10(pdf_KingGlobular_r(rvals, s1, R)/rvals**2) + 2, label='King r')
ax.step(np.log10(bins3[:-1]), np.log10(hist3/bins3[:-1]), label='King r generated projected')       # the malefactor
ax.step(np.log10(bins4[:-1]), np.log10(hist4/bins4[:-1]**2) + 2, label='King r generated')

ax.plot(np.log10(rvals), np.log10(pdf_KingGlobular_rho(rvals, s1, R)/rvals), label='King rho')
ax.step(np.log10(bins5[:-1]), np.log10(hist5/bins5[:-1]), label='King rho generated projected')
ax.legend()
plt.show()

# conclusion: there is a non-trivial transformation between the two, caused by projection onto the plane
# [edit]: the Abel transform gives it.

## schuster power law + king tests
n = 10**6

rvals = np.logspace(-2, np.log10(30), n)
cauchyr = dist.SquaredCauchy_r(n)
cauchyrho = dist.SquaredCauchy_rho(n)
pearsonr = dist.PearsonVII_r(n)
pearsonrho = dist.PearsonVII_rho(n)
kingr = dist.KingGlobular_r(n)
kingrho = dist.KingGlobular_rho(n)

hist, bins = np.histogram(cauchyr, bins='auto', density=True)
hist1, bins1 = np.histogram(cauchyrho, bins='auto', density=True)
hist2, bins2 = np.histogram(pearsonr, bins='auto', density=True)
hist3, bins3 = np.histogram(pearsonrho, bins='auto', density=True)
hist4, bins4 = np.histogram(kingr, bins='auto', density=True)
hist5, bins5 = np.histogram(kingrho, bins='auto', density=True)

fig, ax = plt.subplots()
# ax.step(np.log10(bins[:-1]), np.log10(hist/bins[:-1]**2), label='cauchy r') 
# ax.step(np.log10(bins1[:-1]), np.log10(hist1/bins1[:-1]**2), label='cauchy rho') 
# ax.step(np.log10(rvals), np.log10(dist.pdf_SquaredCauchy_r(rvals)/rvals**2), label='th. cauchy r')
# ax.step(np.log10(rvals), np.log10(dist.pdf_SquaredCauchy_rho(rvals)/rvals), label='th. cauchy rho')   
# ax.step(np.log10(bins2[:-1]), np.log10(hist2/bins2[:-1]**2), label='pearson r') 
# ax.step(np.log10(bins3[:-1]), np.log10(hist3/bins3[:-1]**2), label='pearson rho') 
# ax.step(np.log10(rvals), np.log10(dist.pdf_PearsonVII_r(rvals)/rvals**2), label='pearson')
ax.step(np.log10(bins4[:-1]), np.log10(hist4/bins4[:-1]**2), label='king r') 
ax.step(np.log10(bins5[:-1]), np.log10(hist5/bins5[:-1]), label='king rho') 
ax.step(np.log10(rvals), np.log10(dist.pdf_KingGlobular(rvals)/rvals**2), label='th. cauchy r')
ax.step(np.log10(rvals), np.log10(dist.pdf_KingGlobular(rvals, form='cylindrical')/rvals), label='th. cauchy rho')  
ax.legend()
plt.show()

# projections
def AnglePhi(n=1):
    """Uniformly chosen angle(s) between 0 and 2 pi."""
    return 2*np.pi*np.random.rand(int(n)) 
    
def AngleTheta(n=1):
    """Angle(s) between 0 and pi chosen from a sinus distribution."""
    return np.arccos(2*np.random.rand(int(n)) - 1)
    
def SpherToCart(r, theta, phi):
    """Converts spherical coords to Cartesian. Optimized for arrays."""
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.array([x, y, z])
    
def Distance2D(points):
    return (points[0]**2 + points[1]**2)**(1/2)
    
def CalcRadii(rads):
    phi = AnglePhi(len(rads))
    theta = AngleTheta(len(rads))
    xyz = SpherToCart(rads, theta, phi)
    return Distance2D(xyz)

proj_kingr = CalcRadii(kingr)
hist6, bins6 = np.histogram(proj_kingr, bins='auto', density=True)

fig, ax = plt.subplots()
ax.step(np.log10(bins4[:-1]), np.log10(hist4/bins4[:-1]**2), label='king r') 
ax.step(np.log10(bins5[:-1]), np.log10(hist5/bins5[:-1]), label='king rho') 
ax.step(np.log10(bins6[:-1]), np.log10(hist6/bins6[:-1]), label='proj king r') 
ax.plot(np.log10(rvals), np.log10(dist.pdf_KingGlobular(rvals)/rvals**2), label='th. king r')
ax.plot(np.log10(rvals), np.log10(dist.pdf_KingGlobular(rvals, form='cylindrical')/rvals), label='th. king rho')  
ax.legend()
plt.show()






## simCADO (2) derotator and ADC/AO
src = sim.source.cluster()

sim.run(src, filename='data\DEROTtest100.fits', OBS_EXPTIME=300, INST_DEROT_PERFORMANCE=100)          # percentage, larger effect for longer exptimes
sim.run(src, filename='data\DEROTtest0.fits', OBS_EXPTIME=300, INST_DEROT_PERFORMANCE=0)

fh.PlotFits('DEROTtest100.fits')
fh.PlotFits('DEROTtest0.fits')

##
src = sim.source.cluster(distance=1e6, half_light_radius=3)

sim.run(src, filename='data\ADCtest100.fits', mode='zoom', OBS_EXPTIME=600, INST_ADC_PERFORMANCE=100)            # no effect seen?
sim.run(src, filename='data\ADCtest0.fits', mode='zoom', OBS_EXPTIME=600, INST_ADC_PERFORMANCE=0)

fh.PlotFits('ADCtest100.fits')
fh.PlotFits('ADCtest0.fits')

sim.run(src, filename='data\AOtest100.fits', mode='zoom', OBS_EXPTIME=600, SCOPE_AO_EFFECTIVENESS=100)           # no effect seen?
sim.run(src, filename='data\AOtest0.fits', mode='zoom', OBS_EXPTIME=600, SCOPE_AO_EFFECTIVENESS=0)

fh.PlotFits('AOtest100.fits')
fh.PlotFits('AOtest0.fits')


## making a source (2)
astobj = aoc.AstObject(N_obj=4, age=[8], metal=[0.019], distance=10**6)

# coords in arcsec! - fov 53 arcsec (wide, 4 mas/pixel) or 16 (zoom, 1.5 mas/pixel)
# available filters: B, BrGamma, CH4_169, CH4_227, Fell_166, H, H2_212, H2O_204, Hcont_158, I, J, K, Ks, NH3_153, PaBeta, R, U, V, Y, z
#                 so [U, B, V, R, I, J, H, K] are all there

x_as = list(np.arctan(astobj.objects[:,0]/astobj.distance)*648000/np.pi)            # assumed to be in pc
y_as = list(np.arctan(astobj.objects[:,1]/astobj.distance)*648000/np.pi)            # *648000/np.pi is rad to as

src = sim.source.stars(mags=list(astobj.app_mag[2]-18), x=x_as, y=y_as, filter_name='V', spec_types='M0IV')

sim.run(src, filename='data\my_first_sim.fits', mode='wide', detector_layout='centre', OBS_EXPTIME=600)     # mode=['wide', 'zoom'], detector_layout=['small', 'centre', 'full'] (=FPA_CHIP_LAYOUT)

fh.PlotFits('my_first_sim.fits')


print('\nB-V to spectral type')
print(np.arange(-0.4, 2.4, 0.002))
print(np.unique(sim.source.BV_to_spec_type(np.arange(-0.4, 2.4, 0.002))))

## superslow
astobj = aoc.AstObject(N_obj=3, age=[8], metal=[0.019], distance=10**6)

x_as = np.arctan(astobj.objects[0]/astobj.distance)*648000/np.pi            # assumed to be in pc
y_as = np.arctan(astobj.objects[1]/astobj.distance)*648000/np.pi            # *648000/np.pi is rad to as

14 = sim.source.stars(mags=np.clip(astobj.app_mag[2]-18, 10, 30), x=x_as, y=y_as, filter_name='V', spec_types='F1V')
for i in range(100):
    astobj = aoc.AstObject(N_obj=3, age=[8], metal=[0.019], distance=10**6)
    x_as = np.arctan(astobj.objects[0]/astobj.distance)*648000/np.pi
    y_as = np.arctan(astobj.objects[1]/astobj.distance)*648000/np.pi
    src += sim.source.stars(mags=np.clip(astobj.app_mag[2]-18, 10, 30), x=x_as, y=y_as, filter_name='V', spec_types='F1V')


sim.run(src, filename='data\my_first_sim.fits', mode='zoom', detector_layout='centre', OBS_EXPTIME=60)

fh.PlotFits('my_first_sim.fits')

## faster
n = 300

astobj = aoc.AstObject(N_obj=n, age=[8], metal=[0.019], distance=10**6)

x_as = np.arctan(astobj.objects[0]/astobj.distance)*648000/np.pi            # assumed to be in pc
y_as = np.arctan(astobj.objects[1]/astobj.distance)*648000/np.pi            # *648000/np.pi is rad to as

src = sim.source.stars(mags=np.clip(astobj.app_mag[2,0:3]-18, 10, 30), x=x_as[0:3], y=y_as[0:3], filter_name='V', spec_types='F1V')

for i in range(int(n/3-1)):
    src += sim.source.stars(mags=np.clip(astobj.app_mag[2, 3*i+3:3*i+6]-18, 10, 30), x=x_as[3*i+1:3*i+6], y=y_as[3*i+1:3*i+6], filter_name='V', spec_types='F1V')


# sim.run(src, filename='data\my_first_sim.fits', mode='zoom', detector_layout='centre', OBS_EXPTIME=60)

# fh.PlotFits('my_first_sim.fits')

# Problem resolved: just used the wrong slicing for the coordinates

## calling from Pyzo shell (2)
import sys
import subprocess

subprocess.call([sys.executable, 'constructor.py', '-N 1000', '-rdist', 'Normal', 'KingGlobular', '-rdistpar', '1', '2'])
subprocess.call([sys.executable, 'constructor.py', '-N 1000', '-ages', '9', '10', '9', '-i', '1', '2'])
subprocess.call([sys.executable, 'constructor.py', '-N 1000', '-relN', '5', '4', '2', '-axes', '1', '2', '5', '2', '2', '1'])
##
astobj = aoc.AstObject(N_obj=10000, age=[8], metal=[0.019], distance=10**5)

x_as = list(np.arctan(astobj.objects[:,0]/astobj.distance)*648000/np.pi)            # assumed to be in pc
y_as = list(np.arctan(astobj.objects[:,1]/astobj.distance)*648000/np.pi)            # *648000/np.pi is rad to as

src = sim.source.stars(mags=list(astobj.app_mag[2]), x=x_as, y=y_as, filter_name='V', spec_types='M0IV')

sim.run(src, filename='data\my_first_sim.fits', mode='wide', SCOPE_PSF_FILE='scao', detector_layout='centre', filter_name='Ks', OBS_EXPTIME=2400, OBS_NDIT=1)     # mode=['wide', 'zoom'], detector_layout=['small', 'centre', 'full'] (=FPA_CHIP_LAYOUT)

fh.PlotFits('my_first_sim.fits')
# fh.PrintHdr('my_first_sim.fits')
##
astobj = aoc.AstObject(N_obj=100000, age=[8], metal=[0.019], distance=10**6)

x_as = list(np.arctan(astobj.objects[:,0]/astobj.distance)*648000/np.pi)            # assumed to be in pc
y_as = list(np.arctan(astobj.objects[:,1]/astobj.distance)*648000/np.pi)            # *648000/np.pi is rad to as

src = sim.source.stars(mags=list(astobj.app_mag[2]), x=x_as, y=y_as, filter_name='V', spec_types='M0IV')

sim.run(src, filename='data\my_first_sim.fits', mode='zoom', detector_layout='centre', OBS_EXPTIME=2400, OBS_NDIT=1)     # mode=['wide', 'zoom'], detector_layout=['small', 'centre', 'full'] (=FPA_CHIP_LAYOUT)

fh.PlotFits('my_first_sim.fits')

##
"""
SimCADO is controlled with a series of keyword-value pairs contained in a configuration file. The defaults are the best approximation to MICADO so changing them is not recommended if you want to simulate MICADO images. There are however some which are useful to play around with.
-------
Unfortunately SimCADO doesn't yet have a MCAO PSF
Using the SCAO PSF instead
"""

print(sim.optics.get_filter_set())
sim.commands.dump_defaults()


## stellar classification
import os

with open(os.path.join('tables', 'all_stars.txt')) as file:
    spec_tbl_names  = np.array(file.readline()[:-1].split('\t'))                                # read in the column names
spec_tbl_names = {name: i for i, name in enumerate(spec_tbl_names[1:])}                         # make a dict for ease of use

spec_tbl_types = np.loadtxt(os.path.join('tables', 'all_stars.txt'), dtype=str, usecols=[0], unpack=True)
spec_tbl = np.loadtxt(os.path.join('tables', 'all_stars.txt'), dtype=float, usecols=[1,2,3,4,5,6,7,8], unpack=True)

fig, ax = plt.subplots()
zero_lum_cor = 10**(0.4*(4.74 - spec_tbl[spec_tbl_names['Mbol']])*(spec_tbl[spec_tbl_names['Luminosity']] == 0))
ax.scatter(np.log10(spec_tbl[spec_tbl_names['Temp']]), np.log10(spec_tbl[spec_tbl_names['Luminosity']] + zero_lum_cor))
for i, type in enumerate(spec_tbl_types):
    ax.annotate(type, xy=(np.log10(spec_tbl[spec_tbl_names['Temp'], i]), np.log10(spec_tbl[spec_tbl_names['Luminosity'], i] + zero_lum_cor[i])))
ax.set_xlabel('Temperature')
ax.set_ylabel('Luminosity')
ax.invert_xaxis()
plt.show()

fig, ax = plt.subplots()
ax.scatter(spec_tbl[spec_tbl_names['B-V']], spec_tbl[spec_tbl_names['Mv']])
for i, type in enumerate(spec_tbl_types):
    ax.annotate(type, xy=(spec_tbl[spec_tbl_names['B-V'], i], spec_tbl[spec_tbl_names['Mv'], i]))
ax.set_xlabel('colour')
ax.set_ylabel('magnitude')
ax.invert_yaxis()
plt.show()
##
fig, ax = plt.subplots()
ax.scatter( np.log10(spec_tbl[spec_tbl_names['Temp']]), conv.RadiusToGravity(spec_tbl[spec_tbl_names['Radius']], spec_tbl[spec_tbl_names['Mass']]) )
for i, type in enumerate(spec_tbl_types):
    ax.annotate( type, xy=(np.log10(spec_tbl[spec_tbl_names['Temp'], i]), conv.RadiusToGravity(spec_tbl[spec_tbl_names['Radius']], spec_tbl[spec_tbl_names['Mass']])[i] ) )
ax.set_xlabel('temperature')
ax.set_ylabel('log_g')
ax.invert_xaxis()
plt.show()
##
ys = (conv.RadiusToGravity(spec_tbl[spec_tbl_names['Radius']], spec_tbl[spec_tbl_names['Mass']]) + np.log10(spec_tbl[spec_tbl_names['Luminosity']] + 1) )#/np.log(spec_tbl[spec_tbl_names['Temp']])**3
fig, ax = plt.subplots()
ax.scatter( np.log10(spec_tbl[spec_tbl_names['Temp']]), ys)
for i, type in enumerate(spec_tbl_types):
    ax.annotate( type, xy=(np.log10(spec_tbl[spec_tbl_names['Temp'], i]), ys[i]) )
ax.set_xlabel('temperature')
ax.set_ylabel('')
ax.invert_xaxis()
plt.show()
##
ys = np.log10(spec_tbl[spec_tbl_names['Luminosity']] + 1)

fig, ax = plt.subplots()

for i, type in enumerate(spec_tbl_types):
    if (type[0] in ['O', 'B', 'A', 'F', 'G', 'K', 'M']):
        ax.scatter(np.log10(spec_tbl[spec_tbl_names['Temp'], i]), ys[i])
        ax.annotate( type, xy=(np.log10(spec_tbl[spec_tbl_names['Temp'], i]), ys[i]) )
ax.set_xlabel('temperature')
ax.set_ylabel('log g')
ax.invert_xaxis()
plt.show()

## 3d
from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.scatter(np.log10(spec_tbl[spec_tbl_names['Temp']]), np.log10(spec_tbl[spec_tbl_names['Luminosity']] + 1), np.log10(spec_tbl[spec_tbl_names['Mass']]))
for i, type in enumerate(spec_tbl_types):
    ax.text(np.log10(spec_tbl[spec_tbl_names['Temp'], i]), np.log10(spec_tbl[spec_tbl_names['Luminosity'], i] + 1), np.log10(spec_tbl[spec_tbl_names['Mass'], i]), type)
ax.set_xlabel('Temperature')
ax.set_ylabel('Luminosity')
ax.invert_xaxis()
plt.show()
##
T, L, M, types = [], [], [], []
for i, type in enumerate(spec_tbl_types):
    if (type[0] in ['M', 'K', 'G', 'F', 'A', 'B', 'O']) | (type[:2] == 'DC'):
        types.append(type)
        T.append(np.log10(spec_tbl[spec_tbl_names['Temp'], i]))
        L.append(np.log10(spec_tbl[spec_tbl_names['Luminosity'], i] + 1))
        M.append(np.log10(spec_tbl[spec_tbl_names['Mass'], i]))

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.scatter(T, L, M)
for i, type in enumerate(types):
    ax.text(T[i], L[i], M[i], type)
ax.set_xlabel('Temperature')
ax.set_ylabel('Luminosity')
ax.set_zlabel('Mass')
ax.invert_xaxis()
plt.show()

## spectral type order (Temp)
sort_T = np.argsort(spec_tbl[spec_tbl_names['Temp']])

type_arr = np.array([])
temp_arr = np.array([])
for i in sort_T:
    type = spec_tbl_types[i]
    if type[0] in ['O', 'B', 'A', 'F', 'G', 'K', 'M']:
        type_arr = np.append(type_arr, type)
        temp_arr = np.append(temp_arr, spec_tbl[spec_tbl_names['Temp'], i])

print(temp_arr[[type.startswith('G9') for type in type_arr]])
print(type_arr[[type.startswith('G9') for type in type_arr]])

##
Lum = np.log10(spec_tbl[spec_tbl_names['Luminosity']] + 1e-5*(spec_tbl[spec_tbl_names['Luminosity']] == 0))
# --> replace 0 luminosity 10**-5 for V class, 10**-6 for VI class (to not have the same value in i.e. one M9 type series)
sort_L = np.argsort(Lum)

type_arr = np.array([])
temp_arr = np.array([])
for i in sort_L:
    type = spec_tbl_types[i]
    if type[0] in ['O', 'B', 'A', 'F', 'G', 'K', 'M']:
        if type[:2] in ['M9']:
            type_arr = np.append(type_arr, type)
            temp_arr = np.append(temp_arr, Lum[i])
print(temp_arr)
print(type_arr)
##

spec_letter = ['M', 'K', 'G', 'F', 'A', 'B', 'O']                                               # basic spectral letters (increasing temperature)
T_types = [l + str(n) for l in spec_letter for n in np.arange(9, -1, -1)]                        # the first part of the spectral types (corresponding to increasing temperature) 

# temperature ranges for MS
T_borders = np.array([])
for T, type in zip(spec_tbl[spec_tbl_names['Temp'], sort_T], spec_tbl_types[sort_T]):
    if (type[0] in spec_letter) & (type[2:] == 'V'):
            T_borders = np.append(T_borders, T)                                                   # the lower temperature range borders

## KDTree lookup
import scipy.spatial as sps

lon = np.arange(2, 8, 1)
lat = np.arange(6, 18, 2)
h = np.arange(6, 18, 2)
lonlat = np.column_stack((lon, lat, h))
tree = sps.cKDTree(lonlat)
##
astobj = aoc.AstObject(N_obj=10000, age=[8, 9, 10], metal=[0.019], distance=10**6)

spectra = obg.FindSpectralType(astobj.log_Te, astobj.log_L, np.log10(astobj.M_cur))

##
T, L, M, types = [], [], [], []
for i, type in enumerate(spec_tbl_types):
    if ((type[0] in ['M', 'K', 'G', 'F', 'A', 'B', 'O']) | (type[:2] == 'DC')) & (type[2:] != 'VI'):
        types.append(type)
        T.append(np.log10(spec_tbl[spec_tbl_names['Temp'], i]))
        L.append(0.4*(4.74 - spec_tbl[spec_tbl_names['Mbol'], i]))  # bolometric magnitude converted to luminosity (due to problems with lum=0 in tbl)
        M.append(np.log10(spec_tbl[spec_tbl_names['Mass'], i]))

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.scatter(T, L, M)
ax.scatter(astobj.log_Te, astobj.log_L, np.log10(astobj.M_cur))
ax.set_xlabel('Temperature')
ax.set_ylabel('Luminosity')
ax.set_zlabel('Mass')
ax.invert_xaxis()
plt.show()

##
fig, ax = plt.subplots()
ax.scatter(T, L)
ax.scatter(astobj.log_Te[astobj.log_Te!=1], astobj.log_L[astobj.log_Te!=1])
ax.set_xlabel('Temperature')
ax.set_ylabel('Luminosity')
ax.invert_xaxis()
plt.show()




##
astobj = obg.AstObject(N_obj=10000, age=[8], metal=[0.019], distance=10**5)


x_as = list(np.arctan(astobj.objects[:,0]/astobj.distance)*648000/np.pi)            # assumed to be in pc
y_as = list(np.arctan(astobj.objects[:,1]/astobj.distance)*648000/np.pi)            # *648000/np.pi is rad to as
##
filter = 'I'
src = sim.source.stars(mags=list(astobj.app_mag[np.array(astobj.mag_names)==filter][0]), x=x_as, y=y_as, filter_name=filter, spec_types='M0V')

sim.run(src, filename='data\my_first_sim.fits', mode='wide', SCOPE_PSF_FILE='scao', detector_layout='centre', filter_name=filter, OBS_EXPTIME=4800, OBS_NDIT=1)
fh.PlotFits('my_first_sim.fits')

##
src = sim.source.stars(mags=list(astobj.app_mag[np.array(astobj.mag_names)==filter][0]), x=x_as, y=y_as, filter_name=filter, spec_types=list(astobj.spec_types))

sim.run(src, filename='data\my_first_sim.fits', mode='wide', SCOPE_PSF_FILE='scao', detector_layout='centre', filter_name=filter, OBS_EXPTIME=4800, OBS_NDIT=1)
fh.PlotFits('my_first_sim.fits')

##
fh.PlotFits('Cluster-test-one-spec_type.fits')
fh.PlotFits('Cluster-test-diff-spec_types.fits')


## merged aoc --> obg
astobj = obg.AstObject(N_obj=10000, age=[8], metal=[0.019], distance=10**5)

## plot of IFMR
init_masses = np.arange(0.08, 200, 0.02)
border_masses = np.array([0.85, 2.85, 3.6, 7.2, 11, 30, 50, 90])

fig, ax = plt.subplots()
ax.plot(init_masses, form.RemnantMass(init_masses), label='solar')
ax.plot(init_masses, form.RemnantMass(init_masses, Z=0.014), label='0.014')
ax.plot(init_masses, form.RemnantMass(init_masses, Z=0.008), label='0.008')
ax.plot(init_masses, form.RemnantMass(init_masses, Z=0.001), label='0.001')
ax.plot(init_masses, form.RemnantMass(init_masses, Z=0.0001), label='0.0001')
# ax.scatter(border_masses, form.RemnantMass(border_masses), c='orange')
# ax.scatter(border_masses, form.RemnantMass(border_masses, Z=0.014), c='orange')
# ax.scatter(border_masses, form.RemnantMass(border_masses, Z=0.008), c='orange')
# ax.scatter(border_masses, form.RemnantMass(border_masses, Z=0.001), c='orange')
# ax.scatter(border_masses, form.RemnantMass(border_masses, Z=0.0001), c='orange')
ax.set_xlabel('Initial Mass (M_sun)')
ax.set_ylabel('Final Mass (M_sun)')
plt.legend()
plt.show()

## WD Radius
K_non = 3.16*10**12         # cgs
K_rel = 4.86*10**14         # cgs
G_newt = 6.67*10**-8        # cgs
M_sun = 2*10**33            # g
R_sun = 6.96*10**10         # cm
R_earth = 6.38*10**8        # cm
M_ch = 1.456*M_sun          # g
c = 3*10**10                # cm/s

M = np.arange(0.1, 1.456, 0.001)*M_sun
M2 = np.arange(2.0, 10**3, 10.)*M_sun

R1 = K_non/(G_newt*M**(1/3))*(1 - G_newt**2*M**(4/3)/K_rel**2)**(1/2)
R2 = K_non/(0.4242*G_newt*M**(1/3))*(1 - (M/M_ch)**(4/3))**(1/2)
# mu_e assumed equal to 2
R3 = 0.0126*R_sun*(M/M_sun)**(-1/3)*(1 - (M/M_ch)**(4/3))**(1/2)                                    # the one to use

fig, ax = plt.subplots()
# ax.plot(M/M_sun, R1/R_earth)
ax.plot(M/M_sun, R2/R_earth)
ax.plot(M/M_sun, R3/R_earth)
ax.plot(M/M_sun, 0.0126*R_sun*(M/M_sun)**(-1/3)/R_earth)
ax.set_xlabel('Mass (M_sun)')
ax.set_ylabel('Radius (R_sun)')
plt.show()

## all radii
fig, ax = plt.subplots()
ax.plot(M/M_sun, R3/R_earth)
ax.plot([1.2, 2.0], [11*10**5/R_earth, 11*10**5/R_earth])
ax.plot(M2/M_sun, 2*G_newt*M2/c**2/R_earth)
ax.set_xlabel('Mass (M_sun)')
ax.set_ylabel('Radius (R_earth)')
plt.show()
## solar comparison
fig, ax = plt.subplots()
ax.plot(M/M_sun, R3/R_sun)
ax.plot([1.2, 2.0], [11*10**5/R_sun, 11*10**5/R_sun])
ax.plot(M2/M_sun, 2*G_newt*M2/c**2/R_sun)
ax.plot(M2/M_sun, (M2/M_sun)*4.245*10**-6)
ax.set_xlabel('Mass (M_sun)')
ax.set_ylabel('Radius (R_sun)')
plt.show()


## Remnant Temperatures
M1 = np.append(np.linspace(0.1, 1.456, 20), np.logspace(2.0, 10**3, 20))
R1 = form.RemnantRadius(M1)
t_c = np.logspace(-1, 12, 1000)
T1 = np.zeros([0,len(t_c)])
for M, R in zip(M1, R1):
    Mass = np.array([M for i in range(len(t_c))])
    Radius = np.array([R for i in range(len(t_c))])
    T1 = np.append(T1, [form.RemnantTeff(Mass, Radius, t_c)], axis=0)

fig, ax = plt.subplots()
for T in T1:
    ax.plot(t_c, T)
ax.set_xlabel('cooling time')
ax.set_ylabel('temperature')
ax.loglog()
plt.show()

L1 = np.zeros([0,len(t_c)])
for M, R in zip(M1, R1):
    Mass = np.array([M for i in range(len(t_c))])
    Radius = np.array([R for i in range(len(t_c))])
    L1 = np.append(L1, [form.BBLuminosity(Radius, form.RemnantTeff(Mass, Radius, t_c))], axis=0)

fig, ax = plt.subplots()
for L in L1:
    ax.plot(t_c, L)
ax.set_xlabel('cooling time')
ax.set_ylabel('luminosity')
ax.loglog()
plt.show()



## PSFs
import os
import astropy.io.fits as fits
from astropy.visualization import astropy_mpl_style

with fits.open(os.path.join('images', 'PSF_SCAO.fits')) as hdul:
    image_data = hdul[0].data**(1/32)
    
# plt.style.use(astropy_mpl_style)                                                                # use nice plot parameters
fig, ax = plt.subplots()
cax = ax.imshow(image_data, cmap='gray')
cbar = fig.colorbar(cax)
plt.show()



## distance measures
H0 = 67.                    # km/s/Mpc
c = 3*10**5                 # km/s
d_H = c/H0                  # Mpc

def DC(z):
    zs = np.array([np.logspace(-5, np.log10(z[i]), 10*len(z)) for i in range(len(z)) if z[i] != 0])
    ys = 1/np.sqrt(9*10**-5*(1 + zs)**4 + 0.315*(1 + zs)**3 + 0.685)
    return np.trapz(ys, zs, axis=1)*d_H
    
def DBack(z):
    zs = np.array([np.logspace(-5, np.log10(z[i]), 10*len(z)) for i in range(len(z)) if z[i] != 0])
    ys = 1/np.sqrt(9*10**-5*(1 + zs)**4 + 0.315*(1 + zs)**3 + 0.685)/(1 + zs)
    return np.trapz(ys, zs, axis=1)*d_H
    
def DL(z, d_C):
    return (1 + z)*d_C
    
def DA(z, d_C):
    return d_C/(1 + z)
    
def DP(z):
    return ((1 + z)**2 - 1)/((1 + z)**2 + 1)*d_H #/np.sqrt(9*10**-5*(1 - z)**4 + 0.315*(1 + z)**3 + 0.685)
    
z = np.logspace(-4, 4, 100)
z = z[1:]
d_C = DC(z)
d_back = DBack(z)
d_L = DL(z, d_C)
d_A = DA(z, d_C)
d_P = DP(z)

fig, ax = plt.subplots()
ax.plot(z, d_C*3.2616/10**3, label='Comoving')
ax.plot(z, d_back*3.2616/10**3, label='Lookback')
ax.plot(z, d_L*3.2616/10**3, label='Luminosity')
ax.plot(z, d_A*3.2616/10**3, label='Angular')
ax.plot(z, d_P*(1 + z)*3.2616/10**3, label='Naive Hubble')
# ax.plot([z[0], z[-1]], [d_L[0]*3.2616/10**3, d_L[-1]*3.2616/10**3])                               # linear 'fit' to d_L: 46.31588746864452 x - 0.00396565632131997
ax.set_xlabel('z')
ax.set_ylabel('d (Gly)')
ax.loglog()
plt.legend()
plt.show()

## BB to mag
# B, BrGamma, CH4_169, CH4_227, Fell_166, H, H2_212, H2O_204, Hcont_158, I, J, K, Ks, NH3_153, PaBeta, R, U, V, Y, z
# U, B, V, R, I, J, H, K

import os

c = 3*10**8

all_files = [f for f in os.listdir('tables') if os.path.isfile(os.path.join('tables', f))]
filter_files = [f for f in all_files if f.startswith('TC_filter')]

wvl_data = []
ampl_data = []
for filter in filter_files:
    wvl, put = np.loadtxt(os.path.join('tables', filter), skiprows=3, usecols=(0,1), unpack=True)
    wvl_data.append(wvl)
    ampl_data.append(put)
    

fig, ax = plt.subplots()
for wvl, put, filter in zip(wvl_data, ampl_data, filter_files):
    ax.plot(wvl, put)
    ax.annotate('{}'.format(filter.replace('TC_filter_', '').replace('.dat', '')), (wvl[np.argmax(put)], put[np.argmax(put)]))
ax.set_xlabel('wavelength')
ax.set_ylabel('throughput')
plt.show()

fig, ax = plt.subplots()
for wvl, put, filter in zip(wvl_data, ampl_data, filter_files):
    freq = c/(wvl*10**-6)
    ax.plot(freq, put)
    ax.annotate('{}'.format(filter.replace('TC_filter_', '').replace('.dat', '')), (freq[np.argmax(put)], put[np.argmax(put)]))
ax.set_xlabel('frequency')
ax.set_ylabel('throughput')
plt.show()

##
wavel = np.arange(0.3, 3.0, 0.01)*10**-6
flux = form.PlanckBB(wavel, 5000, var='wavl')


fig, ax = plt.subplots()
for wvl, put, filter in zip(wvl_data, ampl_data, filter_files):
    ax.plot(wvl, put)
    ax.annotate('{}'.format(filter.replace('TC_filter_', '').replace('.dat', '')), (wvl[np.argmax(put)], put[np.argmax(put)]))
ax.plot(wavel*10**6, flux/np.max(flux))
ax.set_xlabel('wavelength')
ax.set_ylabel('throughput')
plt.show()

##
# str1 = 'B, BrGamma, CH4_169, CH4_227, Fell_166, H, H2_212, H2O_204, Hcont_158, I, J, K, Ks, NH3_153, PaBeta, R, U, V, Y, z'.split(', ')
filters = ['B', 'BrGamma', 'CH4_169', 'CH4_227', 'Fell_166', 'H', 'H2_212', 'H2O_204', 'Hcont_158', 'I', 'J', 'K', 'Ks', 'NH3_153', 'PaBeta', 'R', 'U', 'V', 'Y', 'z']

hights = np.zeros([0,3])
eff_band = np.zeros([0,3])

for wvl, put, filter in zip(wvl_data, ampl_data, filter_files):
    cumul = np.cumsum(put)
    middle = np.argmin(np.abs(cumul - np.max(cumul)/2))
    
    low = (put >= 0.5)
    upp = (put >= 0.5)
    
    hights = np.append(hights, [[put[low][0], put[middle], put[upp][-1]]], axis=0)
    eff_band = np.append(eff_band, [[wvl[low][0], wvl[middle], wvl[upp][-1]]], axis=0)
    
widths = eff_band[:, 1] - eff_band[:, 0]

fig, ax = plt.subplots()
for wvl, put, filter in zip(wvl_data, ampl_data, filter_files):
    ax.plot(wvl, put)
    ax.annotate('{}'.format(filter.replace('TC_filter_', '').replace('.dat', '')), (wvl[np.argmax(put)], put[np.argmax(put)]))
ax.scatter(eff_band[:, 1], hights[:, 1])
ax.scatter(eff_band[:, 0], hights[:, 0])
ax.scatter(eff_band[:, 2], hights[:, 2])
ax.set_xlabel('wavelength')
ax.set_ylabel('throughput')
plt.show()


## MIST isochrones
import os

fname = 'MIST_v1.1_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd'
# fname = 'isoc_z0001.dat'

# EEP          log10_isochrone_age_yr                    initial_mass                       star_mass                        log_Teff                           log_g                           log_L                     [Fe/H]_init              [Fe/H]           Bessell_U           Bessell_B           Bessell_V           Bessell_R           Bessell_I             2MASS_J             2MASS_H            2MASS_Ks           Kepler_Kp          Kepler_D51        Hipparcos_Hp             Tycho_B             Tycho_V              Gaia_G             Gaia_BP             Gaia_RP               phase

# log_Teff 5, log_L 7

log_t, log_T, log_L = np.loadtxt(os.path.join('tables', fname), usecols=(1,4,6), unpack=True)

log_t_min = np.min(log_t)                                                                       # minimum available age                                                
log_t_max = np.max(log_t)                                                                       # maximum available age
uni_log_t = np.unique(log_t)                                                                    # unique array of ages

print(log_t_min, log_t_max)



metals = np.array([1.42857E-06, 4.51753E-06, 1.42857E-05, 4.51753E-05, 1.42857E-04, 2.54039E-04, 4.51753E-04, 8.03343E-04, 1.42857E-03, 2.54039E-03, 4.51753E-03, 8.03343E-03, 1.42857E-02, 2.54039E-02, 4.51753E-02])
metal_names = np.array([np.round(z, decimals=-int(np.floor(np.log10(z)))+1) for z in metals])
Fe = np.append(np.arange(-4.0, -2.0, 0.50), np.arange(-2.0, 0.51, 0.25))

##
# opening the file (actual opening lateron)
Z = 0.0014
file_name = os.path.join('tables', 'isoc_Z{1:1.{0}f}.dat'.format(-int(np.floor(np.log10(Z)))+1, Z))    # doesn't check if file for Z exists 

# names to use in the code, and a mapping to the isoc file column names 
code_names = np.array(['log_age', 'M_initial', 'M_actual', 'log_L', 'log_Te', 'log_g', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks'])
mag_names = code_names[6:]                                                                      # names of filters for later reference
mag_num = len(mag_names)
var_names, column_names = np.loadtxt(os.path.join('tables', 'column_names.dat'), dtype=str, unpack=True)

if (len(code_names) != len(var_names)):
    raise SyntaxError('ObjectGen//IsochroneProps: file "column_names.dat" has incorrect names specified. Use: {0}'.format(', '.join(code_names)))
elif np.any(code_names != var_names):
    raise SyntaxError('ObjectGen//IsochroneProps: file "column_names.dat" has incorrect names specified. Use: {0}'.format(', '.join(code_names)))
    
name_dict = {vn: cn for vn, cn in zip(var_names, column_names)}

# find the column names in the isoc file
with open(file_name) as file:
    for line in file:
        if line.startswith('#'):
            header = np.array(line.replace('#', '').split())
        else:
            break

col_dict = {name: col for col, name in enumerate(header) if name in column_names}

var_cols = [col_dict[name_dict[name]] for name in code_names if name not in mag_names]
mag_cols = [col_dict[name_dict[name]] for name in mag_names]

log_t, M_ini, M_act, log_L, log_Te, log_g = np.loadtxt(file_name, usecols=var_cols, unpack=True)
mag = np.loadtxt(file_name, usecols=mag_cols, unpack=True)

fig, ax = plt.subplots()
for age in np.arange(5, 10.3, 0.15):
    where_t = np.where(np.round(log_t, 2) == np.round(age, 2))
    ax.plot(log_Te[where_t], log_L[where_t])
    # ax.annotate(str(np.round(age, 2)), [np.min(log_Te[where_t]), np.max(log_L[where_t])])
ax.set_xlabel('log T (K)')
ax.set_ylabel('log L (L_sun)')
ax.invert_xaxis()
plt.show()

##
astobj = obg.AstObject(N_obj=10**6, age=[8], metal=[0.014], distance=10**4)

vis.HRD(10**astobj.log_Te, astobj.log_L)

## core - half light radius relation
r_par = np.arange(0.1, 20, 0.3)
ages = [8, 9, 10]

r_light_age = []
for age in ages:
    r_light = []
    for r in r_par:
        astobj = obg.AstObject(N_obj=10**5, age=[8], metal=[0.014], distance=10**4, r_dist='KingGlobular', r_dist_par=r)
        r_light.append(astobj.HalfLumRadius(spher=False))
        
    r_light_age.append(r_light)

##

fig, ax = plt.subplots()
ax.plot(r_par, r_light_age[0], label='age 8 glob')
ax.plot(r_par, r_light_age[1], label='age 9 glob')
ax.plot(r_par, r_light_age[2], label='age 10 glob')
ax.set_xlabel('core radius')
ax.set_ylabel('half light radius')
plt.legend()
plt.show()

with open('glob-core-light-radius.txt', 'w') as f:
    f.write('# core radius (s parameter) vs half light radius for 3 ages\n')
    f.write('# {0} {1} {2} {3}\n'.format('r_core', 'age 8', 'age 9', 'age 10'))
    for i in range(len(r_light_age[0])):
        f.write('{0} {1} {2} {3}\n'.format(np.round(r_par[i], decimals=1), r_light_age[0][i], r_light_age[1][i], r_light_age[2][i]))
##
combine_r = np.array(r_light_age).flatten()

a, b = np.polyfit(np.tile(r_par, 3), combine_r, deg=1)


##
import numpy as np
import os

np.memmap(os.path.join('cache', 'test'), )
qtt = np.random.rand(10**8)

qtt = np.append(qtt, np.random.rand(10**8))




## grid
"""
D   800 kpc - 15 Mpc
M   10**5 - 10**7
half light r 1-20 pc    (globular profile)
age 10**10      (as well as much younger)

out to what D can you resolve up to half light radius

foreground stars (models for MW)
"""
M_range = np.arange(1, 102, 20)*10**5                                       # in solar mass
D_range = np.arange(0.8, 15.3, 3.2)*10**6                                   # in pc
r_range = np.round(np.arange(1, 22, 5)/2.9, decimals=3)                     # in pc

par_grid = np.array([[M, D, r] for M in M_range for D in D_range for r in r_range])
par_grid[:, 0:2] = np.log10(par_grid[:, 0:2])


def objsaver(pars):
    M, D, r = pars              # M, D in 10log
    astobj = obg.AstObject(M_tot_init=10**M, age=[10], metal=[0.0014], distance=10**D, r_dist='KingGlobular', r_dist_par=r)
    astobj.SaveTo('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
    return
    
def imgsaver(pars):
    M, D, r = pars              # M, D in 10log
    astobj = obg.AstObject.LoadFrom('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
    src = img.MakeSource(astobj, filter='J')
    image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter='J', ao_mode='scao', filename='grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
    return
    
    
##
# run (part of) the grid    
for pars in par_grid[par_grid[:,0] < 6.7]:
    objsaver(pars)
    
## tests (something is broken)
mag = form.ApparentMag([-2], 10**6)
src = sim.source.stars(mags=[mag], x=[0], y=[0], filter_name='Ks', spec_types='M8IV')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='small', filter='Ks', ao_mode='scao', filename='img_test_save')
fh.PlotFits('img_test_save')
##
# 41274 and up is broken (only shows noise at that point)  [edit: not true]
astobj = obg.AstObject(N_obj=4*10**4, age=[10], metal=[0.0014], distance=10**5, r_dist='KingGlobular', r_dist_par=0.01)
src = img.MakeSource(astobj, filter='Ks')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='small', filter='Ks', ao_mode='scao', filename='img_test_save')
fh.PlotFits('img_test_save')
##
import numpy as np
import ImageGen as img
import simcado as sim
import fitshandler as fh
# could be a good background
n = int(5)
filter = 'Ks'
minmax = 4

xs = np.random.rand(n)*minmax - minmax/2                                                            # coords in as
ys = np.random.rand(n)*minmax - minmax/2                                                            # coords in as

magnitudes = np.linspace(15, 18, n)                    # breaks at a magnitude below 8.2

src = sim.source.stars(mags=magnitudes, x=xs, y=ys, filter_name=filter, spec_types=['M0V'])

image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='small', filter=filter, ao_mode='scao', filename='img_test_save')
fh.PlotFits('img_test_save')

##
filter = 'Ks'

mask = [item.endswith('G2III') | item.endswith('G3II') | item.endswith('G3III') | item.endswith('G4III') | item.endswith('G5III') for item in astobj.spec_names[astobj.spec_types]]

x_as = list((np.arctan(astobj.coords[:, 0]/astobj.d_ang)*648000/np.pi)[mask])                           # original coordinates assumed to be in pc
y_as = list((np.arctan(astobj.coords[:, 1]/astobj.d_ang)*648000/np.pi)[mask])                           # the  *648000/np.pi  is rad to as

magnitudes = astobj.ApparentMagnitude(filter_name=filter)[0]

src = sim.source.stars(mags=magnitudes[mask], 
                        x=x_as, 
                        y=y_as, 
                        filter_name=filter, 
                        spec_types=astobj.spec_names[astobj.spec_types][mask])

image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='small', filter=filter, ao_mode='scao', filename='img_test_save')
fh.PlotFits('img_test_save')

##
# load any astobj
types = astobj.spec_names

n = len(types)
filter = 'Ks'
minmax = 4

mask2 = [item.endswith('G2III') | item.endswith('G3II') | item.endswith('G3III') | item.endswith('G4III') | item.endswith('G5III') for item in types]
mask3 = [(not item.endswith('G2III')) & (not item.endswith('G3II')) & (not item.endswith('G3III')) & (not item.endswith('G4III')) & (not item.endswith('G5III')) & (not item.endswith('II')) & (not item.endswith('Ia')) & (not item.endswith('Ia0')) & (not item.endswith('Ib')) & (not item.endswith('V')) for item in types]
mask4 = [(not item.endswith('G2III')) & (not item.endswith('G3II')) & (not item.endswith('G3III')) & (not item.endswith('G4III')) & (not item.endswith('G5III')) & (not item.startswith('M')) & (not item.startswith('K')) & (not item.startswith('G')) & (not item.startswith('F')) & (not item.startswith('A')) & (not item.startswith('B')) & (not item.startswith('O')) for item in types]
mask5 = [(not item[1:].startswith('0')) & (not item[1:].startswith('1')) & (not item[1:].startswith('2')) & (not item[1:].startswith('3')) & (not item[1:].startswith('4')) & (not item[1:].startswith('5')) & (not item[1:].startswith('6')) & (not item[1:].startswith('7')) & (not item[1:].startswith('8')) & (not item[1:].startswith('9')) for item in types]
mask = [True for i in range(n)]

filter = 'Ks'

xs = np.random.rand(n)*minmax - minmax/2                                                            # coords in as
ys = np.random.rand(n)*minmax - minmax/2                                                            # coords in as

magnitudes = magnitudes = np.linspace(14, 15, n)

src = sim.source.stars(mags=magnitudes[mask], x=xs[mask], y=ys[mask], filter_name=filter, spec_types=types[mask])

# image = sim.run(src, 
#                 filename='data\\img_test_save.fits', 
#                 mode='wide', 
#                 detector_layout='small', 
#                 filter_name=filter, 
#                 SCOPE_PSF_FILE='scao', 
#                 OBS_EXPTIME=1800, 
#                 OBS_NDIT=1,
#                 ATMO_USE_ATMO_BG = "no",
#                 SCOPE_USE_MIRROR_BG = "no",
#                 INST_USE_AO_MIRROR_BG = "no",
#                 FPA_USE_NOISE = "no",
#                 FPA_LINEARITY_CURVE = "none" )
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='small', filter=filter, ao_mode='scao', filename='img_test_save.fits')
fh.PlotFits('img_test_save.fits')

##
# import multiprocessing as mp
# import _PoolParty as pp
# 
# with mp.Pool(processes=1) as pool:
#     pool.map(pp.objsaver, par_grid[par_grid[:,0] < 6.9])
##
import astropy.io.fits as fits
data = fits.getdata('data\\EC_pickles.fits')
# made changes

hdul = fits.open('data\\EC_pickles.fits')
# change: hdul[1].columns.names = data.columns.names
hdul.writeto('data\\EC_pickles2.fits')
##
# look into src.spectra --> lots of nan's
astobj = obg.AstObject(N_obj=4*10**4, age=[10], metal=[0.0014], distance=10**5, r_dist='KingGlobular', r_dist_par=0.01)
src = img.MakeSource(astobj, filter='Ks')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='small', filter='Ks', ao_mode='scao', filename='img_test_save')
fh.PlotFits('img_test_save')

# float32 turned into float64 --> fixed!!


## second IMF tests
def IMF(M, mass=[0.08, 150]):
    """(Forward) Initial Mass Function, normalized to 1 probability.
    Modified Salpeter IMF; actually Kroupa IMF above 0.08 solar mass.
    """
    M_L, M_U = mass
    M_mid = 0.5                                                                                     # fixed turnover position (where slope changes)
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = 1/(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))
    C_U = C_L*M_mid
    return (M < M_mid)*C_L*M**(-1.35) + (M >= M_mid)*C_U*M**(-2.35)
    
def invCIMF(n=1, mass=[0.08, 150]):
    """The inverted cumulative Initial Mass Function. Spits out n masses between lower and upper bound."""
    M_L, M_U = mass
    M_mid = 0.5                                                                                     # fixed turnover position (where slope changes)
    N_dist = np.random.rand(int(n))
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = 1/(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))
    # the mid value in the CDF
    N_mid = C_L/0.35*(M_L**(-0.35) - M_mid**(-0.35))
    # the inverted CDF
    M_a = (M_L**(-0.35) - 0.35*N_dist/C_L)**(-1/0.35)
    M_b = ((1.35/0.35*M_L**(-0.35) - M_mid**(-0.35)/0.35 - 1.35*N_dist/C_L)/M_mid)**(-1/1.35)
    return (N_dist < N_mid)*M_a + (N_dist >= N_mid)*M_b

def MassFraction(M, mass=[0.08, 150]):
    """Returns the fraction of stars in a population above a certain mass M (Msol)."""
    M_L, M_U = mass
    M_mid = 0.5                                                                                     # fixed turnover position (where slope changes)
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = (1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))**-1
    if (M < M_mid):
        f = C_L*(C_mid + M**(-0.35)/0.35 - M_mid*M_U**(-1.35)/1.35)
    else:
        f = C_L*M_mid/1.35*(M**(-1.35) - M_U**(-1.35))
    return f

masses = np.arange(0.08, 100, 0.01)

fig, ax = plt.subplots()
# IMF already has desired behaviour
ax.plot(np.log10(masses), np.log10(IMF(masses)), label='IMF')
ax.plot(np.log10(masses), np.log10(IMF(masses, mass=[1, 150])), label='IMF >1')
ax.plot(np.log10(masses), np.log10(IMF(masses, mass=[0.08, 0.3])), label='IMF <0.3')
ax.set_xlabel('log(M) (Msun)')
ax.set_ylabel('log(N) relative number')
ax.set_title('Output of the IMF')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(masses, IMF(masses), label='IMF')
ax.plot(masses, IMF(masses, mass=[1, 150]), label='IMF >1')
ax.plot(masses, IMF(masses, mass=[0.08, 0.3]), label='IMF <0.3')
# invCIMF also works a charm
ax.hist(invCIMF(n=10000, mass=[1, 150]), bins='auto', density=True, label='IMF >1')
ax.hist(invCIMF(n=10000, mass=[0.08, 0.3]), bins='auto', density=True, label='IMF <0.3')
ax.hist(invCIMF(n=10000), bins='auto', density=True, label='IMF')
ax.set_xlabel('log(M) (Msun)')
ax.set_ylabel('log(N) relative number')
plt.loglog()
plt.legend()
plt.show()

fig, ax = plt.subplots()
frac = []
for m in masses:
    frac.append(MassFraction(m, mass=[0.08, 150]))
ax.plot(masses, frac)
ax.set_xlabel('M (Msun)')
ax.set_ylabel('relative number')
plt.show()

##
# new function MassFraction
masses = np.arange(0.08, 100, 0.01)

fig, ax = plt.subplots()
frac1 = []
frac2 = []
for m in masses:
    frac1.append(form.MassFraction([m, 150], mass=[0.08, 150]))
for m in masses:
    frac2.append(form.MassFraction([0.1, m], mass=[0.08, 150]))
ax.plot(masses, frac1)
ax.plot(masses, frac2)
ax.set_xlabel('M (Msun)')
ax.set_ylabel('relative number')
plt.show()

## test M-L relation
import numpy as np
import astropy.constants as ac
import matplotlib.pyplot as plt
import conversions as conv

masses = np.arange(0.08, 200, 0.01)
lums = conv.MassToLuminosity(masses)
ms = conv.LuminosityToMass(lums, mass=[0.08, 200])

fig, ax = plt.subplots()
ax.plot(masses, lums)
ax.set_xlabel('M (Msun)')
ax.set_ylabel('L (Lsun)')
plt.show()

fig, ax = plt.subplots()
ax.plot(lums, ms)
ax.set_xlabel('L (Lsun)')
ax.set_ylabel('M (Msun)')
plt.show()

# M-T
Ts = conv.MassToTemperature(masses, lums)

fig, ax = plt.subplots()
ax.plot(masses[masses < 31], Ts[masses < 31])
ax.set_xlabel('M (Msun)')
ax.set_ylabel('T (K)')
plt.show()

## compare with isochrones
Z = 0.0014
file_name = os.path.join('tables', 'isoc_Z{1:1.{0}f}.dat'.format(-int(np.floor(np.log10(Z)))+1, Z))    # doesn't check if file for Z exists 

# names to use in the code, and a mapping to the isoc file column names 
code_names = np.array(['log_age', 'M_initial', 'M_actual', 'log_L', 'log_Te', 'log_g', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks'])
mag_names = code_names[6:]                                                                      # names of filters for later reference
mag_num = len(mag_names)
var_names, column_names = np.loadtxt(os.path.join('tables', 'column_names.dat'), dtype=str, unpack=True)

if (len(code_names) != len(var_names)):
    raise SyntaxError('ObjectGen//IsochroneProps: file "column_names.dat" has incorrect names specified. Use: {0}'.format(', '.join(code_names)))
elif np.any(code_names != var_names):
    raise SyntaxError('ObjectGen//IsochroneProps: file "column_names.dat" has incorrect names specified. Use: {0}'.format(', '.join(code_names)))
    
name_dict = {vn: cn for vn, cn in zip(var_names, column_names)}

# find the column names in the isoc file
with open(file_name) as file:
    for line in file:
        if line.startswith('#'):
            header = np.array(line.replace('#', '').split())
        else:
            break

col_dict = {name: col for col, name in enumerate(header) if name in column_names}

var_cols = [col_dict[name_dict[name]] for name in code_names if name not in mag_names]
mag_cols = [col_dict[name_dict[name]] for name in mag_names]

log_t, M_ini, M_act, log_L, log_Te, log_g = np.loadtxt(file_name, usecols=var_cols, unpack=True)
mag = np.loadtxt(file_name, usecols=mag_cols, unpack=True)
##
fig, ax = plt.subplots()
for age in np.arange(5, 10.3, 0.15):
    where_t = np.where(np.round(log_t, 2) == np.round(age, 2))
    # ax.plot(M_act[where_t], 10**log_L[where_t])
    ax.plot(M_ini[where_t], 10**log_L[where_t])
    # ax.annotate(str(np.round(age, 2)), [np.min(log_Te[where_t]), np.max(log_L[where_t])])
ax.plot(masses, lums, c='r')
ax.set_xlabel('log M_init (M_sun)')
ax.set_ylabel('log L (L_sun)')
ax.loglog()
plt.show()

fig, ax = plt.subplots()
for age in np.arange(5, 5.5, 0.15):
    where_t = np.where(np.round(log_t, 2) == np.round(age, 2))
    ax.plot(M_ini[where_t], 10**log_L[where_t])
ax.plot(masses, lums, c='r')
ax.set_xlabel('log M_init (M_sun)')
ax.set_ylabel('log L (L_sun)')
ax.loglog()
plt.show()

fig, ax = plt.subplots()
for age in np.arange(5, 10.3, 0.15):
    where_t = np.where(np.round(log_t, 2) == np.round(age, 2))
    ax.plot(M_ini[where_t], 10**log_Te[where_t])
ax.plot(masses, MassToTemperature(masses, lums), c='r')
ax.set_xlabel('log M_init (M_sun)')
ax.set_ylabel('log T_eff (K)')
ax.loglog()
plt.show()





## limiting magnitude
import numpy as np
import ImageGen as img
import simcado as sim
import fitshandler as fh

n = int(5)
filter = 'Ks'
minmax = 4

xs = np.linspace(-0.5, 0.5, 5)                                                            # coords in as
ys = np.linspace(-0.5, 0.5, 5)                                                            # coords in as

magnitudes = np.linspace(25, 29, n)

# Magnitude 28 seems to be where you loose it

src = sim.source.stars(mags=magnitudes, x=xs, y=ys, filter_name=filter, spec_types=['M0V'])

image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='small', filter=filter, ao_mode='scao', filename='img_test_save')
fh.PlotFits('img_test_save')

## AO tests
import numpy as np
import ImageGen as img
import simcado as sim
import fitshandler as fh

n = int(5)
filter = 'Ks'
minmax = 4

xs = np.linspace(-1, 1, 5)                                                            # coords in as
ys = np.linspace(-1, 1, 5)                                                            # coords in as

magnitudes = np.linspace(12, 18, n)

# Magnitude 28 seems to be where you loose it

src = sim.source.stars(mags=magnitudes, x=xs, y=ys, filter_name=filter, spec_types=['M0V'])

image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='small', filter=filter, ao_mode='PSF_MCAO.fits', filename='img_test_save')
fh.PlotFits('img_test_save', grid=False)


##
abs_mag_lim = form.AbsoluteMag(28, 2*10**8, ext=0)
# bolometric correction is added the second time, when an estimate for Teff is known
lum_lim = conv.MagnitudeToLuminosity(abs_mag_lim)
mass_lim = conv.LuminosityToMass(lum_lim, mass=[0.08, 150])
mag_BC = obg.BolometricCorrection(conv.MassToTemperature(mass_lim, lum_lim))

bol_mag_lim = abs_mag_lim + mag_BC
lum_lim = conv.MagnitudeToLuminosity(bol_mag_lim)
mass_limit = conv.LuminosityToMass(lum_lim, mass=[0.08, 150])

mag_BC = obg.BolometricCorrection(conv.MassToTemperature(mass_limit, lum_lim))
bol_mag_lim = abs_mag_lim + mag_BC
lum_lim = conv.MagnitudeToLuminosity(bol_mag_lim)
mass_limit2 = conv.LuminosityToMass(lum_lim, mass=[0.08, 150])
# fraction_generate = form.MassFraction(mass_limit, mass=[0.08, 150])
print(mass_lim, mass_limit, mass_limit2)
    # remove BC?
    
## 
# using isochrone
age = 7
Z = 0.0014
names = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks']
filter = 'Ks'
M_ini, mag, mag_names = obg.OpenIsochrone(age, Z, columns='mag')

fig, ax = plt.subplots()
ax.plot(M_ini, mag[-1], label=names[-1])
ax.plot([0,5], [2.5,2.5])
ax.set_xlabel('M_ini (M_sun)')
ax.set_ylabel('magnitude')
ax.invert_yaxis()
# ax.loglog()
plt.legend()
plt.show()

# it is multivalued ... 
# mass_lim = np.interp(2.6, -mag[-1], M_ini)

# cutoff = np.argmax((mag[-1, 1:]-mag[-1,:-1]) > 0.01)

mag_lim = 29
dists = np.logspace(1, 8, 10**3)
M_lim = []
for d in dists:
    abs_mag = form.AbsoluteMag(mag_lim, d)
    mask = (mag[-1] < abs_mag + 0.1) #& (mag[-1] > abs_mag - 0.1)
    if not mask.any():
        mask = (mag[-1] == np.min(mag[-1]))
        print('compacting will not work, distance too large.')
    M_lim.append(M_ini[mask][0])


fig, ax = plt.subplots()
ax.plot(dists, M_lim)
ax.set_xlabel('distance (pc)')
ax.set_ylabel('limiting mass (Msun)')
# ax.loglog()
plt.show()

## testing astobj with compact
astobj = obg.AstObject(N_obj=4*10**4, age=[6], metal=[0.0014], distance=10**8, r_dist='KingGlobular', r_dist_par=0.01, compact=True)

print(astobj.N_obj, len(astobj.M_init))
##
astobj = obg.AstObject(N_obj=10**9, age=[6], metal=[0.0014], distance=10**8, r_dist='KingGlobular', r_dist_par=0.01, compact=True)

print(astobj.N_obj, len(astobj.M_init))

##
M, D, r = np.log10(5*10**6), np.log10(0.8*10**6), 0.345              # M, D in 10log
astobj = obg.AstObject(M_tot_init=10**M, age=[10], metal=[0.0014], distance=10**D, r_dist='KingGlobular', r_dist_par=r)
astobj.SaveTo('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
src = img.MakeSource(astobj, filter='J')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r), grid=False)
# zoom
image = img.MakeImage(src, exposure=1800, NDIT=1, view='zoom', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-zoom'.format(M, D, r))
# fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-zoom'.format(M, D, r), grid=False)
##
astobj = obg.AstObject(M_tot_init=10**M, age=[10], metal=[0.0014], distance=10**D, r_dist='KingGlobular', r_dist_par=r, mag_lim=29, compact=True)
astobj.SaveTo('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-compact'.format(M, D, r))
src = img.MakeSource(astobj, filter='J')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-compact'.format(M, D, r))
fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-compact'.format(M, D, r), grid=False)
# zoom
image = img.MakeImage(src, exposure=1800, NDIT=1, view='zoom', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-zoom-compact'.format(M, D, r))
# fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-zoom-compact'.format(M, D, r), grid=False)

##
# D = np.log10(8*10**5)
astobj = obg.AstObject(M_tot_init=10**M, age=[10], metal=[0.0014], distance=10**D, r_dist='KingGlobular', r_dist_par=r, compact=True)
astobj.SaveTo('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-m32-compact'.format(M, D, r))
src = img.MakeSource(astobj, filter='J')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-m32-compact'.format(M, D, r))
fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-m32-compact'.format(M, D, r), grid=False)
# zoom
image = img.MakeImage(src, exposure=1800, NDIT=1, view='zoom', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-m32-zoom-compact'.format(M, D, r))
# fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-m32-zoom-compact'.format(M, D, r), grid=False)




## 99% of the light
age = 6
metal = 0.0014
M_ini, M_act, log_L, log_Te, log_g, mag, mag_names = obg.OpenIsochrone(age, metal, columns='all')
##
L_IMF = 10**log_L*dist.IMF(M_ini)

fig, ax = plt.subplots()
ax.plot(M_ini, L_IMF)
# ax.plot(M_ini, np.cumsum(L_IMF))
# ax.plot(M_ini[::-1], np.cumsum(L_IMF[::-1]))
ax.set_xlabel('initial mass')
ax.set_ylabel('luminosity * IMF')
# ax.loglog()
plt.show()

integral = np.cumsum(L_IMF[::-1])                                                                   # the integral (cumulative sum) starting at the high mass
fraction = 0.9999                                                                                   # depending on age, want following fractions: 6< 0.99, 7= 0.999, 8> 0.9999
M_lim = M_ini[::-1][np.argmin(np.abs(integral - fraction*integral[-1]))]

fig, ax = plt.subplots()
ax.plot(M_ini[::-1], integral)
ax.plot([M_ini[0], M_ini[-1]], [integral[-1]*fraction, integral[-1]*fraction])
ax.plot([M_lim, M_lim], [integral[0], integral[-1]])
ax.set_xlabel('initial mass')
ax.set_ylabel('luminosity * IMF')
# ax.loglog()
plt.show()

print(form.MassFraction([M_lim, M_ini[-1]]))

##
M, D, r = np.log10(5*10**6), np.log10(15*10**6), 0.345              # M, D in 10log
astobj = obg.AstObject(M_tot_init=10**M, age=[10], metal=[0.0014], distance=10**D, r_dist='KingGlobular', r_dist_par=r, compact=True, tot_lum=True)
astobj.SaveTo('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-lum-compact'.format(M, D, r))
src = img.MakeSource(astobj, filter='J')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-lum-compact'.format(M, D, r))
fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-lum-compact'.format(M, D, r), grid=False)
# zoom
image = img.MakeImage(src, exposure=1800, NDIT=1, view='zoom', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-lum-zoom-compact'.format(M, D, r))
# fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-lum-zoom-compact'.format(M, D, r), grid=False)

##
# static mag lim
M, D, r = np.log10(5*10**6), np.log10(0.8*10**6), 0.345              # M, D in 10log
astobj = obg.AstObject(M_tot_init=10**M, age=[10], metal=[0.0014], distance=10**D, r_dist='KingGlobular', r_dist_par=r, compact=True)
astobj.SaveTo('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-static-compact'.format(M, D, r))
src = img.MakeSource(astobj, filter='J')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-static-compact'.format(M, D, r))
fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-static-compact'.format(M, D, r), grid=False)
# zoom
image = img.MakeImage(src, exposure=1800, NDIT=1, view='zoom', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-static-zoom-compact'.format(M, D, r))
# fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-static-zoom-compact'.format(M, D, r), grid=False)

## masslimit by number
frac = np.arange(1, 0, -0.001)
mass = []
for f in frac:
    mass.append(form.MassLimit(f))

fig, ax = plt.subplots()
ax.plot(frac, mass)
ax.set_xlabel('fraction')
ax.set_ylabel('lower mass limit')
# ax.loglog()
plt.show()

##
# number lim
M, D, r = np.log10(5*10**6), np.log10(0.8*10**6), 0.345              # M, D in 10log
astobj = obg.AstObject(M_tot_init=10**M, age=[10], metal=[0.0014], distance=10**D, r_dist='KingGlobular', r_dist_par=r, compact=True, cp_mode='num')
astobj.SaveTo('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-num-compact'.format(M, D, r))
src = img.MakeSource(astobj, filter='J')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-num-compact'.format(M, D, r))
fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-num-compact'.format(M, D, r), grid=False)
# zoom
image = img.MakeImage(src, exposure=1800, NDIT=1, view='zoom', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-num-zoom-compact'.format(M, D, r))
# fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-num-zoom-compact'.format(M, D, r), grid=False)

##
N, D, r = np.log10(2*10**8), np.log10(15*10**6), 0.345              # M, D in 10log
astobj = obg.AstObject(N_obj=10**N, age=[10], metal=[0.0014], distance=10**D, r_dist='KingGlobular', r_dist_par=r, compact=True, cp_mode='num')
astobj.SaveTo('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-num-compact'.format(N, D, r))
src = img.MakeSource(astobj, filter='J')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-num-compact'.format(N, D, r))
fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-num-compact'.format(N, D, r), grid=False)
# zoom
image = img.MakeImage(src, exposure=1800, NDIT=1, view='zoom', chip='centre', filter='J', ao_mode='scao', filename='c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-num-zoom-compact'.format(N, D, r))
# fh.PlotFits('c_test-{0:1.3f}-{1:1.3f}-{2:1.3f}-num-zoom-compact'.format(N, D, r), grid=False)



## save fitsplots
import os
import astropy.io.fits as fits
from astropy.visualization import astropy_mpl_style

def GetData(filename, index=0):
    """Returns the requested data. [NOTE: data[1, 4] gives pixel value at x=5, y=2.] Optional arg: (HDUlist) index"""
    with fits.open(os.path.join('images', filename)) as hdul:
        return hdul[index].data

M, D, r = par_grid[-10]

astobj = obg.AstObject.LoadFrom('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
src = img.MakeSource(astobj, filter='J')
image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter='J', ao_mode='scao', filename='grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
##

def SaveFitsPlot(filename, index=0, colours='gray', grid=True):
    """Saves the plotted image in a fits file. Optional args: (HDUlist) index, colours.
    Can also take image objects directly.
    """
    if isinstance(filename, str):
        if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
            filename += '.fits'
    
        image_data = GetData(filename, index)
    else:
        image_data = filename[index].data
        
    plt.style.use(astropy_mpl_style)                                                                # use nice plot parameters
    fig, ax = plt.subplots(figsize=[12.0, 12.0])
    cax = ax.imshow(image_data, cmap=colours)
    
    ax.grid(grid)
    cbar = fig.colorbar(cax)
    
    if isinstance(filename, str):
        name = os.path.join('images', filename.replace('.fits', '.png').replace('.fit', '.png'))
    else:
        name = os.path.join('images', default_picture_file_name + '.png')
    
    plt.savefig(name, bbox_inches='tight', dpi=600)
    return


SaveFitsPlot('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r), grid=False)



## grid (try #2)
"""
D   800 kpc - 15 Mpc
M   10**5 - 10**7
half light r 1-20 pc    (globular profile)
age 10**10      (as well as much younger)

out to what D can you resolve up to half light radius

foreground stars (models for MW)
"""
M_range = np.arange(1, 102, 20)*10**5                                       # in solar mass
D_range = np.arange(0.8, 15.3, 3.2)*10**6                                   # in pc
r_range = np.round(np.arange(1, 22, 5)/2.9, decimals=3)                     # in pc

par_grid = np.array([[M, D, r] for M in M_range for D in D_range for r in r_range])
par_grid[:, 0:2] = np.log10(par_grid[:, 0:2])


def objsaver(pars):
    M, D, r = pars              # M, D in 10log
    astobj = obg.AstObject(M_tot_init=10**M, age=[10], metal=[0.0014], distance=10**D, r_dist='KingGlobular', r_dist_par=r, compact=True)
    astobj.SaveTo('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
    return
    
def imgsaver(pars):
    M, D, r = pars              # M, D in 10log
    astobj = obg.AstObject.LoadFrom('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
    src = img.MakeSource(astobj, filter='J')
    image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter='J', ao_mode='scao', filename='grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
    fh.SaveFitsPlot('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r), grid=False)
    return
    
    
##
# run the grid    
for pars in par_grid:
    objsaver(pars)
##
# make images
for pars in par_grid:
    imgsaver(pars)


## command test
# python3 constructor.py -struct ellipsoid -N 1000 -ages 9.65 10.0 9.0 -z 0.014 0.0014 0.014 -relN 1 1 1 -D 100.0


## zoom in for grid-7.004-5.903-2.069
M, D, r = 7.004,  5.903, 2.069
fh.PlotFits('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r), grid=False)

## more filters for grid-7.004-5.903-2.069
M, D, r = 7.004,  5.903, 2.069
astobj = obg.AstObject.LoadFrom('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
filters = ['U', 'B', 'V', 'R', 'I', 'H', 'Ks']

for fil in filters:
    src = img.MakeSource(astobj, filter=fil)
    image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', chip='centre', filter=fil, ao_mode='scao', filename='grid-{0:1.3f}-{1:1.3f}-{2:1.3f}-{3}'.format(M, D, r, fil))
    fh.PlotFits('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}-{3}'.format(M, D, r, fil), grid=False)

## RGB
# http://docs.astropy.org/en/stable/visualization/rgb.html
# https://www.astrobetter.com/blog/2010/10/22/making-rgb-images-from-fits-files-with-pythonmatplotlib/
import numpy as np
import matplotlib.pyplot as plt
import fitshandler as fh
import astropy.visualization as avis
import img_scale

M, D, r = 7.004,  5.903, 2.069                                                                      # try: I, J, H, Ks

R_data = fh.GetData('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}-Ks'.format(M, D, r))
G_data = fh.GetData('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}-H'.format(M, D, r))
B_data = fh.GetData('grid-{0:1.3f}-{1:1.3f}-{2:1.3f}'.format(M, D, r))
##
img = np.zeros((R_data.shape[0], R_data.shape[1], 3), dtype=float)

# linear
img[:,:,0] = (R_data - np.min(R_data))/(np.max(R_data) - np.min(R_data))
img[:,:,1] = (G_data - np.min(G_data))/(np.max(G_data) - np.min(G_data))
img[:,:,2] = (B_data - np.min(B_data))/(np.max(B_data) - np.min(B_data))
# seems equivalent
# img[:,:,0] = img_scale.linear(R_data)
# img[:,:,1] = img_scale.linear(G_data)
# img[:,:,2] = img_scale.linear(B_data)

# sqrt
# img[:,:,0] = np.sqrt(R_data - np.min(R_data))/np.sqrt(np.max(R_data) - np.min(R_data))
# img[:,:,1] = np.sqrt(G_data - np.min(G_data))/np.sqrt(np.max(G_data) - np.min(G_data))
# img[:,:,2] = np.sqrt(B_data - np.min(B_data))/np.sqrt(np.max(B_data) - np.min(B_data))
# seems equivalent
# img[:,:,0] = img_scale.sqrt(R_data)
# img[:,:,1] = img_scale.sqrt(G_data)
# img[:,:,2] = img_scale.sqrt(B_data)

# asinh
# img[:,:,0] = np.arcsinh(R_data - np.min(R_data))/np.arcsinh(np.max(R_data) - np.min(R_data))
# img[:,:,1] = np.arcsinh(G_data - np.min(G_data))/np.arcsinh(np.max(G_data) - np.min(G_data))
# img[:,:,2] = np.arcsinh(B_data - np.min(B_data))/np.arcsinh(np.max(B_data) - np.min(B_data))
# seems equivalent
# img[:,:,0] = img_scale.asinh(R_data)
# img[:,:,1] = img_scale.asinh(G_data)
# img[:,:,2] = img_scale.asinh(B_data)
# this is different
# img = avis.make_lupton_rgb(R_data, G_data, B_data)

fig, ax = plt.subplots(figsize=[12.0, 12.0])
ax.imshow(img, origin='upper')
plt.tight_layout()
plt.show()

##
import numpy as np
import matplotlib.pyplot as plt
import simcado as sim
import ObjectGen as obg
import fitshandler as fh
import visualizer as vis
import formulas as form
import distributions as dist
import conversions as conv
import ImageGen as img


# cd documents\documenten_radboud_university\masterstage\SMOC
# cd Documents\GitHub\SMOC







    
    
    

