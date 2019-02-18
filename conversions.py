# Luc IJspeert
# Part of smoc: (coordinate) conversions
##
import numpy as np

'''Conversion between i.e. coordinates. Optimized for converting many numbers at once (ndarray).'''
# global constants
L_0 = 78.70                     # Lsun Luminosity for absolute bolometric magnitude 0
L_sun = 3.828*10**26            # W
R_sun = 6.957*10**8             # m
sigma_SB = 5.670367*10**-8      # W K^-4 m^-2

# global defaults
imf_defaults = [0.08, 150]                                                                          # lower bound, upper bound on mass


def PolToCart(r, theta):
    '''Converts polar coords to Cartesian. Optimized for arrays.'''
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.array([x, y])
    
def CartToPol(x, y):
    '''Converts Cartesian coords to polar. Optimized for arrays.'''
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    return np.array([r, theta])

def SpherToCart(r, theta, phi):
    '''Converts spherical coords to Cartesian. Optimized for arrays.'''
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.array([x, y, z])
    
def CartToSpher(x, y, z):
    '''Converts Cartesian coords to spherical. Optimized for arrays.'''
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)                                                                          # np.arctan(y/x) [doesn't do all the angles]
    return np.array([r, theta, phi])
    
def RotateXZ(positions, angle):
    '''Rotate a set of positions around y axis by 'angle' radians (so xz plane is rotated: positive x towards positive z).'''
    coords = positions.transpose()
    cos = np.cos(angle)
    sin = np.sin(angle)
    x_new = coords[0]*cos - coords[2]*sin
    z_new = coords[2]*cos + coords[0]*sin
    return np.array([x_new, coords[1], z_new]).transpose()

def MtotToNobj(M, mass=imf_defaults):
    '''Converts from mass in a cluster (one stellar population) to number of objects using the implemented IMF.'''
    M_L, M_U = mass
    M_mid = 0.5                                                                                     # fixed turnover position (where slope changes)
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    D_mid = (1/0.35 + 1/0.65)*M_mid**(0.65)
    C_L = 1/(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))
    M_mean = C_L*(D_mid - 1/0.65*M_L**(0.65) - M_mid/0.35*M_U**(-0.35))
    return np.rint(M/M_mean).astype(np.int64)
    
def NobjToMtot(N, mass=imf_defaults):
    '''Converts from number of objects in a cluster (one stellar population) to total mass using the implemented IMF.'''
    M_L, M_U = mass
    M_mid = 0.5                                                                                     # fixed turnover position (where slope changes)
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    D_mid = (1/0.35 + 1/0.65)*M_mid**(0.65)
    C_L = 1/(1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))
    M_mean = C_L*(D_mid - 1/0.65*M_L**(0.65) - M_mid/0.35*M_U**(-0.35))
    return N*M_mean
    
def GravityToRadius(log_g, M):
    '''Converts surface gravity (log g - cgs) to radius (Rsun). Mass must be given in Msun.'''
    G_newt = 274.0                                                                                  # Msun^-1 Rsun^2 m s-2
    g_si = 10**log_g/100                                                                            # convert log g (cgs - cm/s^2) to g (si - m/s^2)
    R_2 = G_newt*M/g_si                                                                             # radius (in Rsun) squared
    return np.sqrt(R_2)
    
def RadiusToGravity(R, M):
    '''Converts radius (Rsun) to surface gravity (log g - cgs). Mass must be given in Msun.'''
    G_newt = 274.0                                                                                  # Msun^-1 Rsun^2 m s-2
    g_si = G_newt*M/R**2                                                                            # surface gravity in m/s^2
    log_g = 2 + np.log10(g_si)                                                                      # convert g (si - m/s^2) to log g (cgs - cm/s^2)
    return log_g
    
def MagnitudeToLuminosity(mag):
    '''Çonverts from bolometric magnitude to luminosity (in Lsun).'''
    return L_0*10**(-0.4*mag)
    
def LuminosityToMagnitude(lum):
    '''Converts from luminosity (in Lsun) to bolometric magnitude.'''
    return -2.5*np.log10(lum/L_0)
    
def TemperatureToRGB(c_temp):
    '''Convert from colour temperature [1000K, 40000K] to RGB values in range [0, 1]. Expects an array.
    Below 1000K, return dark grey (0.2, 0.2, 0.2).
    Input: array-like
    Output: numpy.ndarray
    '''
    if isinstance(c_temp, (list, tuple)):
        c_temp = np.array(c_temp)
    
    # edited from source code at: http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    low = (c_temp < 1000)
    high = (c_temp > 40000)
    
    c_temp = np.clip(c_temp, 1000, 40000)                                                           # temperatures must be within: 1000 - 40000 K
    c_temp = c_temp/100                                                                             # algorithm uses temp/100
    
    # calculate red:
    red = (c_temp < 66)*255 
    red[c_temp >= 66] = 329.698727446*((c_temp - 60)[c_temp >= 66]**-0.1332047592)
    
    # calculate green:
    green = (c_temp <= 66)*c_temp
    green[c_temp <= 66] = 99.4708025861*np.log(green[c_temp <= 66]) - 161.1195681661
    green[c_temp > 66] = 288.1221695283*((c_temp - 60)[c_temp > 66]**-0.0755148492)
        
    # calculate blue:
    blue = (c_temp >= 66)*255
    blue[c_temp <= 19] = 0
    blue[(c_temp > 19) & (c_temp < 66)] = 138.5177312231*np.log((c_temp - 10)[(c_temp > 19) & (c_temp < 66)]) - 305.0447927307
    
    # cut off values and convert to floats
    R = np.clip(red, 0, 255)/255
    G = np.clip(green, 0, 255)/255
    B = np.clip(blue, 0, 255)/255
    
    R[low] = 0.2
    G[low] = 0.2
    B[low] = 0.2
    
    return np.array([R, G, B])
    
def WavelengthToRGB(wavelength, maxI=1):
    '''Converts wavelength in nm to RGB values (approximation). maxI is the maximum output of colour values.
    Input: array-like
    Output: numpy.ndarray
    ''' 
    if isinstance(wavelength, (list, tuple)):
        wavelength = np.array(wavelength)
    
    # editeed from source code at: http://www.efg2.com/Lab/ScienceAndEngineering/Spectra.htm
    Gamma        = 0.8                                                                              # determines gradient between colours
    IntensityMax = maxI                                                                             # determines max output value
    
    red = np.zeros(len(wavelength))
    green = np.zeros(len(wavelength))
    blue = np.zeros(len(wavelength))
    
    red[(wavelength >= 380) & (wavelength < 440)] = (440 - wavelength[(wavelength >= 380) & (wavelength < 440)])/(440 - 380)
    green[(wavelength >= 380) & (wavelength < 440)] = 0.0
    blue[(wavelength >= 380) & (wavelength < 440)] = 1.0
    
    red[(wavelength >= 440) & (wavelength < 490)] = 0.0
    green[(wavelength >= 440) & (wavelength < 490)] = (wavelength[(wavelength >= 440) & (wavelength < 490)] - 440)/(490 - 440)
    blue[(wavelength >= 440) & (wavelength < 490)] = 1.0
    
    red[(wavelength >= 490) & (wavelength < 510)] = 0.0
    green[(wavelength >= 490) & (wavelength < 510)] = 1.0
    blue[(wavelength >= 490) & (wavelength < 510)] = (510 - wavelength[(wavelength >= 490) & (wavelength < 510)])/(510 - 490)
    
    red[(wavelength >= 510) & (wavelength < 580)] = (wavelength[(wavelength >= 510) & (wavelength < 580)] - 510)/(580 - 510)
    green[(wavelength >= 510) & (wavelength < 580)] = 1.0
    blue[(wavelength >= 510) & (wavelength < 580)] = 0.0
    
    red[(wavelength >= 580) & (wavelength < 645)] = 1.0
    green[(wavelength >= 580) & (wavelength < 645)] = (645 - wavelength[(wavelength >= 580) & (wavelength < 645)])/(645 - 580)
    blue[(wavelength >= 580) & (wavelength < 645)] = 0.0
    
    red[(wavelength >= 645) & (wavelength <= 780)] = 1.0
    green[(wavelength >= 645) & (wavelength <= 780)] = 0.0
    blue[(wavelength >= 645) & (wavelength <= 780)] = 0.0
    
    # Let the intensity fall off near the vision limits
    factor = np.zeros(len(wavelength))
    factor[(wavelength >= 380) & (wavelength < 420)] = 0.3 + 0.7*(wavelength[(wavelength >= 380) & (wavelength < 420)] - 380)/(420 - 380)
    factor[(wavelength >= 420) & (wavelength < 700)] = 1.0
    factor[(wavelength >= 700) & (wavelength <= 780)] = 0.3 + 0.7*(780 - wavelength[(wavelength >= 700) & (wavelength <= 780)])/(780 - 700)
  
    def Adjust(colour, factor):
        # Don't want 0^x = 1 for x != 0
        return (colour != 0.0)*IntensityMax*(colour*factor)**Gamma
            
    R = Adjust(red,   factor)
    G = Adjust(green, factor)
    B = Adjust(blue,  factor)
  
    return np.array([R, G, B])














