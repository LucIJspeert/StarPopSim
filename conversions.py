"""Conversion between i.e. coordinates.
Optimized for converting many numbers at once (ndarray).
"""
import numpy as np
import utils


# global constants
L_0 = 78.70                     # Lsun      Luminosity for absolute bolometric magnitude 0
L_sun = 3.828*10**26            # W         solar luminosity
R_sun = 6.9551*10**8            # m         solar radius
sigma_SB = 5.670367*10**-8      # W K^-4 m^-2
G_newt = 274.0                  # Msun^-1 Rsun^2 m s-2
rad_as = 648000/np.pi           # radians to arcseconds
as_rad = np.pi/648000           # arcseconds to radians

# global defaults
default_imf_par = [0.08, 150]   # M_sun     lower bound, upper bound on mass


def pol_to_cart(r, theta):
    """Converts polar coords to Cartesian. Optimized for arrays."""
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.array([x, y])


def cart_to_pol(x, y):
    """Converts Cartesian coords to polar. Optimized for arrays."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    return np.array([r, theta])


def spher_to_cart(r, theta, phi):
    """Converts spherical coords to Cartesian. Optimized for arrays."""
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.array([x, y, z])


def cart_to_spher(x, y, z):
    """Converts Cartesian coords to spherical. Optimized for arrays."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)  # np.arctan(y/x) [doesn't do all the angles]
    return np.array([r, theta, phi])


def rotate_xz(coords, angle):
    """Rotate a set of positions around y axis by 'angle' radians 
    (so xz plane is rotated: positive x towards positive z).
    First axis is assumed to correspond to x,y,z (so x=coords[0]).
    """
    cos = np.cos(angle)
    sin = np.sin(angle)
    x_new = coords[0]*cos - coords[2]*sin
    z_new = coords[2]*cos + coords[0]*sin
    return np.array([x_new, coords[1], z_new])
    
    
def parsec_to_arcsec(x, distance):
    """Convert from distances (x) perpendicular to the distance (d)
    (both in parsec, or otherwise the same units) to arcseconds.
    """
    return np.arctan2(x, distance)*rad_as
    
    
def arcsec_to_parsec(x, distance):
    """Convert from arcseconds (x) perpendicular to the distance (d)
    (in parsec) to distances in parsec.
    """
    return np.tan(x*as_rad)*distance
    
    
def parsec_to_d_modulus(distance):
    """Compute the distance modulus given a distance in pc."""
    return 5*np.log10(distance/10)
    
    
def d_modulus_to_parsec(mod):
    """Compute the distance in pc given a distance modulus."""
    return 10**(mod/5 + 1)


def m_tot_to_n_stars(M, imf=None):
    """Converts from mass in a cluster (per single stellar population)
    to number of objects using the implemented IMF.
    """
    M = np.atleast_1d(M)
    if imf is None:
        imf = np.full([len(M), len(default_imf_par)], default_imf_par)
    else:
        imf = np.atleast_2d(imf)
    M_low, M_high = imf[:, 0], imf[:, 1]
    M_mid = 0.5  # fixed turnover position (where slope changes)
    c_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    d_mid = (1/0.35 + 1/0.65)*M_mid**(0.65)
    c_low = 1/(1/0.35*M_low**(-0.35) + c_mid - M_mid/1.35*M_high**(-1.35))
    M_mean = c_low*(d_mid - 1/0.65*M_low**(0.65) - M_mid/0.35*M_high**(-0.35))
    return np.rint(M/M_mean).astype(np.int64)


def n_stars_to_m_tot(N, imf=None):
    """Converts from number of objects in a cluster (one stellar population) 
    to total mass using the implemented IMF.
    """
    N = np.atleast_1d(N)
    if imf is None:
        imf = np.full([len(N), len(default_imf_par)], default_imf_par)
    else:
        imf = np.atleast_2d(imf)
    M_low, M_high = imf[:, 0], imf[:, 1]
    M_mid = 0.5  # fixed turnover position (where slope changes)
    c_mid = (1/1.35 - 1/0.35) * M_mid**(-0.35)
    d_mid = (1/0.35 + 1/0.65) * M_mid**(0.65)
    c_low = 1/(1/0.35 * M_low**(-0.35) + c_mid - M_mid/1.35 * M_high**(-1.35))
    M_mean = c_low * (d_mid - 1/0.65 * M_low**(0.65) - M_mid/0.35 * M_high**(-0.35))
    return N * M_mean


def gravity_to_radius(log_g, M):
    """Converts surface gravity (log g - cgs) to radius (Rsun). Mass must be given in Msun."""
    # convert log g (cgs - cm/s^2) to g (si - m/s^2)
    g_si = 10**log_g/100
    # radius (in Rsun) squared
    R_2 = G_newt*M/g_si
    return np.sqrt(R_2)


def radius_to_gravity(R, M):
    """Converts radius (Rsun) to surface gravity (log g - cgs). Mass must be given in Msun."""
    # surface gravity in m/s^2
    g_si = G_newt*M/R**2
    # convert g (si - m/s^2) to log g (cgs - cm/s^2)
    log_g = 2 + np.log10(g_si)
    return log_g


def mag_to_lum(mag):
    """Converts from bolometric magnitude to luminosity (in Lsun)."""
    return L_0*10**(-0.4*mag)


def lum_to_mag(lum):
    """Converts from luminosity (in Lsun) to bolometric magnitude."""
    return -2.5*np.log10(lum/L_0)
    
    
def flux_to_mag(flux, filters=None):
    """Converts spectral flux density (in W/m^3) to magnitude in certain filters."""
    if filters is not None:
        zero_point_flux = utils.open_photometric_data(columns=['zp_flux'], filters=filters)
        mag = -2.5*np.log10(flux/zero_point_flux)
    else:
        # just convert raw input
        mag = -2.5*np.log10(flux)
    
    return mag


def temperature_to_rgb(c_temp):
    """Convert from colour temperature [1000K, 40000K] to RGB values in range [0, 1]. 
    Expects an array.
    Below 1000K, return dark grey (0.2, 0.2, 0.2).
    Input: array-like
    Output: numpy.ndarray
    """
    if hasattr(c_temp, '__len__'):
        c_temp = np.array(c_temp)
    
    # edited from source code at: 
    # http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    low = (c_temp < 1000)
    high = (c_temp > 40000)
    
    # temperatures must be within: 1000 - 40000 K. algorithm uses temp/100
    c_temp = np.clip(c_temp, 1000, 40000)
    c_temp = c_temp/100
    
    # calculate red:
    red = (c_temp < 66)*255 
    red[c_temp >= 66] = 329.698727446*((c_temp - 60)[c_temp >= 66]**(-0.1332047592))
    
    # calculate green:
    green = (c_temp <= 66)*c_temp
    green[c_temp <= 66] = 99.4708025861*np.log(green[c_temp <= 66]) - 161.1195681661
    green[c_temp > 66] = 288.1221695283*((c_temp - 60)[c_temp > 66]**(-0.0755148492))
        
    # calculate blue:
    blue = (c_temp >= 66)*255
    blue[c_temp <= 19] = 0
    c_temp_range = (c_temp - 10)[(c_temp > 19) & (c_temp < 66)]
    blue[(c_temp > 19) & (c_temp < 66)] = 138.5177312231*np.log(c_temp_range) - 305.0447927307
    
    # cut off values and convert to floats
    red = np.clip(red, 0, 255)/255
    green = np.clip(green, 0, 255)/255
    blue = np.clip(blue, 0, 255)/255
    
    red[low] = 0.2
    green[low] = 0.2
    blue[low] = 0.2
    return np.array([red, green, blue])


def wavelength_to_rgb(wavelength, max_intensity=1):
    """Converts wavelength in nm to RGB values (approximation). 
    maxI is the maximum output of colour values.
    Input: array-like
    Output: numpy.ndarray
    """ 
    if hasattr(wavelength, '__len__'):
        wavelength = np.array(wavelength)
    
    # edited from source code at: http://www.efg2.com/Lab/ScienceAndEngineering/Spectra.htm
    # Gamma determines gradient between colours, max_intensity determines max output value
    gamma = 0.8
    
    red = np.zeros(len(wavelength))
    green = np.zeros(len(wavelength))
    blue = np.zeros(len(wavelength))
    
    mask_1 = (wavelength >= 380) & (wavelength < 440)
    red[mask_1] = (440 - wavelength[mask_1])/(440 - 380)
    green[mask_1] = 0.0
    blue[mask_1] = 1.0
    
    mask_2 = (wavelength >= 440) & (wavelength < 490)
    red[mask_2] = 0.0
    green[mask_2] = (wavelength[mask_2] - 440)/(490 - 440)
    blue[mask_2] = 1.0
    
    mask_3 = (wavelength >= 490) & (wavelength < 510)
    red[mask_3] = 0.0
    green[mask_3] = 1.0
    blue[mask_3] = (510 - wavelength[mask_3])/(510 - 490)
    
    mask_4 = (wavelength >= 510) & (wavelength < 580)
    red[mask_4] = (wavelength[mask_4] - 510)/(580 - 510)
    green[mask_4] = 1.0
    blue[mask_4] = 0.0
    
    mask_5 = (wavelength >= 580) & (wavelength < 645)
    red[mask_5] = 1.0
    green[mask_5] = (645 - wavelength[mask_5])/(645 - 580)
    blue[mask_5] = 0.0
    
    mask_6 = (wavelength >= 645) & (wavelength <= 780)
    red[mask_6] = 1.0
    green[mask_6] = 0.0
    blue[mask_6] = 0.0
    
    # Let the intensity fall off near the vision limits
    mask_7 = (wavelength >= 380) & (wavelength < 420)
    mask_8 = (wavelength >= 420) & (wavelength < 700)
    mask_9 = (wavelength >= 700) & (wavelength <= 780)
    factor = np.zeros_like(wavelength)
    factor[mask_7] = 0.3 + 0.7*(wavelength[mask_7] - 380)/(420 - 380)
    factor[mask_8] = 1.0
    factor[mask_9] = 0.3 + 0.7*(780 - wavelength[mask_9])/(780 - 700)
  
    def adjust(colour, reduce_factor):
        # Don't want 0^x = 1 for x != 0
        return (colour != 0.0)*max_intensity*(colour*reduce_factor)**gamma
            
    red = adjust(red, factor)
    green = adjust(green, factor)
    blue = adjust(blue, factor)
    return np.array([red, green, blue])














