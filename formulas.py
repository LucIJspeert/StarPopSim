# Luc IJspeert
# Part of smoc: (physical) formulas
##
"""Formulas that are not just conversions (to keep things more organized). 
Most functions optimized for processing many numbers at once (ndarray).
"""
import numpy as np


# global constants
c_light = 299792458.0           # m/s speed of light 
h_plank = 6.62607004*10**-34    # J s Planck constant
k_B = 1.38064852*10**-23        # J/K Boltzmann constant
Z_sun = 0.019                   # metallicity
T_sun = 5772                    # K (effective temp)
M_ch = 1.456                    # in M_sun
H_0 = 67                        # km/s/Mpc
d_H = c_light*10**3/H_0         # Hubble distance in pc
om_r = 9*10**-5                 # omega radiation
om_m = 0.315                    # omega matter
om_l = 0.685                    # omega lambda ('dark energy')

# global defaults
imf_defaults = [0.08, 150]      # lower bound, upper bound on mass


def Distance(points, position=[0,0,0]):
    """Calculates the distance between a set of points and a position (default at origin)."""
    coords = points.transpose()                                                                     # change array to [xs, ys, zs]
    return ((coords[0] - position[0])**2 
            + (coords[1] - position[1])**2 
            + (coords[2] - position[2])**2)**(1/2)


def Distance2D(points):
    """Calculates the 2 dimensional distance to the origin of a point set in the xy plane."""
    coords = points.transpose()                                                                     # take only x and y coordinates
    return (coords[0]**2 + coords[1]**2)**(1/2)


def PlanckBB(nu, T, var='freq'):
    """Planck distribution of Black Body radiation. If var='wavl', use wavelength instead.
    Units are either W/sr^1/m^2/Hz^1 or W/sr^1/m^3 (for freq/wavl).
    """
    if (var == 'wavl'):
        lam = nu
        spec = (2*h_plank*c_light**2/lam**5)/(np.e**((h_plank*c_light)/(lam*k_B*T)) - 1)
    else:
        spec = (2*h_plank*nu**3/c_light**2)/(np.e**((h_plank*nu)/(k_B*T)) - 1)                      # else assume freq
        
    return spec


def DComoving(z):
    """Gives the comoving distance (in pc) given the redshift z."""
    num_points = 10**5
    
    if isinstance(z, list):
        z = np.array(z)                                                                             # just making sure it works for lists
    
    if isinstance(z, (np.ndarray)):
        N = int(np.max([100, num_points/len(z)]))                                                   # integrate with at least 100 points
        zs = np.array([np.logspace(-5, np.log10(z[i]), N) for i in range(len(z)) if z[i] != 0])
        ys = 1/np.sqrt(om_r*(1 + zs)**4 + om_m*(1 + zs)**3 + om_l)
        integral =  np.trapz(ys, zs, axis=1)
    else:
        zs = np.logspace(-5, np.log10(z), num_points)
        ys = 1/np.sqrt(om_r*(1 + zs)**4 + om_m*(1 + zs)**3 + om_l)
        integral =  np.trapz(ys, zs)
        
    return integral*d_H


def DLToRedshift(dist):
    """Calculates the redshift assuming luminosity distance (in pc) is given.
    Quite slow for arrays (usually not what it would be used for anyway).
    """
    num_points = 1000
    
    if isinstance(dist, list):
        dist = np.array(dist)                                                                       # just making sure it works for lists
    
    z_est = 7.04208*10**-11*dist                                                                    # first estimate using a linear formula
    
    if isinstance(dist, (np.ndarray)):
        z_ranges = np.array([np.linspace(0.1*z, 10*z, num_points) for z in z_est])
        z_best = []
        for d, z_range in zip(dist, z_ranges):
            z_best.append(z_range[np.argmin(np.abs(DLuminosity(z_range) - d))])
        z_best = np.array(z_best)
    else:
        z_range = np.linspace(0.1*z_est, 10*z_est, num_points)
        z_best = z_range[np.argmin(np.abs(DLuminosity(z_range) - dist))]
        
    return z_best


def DLuminosity(z):
    """Gives the luminosity distance (in pc) to the object given the redshift z."""
    if isinstance(z, list):
        z = np.array(z)                                                                             # just making sure it works for lists
        
    d_C = DComoving(z)
    return (1 + z)*d_C


def DAngular(z):
    """Gives the angular diameter distance (in pc) to the object given the redshift z."""
    if isinstance(z, list):
        z = np.array(z)                                                                             # just making sure it works for lists
        
    d_C = DComoving(z)
    return d_C/(1 + z)


def ApparentMag(mag, dist, ext=0, sigma=[0]):
    """Compute the apparent magnitude for the absolute magnitude plus a distance (in pc!). 
    sigma defines individual distances relative to the distance of the objects distance.
    ext is an optional extinction to add (waveband dependent).
    """
    if (np.shape(mag) != ()):
        if (len(sigma) == len(mag)):
            z_coord = np.array(sigma)                                                               # can use given coordinates
        else:
            z_coord = np.array([0])                                                                 # define z coord relative to dist 
    else:
        z_coord = 0
        
    return mag + 5*np.log10((dist + z_coord)/10.) + ext                                             # sigma and dist in pc!


def AbsoluteMag(mag, dist, ext=0):
    """Compute the absolute magnitude for the apparant magnitude plus a distance (in pc!).
    ext is an optional extinction to subtract (waveband dependent).
    """
    return mag - 5*np.log10(dist/10.) - ext                                                         # dist in pc!


def MSLifetime(M):
    """Estimate the MS lifetime in years of a star of certain initial mass M (in Msun).
    Applies to stars from 0.1 to 50 Msun (strictly speaking)
    """
    return 10**10*(M)**-2.5


def BBLuminosity(R, T_eff):
    """Luminosity (in Lsun) of a Black Body with given radius (Rsun) and temperature (K)"""
    return R**2*(T_eff/T_sun)**4


def MassFraction(mass_limits, imf=imf_defaults):
    """Returns the fraction of stars in a population above and below certain mass_limits (Msol)."""
    M_L, M_U = imf
    M_mid = 0.5                                                                                     # fixed turnover position (where slope changes)
    M_lim_low, M_lim_high = mass_limits
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = (1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))**-1
    
    if (M_lim_low > M_mid):
        f = C_L*M_mid/1.35*(M_lim_low**(-1.35) - M_lim_high**(-1.35))
    elif (M_lim_high < M_mid):
        f = C_L/0.35*(M_lim_low**(-0.35) - M_lim_high**(-0.35))
    else:
        f = C_L*(C_mid + M_lim_low**(-0.35)/0.35 - M_mid*M_lim_high**(-1.35)/1.35)
    
    return f


def MassLimit(frac, M_max=None, imf=imf_defaults):
    """Returns the lower mass limit to get a certain fraction of stars generated."""
    M_L, M_U = imf
    M_mid = 0.5  
    if (M_max is None):
        M_max = M_U
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = (1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))**-1
    # the mid value in the CDF
    N_mid = C_L*M_mid/1.35*(M_mid**(-1.35) - M_U**(-1.35))
    
    if (frac < N_mid):
        M_lim = (1.35*frac/(C_L*M_mid) + M_max**(-1.35))**(-1/1.35)
    else:
        M_lim = (0.35*(frac/C_L - C_mid + M_mid/1.35*M_max**(-1.35)))**(-1/0.35)
        
    return M_lim


def RemnantMass(M_init, Z=Z_sun):
    """Very rough estimate of the final (remnant) mass given an initial (ZAMS) mass.
    Also metallicity dependent.
    Taken from various papers
    """
    if isinstance(M_init, (list, tuple)):
        M_init = np.array(M_init)
    
    Z_f = Z/Z_sun                                                                                   # metallicity as fraction of solar
    
    # define the mass intervals
    mask_1 = M_init < 0.85
    mask_2 = (M_init >= 0.85) & (M_init < 2.85)
    mask_3 = (M_init >= 2.85) & (M_init < 3.6)
    mask_4 = (M_init >= 3.6) & (M_init < 7.56)                                                      # upper border has changed slightly
    mask_5 = (M_init >= 7.56) & (M_init < 11)                                                       # lower border has changed slightly
    mask_6 = (M_init >= 11) & (M_init < 30)
    mask_7 = (M_init >= 30) & (M_init < 50)
    mask_8 = (M_init >= 50) & (M_init < 90)
    mask_9 = M_init >= 90
    mask_789 = np.any([mask_7, mask_8, mask_9], axis=0)                                             # needed for the high mass formulae
    mask_789_7 = mask_7[mask_789]                                                                   # to select 8 from 789
    mask_789_8 = mask_8[mask_789]                                                                   # to select 8 from 789
    mask_789_9 = mask_9[mask_789]                                                                   # to select 9 from 789
    
    M_08 = 0.655*M_init[mask_1]                                                                     # below 0.85 Msun relation is continued linearly through zero
    # WD masses
    M_08_28 = 0.08*M_init[mask_2] + 0.489                                                           # mass range 0.85-2.85 [J.D.Cummings 2018]
    M_28_36 = 0.187*M_init[mask_3] + 0.184                                                          # mass range 2.85-3.60 [J.D.Cummings 2018]
    M_36_72 = 0.107*M_init[mask_4] + 0.471                                                          # mass range 3.60-7.20 [J.D.Cummings 2018]
    # NS/BH masses
    M_72_11 = 1.28                                                                                  # mass range 7.20-11 (to close the gap, based on [C.L.Fryer 2011])
    M_11_30 = (1.1 + 0.2*np.exp((M_init[mask_6] - 11)/4) 
              - (2.0 + Z_f)*np.exp(0.4*(M_init[mask_6] - 26)))                                      # mass range 11-30 [C.L.Fryer 2011]
    
    M_30_50_1 = 33.35 + (4.75 + 1.25*Z_f)*(M_init[mask_789] - 34)
    M_30_50_2 = M_init[mask_789] - (Z_f)**0.5*(1.3*M_init[mask_789] - 18.35)
    M_30_50 = np.min([M_30_50_1, M_30_50_2], axis=0)                                                # mass range 30-50 [C.L.Fryer 2011]
    M_50_90_high = 1.8 + 0.04*(90 - M_init[mask_8])                                                 # mass range 50-90, high Z [C.L.Fryer 2011]
    M_90_high = 1.8 + np.log10(M_init[mask_9] - 89)                                                 # mass above 90, high Z [C.L.Fryer 2011]
    M_50_90_low = np.max([M_50_90_high, M_30_50[mask_789_8]], axis=0)                               # mass range 50-90, low Z [C.L.Fryer 2011]
    M_90_low = np.max([M_90_high, M_30_50[mask_789_9]], axis=0)                                     # mass above 90, high Z [C.L.Fryer 2011]
    
    M_remnant = np.zeros_like(M_init, dtype=float)
    
    M_remnant[mask_1] = M_08
    M_remnant[mask_2] = M_08_28
    M_remnant[mask_3] = M_28_36
    M_remnant[mask_4] = M_36_72
    M_remnant[mask_5] = M_72_11
    M_remnant[mask_6] = M_11_30
    M_remnant[mask_7] = M_30_50[mask_789_7]
    if (Z >= Z_sun):
        M_remnant[mask_8] = M_50_90_high
        M_remnant[mask_9] = M_90_high
    else:
        M_remnant[mask_8] = M_50_90_low
        M_remnant[mask_9] = M_90_low
        
    return M_remnant


def RemnantRadius(M_rem):
    """Approximate radius (in Rsun) of remnant with mass M_rem (in Msun)."""
    R_remnant = np.zeros_like(M_rem)
    
    mask_WD = M_rem < 1.2
    R_remnant[mask_WD] = 0.0126*(M_rem[mask_WD])**(-1/3)*(1 - (M_rem[mask_WD]/M_ch)**(4/3))**(1/2)  # WD radius: https://www.astro.princeton.edu/~burrows/classes/403/white.dwarfs.pdf
    R_remnant[(M_rem > 1.2) & (M_rem <= 2.0)] = 1.58*10**-5                                         # NS radius: uncertain due to e.o.s. but mostly close to constant 11 km
    mask_BH = (M_rem > 2.0)
    R_remnant[mask_BH] = 4.245*10**-6*M_rem[mask_BH]                                                # BH radius: Schwarzschild radius
    
    return R_remnant


def RemnantTeff(M_rem, R_rem, t_cool, BH_mass=2.0):
    """Approximation of the effective temperature. 
    M_rem and R_rem in solar units and t_cool in years.
    These have to be arrays of the same length.
    Based on WD cooling from Althaus et al. 2010: https://arxiv.org/abs/1007.2659
    """
    A = 12                                                                                          # (mean) atomic weight (assumes C WD)
    
    Temps = np.zeros_like(M_rem)
    
    Temps[M_rem > 2] = 10**-8                                                                       # above mass M_rem=BH_mass is considered a BH
    
    mask = (M_rem <= 2)
    Temps[mask] = T_sun*(10**8/(A*t_cool[mask]))**(7/20)*(M_rem[mask])**(1/4)*(R_rem[mask])**(-1/2)
    
    return Temps


def RemnantMagnitudes(T_rem, L_rem):
    """Estimates very roughly what the magnitudes should be in various filters,
    from the temperatures (K) and the luminosities (Lsun).
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    