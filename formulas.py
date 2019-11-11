# Luc IJspeert
# Part of starpopsim: (physical) formulas
##
"""Formulas that are not just conversions (to keep things more organized). 
Most functions optimized for processing many numbers at once (ndarray).
"""
import numpy as np

import utils
import conversions as conv


# global constants
c_light = 299792458.0           # m/s       speed of light 
G_newt = 274.0                  # Msun^-1 Rsun^2 m s-2
h_plank = 6.62607004*10**-34    # J s       Planck constant
k_B = 1.38064852*10**-23        # J/K       Boltzmann constant
parsec = 3.086*10**16           # m         parallax second
year = 31557600                 # s         length of year
Z_sun = 0.019                   # frac      solar metallicity
T_sun = 5772                    # K         solar effective temp
L_sun = 3.828*10**26            # W         solar luminosity
R_sun = 6.9551*10**8            # m         solar radius
M_ch = 1.456                    # M_sun     Chandrasekhar mass
H_0 = 67                        # km/s/Mpc  Hubble constant at present epoch
d_H = c_light*10**3/H_0         # pc        Hubble distance_3d
om_r = 9*10**-5                 # frac      omega radiation
om_m = 0.315                    # frac      omega matter
om_l = 0.685                    # frac      omega lambda ('dark energy')

# global defaults
default_imf_par = [0.08, 150]   # M_sun     lower bound, upper bound on mass
NS_mass = 1.2                   # M_sun     lower mass bound for NSs
BH_mass = 2.0                   # M_sun     lower mass bound for BHs


def distance_3d(points, position=None):
    """Calculates the 3 dimensional distance between a set of points and a position (default at origin)."""
    if not position:
        position = [0, 0, 0]
    # change array to [xs, ys, zs]
    coords = points.transpose()
    return ((coords[0] - position[0])**2 + (coords[1] - position[1])**2 + (coords[2] - position[2])**2)**(1/2)


def distance_2d(points, position=None):
    """Calculates the 2 dimensional distance between a set of points and a position (default at origin)."""
    if not position:
        position = [0, 0]
    # change array to [xs, ys]
    coords = points.transpose()
    return ((coords[0] - position[0])**2 + (coords[1] - position[1])**2)**(1/2)


def PlanckBB(nu, T, var='freq', allow_overflow=False):
    """Planck distribution of Black Body radiation. If var='wavl', use wavelength instead (in m).
    Units are either W/sr^1/m^2/Hz^1 or W/sr^1/m^3 (for freq/wavl).
    """
    if not allow_overflow:
        T = T + 2.2e-5/(nu) * (T*nu < 2.2e-5)                                                       # increase T where it would lead to overflows
    
    if (var == 'wavl'):
        lam = nu
        spec = (2*h_plank*c_light**2/lam**5)/(np.e**((h_plank*c_light)/(lam*k_B*T)) - 1)
    else:
        spec = (2*h_plank*nu**3/c_light**2)/(np.e**((h_plank*nu)/(k_B*T)) - 1)                      # else assume freq
        
    return spec


def BBMagnitude(T_eff, R, filters, filter_means=None):
    """Calculates roughly the magnitude (Vega) of a black body with temperature T (in K) 
    and radius R (in Rsun), in a set of filters (list of names).
    filter means can be specified to shift the filter wavelengths (mimicking redshift).
    """
    phot_dat = utils.open_photometric_data(columns=['mean', 'width'], filters=filters)
    if filter_means is not None:
        lam = filter_means
    else:
        lam = phot_dat['mean']
    width = phot_dat['width']
    lam_arr = np.linspace(lam - width/2, lam + width/2, 10).T
    
    # make shapes broadcastable, if not dealing with single numbers
    if hasattr(T_eff, '__len__'):
        T_eff = T_eff.reshape((len(T_eff),) + (1,)*(len(np.shape(lam_arr)) - (len(filters) == 1)))
    if hasattr(R, '__len__'):
        R = R.reshape((len(R),) + (1,)*(len(np.shape(lam)) - (len(filters) == 1)))
    
    spec_radiance = PlanckBB(lam_arr, T_eff, var='wavl')                                            # W/sr^1/m^3
    spec_radiance = np.mean(spec_radiance, axis=-1)                                                 # take the mean
    spec_flux_density = np.pi*spec_radiance                                                         # W/m^3     integrate d(Ohm)d(nu)
    spec_flux = spec_flux_density*(R*R_sun)**2                                                      # W/m       times surface star
    calibrated_spec_flux_density = spec_flux/(10*parsec)**2                                         # W/m^3     at ten pc
    mag = conv.FluxToMag(calibrated_spec_flux_density, filters=filters)
    return mag


def BBLuminosity(T_eff, R):
    """Luminosity (in Lsun) of a Black Body with given radius (in Rsun) and temperature (in K)"""
    return R**2*(T_eff/T_sun)**4


def LightTravelTime(z, points=1e4):
    """Gives the time it takes light (in yr) to travel from a certain redshift z.
    points is the number of steps for integration: higher=more precision, lower=faster.
    """
    min_log_z = -5                                                                                  # default minimum valid (log)redshift
    len_flag = hasattr(z, '__len__')
    
    # making sure it works for many different types of input
    if len_flag:
        z = np.array(z, dtype=np.float)                                                                
    else:
        z = np.array([z], dtype=float)
    
    # make sure there are no zeros or negative output
    zeros_z = (z == 0)
    below_min = (z < 10**(min_log_z + 1)) & (z != 0)
    z[zeros_z] = 10**min_log_z
    min_log_z = np.full_like(z, min_log_z)
    min_log_z[below_min] = np.log10(z[below_min]) - 5
    
    # do the actual calculation
    zs = np.logspace(min_log_z, np.log10(z), int(points))
    ys = 1/((1 + zs)*np.sqrt(om_r*(1 + zs)**4 + om_m*(1 + zs)**3 + om_l))
    integral = np.trapz(ys, zs, axis=0)
    if len_flag:
        time = integral*d_H*parsec/(c_light*year)
    else:
        time = integral[0]*d_H*parsec/(c_light*year)
    
    return time

    
def DComoving(z, points=1e4):
    """Gives the comoving distance_3d (in pc) given the redshift z.
    points is the number of steps for integration: higher=more precision, lower=faster.
    """
    min_log_z = -5                                                                                  # default minimum valid (log)redshift
    len_flag = hasattr(z, '__len__')
    
    # making sure it works for many different types of input
    if len_flag:
        z = np.array(z, dtype=np.float)                                                                
    else:
        z = np.array([z], dtype=float)
    
    # make sure there are no zeros or negative output
    zeros_z = (z == 0)
    below_min = (z < 10**(min_log_z + 1)) & (z != 0)
    z[zeros_z] = 10**min_log_z
    min_log_z = np.full_like(z, min_log_z)
    min_log_z[below_min] = np.log10(z[below_min]) - 5
    
    # do the actual calculation
    zs = np.logspace(min_log_z, np.log10(z), int(points))
    ys = 1/np.sqrt(om_r*(1 + zs)**4 + om_m*(1 + zs)**3 + om_l)
    integral =  np.trapz(ys, zs, axis=0)
    if len_flag:
        distance = integral*d_H
    else:
        distance = integral[0]*d_H
    
    return distance


def DLuminosity(z, points=1e4):
    """Gives the luminosity distance_3d (in pc) to the object given the redshift z.
    points is the number of steps for integration: higher=more precision, lower=faster.
    """
    # making sure it works for different types of input
    if hasattr(z, '__len__'):
        z = np.array(z)
        
    d_C = DComoving(z, points=points)
    return (1 + z)*d_C


def DAngular(z, points=1e4):
    """Gives the angular diameter distance_3d (in pc) to the object given the redshift z.
    points is the number of steps for integration: higher=more precision, lower=faster.
    """
    # making sure it works for different types of input
    if hasattr(z, '__len__'):
        z = np.array(z)
        
    d_C = DComoving(z, points=points)
    return d_C/(1 + z)


def DLToRedshift(dist, points=1e3):
    """Calculates the redshift assuming luminosity distance_3d (in pc) is given.
    Quite slow for arrays (usually not what it would be used for anyway).
    points is the number of steps for which z is calculated.
    """
    # this formula is a 'conversion' strictly speaking
    # but it is not really meant for number crunching 
    # and it belongs with the other distance_3d formulas
    num_dist = 10**3                                                                                # precision for distance_3d calculation
    len_flag = hasattr(dist, '__len__')
    
    # making sure it works for many different types of input
    if len_flag:
        dist = np.array(dist)
    else:
        dist = np.array([dist])
    
    z_est = 7.04208*10**-11*dist                                                                    # first estimate using a linear formula
    
    z_range = np.linspace(0.1*z_est, 10*z_est + 0.1, int(points))
    arg_best = np.argmin(np.abs(DLuminosity(z_range, points=num_dist) - dist), axis=0)
    z_best = z_range[arg_best, np.arange(len(arg_best))]
    if len_flag:
        z_best = z_range[arg_best, np.arange(len(arg_best))]
    else:
        z_best = z_range[arg_best, np.arange(len(arg_best))][0]
    
    return z_best


def ApparentMag(mag, dist, ext=0):
    """Compute the apparent magnitude for the absolute magnitude plus a distance_3d (in pc!).
    ext is an optional extinction to add (waveband dependent).
    """
    return mag + 5*np.log10(dist/10) + ext                                                          # dist in pc!


def AbsoluteMag(mag, dist, ext=0):
    """Compute the absolute magnitude for the apparant magnitude plus a distance_3d (in pc!).
    ext is an optional extinction to subtract (waveband dependent).
    """
    return mag - 5*np.log10(dist/10) - ext                                                          # dist in pc!


def MassFraction(mass_limits, imf=None):
    """Returns the fraction of stars in a population above and below certain mass_limits (Msol)."""
    if not imf:
        imf = default_imf_par
    M_L, M_U = imf
    # fixed turnover position (where slope changes)
    M_mid = 0.5
    M_lim_low, M_lim_high = mass_limits
    
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = (1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))**(-1)
    
    if (M_lim_low > M_mid):
        f = C_L*M_mid/1.35*(M_lim_low**(-1.35) - M_lim_high**(-1.35))
    elif (M_lim_high < M_mid):
        f = C_L/0.35*(M_lim_low**(-0.35) - M_lim_high**(-0.35))
    else:
        f = C_L*(C_mid + M_lim_low**(-0.35)/0.35 - M_mid*M_lim_high**(-1.35)/1.35)
    
    return f


def MassLimit(frac, M_max=None, imf=None):
    """Returns the lower mass limit to get a certain fraction of stars generated."""
    if not imf:
        imf = default_imf_par
    M_L, M_U = imf
    M_mid = 0.5  
    if (M_max is None):
        M_max = M_U
    
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = (1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))**(-1)
    # the mid value in the CDF
    N_mid = C_L*M_mid/1.35*(M_mid**(-1.35) - M_U**(-1.35))
    
    if (frac < N_mid):
        M_lim = (1.35*frac/(C_L*M_mid) + M_max**(-1.35))**(-1/1.35)
    else:
        M_lim = (0.35*(frac/C_L - C_mid + M_mid/1.35*M_max**(-1.35)))**(-1/0.35)
        
    return M_lim


def RemnantTime(M_init, age, Z):
    """Calculates approximately how long the remnant has been a remnant.
    Uses the isochrone files. Takes linear age or log(age) in years.
    """
    if (age <= 12):                                                                 # determine if stellar age in logarithm or not
        lin_age = 10**age                                                           # if so, go back to linear
    else:
        lin_age = age
    
    iso_log_t, iso_M_init = utils.open_isochrones_file(Z, columns=['log_age', 'M_initial'])
    t_steps = np.unique(iso_log_t)
    max_M_init = np.array([np.max(iso_M_init[iso_log_t == time]) for time in t_steps])
    
    indices = np.searchsorted(max_M_init[::-1], M_init)
    birth_time = 10**t_steps[::-1][indices]                                                         # time of remnant birth relative to age
    remnant_time = lin_age - birth_time                                                             # time since remnant birth
    
    if hasattr(M_init, '__len__'):
        remnant_time[remnant_time < 1] = 1                                                          # avoid negatives
    elif (remnant_time < 1):
        remnant_time = 1
    return remnant_time


def RemnantMass(M_init, Z=Z_sun):
    """Very rough estimate of the final (remnant) mass given an initial (ZAMS) mass.
    Also metallicity dependent.
    Taken from various papers stated in the code comments.
    """
    if hasattr(M_init, '__len__'):
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
    R_rem = np.zeros_like(M_rem)
    
    # masks for selecting the right remnants
    mask_WD = (M_rem < NS_mass)
    mask_NS = (M_rem >= NS_mass) & (M_rem < BH_mass)
    mask_BH = (M_rem >= BH_mass)
    
    R_rem[mask_WD] = 0.0126*(M_rem[mask_WD])**(-1/3)*(1 - (M_rem[mask_WD]/M_ch)**(4/3))**(1/2)      # WD radius: https://www.astro.princeton.edu/~burrows/classes/403/white.dwarfs.pdf
    R_rem[mask_NS] = 1.58*10**-5                                                                    # NS radius: uncertain due to e.o.s. but mostly close to constant 11 km
    R_rem[mask_BH] = 4.245*10**-6*M_rem[mask_BH]                                                    # BH radius: Schwarzschild radius
    
    return R_rem


def RemnantTeff(M_rem, R_rem, t_cool):
    """Approximation (very rough) of the effective temperature in Kelvin. 
    M_rem and R_rem in solar units and t_cool in years.
    These have to be arrays of the same length.
    Based on various sources stated in the source comments.
    """
    T_rem = np.zeros_like(M_rem)
    
    # parameter for WD cooling
    A = 12                                                                                          # (mean) atomic weight (assumes C WD)
    # parameters for NS cooling
    a_ns = 73
    s_ns = 2e-6
    q_ns = 1e-53
    
    # masks for selecting the right remnants
    mask_WD = (M_rem < NS_mass)
    mask_NS = (M_rem >= NS_mass) & (M_rem < BH_mass)
    mask_BH = (M_rem >= BH_mass)
    
    T_rem[mask_WD] = (T_sun*(10**8/(A*t_cool[mask_WD]))**(7/20)*(M_rem[mask_WD])**(1/4)
                     *(R_rem[mask_WD])**(-1/2))                                                     # WD temperature [Althaus et al., 2010] https://arxiv.org/abs/1007.2659
    
    # T_rem[mask_NS] = 2*10**(32/5)*t_cool[mask_NS]**(-2/5)                                         # [crude approx. https://pdfs.semanticscholar.org/2c10/e76c6c264161c48a4742d4c3ba80ed7fbc3f.pdf]
    log_g = conv.RadiusToGravity(R_rem[mask_NS], M_rem[mask_NS])
    T_rem[mask_NS] = ((10**6*a_ns*10**log_g/10**14)**(1/4) 
                     * ((s_ns/q_ns)/(np.e**(6*s_ns*t_cool[mask_NS]) - 1))**(1/12) 
                     * (1 - 2*G_newt*M_rem[mask_NS]/(R_rem[mask_NS]*c_light**2))**(1/4))            # quite involved this... [Ofengeim and Yakovlev, 2017] http://www.ioffe.ru/astro/Stars/Paper/ofengeim_yak17mn.pdf
    #todo: fix above formula to not overflow
    T_rem[mask_NS] += 10**-6                                                                        # avoid invalid logarithms [also, above formula encounters overflows in power]
    
    T_rem[mask_BH] = 6.169*10**(-8)/M_rem[mask_BH]                                                  # temperature due to Hawking radiation (derivation on wikipedia)
    
    return T_rem
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    