"""Distribution definitions to randomly draw numbers from. Optimized for drawing many numbers at once (array).
Distributions ending in _r are 3D radial (spherical) variants of the 1D (cartesian) versions.
Distributions ending in _rho are 2D radial (cylindrical) variants.
"""
import numpy as np
import scipy.special as sps


# global defaults
default_imf_par = [0.08, 150]   # M_sun     lower bound, upper bound on mass


def pdf_kroupa_imf(M, imf=None):
    """(Forward) Initial Mass Function (=probability density function), 
    normalized to 1 probability.
    Modified Salpeter IMF; actually Kroupa IMF above 0.08 solar mass.

    M: int or array of float
    imf: 1D or 2D array of float (in pairs of 2), optional

    out:
        array of float
    """
    M = np.atleast_1d(M)
    if not imf:
        imf = np.full([len(M), len(default_imf_par)], default_imf_par)
    else:
        imf = np.atleast_2d(imf)
    M_low, M_high = imf[:, 0], imf[:, 1]
    M_mid = 0.5  # fixed turnover position (where slope changes)
    c_mid = (1/1.35 - 1/0.35) * M_mid**(-0.35)
    c_low = (1/0.35 * M_low**(-0.35) + c_mid - M_mid/1.35 * M_high**(-1.35))**-1
    c_high = c_low * M_mid
    return (M < M_mid) * c_low * M**(-1.35) + (M >= M_mid) * c_high * M**(-2.35)


def kroupa_imf(n=1, imf=None):
    """Generate masses distributed like the Kroupa Initial Mass Function.
    Spits out n masses between lower and upper bound in 'imf'.

    n: int or array of int
    imf: 1D or 2D array of float (in pairs of 2), optional

    out:
        array of float
    """
    n = np.atleast_1d(n).astype(int)
    if not imf:
        imf = np.full([len(n), len(default_imf_par)], default_imf_par)
    else:
        imf = np.atleast_2d(imf)
    M_low, M_high = imf[:, 0], imf[:, 1]
    M_mid = 0.5  # fixed turnover position (where slope changes)
    n_uniform = np.random.random_sample(n)
    # same constants as are in the IMF:
    c_mid = (1/1.35 - 1/0.35) * M_mid**(-0.35)
    c_low = (1/0.35 * M_low**(-0.35) + c_mid - M_mid/1.35 * M_high**(-1.35))**-1
    # the mid value in the CDF
    n_mid = c_low/0.35 * (M_low**(-0.35) - M_mid**(-0.35))
    # the inverted CDF (for masses below M_mid (a) and for masses above M_mid (b))
    M_a = (M_low**(-0.35) - 0.35 * n_uniform/c_low)**(-1/0.35)
    M_b = ((1.35/0.35 * M_low**(-0.35) - M_mid**(-0.35)/0.35 - 1.35 * n_uniform/c_low)/M_mid)**(-1/1.35)
    return (n_uniform < n_mid) * M_a + (n_uniform >= n_mid) * M_b


# below: general distributions (i.e. for Cartesian coordinates)
def uniform(n=1, min_val=0, max_val=1, power=1):
    """Uniformly distributed parameter between min and max.
    power=2 gives the cylindrical distribution, power=3 the spherical one.
    """
    c = (max_val**power - min_val**power)
    return (c*np.random.rand(int(n)) + min_val**power)**(1/power)


def normal(n=1, mean=0.0, s=1.0):
    """normal (gaussian) distribution with some mean and width (s=sigma). Draws n numbers."""
    return np.random.normal(mean, s, n)


def log_normal(n=1, mean=0.0, s=1.0):
    """Log-normal distribution with some mean and width (s=sigma). Draws n numbers."""
    #  base 10! (numpy gives 'e**np.random.normal')
    return 10.0**np.random.normal(mean, s, n)


def ln_normal(n=1, mean=0.0, s=1.0):
    """Natural-log-normal distribution with some mean and width (s=sigma). Draws n numbers."""
    # turns out 10.0**np.random.normal is a tad bit slower. 
    # Note: these are a different base! (numpy gives e**np.random.normal)
    return np.random.lognormal(mean, s, n)


def exponential(n=1, s=1.0):
    """exponential distribution with scale height s. Draws n numbers."""
    # self made function -s*np.log(1-np.random.rand(int(n))) is a bit slower
    return np.random.exponential(s, n)


def double_exponential(n=1, s=1.0, origin=0.0):
    """Double sided exponential distribution with scale height s. 
    Draws n numbers around origin.
    """
    return np.random.laplace(origin, s, n)


def power_law(n=1, power=-2.0, min_val=1e-9, max_val=1):
    """Power law distribution with index power (<-1). Draws n numbers between min and max."""
    n_uniform = np.random.rand(int(n))
    p1 = power + 1
    c = max_val**p1 - min_val**p1
    return (n_uniform*c + min_val**p1)**(1/p1)


# below: distributions for spherical/cylindrical coordinates (marked '_r')
def angle_phi(n=1):
    """Uniformly chosen angle(s) between 0 and 2 pi."""
    return 2*np.pi*np.random.rand(int(n)) 


def angle_theta(n=1, min_val=0, max_val=np.pi):
    """Angle(s) between min and max chosen from a sine distribution.
    Default is between 0 and pi for a whole sphere.
    """
    n_uniform = np.random.rand(int(n))
    return np.arccos(np.cos(min_val) - n_uniform*(np.cos(min_val) - np.cos(max_val)))


def exponential_r(n=1, s=1.0):
    """Radial exponential distribution with scale height s. Draws n numbers.
    For both spherical and cylindrical distribution.
    """
    r_vals = np.logspace(-3, 4, 1000)  # short to medium tail (max radius 10**4!)
    N_vals_exp = cdf_exponential(r_vals, s)
    return np.interp(np.random.rand(int(n)), N_vals_exp, r_vals)


def normal_r(n=1, s=1.0, spher=True):
    """Radial normal (gaussian) distribution with scale height s. Draws n numbers.
    For either spherical or cylindrical distribution (keyword spher).
    """
    r_vals = np.logspace(-3, 4, 1000)  # quite short tail (max radius 10**4!)
    n_vals = cdf_normal(r_vals, s, spher=spher)
    return np.interp(np.random.rand(int(n)), n_vals, r_vals)


def squared_cauchy_r(n=1, s=1.0, spher=True):
    """Radial squared Cauchy distribution (Schuster with m=2) with scale height s. 
    Draws n numbers. For either spherical or cylindrical distribution (keyword spher).
    """
    r_vals = np.logspace(-3, 6, 1000)  # very long tail (max radius 10**6!)
    n_vals = cdf_squared_cauchy(r_vals, s, spher=spher)
    return np.interp(np.random.rand(int(n)), n_vals, r_vals)


def pearson_vii_r(n=1, s=1.0, spher=True):
    """Radial Pearson type VII distribution (Schuster with m=2.5) with scale height s. 
    Draws n numbers. For either spherical or cylindrical distribution (keyword spher).
    """
    r_vals = np.logspace(-3, 4, 1000)  # medium tail (max radius 10**4!)
    n_vals = cdf_pearson_vii(r_vals, s, spher=spher)
    return np.interp(np.random.rand(int(n)), n_vals, r_vals)


def king_globular_r(n=1, s=1.0, R=None, spher=True):
    """Radial King distribution for globular clusters with scale height s and outter radius R. 
    Draws n numbers. For either spherical or cylindrical distribution (keyword spher).
    """
    if (R is None):
        # typical globular cluster has R/s ~ 30
        R = 30*s
    r_vals = np.logspace(-2, np.log10(R), 1000)
    n_vals = cdf_king_globular(r_vals, s, R, spher=spher)
    return np.interp(np.random.rand(int(n)), n_vals, r_vals)


# below: pdf and cdf distributions for the distributions using interpolation
def pdf_exponential(r, s=1.0):
    """pdf of radial exponential distribution."""
    # same for spherical/cylindrical
    rs = r/s
    pdf = rs**2/(2*s)*np.exp(-rs)
    return pdf


def cdf_exponential(r, s=1.0):
    """cdf of radial exponential distribution."""
    # same for spherical/cylindrical
    rs = r/s
    cdf = 1 - (rs**2/2 + rs + 1)*np.exp(-rs)
    return cdf


def pdf_normal(r, s=1.0, spher=True):
    """pdf of radial normal distribution (spherical or cylindrical)."""
    if not spher:
        rs = r/s
        pdf = 2*rs/s*np.exp(-rs**2)
    else:
        rs2 = (r/s)**2
        pdf = 4/(np.sqrt(np.pi)*s)*rs2*np.exp(-rs2)
    return pdf


def cdf_normal(r, s=1.0, spher=True):
    """cdf of radial normal distribution (spherical or cylindrical)."""
    if not spher:
        rs2 = (r/s)**2
        cdf = 1 - np.exp(-rs2)
    else:
        rs = r/s
        cdf = sps.erf(rs) - 2/np.sqrt(np.pi)*rs*np.exp(-rs**2)
    return cdf


def pdf_gamma3(r, s=1.0, spher=True):
    """pdf of radial gamma=3 distribution (spherical or cylindrical)."""
    if not spher:
        rs = r/s
        pdf = rs/s*(1 + rs**2)**(-3/2)
    else:
        rs2 = (r/s)**2
        pdf = 4/(np.pi*s)*rs2*(1 + rs2)**(-2)
    return pdf


def cdf_gamma3(r, s=1.0, spher=True):
    """cdf of radial gamma=3 distribution (spherical or cylindrical)."""
    if not spher:
        rs2 = (r/s)**2
        cdf = 1 - (1 + rs2)**(-1/2)
    else:
        rs = r/s
        cdf = 2/np.pi*(np.arctan(rs) - rs/(1 + rs**2))
    return cdf


def pdf_squared_cauchy(r, s=1.0, spher=True):
    """pdf of radial squared Cauchy distribution (spherical or cylindrical)."""
    if not spher:
        # the actual observational profile
        rs = r/s
        pdf = 2/s*rs*(1 + rs**2)**(-2)
    else:
        # the spherical equivalent
        rs2 = (r/s)**2
        pdf = 3/s*rs2*(1 + rs2)**(-5/2)
    return pdf


def cdf_squared_cauchy(r, s=1.0, spher=True):
    """cdf of radial squared Cauchy distribution (spherical or cylindrical)."""
    if not spher:
        rs2 = (r/s)**2
        cdf = rs2*(1 + rs2)**(-1)
    else:
        rs = r/s
        cdf = rs**3*(1 + rs**2)**(-3/2)
    return cdf


def pdf_pearson_vii(r, s=1.0, spher=True):
    """pdf of radial Pearson type VII distribution (spherical or cylindrical)."""
    if not spher:
        rs = r/s
        pdf = 3/s*rs*(1 + rs**2)**(-5/2)
    else:
        rs2 = (r/s)**2
        pdf = 16/(np.pi*s)*rs2*(1 + rs2)**(-3)
    return pdf


def cdf_pearson_vii(r, s=1.0, spher=True):
    """cdf of radial Pearson type VII distribution."""
    if not spher:
        rs2 = (r/s)**2
        cdf = 1 - (1 + rs2)**(-3/2)
    else:
        rs = r/s
        rs2 = (r/s)**2
        cdf = 2/np.pi*(rs*((1 + rs2)**(-1) - 2*(1 + rs2)**(-2)) + np.arctan(rs))
    return cdf


def pdf_king_globular(r, s=1.0, R=None, spher=True):
    """pdf of radial King distribution for Globular clusters (spherical or cylindrical)."""
    if (R is None):
        # typical globular cluster has R/s ~ 30
        R = 30*s
    
    rs2 = (r/s)**2
    Rs2 = (R/s)**2
    c2 = (1 + Rs2)**(-1/2)
    
    if not spher:
        c = (np.log(1 + Rs2)/2 + 2*c2 - 2 + Rs2/2*c2**2)**(-1)
        pdf = c/s*r/s*(1/(1 + rs2)**(1/2) - c2)**2
    else:
        Rs = R/s
        c3 = (4 - np.pi)/(2*np.pi)*c2**3
        c = (np.arcsinh(Rs)/2 + 2*c2/np.pi*np.arctan(Rs) - (1/2 + 2/np.pi)*c2*Rs + c3/3*Rs**3)**(-1)
        pdf = c/s*rs2*((1 + rs2)**(-3/2)/2 - 2*c2/np.pi*(1 + rs2)**(-1) + c3)
        
    return pdf


def cdf_king_globular(r, s=1.0, R=None, spher=True):
    """cdf of radial King distribution for Globular clusters (spherical or cylindrical)."""
    if (R is None):
        # typical globular cluster has R/s ~ 30
        R = 30*s
        
    # make sure r doesn't go above R (cdf is wrong there)
    r = np.clip(r, 0, R)
    
    rs2 = (r/s)**2
    Rs2 = (R/s)**2
    c2 = (1 + Rs2)**(-1/2)
    
    if not spher:
        c = (np.log(1 + Rs2)/2 + 2*c2 - 2 + Rs2/2*c2**2)**(-1)
        cdf = c*(np.log(1 + rs2)/2 + 2*c2*(1 - (1 + rs2)**(1/2)) + rs2/2*c2**2)
    else:
        rs = r/s
        Rs = R/s
        C3 = (4 - np.pi)/(2*np.pi)*c2**3
        c = (np.arcsinh(Rs)/2 + 2*c2/np.pi*np.arctan(Rs) - (1/2 + 2/np.pi)*c2*Rs + C3/3*Rs**3)**(-1)
        cdf = c*(np.arcsinh(rs)/2 - rs/2/(1 + rs2)**(1/2) + 2*c2/np.pi*np.arctan(rs) - 2*c2/np.pi*rs + C3/3*rs**3)
        
    return cdf




















    
    
    
    
    
    
    









