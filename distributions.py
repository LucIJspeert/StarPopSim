# Luc IJspeert
# Part of starpopsim: distribution functions
##
"""Distribution definitions to randomly draw numbers from. Optimized for drawing many numbers at once (array).
Distributions ending in _r are 3D radial (spherical) variants of the 1D (cartesian) versions.
Distributions ending in _rho are 2D radial (cylindrical) variants.
"""
import numpy as np
import scipy.special as sps


# global defaults
imf_defaults = [0.08, 150]      # lower bound, upper bound on mass


def pdf_KroupaIMF(M, imf=imf_defaults):
    """(Forward) Initial Mass Function (=probability density function), 
    normalized to 1 probability.
    Modified Salpeter IMF; actually Kroupa IMF above 0.08 solar mass.
    """
    M_L, M_U = imf
    # fixed turnover position (where slope changes)
    M_mid = 0.5
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = (1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))**-1
    C_U = C_L*M_mid
    return (M < M_mid)*C_L*M**(-1.35) + (M >= M_mid)*C_U*M**(-2.35)


def KroupaIMF(n=1, imf=imf_defaults):
    """Generate masses distributed like the Kroupa Initial Mass Function. 
    Spits out n masses between lower and upper bound in 'imf'.
    """
    M_L, M_U = imf
    # fixed turnover position (where slope changes)
    M_mid = 0.5
    N_dist = np.random.rand(int(n))
    # same constants as are in the IMF:
    C_mid = (1/1.35 - 1/0.35)*M_mid**(-0.35)
    C_L = (1/0.35*M_L**(-0.35) + C_mid - M_mid/1.35*M_U**(-1.35))**-1
    # the mid value in the CDF
    N_mid = C_L/0.35*(M_L**(-0.35) - M_mid**(-0.35))
    # the inverted CDF
    M_a = (M_L**(-0.35) - 0.35*N_dist/C_L)**(-1/0.35)
    M_b = ((1.35/0.35*M_L**(-0.35) - M_mid**(-0.35)/0.35 - 1.35*N_dist/C_L)/M_mid)**(-1/1.35)
    return (N_dist < N_mid)*M_a + (N_dist >= N_mid)*M_b

# below: general distributions (i.e. for Cartesian coordinates)

def Uniform(n=1, min=0, max=1, power=1):
    """Uniformly distributed parameter between min and max.
    power=2 gives the cylindrical distribution, power=3 the spherical one.
    """
    C = (max**power - min**power)
    return (C*np.random.rand(int(n)) + min**power)**(1/power)


def Normal(n=1, mean=0.0, s=1.0):
    """Normal (gaussian) distribution with some mean and width (s=sigma). Draws n numbers."""
    return np.random.normal(mean, s, n)


def LogNorm(n=1, mean=0.0, s=1.0):
    """Log-normal distribution with some mean and width (s=sigma). Draws n numbers."""
    #  base 10! (numpy gives 'e**np.random.normal')
    return 10.0**np.random.normal(mean, s, n)


def LnNorm(n=1, mean=0.0, s=1.0):
    """Natural-log-normal distribution with some mean and width (s=sigma). Draws n numbers."""
    # turns out 10.0**np.random.normal is a tad bit slower. 
    # Note: these are a different base! (numpy gives e**np.random.normal)
    return np.random.lognormal(mean, s, n)


def Exponential(n=1, s=1.0):
    """Exponential distribution with scale height s. Draws n numbers."""
    # self made function -s*np.log(1-np.random.rand(int(n))) is a bit slower
    return np.random.exponential(s, n)


def DoubleExp(n=1, s=1.0, origin=0.0):
    """Double sided exponential distribution with scale height s. 
    Draws n numbers around origin.
    """
    return np.random.laplace(origin, s, n)


def PowerLaw(n=1, power=-2.0, min=1e-9, max=1):
    """Power law distribution with index power (<-1). Draws n numbers between min and max."""
    N_dist = np.random.rand(int(n))
    p1 = power + 1
    C = max**p1 - min**p1
    return (N_dist*C + min**p1)**(1/p1)
    
# below: distributions for spherical/cylindrical coordinates (marked '_r')

def AnglePhi(n=1):
    """Uniformly chosen angle(s) between 0 and 2 pi."""
    return 2*np.pi*np.random.rand(int(n)) 


def AngleTheta(n=1):
    """Angle(s) between 0 and pi chosen from a sine distribution."""
    return np.arccos(2*np.random.rand(int(n)) - 1)


def Exponential_r(n=1, s=1.0):
    """Radial exponential distribution with scale height s. Draws n numbers.
    For both spherical and cylindrical distribution.
    """
    r_vals = np.logspace(-3, 4, 1000)                                                               # short to medium tail (max radius 10**4!)
    N_vals_exp = cdf_Exponential(r_vals, s)
    return np.interp(np.random.rand(int(n)), N_vals_exp, r_vals)


def Normal_r(n=1, s=1.0, spher=True):
    """Radial normal (gaussian) distribution with scale height s. Draws n numbers.
    For either spherical or cylindrical distribution (keyword spher).
    """
    r_vals = np.logspace(-3, 4, 1000)                                                               # quite short tail (max radius 10**4!)
    N_vals_norm = cdf_Normal(r_vals, s, spher=spher)
    return np.interp(np.random.rand(int(n)), N_vals_norm, r_vals)


def SquaredCauchy_r(n=1, s=1.0, spher=True):
    """Radial squared Cauchy distribution (Schuster with m=2) with scale height s. 
    Draws n numbers. For either spherical or cylindrical distribution (keyword spher).
    """
    r_vals = np.logspace(-3, 6, 1000)                                                               # very long tail (max radius 10**6!)
    N_vals = cdf_Squaredcauchy(r_vals, s, spher=spher)
    return np.interp(np.random.rand(int(n)), N_vals, r_vals)


def PearsonVII_r(n=1, s=1.0, spher=True):
    """Radial Pearson type VII distribution (Schuster with m=2.5) with scale height s. 
    Draws n numbers. For either spherical or cylindrical distribution (keyword spher).
    """
    r_vals = np.logspace(-3, 4, 1000)                                                               # medium tail (max radius 10**4!)
    N_vals = cdf_PearsonVII(r_vals, s, spher=spher)
    return np.interp(np.random.rand(int(n)), N_vals, r_vals)

def KingGlobular_r(n=1, s=1.0, R=None, spher=True):
    """Radial King distribution for globular clusters with scale height s and outter radius R. 
    Draws n numbers. For either spherical or cylindrical distribution (keyword spher).
    """
    if (R is None):
        R = 30*s                                                                                    # typical globular cluster has R/s ~ 30
    r_vals = np.logspace(-2, np.log10(R), 1000)
    N_vals = cdf_KingGlobular(r_vals, s, R, spher=spher)
    return np.interp(np.random.rand(int(n)), N_vals, r_vals)

# below: pdf and cdf distributions for the distributions using interpolation
    
def pdf_Exponential(r, s=1.0):
    """pdf of radial exponential distribution."""
    # same for spherical/cylindrical
    rs = r/s
    pdf = rs**2/(2*s)*np.exp(-rs)
    return pdf


def cdf_Exponential(r, s=1.0):
    """cdf of radial exponential distribution."""
    # same for spherical/cylindrical
    rs = r/s
    cdf = 1 - (rs**2/2 + rs + 1)*np.exp(-rs)
    return cdf


def pdf_Normal(r, s=1.0, spher=True):
    """pdf of radial normal distribution (spherical or cylindrical)."""
    if not spher:
        rs = r/s
        pdf = 2*rs/s*np.exp(-rs**2)
    else:
        rs2 = (r/s)**2
        pdf = 4/(np.sqrt(np.pi)*s)*rs2*np.exp(-rs2)
    return pdf


def cdf_Normal(r, s=1.0, spher=True):
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
        cdf = 1- (1 + rs2)**(-1/2)
    else:
        rs = r/s
        cdf = 2/np.pi*(np.arctan(rs) - rs/(1 + rs**2))
    return cdf


def pdf_SquaredCauchy(r, s=1.0, spher=True):
    """pdf of radial squared Cauchy distribution (spherical or cylindrical)."""
    if not spher:
        rs = r/s
        pdf = 2/s*rs*(1 + rs**2)**(-2)                                                              # the actual observational profile
    else:
        rs2 = (r/s)**2
        pdf = 3/s*rs2*(1 + rs2)**(-5/2)                                                             # the spherical equivalent
    return pdf


def cdf_Squaredcauchy(r, s=1.0, spher=True):
    """cdf of radial squared Cauchy distribution (spherical or cylindrical)."""
    if not spher:
        rs2 = (r/s)**2
        cdf = rs2*(1 + rs2)**(-1)
    else:
        rs = r/s
        cdf = rs**3*(1 + rs**2)**(-3/2)
    return cdf


def pdf_PearsonVII(r, s=1.0, spher=True):
    """pdf of radial Pearson type VII distribution (spherical or cylindrical)."""
    if not spher:
        rs = r/s
        pdf = 3/s*rs*(1 + rs**2)**(-5/2)
    else:
        rs2 = (r/s)**2
        pdf = 16/(np.pi*s)*rs2*(1 + rs2)**(-3)
    return pdf


def cdf_PearsonVII(r, s=1.0, spher=True):
    """cdf of radial Pearson type VII distribution."""
    if not spher:
        rs2 = (r/s)**2
        cdf = 1 - (1 + rs2)**(-3/2)
    else:
        rs = r/s
        rs2 = (r/s)**2
        cdf = 2/np.pi*(rs*((1 + rs2)**(-1) - 2*(1 + rs2)**(-2)) + np.arctan(rs))
    return cdf


def pdf_KingGlobular(r, s=1.0, R=None, spher=True):
    """pdf of radial King distribution for Globular clusters (spherical or cylindrical)."""
    if (R is None):
        # typical globular cluster has R/s ~ 30
        R = 30*s
    
    rs2 = (r/s)**2
    Rs2 = (R/s)**2
    C2 = (1 + Rs2)**(-1/2)
    
    if not spher:
        C = (np.log(1 + Rs2)/2 + 2*C2 - 2 + Rs2/2*C2**2)**(-1)
        pdf = C/s*r/s*(1/(1 + rs2)**(1/2) - C2)**2
    else:
        Rs = R/s
        C3 = (4 - np.pi)/(2*np.pi)*C2**3
        C = (np.arcsinh(Rs)/2 + 2*C2/np.pi*np.arctan(Rs) 
            - (1/2 + 2/np.pi)*C2*Rs + C3/3*Rs**3)**(-1)
            
        pdf = C/s*rs2*((1 + rs2)**(-3/2)/2 - 2*C2/np.pi*(1 + rs2)**(-1) + C3)
        
    return pdf


def cdf_KingGlobular(r, s=1.0, R=None, spher=True):
    """cdf of radial King distribution for Globular clusters (spherical or cylindrical)."""
    if (R is None):
        # typical globular cluster has R/s ~ 30
        R = 30*s
        
    # make sure r doesn't go above R (cdf is wrong there)
    r = np.clip(r, 0, R)
    
    rs2 = (r/s)**2
    Rs2 = (R/s)**2
    C2 = (1 + Rs2)**(-1/2)
    
    if not spher:
        C = (np.log(1 + Rs2)/2 + 2*C2 - 2 + Rs2/2*C2**2)**(-1)
        cdf = C*(np.log(1 + rs2)/2 + 2*C2*(1 - (1 + rs2)**(1/2)) + rs2/2*C2**2)  
    else:
        rs = r/s
        Rs = R/s
        C3 = (4 - np.pi)/(2*np.pi)*C2**3
        C = (np.arcsinh(Rs)/2 + 2*C2/np.pi*np.arctan(Rs) 
            - (1/2 + 2/np.pi)*C2*Rs + C3/3*Rs**3)**(-1)
            
        cdf = C*(np.arcsinh(rs)/2 - rs/2/(1 + rs2)**(1/2) + 2*C2/np.pi*np.arctan(rs) 
              - 2*C2/np.pi*rs + C3/3*rs**3)
        
    return cdf




















    
    
    
    
    
    
    









