"""This module defines the Astronomical Object class and subclasses that hold
all the information for the simulated astronomical object. 
It also contains all the checks performed on the user input.
It defines the functions that can be performed on the astronomical object.
As well as the functions to make/generate (parts of) the astronomical object,
like the properties of the individual objects (stars/WDs/NSs/BHs; from here on: stars).
Several options for the distribution of stars are available; 
personal profiles might be added to the distributions module.
"""
import os
import inspect
import fnmatch
import pickle
import warnings

import numpy as np
import scipy.spatial as sps
import scipy.interpolate as spi

import utils
import distributions as dist
import conversions as conv
import formulas as form
import visualizer as vis


# global constants
M_bol_sun = 4.74                # bolometric magnitude of the sun
as_rad = np.pi/648000           # arcseconds to radians


# global defaults
default_rdist = 'normal'        # see distributions module for a full list of options
default_imf_par = [0.08, 150]   # M_sun     lower bound, upper bound on mass
default_mag_lim = 32            # magnitude limit for compact mode
limiting_number = 10**7         # maximum number of stars for compact mode 


class Stars(object):
    """Generates populations of stars and contains all the information about them.
    Also functions that can be performed on the stars are defined here.
    
    Note:
        N_stars and M_tot_init cannot be zero simultaneously.
        ages and metal are both not optional parameters (they might look like they are).
        'array' here means numpy.ndarray, while lists (or even single entries) are also accepted
    # https://quantecon.org/wiki-py-docstrings/

    Arguments
    ---------

    n_stars (array of int): the total number of stars to make per stellar population.
    M_tot_init (array of float): the total mass in stars to produce per population (in Msol).
    ages (array of float): stellar age for each population (in linear or 10log yr).
        This is the maximum age when sfh is used.
    metal (array of float): metallicity for each population.
    sfh (array of str and None, optional): type of star formation history to use per population.
    min_ages (array of float): minimum ages to use in star formation histories, per population
    tau_sfh (array of float): characteristic timescales for star formation per population
    imf_par (array of float, optional): 2D array of two parameters per stellar population.
        First parameter is lowest mass, second is highest mass to make.
    r_dist (array of str): type of radial distribution to use per population.
    r_dist_par (array of dict): parameter values for the radial distributions specified
        (one dictionary per population).
    incl (array of float): inclination values per population (in radians)
    ellipse_axes (array of float): 2D array of one set of axes per population.
        Relative (x,y,z)-sizes to stretch the distributions; volume is kept constant.
    spiral_arms (array of int): number of spiral arms (per population).
    spiral_bulge (array of float): relative proportion of the central bulge (per population).
    spiral_bar (array of float): relative proportion of the central bar (per population).
    compact_mode (array of str and None): generate only a fraction of the total number of stars.
        choose 'num' or 'mag' for number or magnitude limited.

    Attributes
    ----------

    All of the arguments described above are stored as attributes.
    origin (array of float): origin of the stellar distribution (for each population). Shape is [3, n_pop]
    coords (array of float): 2D array of the cartesian coordinates of the stars. Shape is [3, n_stars].
    M_init (array of float): the individual masses of the stars in Msol.
    M_diff (array of float): difference between given and generated mass per population (only when n_stars=0).
    mag_names (array of str): the filter names of the corresponding default set of supported magnitudes.
    spec_names (array of str): spectral type names corresponding to the reference numbers in spectral_types.
    fraction_generated (array of float): part of the total number of stars that has actually been generated
        per population (when compact mode is used).
    
    Returns
    Stars object

    """    
    def __init__(self, n_stars=0, M_tot_init=0, ages=None, metal=None, imf_par=None, sfh=None,
                 min_ages=None, tau_sfh=None, origin=None, incl=None, r_dist=None, r_dist_par=None,
                 ellipse_axes=None, spiral_arms=None, spiral_bulge=None, spiral_bar=None,
                 compact_mode=None):

        # cast input to right formats, and perform some checks. first find the intended number of populations
        n_pop = utils.check_number_of_populations(n_stars, M_tot_init, ages, metal)
        self.ages = utils.cast_ages(ages, n_pop)
        self.metal = utils.cast_metallicities(metal, n_pop)
        self.imf_param = utils.cast_imf_parameters(imf_par, n_pop, fill_value=default_imf_par)
        self.imf_param = utils.check_lowest_imf_mass(self.imf_param, self.ages, self.metal)
        self.M_tot_init = utils.cast_m_total(M_tot_init, n_pop)
        self.n_stars = utils.check_and_cast_n_stars(n_stars, self.M_tot_init, n_pop, self.imf_param)
        self.sfhist = utils.cast_sfhistory(sfh, n_pop)
        self.min_ages = min_ages
        self.tau_sfh = tau_sfh

        # parameters defining the shape
        self.origin = utils.cast_translation(origin, n_pop).T
        self.inclination = utils.cast_inclination(incl, n_pop)
        self.r_dist_types = utils.cast_radial_dist_type(r_dist, n_pop)
        self.r_dist_types = utils.check_radial_dist_type(self.r_dist_types)
        self.r_dist_param = utils.cast_radial_dist_param(r_dist_par, self.r_dist_types, n_pop)
        self.ellipse_axes = utils.cast_ellipse_axes(ellipse_axes, n_pop)
        self.spiral_arms = spiral_arms
        self.spiral_bulge = spiral_bulge
        self.spiral_bar = spiral_bar
        
        # properties that are derived/generated
        self.coords = np.empty([3, 0])
        self.M_init = np.array([])
        self.M_diff = np.zeros(n_pop)
        self.mag_names = utils.get_supported_filters()
        self.spec_names = np.array([])
        
        # compact mode parameters
        self.compact_mode = compact_mode
        self.fraction_generated = np.ones(n_pop)
        
        # semi-private
        # the actual imf parameters to use for each population (limits imposed by compacting)
        self.gen_imf_param = self.imf_param
        # actually generated number of stars per population (for compact mode/sfhist)
        self.gen_n_stars = self.n_stars
        # the actual ages to be generated for each population (for sfhist)
        self.gen_ages = self.ages

        # actually generate the stars
        self.GenerateStars()
        return
    
    def __repr__(self):
        """Unambiguous representation of what this object is."""
        repr = (f'Stars(N_stars={self.n_stars!r}, M_tot_init={self.M_tot_init!r}, '
                f'ages={self.ages!r}, metal={self.metal!r}, sfh={self.sfhist!r}, '
                f'min_ages={self.min_ages!r}, tau_sfh={self.tau_sfh!r}, '
                f'imf_par={self.imf_param!r}, incl={self.inclination!r}, '
                f'r_dist={self.r_dist_types!r}, r_dist_par={self.r_dist_param!r}, '
                f'ellipse_axes={self.ellipse_axes!r}, spiral_arms={self.spiral_arms!r}, '
                f'spiral_bulge={self.spiral_bulge!r}, spiral_bar={self.spiral_bar!r}, '
                f'compact_mode={self.compact_mode!r})')
        return repr
    
    def __str__(self):
        """Nice-to-read representation of what this object is"""
        string = ('Stars object with parameters:\n'
                  f'N_stars:        {self.n_stars!s}\n'
                  f'M_tot_init:     {np.round(self.M_tot_init, 1)!s}\n'
                  f'ages:           {self.ages!s}\n'
                  f'metal:          {self.metal!s}\n'
                  f'sfh:            {self.sfhist!s}\n'
                  f'min_ages:       {self.min_ages!s}\n'
                  f'tau_sfh:        {self.tau_sfh!s}\n'
                  f'imf_par:        {[list(item) for item in self.imf_param]!s}\n'
                  f'incl:           {self.inclination!s}\n'
                  f'r_dist:         {self.r_dist_types!s}\n'
                  f'r_dist_par:     {self.r_dist_param!s}\n'
                  f'ellipse_axes:   {[list(item) for item in self.ellipse_axes]!s}\n'
                  f'spiral_arms:    {self.spiral_arms!s}\n'
                  f'spiral_bulge:   {self.spiral_bulge!s}\n'
                  f'spiral_bar:     {self.spiral_bar!s}\n'
                  f'compact_mode:   {self.compact_mode!s}\n')
        return string
        
    def __add__(self, other):
        """Appends two Stars objects to each other."""
        # Need to append all the properties.
        self.n_stars = np.append(self.n_stars, other.n_stars)
        self.M_tot_init = np.append(self.M_tot_init, other.M_tot_init)
        self.ages = np.append(self.ages, other.ages)
        self.metal = np.append(self.metal, other.metal)
        self.rel_number = np.append(self.rel_number, other.rel_number)
        self.sfhist = np.append(self.sfhist, other.sfhist)
        self.min_ages = np.append(self.min_ages, other.min_ages)
        self.tau_sfh = np.append(self.tau_sfh, other.tau_sfh)
        self.imf_param = np.append(self.imf_param, other.imf_param, axis=0)
        self.origin = np.append(self.origin, other.origin, axis=0)
        self.inclination = np.append(self.inclination, other.inclination)
        self.r_dist_types = np.append(self.r_dist_types, other.r_dist_types)
        self.r_dist_param = np.append(self.r_dist_param, other.r_dist_param)
        self.ellipse_axes = np.append(self.ellipse_axes, other.ellipse_axes, axis=0)
        self.spiral_arms = np.append(self.spiral_arms, other.spiral_arms)
        self.spiral_bulge = np.append(self.spiral_bulge, other.spiral_bulge)
        self.spiral_bar = np.append(self.spiral_bar, other.spiral_bar)
        self.compact_mode = np.append(self.compact_mode, other.compact_mode)
        # non-user defined:
        self.coords = np.append(self.coords, other.coords, axis=0)
        self.M_init = np.append(self.M_init, other.M_init)
        self.M_diff = np.append(self.M_diff, other.M_diff)
        self.fraction_generated = np.append(self.fraction_generated, other.fraction_generated)
        self.gen_imf_param = np.append(self.gen_imf_param, other.gen_imf_param, axis=0)
        self.gen_n_stars = np.append(self.gen_n_stars, other.gen_n_stars)
        self.gen_ages = np.append(self.gen_ages, other.gen_ages)
        # check if both have performance mode attributes (if not, deletes attributes from self)
        if (hasattr(self, 'current_masses') & hasattr(other, 'current_masses')):
            self.current_masses = np.append(self.current_masses, other.current_masses)
        elif hasattr(self, 'current_masses'):
            del self.current_masses
            
        if (hasattr(self, 'stellar_radii') & hasattr(other, 'stellar_radii')):
            self.stellar_radii = np.append(self.stellar_radii, other.stellar_radii)
        elif hasattr(self, 'stellar_radii'):
            del self.stellar_radii
        
        if (hasattr(self, 'log_luminosities') & hasattr(other, 'log_luminosities')):
            self.log_luminosities = np.append(self.log_luminosities, other.log_luminosities)
        elif hasattr(self, 'log_luminosities'):
            del self.log_luminosities
        
        if (hasattr(self, 'log_temperatures') & hasattr(other, 'log_temperatures')):
            self.log_temperatures = np.append(self.log_temperatures, other.log_temperatures)
        elif hasattr(self, 'log_temperatures'):
            del self.log_temperatures
        
        if (hasattr(self, 'absolute_magnitudes') & hasattr(other, 'absolute_magnitudes') &
            np.all(self.mag_names == other.mag_names)):
            self.absolute_magnitudes = np.append(self.absolute_magnitudes, 
                                                 other.absolute_magnitudes)
        elif hasattr(self, 'absolute_magnitudes'):
            del self.absolute_magnitudes
        
        if (hasattr(self, 'apparent_magniutdes') & hasattr(other, 'apparent_magniutdes') &
            np.all(self.mag_names == other.mag_names)):
            self.apparent_magniutdes = np.append(self.apparent_magniutdes, 
                                                 other.apparent_magniutdes)
        elif hasattr(self, 'apparent_magniutdes'):
            del self.apparent_magniutdes
        self.mag_names = np.unique(np.append(self.mag_names, other.mag_names))
        
        if (hasattr(self, 'spectral_types') & hasattr(other, 'spectral_types') &
            np.all(self.spec_names == other.spec_names)):
            self.spectral_types = np.append(self.spectral_types, other.spectral_types)
        elif hasattr(self, 'spectral_types'):
            del self.spectral_types
        self.spec_names = np.unique(np.append(self.spec_names, other.spec_names))
        
        return self
        
    # def __radd__(self, other):
    #     """Reverse add (for when adding doesn't work. e.g. in sum(a,b,c))."""
    #     if other == 0:
    #         return self
    #     else:
    #         return self.__add__(other)
    
    def GenerateStars(self):
        """Generate the masses and positions of the stars."""
        # assign the right values for generation (start of generating sequence)
        if self.compact_mode:
            # todo: fix this, needs functions instead of the not-yet-declared vars (and more)
            self.gen_imf_param, self.fraction_generated = compactify(self.compact_mode, self.n_stars, self.mag_limit,
                                                                     imf=self.imf_param)
            self.gen_n_stars = np.rint(self.n_stars*self.fraction_generated).astype(int)
            '''
            # check if compact mode is on
            self.fraction_generated = np.ones_like(self.n_stars, dtype=float)                        # set to ones initially
            mass_limit = np.copy(self.imf_param[:])                                                     # set to imf params initially
            
            if self.compact_mode:
                if (self.mag_limit is None):
                    self.mag_limit = default_mag_lim                                                    # if not specified, use default
                
                for i, pop_num in enumerate(self.n_stars):
                    if (self.compact_mode == 'mag'):
                        mass_limit[i] = magnitude_limited(self.ages[i], self.metal[i], 
                                                            mag_lim=self.mag_limit, distance=self.d_lum, 
                                                            ext=self.extinction, filter='Ks')
                    else:
                        mass_limit[i] = number_limited(self.n_stars, self.ages[i], 
                                                        self.metal[i], imf=self.imf_param[i])
                    
                    if (mass_limit[i, 1] > self.imf_param[i, 1]):
                        mass_limit[i, 1] = self.imf_param[i, 1]                                         # don't increase the upper limit!
                    
                    self.fraction_generated[i] = form.mass_fraction_from_limits(mass_limit[i], 
                                                                    imf=self.imf_param[i])
                if np.any(mass_limit[:, 0] > mass_limit[:, 1]):                                       
                    raise RuntimeError('compacting failed, '
                                        'mass limit raised above upper mass.')                              # lower limit > upper limit!
                elif np.any(self.n_stars*self.fraction_generated < 10):                                  # don't want too few stars
                    raise RuntimeError('Population compacted to <10, '
                                        'try not compacting or generating a higher number of stars.')
            '''
        if not np.all(self.sfhist == None):
            rel_num, ages = star_form_history(self.gen_ages, min_age=self.min_ages, sfr=self.sfhist,
                                              Z=self.metal, tau=self.tau_sfh)
            self.gen_n_stars = np.rint(rel_num*self.n_stars).astype(int)
            self.gen_ages = ages
        
        # generate the positions, masses   
        for i, pop_num in enumerate(self.gen_n_stars):
            coords_i = gen_spherical(pop_num, dist_type=self.r_dist_types[i], **self.r_dist_param[i])
            M_init_i, M_diff_i = gen_star_masses(pop_num, 0, imf=self.gen_imf_param[i])
            self.coords = np.append(self.coords, coords_i, axis=0)                           
            self.M_init = np.append(self.M_init, M_init_i)
            self.M_diff += M_diff_i                                                                 # gen_star_masses already gives diff in Mass (=estimate since no mass was given)
        
        # if only N_stars was given, set M_tot_init to the total generated mass
        mass_generated = np.sum(self.M_init)
        if (self.M_tot_init == 0):
            self.M_tot_init = mass_generated
        else:
            self.M_diff = mass_generated - self.M_tot_init                                          # if negative: too little mass generated, else too much
            self.M_tot_init = mass_generated                                                        # set to actual initial mass
        return
        
    def AddInclination(self, incl):
        """Put the object at an inclination w.r.t. the observer, in radians. 
        The given angle is measured from the x-axis towards the z-axis (which is the l.o.s.).
        Can be specified for each population separately or for all at once.
        This is additive (given angles stack with previous ones).
        """
        # check the input, make sure it is an array of the right size
        n_pop = len(self.n_stars)
        incl = utils.cast_inclination(incl, n_pop)
        
        # check for existing inclination and record the new one
        if hasattr(self, 'inclination'):
            self.inclination += incl
        else:
            self.inclination = incl
            
        if np.any(self.inclination > 2*np.pi):
            if np.any(incl > 2*np.pi):
                warnings.warn('objectgenerator//AddInclination: inclination angle over 2pi '
                              'detected, make sure to use radians!')
            self.inclination = np.mod(self.inclination, 2*np.pi)
        
        # rotate the XZ plane (x axis towards positive z)
        if not np.all(self.inclination == 0):
            indices = np.repeat(np.arange(n_pop), self.n_stars)                                  # population index per star
            self.coords = conv.rotate_xz(self.coords, self.inclination[indices])
        return
        
    def AddEllipticity(self, axes):
        """Give the object axes (cartesian) different relative proportions. 
        The given axes (a,b,c) are taken as relative sizes, the volume of the object is conserved.
        Can be specified for each population separately or for all at once.
        This is additive (given axis sizes stack with previous ones).
        """
        # check the input, make sure it is an array of the right size
        n_pop = len(self.n_stars)
        axes = utils.cast_ellipse_axes(axes, n_pop)
        axes_shape = np.shape(axes)
        
        # check for existing ellipse axes and record the new ones
        if hasattr(self, 'ellipse_axes'):
            self.ellipse_axes *= axes
        else:
            self.ellipse_axes = axes
        
        # if (len(np.unique(self.ellipse_axes)) == 1):
        #     return                                               # no relative change is made
        # elif (axes_shape[0] == 1):
        #     self.coords = self.coords*axes/np.prod(axes)**(1/3)  # convert to ellipsoid (keeps volume conserved)
        # else:
        #     index = np.cumsum(np.append([0], self.gen_n_stars))  # indices defining the different populations
        #     for i, axes_i in enumerate(self.ellipse_axes):
        #         ind_1, ind_2 = index[i], index[i+1]
        #         self.coords[ind_1:ind_2] = self.coords[ind_1:ind_2]*axes_i/np.prod(axes_i)**(1/3)
        
        if not (len(np.unique(axes)) == 1):
            # population index per star
            indices = np.repeat(np.arange(n_pop), self.n_stars)
            # convert to ellipsoid (keeps volume conserved)
            self.coords = self.coords * (axes / np.prod(axes)**(1 / 3))[indices]
        return
        
    def AddTranslation(self, translation):
        """Translate the origin of the stellar distribution.
        Can be specified for each population separately or for all at once.
        """
        # check the input, make sure it is an array of the right size
        n_pop = len(self.n_stars)
        translation = utils.cast_translation(translation, n_pop)
        # record the new translation
        self.origin += translation.T
        # population index per star
        indices = np.repeat(np.arange(n_pop), self.n_stars)
        # move the stars (assuming shape (n_pop, 3) --> cast to (3, #stars))
        self.coords = self.coords + translation[indices].T
        return
    
    def CurrentMasses(self, realistic_remnants=True):
        """Gives the current masses of the stars in Msun.
        Uses isochrone files and the given initial masses of the stars.
        Stars should not have a lower initial mass than the lowest mass in the isochrone file.
        """
        if hasattr(self, 'current_masses'):
            M_cur = self.current_masses
        else:
            M_cur = np.array([])
            index = np.cumsum(np.append([0], self.gen_n_stars))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_M_act = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial', 'M_current'])   # get the isochrone values
                M_init_i = self.M_init[index[i]:index[i+1]]
                M_cur_i = np.interp(M_init_i, iso_M_ini, iso_M_act, right=0)                        # (right) return 0 for stars heavier than available in isoc file (dead stars)
                
                # give estimates for remnant masses (replacing the 0 above)
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])               # approx. remnant masses (depend on Z)
                    M_cur_i[remnants_i] = r_M_cur_i                                                 # fill in the values
                
                M_cur = np.append(M_cur, M_cur_i)  
        
        return M_cur
        
    def StellarRadii(self, realistic_remnants=True):
        """Gives the stellar radii of the stars in Rsun.
        Uses isochrone files and the given initial masses of the stars.
        Stars should not have a lower initial mass than the lowest mass in the isochrone file.
        """
        if hasattr(self, 'stellar_radii'):
            R_cur = self.stellar_radii
        else:
            R_cur = np.array([])
            index = np.cumsum(np.append([0], self.gen_n_stars))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_log_g = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial', 'log_g'])       # get the isochrone values
                iso_R_cur = conv.gravity_to_radius(iso_log_g, iso_M_ini)
                M_init_i = self.M_init[index[i]:index[i+1]]
                R_cur_i = np.interp(M_init_i, iso_M_ini, iso_R_cur, right=0)                        # (right) return 0 for stars heavier than available in isoc file (dead stars)
                
                # give estimates for remnant radii (replacing the 0 above)
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])               # approx. remnant masses (depend on Z)
                    r_R_cur_i = form.remnant_radius(r_M_cur_i)                                       # approx. remnant radii
                    R_cur_i[remnants_i] = r_R_cur_i                                                 # fill in the values
                
                R_cur = np.append(R_cur, R_cur_i)  
        
        return R_cur
        
    def LogLuminosities(self, realistic_remnants=True):
        """Gives the logarithm of the luminosity of the stars in Lsun.
        Uses isochrone files and the given initial masses of the stars.
        realistic_remnants gives estimates for remnant luminosities. Set False to save time.
        Stars should not have a lower initial mass than the lowest mass in the isochrone file.
        """
        if hasattr(self, 'log_luminosities'):
            log_L = self.log_luminosities
        else:
            log_L = np.array([])
            index = np.cumsum(np.append([0], self.gen_n_stars))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_log_L = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial', 'log_L'])       # get the isochrone values
                M_init_i = self.M_init[index[i]:index[i+1]]
                log_L_i = np.interp(M_init_i, iso_M_ini, iso_log_L, right=-9)                       # (right) return -9 --> L = 10**-9 Lsun (for stars heavier than available)
                
                # give estimates for remnant luminosities (replacing the -9 above)
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    remnant_time = form.remnant_time(M_init_i[remnants_i], age, self.metal[i])       # approx. time that the remnant had to cool
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])               # approx. remnant masses
                    r_R_cur_i = form.remnant_radius(r_M_cur_i)                                       # approx. remnant radii
                    r_Te_i = form.remnant_temperature(r_M_cur_i, r_R_cur_i, remnant_time)                   # approx. remnant temperatures
                    r_log_L_i = np.log10(form.bb_luminosity(r_Te_i, r_R_cur_i))                      # remnant luminosity := BB radiation
                    log_L_i[remnants_i] = r_log_L_i                                                 # fill in the values
                
                log_L = np.append(log_L, log_L_i)  
            
        return log_L
        
    def LogTemperatures(self, realistic_remnants=True):
        """Gives the logarithm of the effective temperature of the stars in K.
        Uses isochrone files and the given initial masses of the stars.
        realistic_remnants gives estimates for remnant temperatures. Set False to save time.
        Stars should not have a lower initial mass than the lowest mass in the isochrone file.
        """
        if hasattr(self, 'log_temperatures'):
            log_Te = self.log_temperatures
        else:
            log_Te = np.array([])
            index = np.cumsum(np.append([0], self.gen_n_stars))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_log_Te = utils.stellar_isochrone(age, self.metal[i],
                                                                columns=['M_initial', 'log_Te'])     # get the isochrone values
                M_init_i = self.M_init[index[i]:index[i+1]]
                log_Te_i = np.interp(M_init_i, iso_M_ini, iso_log_Te, right=1)                      # (right) return 1 --> Te = 10 K (for stars heavier than available)
                
                # give estimates for remnant temperatures (replacing the 1 above)
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    remnant_time = form.remnant_time(M_init_i[remnants_i], age, self.metal[i])       # approx. time that the remnant had to cool
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])               # approx. remnant masses
                    r_R_cur_i = form.remnant_radius(r_M_cur_i)                                       # approx. remnant radii
                    r_log_Te_i = np.log10(form.remnant_temperature(r_M_cur_i, r_R_cur_i, remnant_time))     # approx. remnant temperatures
                    log_Te_i[remnants_i] = r_log_Te_i                                               # fill in the values
                
                log_Te = np.append(log_Te, log_Te_i)  
        
        return log_Te
        
    def AbsoluteMagnitudes(self, filters=None, realistic_remnants=True):
        """Gives the absolute magnitudes of the stars using isochrone files and the initial masses.
        A list of filters can be specified; None will result in all available magnitudes.
        Realistic remnants can be emulated by using black body spectra.
        Stars should not have a lower initial mass than the lowest mass in the isochrone file.
        """
        if filters is None:
            filters = self.mag_names
        
        if hasattr(self, 'absolute_magnitudes'):
            abs_mag = self.absolute_magnitudes[:, utils.get_filter_mask(filters)]
            if (len(filters) == 1):
                abs_mag = abs_mag.flatten()                                                         # correct for 2D array
        else:
            abs_mag = np.empty((0,) + (len(filters),)*(len(filters) != 1))
            index = np.cumsum(np.append([0], self.gen_n_stars))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial'])
                iso_mag = utils.stellar_isochrone(age, self.metal[i], columns=filters).T             # get the isochrone values
                
                M_init_i = self.M_init[index[i]:index[i+1]]                                         # select the masses of one population
                interper = spi.interp1d(iso_M_ini, iso_mag, bounds_error=False, 
                                        fill_value=30, axis=0)                                      # (fill) return 30 --> L of less than 10**-9 (for stars heavier than available)
                mag_i = interper(M_init_i)
                
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    remnant_time = form.remnant_time(M_init_i[remnants_i], age, self.metal[i])       # approx. time that the remnant had to cool
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])               # approx. remnant masses
                    r_R_cur_i = form.remnant_radius(r_M_cur_i)                                       # approx. remnant radii
                    r_Te_i = form.remnant_temperature(r_M_cur_i, r_R_cur_i, remnant_time)                   # approx. remnant temperatures
                    mag_i[remnants_i] = form.bb_magnitude(r_Te_i, r_R_cur_i, filters)                # approx. remnant magnitudes
                
                abs_mag = np.append(abs_mag, mag_i, axis=0)
        
        return abs_mag
    
    def ApparentMagnitudes(self, distance, filters=None, extinction=0, add_redshift=False, redshift=None):
        """Computes the apparent magnitude from the absolute magnitude and the individual distances 
        (in pc). Needs the luminosity distance to the astronomical object (in pc).
        A list of filters can be specified; None will result in all available magnitudes.
        Redshift can be roughly included (uses black body spectra).
        """
        if filters is None:
            filters = self.mag_names
        
        if hasattr(self, 'apparent_magnitudes'):
            app_mag = self.apparent_magnitudes[:, utils.get_filter_mask(filters)]
            if (len(filters) == 1):
                app_mag = app_mag.flatten()                                                         # correct for 2D array
        else:
            if (distance > 100*np.abs(np.min(self.coords[:,2]))):                                   # distance 'much larger' than individual variations
                true_dist = distance - self.coords[:,2]                                             # approximate the individual distance_3d to each star with the z-coordinate
            else:
                true_dist = form.distance_3d(self.coords, np.array([0, 0, distance]))                  # distance is now properly calculated for each star
            
            true_dist = true_dist.reshape((len(true_dist),) + (1,)*(len(filters) > 1))              # fix dimension for broadcast
            
            # add redshift (rough approach)
            if add_redshift:
                filter_means = utils.open_photometric_data(columns=['mean'], filters=filters)
                shifted_filters = (1 + redshift)*filter_means
                R_cur = self.StellarRadii(realistic_remnants=True)
                T_eff = 10**self.LogTemperatures(realistic_remnants=True)
                abs_mag = form.bb_magnitude(T_eff, R_cur, filters, filter_means=shifted_filters)
            else:
                abs_mag = self.AbsoluteMagnitudes(filters=filters)
            
            app_mag = form.apparent_magnitude(abs_mag, true_dist, ext=extinction)                     # true_dist in pc!
        
        return app_mag
        
    def SpectralTypes(self, realistic_remnants=True):
        """Gives the spectral types (as indices, to conserve memory) for the stars and
        the corresponding spectral type names.
        Uses isochrone files and the given initial masses of the stars.
        Stars should not have a lower initial mass than the lowest mass in the isochrone file.
        """
        if hasattr(self, 'spectral_types'):
            spec_indices = self.spectral_types
            spec_names = self.spec_names
        else:
            log_T_eff = self.LogTemperatures(realistic_remnants=realistic_remnants)
            log_L = self.LogLuminosities(realistic_remnants=realistic_remnants)
            log_M_cur = np.log10(self.CurrentMasses(realistic_remnants=realistic_remnants))
            
            spec_indices, spec_names = find_spectral_type(log_T_eff, log_L, log_M_cur)                # assign spectra to the stars
            
            # if this is run for the first time, save the spectral names
            if len(self.spec_names == 0):
                self.spec_names = spec_names
        
        return spec_indices, spec_names
    
    def Remnants(self):
        """Gives the indices of the positions of remnants (not white dwarfs) as a boolean array. 
        (WD should be handled with the right isochrone files, but NS/BHs are not)
        """
        if hasattr(self, 'remnants'):
            remnants = self.remnants
        else:
            remnants = np.array([], dtype=bool)
            index = np.cumsum(np.append([0], self.gen_n_stars))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                iso_M_ini = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial'])
                max_mass = np.max(iso_M_ini)                                                        # maximum initial mass in isoc file
                remnants_i = (self.M_init[index[i]:index[i+1]] > max_mass)
                remnants = np.append(remnants, remnants_i)
        
        return remnants
    
    def TotalCurrentMass(self):
        """Returns the total current mass in stars (in Msun)."""
        return np.sum(self.CurrentMasses())
    
    def TotalLuminosity(self):
        """Returns log of the total luminosity from stars (in Lsun)."""
        return np.log10(np.sum(10**self.LogLuminosities(realistic_remnants=False)))                 # remnants don't add much, so leave them as is
        
    def CoordsArcsec(self, distance):
        """Returns coordinates converted to arcseconds (from pc). Needs the angular distance
        to the object (in pc).
        """
        return conv.parsec_to_arcsec(self.coords, distance)
        
    def OrbitalRadii(self, distance, unit='pc', spher=False):
        """Returns the radial coordinate of the stars (spherical or cylindrical) in pc/as."""
        if spher:
            radii = form.distance_3d(self.coords)
        else:
            radii = form.distance_2d(self.coords)
        
        if (unit == 'as'):                                                                          # convert to arcsec if wanted
            radii = conv.parsec_to_arcsec(radii, distance)
        
        return radii
    
    def OrbitalVelocities(self):
        """"""
        # todo: make this
    
    def HalfMassRadius(self, spher=False):
        """Returns the (spherical or cylindrical) half mass radius in pc/as."""
        M_cur = self.CurrentMasses()
        tot_mass = np.sum(M_cur)                                                                    # do this locally, to avoid unnecesairy overhead
        
        if spher:
            r_star = form.distance_3d(self.coords)                                                     # spherical radii of the stars
        else:
            r_star = form.distance_2d(self.coords)                                                   # cylindrical radii of the stars
            
        indices = np.argsort(r_star)                                                                # indices that sort the radii
        r_sorted = r_star[indices]                                                                  # sorted radii
        mass_sorted = M_cur[indices]                                                                # masses sorted for radius
        mass_sum = np.cumsum(mass_sorted)                                                           # cumulative sum of sorted masses
        hmr = np.max(r_sorted[mass_sum <= tot_mass/2])                                              # 2D/3D radius at half the mass
        return hmr
        
    def HalfLumRadius(self, spher=False):
        """Returns the (spherical or cylindrical) half luminosity radius in pc/as."""
        lum = 10**self.LogLuminosities(realistic_remnants=False)
        
        tot_lum = np.sum(lum)                                                                       # do this locally, to avoid unnecesairy overhead
        
        if spher:
            r_star = form.distance_3d(self.coords)                                                     # spherical radii of the stars
        else:
            r_star = form.distance_2d(self.coords)                                                   # cylindrical radii of the stars
            
        indices = np.argsort(r_star)                                                                # indices that sort the radii
        r_sorted = r_star[indices]                                                                  # sorted radii
        lum_sorted = lum[indices]                                                                   # luminosities sorted for radius
        lum_sum = np.cumsum(lum_sorted)                                                             # cumulative sum of sorted luminosities
        hlr = np.max(r_sorted[lum_sum <= tot_lum/2])                                                # 2D/3D radius at half the luminosity
        return hlr
        
    def PerformanceMode(self, full=False, turn_off=False):
        """Sacrifices memory usage for performance during the simulation of images or other tasks.
        The full set of variables like luminosity and temperature can be stored or only
            a selection needed for imaging, by setting full to True or False.
        If for some reason this mode has to be turned off (delete stored data), set turn_off=True.
        """
        if (not hasattr(self, 'current_masses') & full):
            self.current_masses = self.CurrentMasses(realistic_remnants=True)
        elif (turn_off & full):
            del self.current_masses
            
        if (not hasattr(self, 'stellar_radii') & full):
            self.stellar_radii = self.StellarRadii(realistic_remnants=True)
        elif (turn_off & full):
            del self.stellar_radii
            
        if (not hasattr(self, 'log_luminosities') & full):
            self.log_luminosities = self.LogLuminosities(realistic_remnants=True)
        elif (turn_off & full):
            del self.log_luminosities
            
        if (not hasattr(self, 'log_temperatures') & full):
            self.log_temperatures = self.LogTemperatures(realistic_remnants=True)
        elif (turn_off & (full == True)):
            del self.log_temperatures
        
        if (not hasattr(self, 'absolute_magnitudes') & full):
            self.absolute_magnitudes = self.AbsoluteMagnitudes(realistic_remnants=True)
        elif (turn_off & full):
            del self.absolute_magnitudes
        
        if not hasattr(self, 'apparent_magniutdes'):
            self.apparent_magniutdes = self.ApparentMagnitudes(add_redshift=True)
        elif turn_off:
            del self.apparent_magniutdes
        
        if not hasattr(self, 'spectral_types'):
            self.spectral_types, names = self.SpectralTypes(realistic_remnants=True)
        elif turn_off:
            del self.spectral_types
        
        return


class Gas():
    """"""
    def __init__(self):
        pass
    
    def __repr__(self):
        pass
    
    def __str__(self):
        pass
    
    def __add__(self):
        pass
    
    def AddInclination(self, incl):
        """Put the object at an inclination w.r.t. the observer, in radians. 
        The given angle is measured from the x-axis towards the z-axis (which is the l.o.s.).
        Can be specified for each cloud separately or for all at once.
        This is additive (given angles stack with previous ones).
        """
    
    def AddEllipticity(self, axes):
        """Give the object axes (cartesian) different relative proportions. 
        The given axes (a,b,c) are taken as relative sizes, the volume of the object is conserved.
        Can be specified for each cloud separately or for all at once.
        This is additive (given axis sizes stack with previous ones).
        """
    
    
class Dust():
    """"""
    def __init__(self):
        pass
    
    def __repr__(self):
        pass
    
    def __str__(self):
        pass
    
    def __add__(self):
        pass
    
    def AddInclination(self, incl):
        """Put the object at an inclination w.r.t. the observer, in radians. 
        The given angle is measured from the x-axis towards the z-axis (which is the l.o.s.).
        Can be specified for each cloud separately or for all at once.
        This is additive (given angles stack with previous ones).
        """
    
    def AddEllipticity(self, axes):
        """Give the object axes (cartesian) different relative proportions. 
        The given axes (a,b,c) are taken as relative sizes, the volume of the object is conserved.
        Can be specified for each cloud separately or for all at once.
        This is additive (given axis sizes stack with previous ones).
        """


class AstronomicalObject(object):
    """Base class for astronomical objects like clusters and galaxies.
    Contains the basic information and functionality. It is also a composite of the different
    component classes: Stars, Gas and Dust.
    
    Note:
        takes in all the kwargs necessary for the component classes
        
    Args:
        distance
        d_type
        extinct
        **kwargs: the kwargs are used in initiation of the component classes
    
    Attributes:
        AddFieldStars
        AddBackGround
        AddNGS
        SaveTo
        LoadFrom
    """
    def __init__(self, distance=10, d_type='l', extinct=0, **kwargs):
        
        self.d_type = d_type                                                                        # distance type [l for luminosity, z for redshift]
        if (self.d_type == 'z'):
            self.redshift = distance                                                                # redshift for the object
            self.d_lum = form.d_luminosity(self.redshift)                                            # luminosity distance to the object (in pc)
        else:
            self.d_lum = distance                                                                   # luminosity distance to the object (in pc)
            self.redshift = form.d_luminosity_to_redshift(self.d_lum)                                           # redshift for the object
        
        self.d_ang = form.d_angular(self.redshift)                                                   # angular distance (in pc)
        self.extinction = extinct                                                                   # extinction between source and observer
        
        # initialise the component classes (pop the right kwargs per object)
        stars_args = [k for k, v in inspect.signature(Stars).parameters.items()]
        stars_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in stars_args}
        if stars_dict:
            self.stars = Stars(**stars_dict)                                                        # the stellar component
        
        gas_args = [k for k, v in inspect.signature(Gas).parameters.items()]
        gas_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in gas_args}
        if gas_dict:
            self.gas = Gas(**gas_dict)                                                              # the gaseous component
        
        dust_args = [k for k, v in inspect.signature(Dust).parameters.items()]
        dust_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in dust_args}
        if dust_dict:
            self.dust = Dust(**dust_dict)                                                           # the dusty component
        
        super().__init__(**kwargs)                                                                  # inheritance
        return
    
    def __repr__(self):
        pass
    
    def __str__(self):
        pass
    
    def __add__(self):
        pass

    def parsec_to_arcsec(self, length):
        """Convert a length (pc) to angular size on the sky (as). Can be used for coordinates and radii."""
        return conv.parsec_to_arcsec(length, self.d_ang)

    def AddFieldStars(self, n=10, fov=53, sky_coords=None):
        """Adds (Milky Way) field stars to the object in the given f.o.v (in as).
        [WIP] Nothing is done with sky_coords at the moment.
        """
        # self.field_stars = np.array([pos_x, pos_y, mag, filt, spec_types])
        # todo: WIP
        # add like a query to Gaia data, 
        # default: some random distribution in the shape of the fov
        fov_rad = fov*as_rad
        phi = dist.uniform(n=n, min_val=0, max_val=2*np.pi, power=1)
        theta = dist.angle_theta(n=n)
        radius = dist.uniform(n=n, min_val=10, max_val=10**3, power=3)                                      # uniform between 10 pc and a kpc
        coords = conv.spher_to_cart(radius, theta, phi)
        distance = self.d_lum - np.abs(coords[:,3])
        
        mag = dist.uniform(n=n, min_val=5, max_val=20, power=1)                                             # placeholder for magnitudes
        filt = np.full_like(mag, 'V', dtype='U5')                                                   # placeholder for filter
        spec_types = np.full_like(mag, 'G0V', dtype='U5')                                           # placeholder for spectral types
        self.field_stars = np.array([coords[:,0], coords[:,1], mag, filt, spec_types])
        return
        
    def AddBackGround(self, n=10, fov=53):
        """Adds additional structures to the background, like more clusters or galaxies. 
        These will be unresolved. Give the imaging f.o.v (in as).
        """
        # self.back_ground = 
        # todo: WIP
        return
        
    def AddNGS(self, mag=13, filter='V', pos_x=None, pos_y=None, spec_types='G0V'):
        """Adds one or more natural guide star(s) for the adaptive optics.
        The SCAO mode can only use one NGS that has to be within a magnitude of 10-16 (optical).
        The MCAO mode can track multiple stars. Positions can be specified manually (in as!),
        as well as the spectral types for these stars.
        [The automatically generated positions are specific to the ELT!]
        """
        if hasattr(mag, '__len__'):
            mag = np.array(mag)
        else:
            mag = np.array([mag])
        
        # give them positions
        if ((pos_x is not None) & (pos_y is not None)):
            if hasattr(pos_x, '__len__'):
                pos_x = np.array(pos_x)
            else:
                pos_x = np.array([pos_x])
            if hasattr(pos_y, '__len__'):
                pos_y = np.array(pos_y)
            else:
                pos_y = np.array([pos_y])
        elif (len(mag) == 1):
            # just outside fov for 1.5 mas/p scale, but inside patrol field of scao
            pos_x = np.array([7.1])                                                                 # x position in as
            pos_y = np.array([10.6])                                                                # y position in as
        else:
            # assume we are using MCAO now. generate random positions within patrol field
            angle = dist.angle_phi(n=len(mag))
            radius = dist.uniform(n=len(mag), min_val=46, max_val=90, power=2)                              # radius of patrol field in as
            pos_x_y = conv.pol_to_cart(radius, angle)
            pos_x = pos_x_y[0]
            pos_y = pos_x_y[1]
        
        # give them a spectral type
        if isinstance(spec_types, str):
            spec_types = np.full_like(mag, spec_types, dtype='U5')
        elif hasattr(spec_types, '__len__'):
            spec_types = np.array(spec_types)
        else:
            spec_types = np.full_like(mag, 'G0V', dtype='U5')                                           
        
        filt = np.full_like(mag, filter, dtype='U5')                                                # the guide stars can be in a different filter than the observations 
        
        self.natural_guide_stars = [pos_x, pos_y, mag, filt, spec_types]
        return

    def PlotStars2D(self, title='Scatter', xlabel='x', ylabel='y', axes='xy', colour='blue', filter='V', theme='dark1',
               show=True):
        """Make a plot of the object positions in two dimensions
        Set colour to 'temperature' for a temperature representation.
        Set filter to None to avoid markers scaling in size with magnitude.
        Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but
            saveable dark plot, 'fits' for a plot that resembles a .fits image,
            and None for normal light colours.
        """
        # todo: fix the plotting functions for their new class
        if (filter is not None):
            mags = self.ApparentMagnitudes(filters=filter)
        else:
            mags = None

        if (colour == 'temperature'):
            temps = 10**self.LogTemperatures()
        else:
            temps = None

        vis.scatter_2d(self.coords, title=title, xlabel=xlabel, ylabel=ylabel, axes=axes, colour=colour, T_eff=temps,
                       mag=mags, theme=theme, show=show)
        return

    def PlotStars3D(self, title='Scatter', xlabel='x', ylabel='y', colour='blue', filter='V', theme='dark1', show=True):
        """Make a plot of the object positions in three dimensions.
        Set colour to 'temperature' for a temperature representation.
        Set filter to None to avoid markers scaling in size with magnitude.
        Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but
            saveable dark plot, and None for normal light colours.
        """
        if (filter is not None):
            mags = self.ApparentMagnitudes(filters=filter)
        else:
            mags = None

        if (colour == 'temperature'):
            temps = 10**self.LogTemperatures()
        else:
            temps = None

        vis.scatter_3d(self.coords, title=title, xlabel=xlabel, ylabel=ylabel, colour=colour, T_eff=temps, mag=mags,
                       theme=theme, show=show)
        return

    def PlotHRD(self, title='hr_diagram', colour='temperature', theme='dark1', show=True):
        """Make a plot of stars in an HR diagram.
        Set colour to 'temperature' for a temperature representation.
        Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but
            saveable dark plot, and None for normal light colours.
        """
        r_mask = np.invert(self.Remnants())
        temps = 10**self.LogTemperatures()
        lums = self.LogLuminosities()

        vis.hr_diagram(T_eff=temps, log_lum=lums, title=title, xlabel='Temperature (K)',
                       ylabel=r'Luminosity log($L/L_\odot$)', colour=colour, theme=theme, mask=r_mask, show=show)
        return

    def PlotCMD(self, x='B-V', y='V', title='cm_diagram', colour='blue', theme=None, show=True):
        """Make a plot of the stars in a cm_diagram
        Set x and y to the colour and magnitude to be used (x needs format 'A-B')
        Set colour to 'temperature' for a temperature representation.
        """
        x_filters = x.split('-')
        mag_A = self.ApparentMagnitudes(filters=x_filters[0])
        mag_B = self.ApparentMagnitudes(filters=x_filters[1])
        c_mag = mag_A - mag_B

        if (y == x_filters[0]):
            mag = mag_A
        elif (y == x_filters[1]):
            mag = mag_B
        else:
            mag = self.ApparentMagnitudes(filters=y)

        vis.cm_diagram(c_mag, mag, title=title, xlabel=x, ylabel=y,
                       colour=colour, T_eff=None, theme=theme, adapt_axes=True, mask=None, show=show)
        return

    def SaveTo(self, filename):
        """Saves the class to a file."""
        if (filename[-4:] != '.pkl'):
            filename += '.pkl'
        
        if os.path.isdir('objects'):
            with open(os.path.join('objects', filename), 'wb') as output:
                pickle.dump(self, output, -1)
        else:                                                                                       # if for some reason the objects folder isn't there
            with open(filename, 'wb') as output:
                pickle.dump(self, output, -1)
        return
        
    def LoadFrom(filename):
        """Loads the class from a file."""
        if (filename[-4:] != '.pkl'):
            filename += '.pkl'
        
        if os.path.isdir('objects'):
            with open(os.path.join('objects', filename), 'rb') as input:
                data = pickle.load(input)
        else:                                                                                       # if for some reason the objects folder isn't there
            with open(filename, 'rb') as input:
                data = pickle.load(input)
        return data


class StarCluster(AstronomicalObject):
    """For generating star clusters."""    
    def __init__(**kwargs):
        
        super().__init__(**kwargs)
        return


class EllipticalGalaxy(AstronomicalObject):
    """For generating elliptical galaxies."""    
    def __init__(**kwargs):
        
        super().__init__(**kwargs)
        return


class SpiralGalaxy(AstronomicalObject):
    """For generating spiral galaxies."""    
    def __init__(**kwargs):
        
        super().__init__(**kwargs)
        return


def gen_spherical(N_stars, dist_type=default_rdist, **kwargs):
    """Make a spherical distribution of stars using the given radial distribution type.
    Takes additional parameters for the r-distribution function (i.e. scale length s).
    """
    # check if the dist type exists
    if (dist_type[-2:] != '_r'):
        dist_type += '_r'                                                                           # add the r to the end for radial version
        
    dist_list = list(set(fnmatch.filter(dir(dist), '*_r')))
    
    if (dist_type not in dist_list):
        warnings.warn(('objectgenerator//gen_spherical: Specified distribution type does not exist. '
                       'Using default (={})').format(default_rdist), SyntaxWarning)
        dist_type = default_rdist + '_r'
    
    # check if right parameters given    
    sig = inspect.signature(getattr(dist, dist_type))                                               # parameters of the dist function (includes n)
    dict = kwargs.copy()                                                                            # need a copy for popping in iteration
    for key in dict:
        if key not in sig.parameters:
            warnings.warn(('objectgenerator//gen_spherical: Wrong keyword given in distribution '
                           'parameters. Deleted entry.\n    {0} = {1}'
                           ).format(key, kwargs.pop(key, None)), SyntaxWarning)
    
    r_dist = getattr(dist, dist_type)(n=N_stars, **kwargs)                                           # the radial distribution
    phi_dist = dist.angle_phi(N_stars)                                                               # dist for angle with x axis
    theta_dist = dist.angle_theta(N_stars)                                                           # dist for angle with z axis
    
    return conv.spher_to_cart(r_dist, theta_dist, phi_dist)


def gen_spiral(N_stars):
    """Make a spiral galaxy."""
    #todo: add this
    

def gen_spiral_arms():
    """Make spiral arms."""
    

def gen_irregular(N_stars):
    """Make an irregular galaxy"""
    

def gen_star_masses(N_stars=0, M_tot=0, imf=None):
    """Generate masses using the Initial Mass Function. 
    Either number of stars or total mass should be given.
    imf defines the lower and upper bound to the masses generated in the IMF.
    Also gives the difference between the total generated mass and the input mass 
        (or estimated mass when using N_stars).
    """
    if not imf:
        imf = default_imf_par

    # check input (N_stars or M_tot?)
    if (N_stars == 0) & (M_tot == 0):
        warnings.warn(('objectgenerator//gen_star_masses: Input mass and number of stars '
                       'cannot be zero simultaniously. Using N_stars=10'), SyntaxWarning)
        N_stars = 10
    elif (N_stars == 0):                                                                            # a total mass is given
        N_stars = conv.m_tot_to_n_stars(M_tot, imf)                                                     # estimate the number of stars to generate
        
    # mass
    M_init = dist.kroupa_imf(N_stars, imf)                                                           # assign initial masses using IMF
    M_tot_gen = np.sum(M_init)                                                                      # total generated mass (will differ from input mass, if given)
    
    if (M_tot != 0):
        M_diff = M_tot_gen - M_tot
    else:
        M_tot_est = conv.n_stars_to_m_tot(N_stars, imf)
        M_diff = M_tot_gen - M_tot_est                                                              # will give the difference to the estimated total mass for n stars
        
    return M_init, M_diff


def star_form_history(max_age, min_age=1, sfr='exp', Z=0.014, tau=1e10):
    """Finds the relative number of stars to give each (logarithmic) age step up to a 
    maximum given age (starting from a minimum age if desired).
    The star formation rate (sfr) can be 'exp' or 'lin-exp'.
    tau is the decay timescale of the star formation (in yr).
    """
    if not hasattr(max_age, '__len__'):
        max_age = np.array([max_age])
    else:
        max_age = np.array(max_age)
    if not hasattr(min_age, '__len__'):
        min_age = np.full_like(max_age, min_age)
    else:
        min_age = np.array(min_age)
    if not hasattr(Z, '__len__'):
        Z = np.full_like(max_age, Z, dtype=float)
    if isinstance(sfr, str):
        sfr = np.full_like(max_age, sfr, dtype='U10')
    if not hasattr(tau, '__len__'):
        tau = np.full_like(max_age, tau, dtype=float)
    
    if np.all(max_age > 12):                                                                        # determine if logarithm or not
        max_age = np.log10(max_age)
    if np.all(min_age > 12):
        min_age = np.log10(min_age)
    
    rel_num = np.array([])
    log_ages_used = np.array([])
    for i in range(len(max_age)):
        log_ages = np.unique(utils.open_isochrones_file(Z[i], columns=['log_age']))                   # avaiable ages
        uni_log_ages = np.unique(log_ages)
        log_ages_used_i = uni_log_ages[(uni_log_ages <= max_age[i]) & (uni_log_ages >= min_age[i])] # log t's to use (between min/max ages)
        ages_used = 10**log_ages_used_i                                                             # age of each SSP
    
        if (sfr[i] == 'exp'):
            psi = np.exp((ages_used - 10**min_age[i])/tau[i])                                       # Star formation rates (relative)
        elif(sfr[i] == 'lin-exp'):
            t0 = 10**np.max(uni_log_ages)                                                           # represents the start of time
            psi = ((t0 - ages_used - 10**min_age[i])
                   /tau[i]*np.exp((ages_used - 10**min_age[i])/tau[i]))                             # Star formation rates (relative)
        else:
            # for when None is given
            log_ages_used_i = max_age[i]
            psi = 1
        
        rel_num = np.append(rel_num, psi/np.sum(psi))                                               # relative number in each generation
        log_ages_used = np.append(log_ages_used, log_ages_used_i)
    return rel_num, log_ages_used


def find_spectral_type(T_eff, Lum, Mass):
    """Finds the spectral type of a star from its properties using a table.
    T_eff: effective temperature (K)
    Lum: logarithm of the luminosity in Lsun
    Mass: log of the mass in Msun
    Returns the reference indices (ndarray) and an ndarray of corresponding types.
     (so type_selection[indices] gives the full array of names)
    """
    # Stellar_Type    Mass        Luminosity	Radius       Temp    B-V	Mv	BC(Temp) Mbol
    # None            Mstar/Msun  Lstar/Lsun	Rstar/Rsun   K       Color	Mag	Corr	 Mag
    
    with open(os.path.join('tables', 'all_stars.txt')) as file:
        spec_tbl_names  = np.array(file.readline()[:-1].split('\t'))                                # read in the column names
    spec_tbl_names = {name: i for i, name in enumerate(spec_tbl_names[1:])}                         # make a dict for ease of use
    
    spec_tbl_types = np.loadtxt(os.path.join('tables', 'all_stars.txt'), dtype=str, 
                                usecols=[0], unpack=True)
    spec_tbl = np.loadtxt(os.path.join('tables', 'all_stars.txt'), dtype=float, 
                          usecols=[1,2,3,4,5,6,7,8], unpack=True)
    
    spec_letter = ['M', 'K', 'G', 'F', 'A', 'B', 'O']                                               # basic spectral letters (increasing temperature)
    
    # collect the relevant spectral types 
    # (C, R, N, W, S and most D are disregarded, 
    # includes DC - WD with no strong spectral features, also leaves out subdwarfs)
    T_tbl, L_tbl, M_tbl, type_selection = [], [], [], []
    for i, type in enumerate(spec_tbl_types):
        if ((type[0] in spec_letter) | (type[:2] == 'DC')) & (type[2:] != 'VI'):
            type_selection.append(type)
            T_tbl.append(np.log10(spec_tbl[spec_tbl_names['Temp'], i]))
            L_tbl.append(0.4*(M_bol_sun - spec_tbl[spec_tbl_names['Mbol'], i]))                     # convert to L from Mbol (to avoid L=0 in tbl)              
            M_tbl.append(np.log10(spec_tbl[spec_tbl_names['Mass'], i]))
    
    type_selection = np.array(type_selection)
    T_tbl = np.array(T_tbl)
    L_tbl = np.array(L_tbl)
    M_tbl = np.array(M_tbl)
    
    data_points = np.column_stack([T_eff, Lum, Mass])
    tbl_grid = np.column_stack([T_tbl, L_tbl, M_tbl])                                               # 3d data grid with points (T, L, M)
    tbl_tree = sps.cKDTree(tbl_grid)                                                                # K-Dimensional lookup Tree
    dists, indices = tbl_tree.query(data_points, k=1)                                               # the distances to and indices of the closest points
    
    indices[dists == np.inf] = np.where(type_selection == 'DC9')[0][0]                              # correct for missing neighbours
    
    # WD are taken care of this way, but what to do with NS/BH? 
    # (now they would probably get WD spectra)
    return indices, type_selection


def compactify(compact_mode, n_stars, mag_limit, imf=None):
    """Reduces the total amount of stars generated by decreasing the range of stellar initial masses."""
    if not imf:
        imf = default_imf_par

    # check if compact mode is on
    fraction_generated = np.ones_like(self.n_stars, dtype=float)  # set to ones initially
    mass_limit = np.copy(self.imf_param[:])  # set to imf params initially

    if (mag_limit is None):
        mag_limit = default_mag_lim  # if not specified, use default

    for i, pop_num in enumerate(self.n_stars):
        if (compact_mode == 'mag'):
            mass_limit[i] = magnitude_limited(self.ages[i], self.metal[i], mag_lim=mag_limit, distance=self.d_lum,
                                              ext=self.extinction, filter='Ks')
        else:
            mass_limit[i] = number_limited(self.n_stars, self.ages[i], self.metal[i], imf=self.imf_param[i])

        if (mass_limit[i, 1] > imf[i, 1]):
            # don't increase the upper limit!
            mass_limit[i, 1] = imf[i, 1]

        fraction_generated[i] = form.mass_fraction_from_limits(mass_limit[i], imf=imf[i])
    if np.any(mass_limit[:, 0] > mass_limit[:, 1]):
        raise RuntimeError('compacting failed, '
                           'mass limit raised above upper mass.')  # lower limit > upper limit!
    elif np.any(n_stars*fraction_generated < 10):  # don't want too few stars
        raise RuntimeError('Population compacted to <10, '
                           'try not compacting or generating a higher number of stars.')
    return mass_limit, fraction_generated


def number_limited(N, age, Z, imf=None):
    """Retrieves the lower mass limit for which the number of stars does not exceed 10**7. 
    Will also give an upper mass limit based on the values in the isochrone.
    The intended number of generated stars, age and metallicity are needed.
    """
    if not imf:
        imf = default_imf_par

    fraction = np.clip(limiting_number/N, 0, 1)                                                     # fraction of the total number of stars to generate
    
    M_ini = utils.stellar_isochrone(age, Z, columns=['M_initial'])                                   # get the isochrone values
    mass_lim_high = M_ini[-1]                                                                       # highest value in the isochrone
    
    mass_lim_low = form.mass_limit_from_fraction(fraction, M_max=mass_lim_high, imf=imf)
    
    return mass_lim_low, mass_lim_high


def magnitude_limited(age, Z, mag_lim=default_mag_lim, distance=10, ext=0, filter='Ks'):
    """Retrieves the lower mass limit for which the given magnitude threshold is met. 
    Will also give an upper mass limit based on the values in the isochrone.
    Works only for resolved stars. 
    If light is integrated along the line of sight (crowded fields), 
    then don't use this method! Resulting images would not be accurate!
    distance, age, metallicity and extinction of the population of stars are needed.
    A filter must be specified in which the given limiting magnitude is measured.
    """
    iso_M_ini = utils.stellar_isochrone(age, Z, columns=['M_initial'])                               # get the isochrone values
    mag_vals = utils.stellar_isochrone(age, Z, columns=[filter])

    abs_mag_lim = form.absolute_magnitude(mag_lim, distance, ext=ext)                                             # calculate the limiting absolute magnitude
    mask = (mag_vals < abs_mag_lim + 0.1)                                                           # take all mag_vals below the limit
    if not mask.any():
        mask = (mag_vals == np.min(mag_vals))                                                       # if limit too high (too low in mag) then it will break
        warnings.warn(('objectgenerator//magnitude_limited: compacting will not work, '
                       'distance too large.'), RuntimeWarning)

    mass_lim_low = iso_M_ini[mask][0]                                                               # the lowest mass where mask==True
    
    mass_lim_high = iso_M_ini[-1]                                                                   # highest value in the isochrone
    
    return mass_lim_low, mass_lim_high


def remnants_single_pop(M_init, age, Z):
    """Gives the positions of the remnants in a single population (as a boolean mask)."""
    iso_M_ini = utils.stellar_isochrone(age, Z, columns=['M_initial'])
    max_mass = np.max(iso_M_ini)                                                                    # maximum initial mass in isoc file
    return (M_init > max_mass)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


