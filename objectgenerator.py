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
import copy
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


class Stars():
    """Generates populations of stars and contains all the information about them.
    Also functions that can be performed on the stars are defined here.

    Notes
    -----
    N_stars and M_tot_init cannot be zero simultaneously.
    ages and metal are both not optional parameters (they might look like they are).
    'array' here means numpy.ndarray, while lists (or even single entries) are also accepted
    # https://quantecon.org/wiki-py-docstrings/


    Arguments
    ---------
    n_stars: array of int
        the total number of stars to make per stellar population.

    M_tot_init (array of float):
        the total mass in stars to produce per population (in Msol).

    ages (array of float):
        stellar age for each population (in linear or 10log yr).
        This is the maximum age when sfh is used.

    metal (array of float):
        metallicity for each population.

    sfh (array of str and None, optional):
        type of star formation history to use per population.

    min_ages (array of float):
        minimum ages to use in star formation histories, per population

    tau_sfh (array of float):
        characteristic timescales for star formation per population

    imf_par (array of float, optional):
        2D array of two parameters per stellar population.
        First parameter is lowest mass, second is highest mass to make.

    r_dist (array of str):
        type of radial distribution to use per population.

    r_dist_par (array of dict):
        parameter values for the radial distributions specified
        (one dictionary per population).

    incl (array of float):
        inclination values per population (in radians)

    ellipse_axes (array of float):
        2D array of one set of axes per population.
        Relative (x,y,z)-sizes to stretch the distributions; volume is kept constant.

    spiral_arms (array of int):
        number of spiral arms (per population).

    spiral_bulge (array of float):
        relative proportion of the central bulge (per population).

    spiral_bar (array of float):
        relative proportion of the central bar (per population).

    compact_mode (array of str and None):
        generate only a fraction of the total number of stars.
        choose 'num' or 'mag' for number or magnitude limited.


    Attributes
    ----------
    Most of the arguments described above are stored as attributes.
    (exceptions: M_tot_init)

    origin (array of float):
        origin of the stellar distribution (for each population). Shape is [3, n_pop]

    coords (array of float):
        2D array of the cartesian coordinates of the stars. Shape is [3, n_stars].

    M_init (array of float):
        the individual masses of the stars in Msol.

    mag_names (array of str):
        the filter names of the corresponding default set of supported magnitudes.

    _spec_names (array of str):
        spectral type names corresponding to the reference numbers in _spectral_types.

    fraction_generated (array of float):
        part of the total number of stars that has actually been generated
        per population (when compact mode is used).


    Returns
    -------
    object
        Contains all information about the stars. Stars are not yet generated when the object is initialized.

    """    
    def __init__(self, n_stars=0, M_tot_init=0, ages=None, metal=None, imf_par=None, sfh=None, min_ages=None,
                 tau_sfh=None, origin=None, incl=None, r_dist=None, r_dist_par=None, ellipse_axes=None,
                 spiral_arms=None, spiral_bulge=None, spiral_bar=None, compact_mode=None):

        # cast input to right formats, and perform some checks. first find the intended number of populations
        n_pop = utils.check_number_of_populations(n_stars, M_tot_init, ages, metal)
        self.ages = utils.cast_simple_array(ages, n_pop, error='No age was defined.')
        self.metal = utils.cast_simple_array(metal, n_pop, error='No metallicity was defined.')
        self.imf_param = utils.cast_imf_parameters(imf_par, n_pop, fill_value=default_imf_par)
        self.imf_param = utils.check_lowest_imf_mass(self.imf_param, self.ages, self.metal)
        self.n_stars = utils.cast_and_check_n_stars(n_stars, M_tot_init, n_pop, self.imf_param)
        self.sfhist = utils.cast_simple_array(sfh, n_pop, fill_value=None,
                                              warning='Excess SFH types discarded.')
        self.min_ages = utils.cast_simple_array(min_ages, n_pop, fill_value=None,
                                                warning='Excess minimum ages discarded.')
        self.tau_sfh = utils.cast_simple_array(tau_sfh, n_pop, fill_value=None,
                                               warning='Excess tau sfh values discarded.')

        # parameters defining the shape
        self.origin = utils.cast_translation(origin, n_pop).T
        self.inclination = utils.cast_simple_array(incl, n_pop, fill_value=0,
                                                   warning='Excess inclination values discarded.')
        self.r_dist_types = utils.cast_simple_array(r_dist, n_pop, fill_value=default_rdist,
                                                    warning='Excess radial distribution types discarded.')
        self.r_dist_types = utils.check_radial_dist_type(self.r_dist_types)
        self.r_dist_param = utils.cast_radial_dist_param(r_dist_par, self.r_dist_types, n_pop)
        self.ellipse_axes = utils.cast_ellipse_axes(ellipse_axes, n_pop)
        self.spiral_arms = utils.cast_simple_array(spiral_arms, n_pop, fill_value=0,
                                                   warning='Excess spiral arm values discarded.')
        self.spiral_bulge = utils.cast_simple_array(spiral_bulge, n_pop, fill_value=0,
                                                    warning='Excess spiral bulge values discarded.')
        self.spiral_bar = utils.cast_simple_array(spiral_bar, n_pop, fill_value=0,
                                                  warning='Excess spiral bar values discarded.')
        
        # properties that are derived/generated
        self.coords = np.empty([3, 0])
        self.M_init = np.array([])
        self.mag_names = utils.get_supported_filters()
        
        # compact mode parameters
        self.compact_mode = compact_mode
        self.fraction_generated = np.ones(n_pop)
        
        # semi-private
        self._current_masses = None
        self._stellar_radii = None
        self._surface_gravity = None
        self._log_luminosities = None
        self._log_temperatures = None
        self._absolute_magnitudes = None
        self._spectral_types = None
        self._spec_names = None
        self._remnants = None
        # the actual imf parameters to use for each population (limits imposed by compacting)
        self.gen_imf_param = self.imf_param
        # actually generated number of stars per population (for compact mode/sfhist)
        self.gen_n_stars = self.n_stars
        # the actual ages to be generated for each population (for sfhist)
        self.gen_ages = self.ages
        return

    def __repr__(self):
        """Unambiguous representation of what this object is."""
        r = (f'Stars(N_stars={self.n_stars!r}, '
             f'ages={self.ages!r}, '
             f'metal={self.metal!r}, '
             f'sfh={self.sfhist!r}, '
             f'min_ages={self.min_ages!r}, '
             f'tau_sfh={self.tau_sfh!r}, '
             f'imf_par={self.imf_param!r}, '
             f'incl={self.inclination!r}, '
             f'r_dist={self.r_dist_types!r}, '
             f'r_dist_par={self.r_dist_param!r}, '
             f'ellipse_axes={self.ellipse_axes!r}, '
             f'spiral_arms={self.spiral_arms!r}, '
             f'spiral_bulge={self.spiral_bulge!r}, '
             f'spiral_bar={self.spiral_bar!r}, '
             f'compact_mode={self.compact_mode!r})')
        return r
    
    def __str__(self):
        """Nice-to-read representation of what this object is"""
        s = ('Stars object with parameters:\n'
             f'N_stars:        {self.n_stars!s}\n'
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
        return s
        
    def __add__(self, other):
        """Appends two Stars objects to each other."""
        # Need to append all the properties.
        # the ones below can be appended in a simple append
        append = ['n_stars', 'ages', 'metal', 'sfhist', 'min_ages', 'tau_sfh', 'inclination',
                  'r_dist_types', 'r_dist_param', 'spiral_arms', 'spiral_bulge', 'spiral_bar', 'compact_mode',
                  'M_init', 'fraction_generated', 'gen_n_stars', 'gen_ages']
        # the ones below need axis 0 specified
        append_axis_zero = ['imf_param', 'origin', 'ellipse_axes', 'coords', 'gen_imf_param']
        # the ones below need to be checked for non-None value
        append_if = ['_current_masses', '_stellar_radii', '_surface_gravity', '_log_luminosities',
                     '_log_temperatures', '_remnants']
        # the ones below need the extra check of having all columns/names exist
        append_if_all = ['_absolute_magnitudes', '_spectral_types']
        # these are never appended since their equality is ensured (but need to correspond 1to1 with append_if_all)
        append_names = ['mag_names', '_spec_names']

        sum = copy.copy(self)  # don't alter the original
        for attr in append:
            setattr(sum, attr, np.append(getattr(self, attr), getattr(other, attr)))
        for attr in append_axis_zero:
            setattr(sum, attr, np.append(getattr(self, attr), getattr(other, attr), axis=0))
        for attr in append_if:
            if getattr(self, attr) and getattr(other, attr):
                setattr(sum, attr, np.append(getattr(self, attr), getattr(other, attr)))
            else:
                setattr(sum, attr, None)
        for attr, names in zip(append_if_all, append_names):
            if getattr(self, attr) and getattr(other, attr) and np.all(getattr(self, names) == getattr(other, names)):
                setattr(sum, attr, np.append(getattr(self, attr), getattr(other, attr)))
            else:
                setattr(sum, attr, None)
        return sum
        
    def __radd__(self, other):
        """Reverse add (for when adding doesn't work. e.g. in sum(a,b,c))."""
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def generate_stars(self, d_lum=None, extinction=None):
        """Generate the masses and positions of the stars. Not automatically done on initiate."""
        # assign the right values for generation (start of generating sequence)
        if self.compact_mode:
            self.gen_imf_param, self.fraction_generated = self.compact_parameters(d_lum=d_lum, extinction=extinction)
            self.gen_n_stars = np.rint(self.n_stars * self.fraction_generated).astype(int)

        if not np.all(self.sfhist == None):
            rel_num, ages = star_form_history(self.gen_ages, min_age=self.min_ages, sfr=self.sfhist,
                                              Z=self.metal, tau=self.tau_sfh)
            self.gen_n_stars = np.rint(rel_num * self.n_stars).astype(int)
            self.gen_ages = ages
        
        # generate the positions, masses   
        self.coords = gen_spherical(self.gen_n_stars, dist_types=self.r_dist_types, kwargs_list=self.r_dist_param)
        self.M_init = gen_star_masses(self.gen_n_stars, imf=self.gen_imf_param)
        return
        
    def add_inclination(self, incl):
        """Put the object at an inclination w.r.t. the observer, in radians. 
        The given angle is measured from the x-axis towards the z-axis (which is the l.o.s.).
        Can be specified for each population separately or for all at once.
        This is additive (given angles stack with previous ones).
        """
        # check the input, make sure it is an array of the right size
        n_pop = len(self.n_stars)
        incl = utils.cast_simple_array(incl, n_pop, fill_value=0, warning='Excess inclination values discarded.')
        
        # check for existing inclination and record the new one
        if hasattr(self, 'inclination'):
            self.inclination += incl
        else:
            self.inclination = incl
            
        if np.any(self.inclination > 2*np.pi):
            if np.any(incl > 2*np.pi):
                warnings.warn('Inclination angle over 2pi detected, make sure to use radians!')
            self.inclination = np.mod(self.inclination, 2*np.pi)
        
        # rotate the XZ plane (x axis towards positive z)
        if not np.all(self.inclination == 0):
            indices = np.repeat(np.arange(n_pop), self.n_stars)  # population index per star
            self.coords = conv.rotate_xz(self.coords, self.inclination[indices])
        return
        
    def add_ellipticity(self, axes):
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
        
        if not (len(np.unique(axes)) == 1):
            indices = np.repeat(np.arange(n_pop), self.n_stars)  # population index per star
            # convert to ellipsoid (keeps volume conserved)
            self.coords = self.coords * (axes / np.prod(axes)**(1 / 3))[indices]
        return
        
    def add_translation(self, translation):
        """Translate the origin of the stellar distribution.
        Can be specified for each population separately or for all at once.
        """
        # check the input, make sure it is an array of the right size
        n_pop = len(self.n_stars)
        translation = utils.cast_translation(translation, n_pop)
        self.origin += translation.T  # record the new translation
        indices = np.repeat(np.arange(n_pop), self.n_stars)  # population index per star
        # move the stars (assuming shape (n_pop, 3) --> cast to (3, #stars))
        self.coords = self.coords + translation[indices].T
        return
    
    def current_masses(self, realistic_remnants=True, performance_mode=True):
        """Gives the current masses of the stars in Msun.
        Uses isochrone files and the given initial masses of the stars.
        Stars should not have a lower initial mass than the lowest mass in the isochrone file.
            (goes for all functions that use the isochrones)
        performance_mode: save the data returned by this function in memory for faster access later.
        """
        if self._current_masses is not None:
            M_cur = self._current_masses
        else:
            M_cur = np.array([])
            index = np.cumsum(np.append([0], self.gen_n_stars))  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_M_act = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial', 'M_current'])
                # select the masses of one population
                M_init_i = self.M_init[index[i]:index[i+1]]
                # arg 'right': return 0 for stars heavier than available in isoc file (dead stars)
                M_cur_i = np.interp(M_init_i, iso_M_ini, iso_M_act, right=0)
                
                # give estimates for remnant masses (replacing the 0 above)
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])
                    M_cur_i[remnants_i] = r_M_cur_i
                
                M_cur = np.append(M_cur, M_cur_i)

        # turn performance mode on or off (meaning to save or to delete the data from memory)
        if performance_mode and (self._current_masses is None):
            self._current_masses = M_cur
        elif (not performance_mode) and (self._current_masses is not None):
            self._current_masses = None
        return M_cur
        
    def stellar_radii(self, realistic_remnants=True, performance_mode=True):
        """Gives the stellar radii of the stars in Rsun.
        Uses isochrone files and the given initial masses of the stars.
        performance_mode: save the data returned by this function in memory for faster access later.
        """
        if self._stellar_radii is not None:
            R_cur = self._stellar_radii
        else:
            R_cur = np.array([])
            index = np.cumsum(np.append([0], self.gen_n_stars))  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_log_g = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial', 'log_g'])
                iso_R_cur = conv.gravity_to_radius(iso_log_g, iso_M_ini)
                M_init_i = self.M_init[index[i]:index[i+1]]
                # arg 'right': return 0 for stars heavier than available in isoc file (dead stars)
                R_cur_i = np.interp(M_init_i, iso_M_ini, iso_R_cur, right=0)
                
                # give estimates for remnant radii (replacing the 0 above)
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])
                    r_R_cur_i = form.remnant_radius(r_M_cur_i)
                    R_cur_i[remnants_i] = r_R_cur_i
                
                R_cur = np.append(R_cur, R_cur_i)

        # turn performance mode on or off (meaning to save or to delete the data from memory)
        if performance_mode and (self._stellar_radii is None):
            self._stellar_radii = R_cur
        elif (not performance_mode) and (self._stellar_radii is not None):
            self._stellar_radii = None
        return R_cur

    def surface_gravity(self, realistic_remnants=True, performance_mode=True):
        """Gives the logarithm of the surface gravity of the stars (in log cgs units).
        Uses isochrone files and the given initial masses of the stars.
        performance_mode: save the data returned by this function in memory for faster access later.
        """
        if self._surface_gravity is not None:
            log_g = self._surface_gravity
        else:
            log_g = np.array([])
            index = np.cumsum(np.append([0], self.gen_n_stars))  # indices defining the different populations

            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_log_g = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial', 'log_g'])
                M_init_i = self.M_init[index[i]:index[i + 1]]
                # arg 'right': return 0 for stars heavier than available in isoc file (dead stars)
                log_g_i = np.interp(M_init_i, iso_M_ini, iso_log_g, right=0)

                # give estimates for remnant radii (replacing the 0 above)
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])
                    r_R_cur_i = form.remnant_radius(r_M_cur_i)
                    r_log_g_i = conv.radius_to_gravity(r_R_cur_i, r_M_cur_i)
                    log_g_i[remnants_i] = r_log_g_i

                log_g = np.append(log_g, log_g_i)

        # turn performance mode on or off (meaning to save or to delete the data from memory)
        if performance_mode and (self._surface_gravity is None):
            self._surface_gravity = log_g
        elif (not performance_mode) and (self._surface_gravity is not None):
            self._surface_gravity = None
        return log_g
        
    def log_luminosities(self, realistic_remnants=True, performance_mode=True):
        """Gives the logarithm of the luminosity of the stars in Lsun.
        Uses isochrone files and the given initial masses of the stars.
        realistic_remnants gives estimates for remnant luminosities. Set False to save time.
        performance_mode: save the data returned by this function in memory for faster access later.
        """
        if self._log_luminosities is not None:
            log_L = self._log_luminosities
        else:
            log_L = np.array([])
            index = np.cumsum(np.append([0], self.gen_n_stars))  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_log_L = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial', 'log_L'])
                M_init_i = self.M_init[index[i]:index[i+1]]
                # arg 'right': return -9 --> L = 10**-9 Lsun (for stars heavier than available)
                log_L_i = np.interp(M_init_i, iso_M_ini, iso_log_L, right=-9)
                
                # give estimates for remnant luminosities (replacing the -9 above)
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    remnant_time = form.remnant_time(M_init_i[remnants_i], age, self.metal[i])
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])
                    r_R_cur_i = form.remnant_radius(r_M_cur_i)
                    r_Te_i = form.remnant_temperature(r_M_cur_i, r_R_cur_i, remnant_time)
                    r_log_L_i = np.log10(form.bb_luminosity(r_Te_i, r_R_cur_i))  # remnant luminosity := BB radiation
                    log_L_i[remnants_i] = r_log_L_i
                
                log_L = np.append(log_L, log_L_i)

        # turn performance mode on or off (meaning to save or to delete the data from memory)
        if performance_mode and (self._log_luminosities is None):
            self._log_luminosities = log_L
        elif (not performance_mode) and (self._log_luminosities is not None):
            self._log_luminosities = None
        return log_L
        
    def log_temperatures(self, realistic_remnants=True, performance_mode=True):
        """Gives the logarithm of the effective temperature of the stars in K.
        Uses isochrone files and the given initial masses of the stars.
        realistic_remnants gives estimates for remnant temperatures. Set False to save time.
        performance_mode: save the data returned by this function in memory for faster access later.
        """
        if self._log_temperatures is not None:
            log_Te = self._log_temperatures
        else:
            log_Te = np.array([])
            index = np.cumsum(np.append([0], self.gen_n_stars))  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_log_Te = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial', 'log_Te'])
                M_init_i = self.M_init[index[i]:index[i+1]]
                # arg 'right': return 1 --> Te = 10 K (for stars heavier than available)
                log_Te_i = np.interp(M_init_i, iso_M_ini, iso_log_Te, right=1)
                
                # give estimates for remnant temperatures (replacing the 1 above)
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    remnant_time = form.remnant_time(M_init_i[remnants_i], age, self.metal[i])
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])
                    r_R_cur_i = form.remnant_radius(r_M_cur_i)
                    r_log_Te_i = np.log10(form.remnant_temperature(r_M_cur_i, r_R_cur_i, remnant_time))
                    log_Te_i[remnants_i] = r_log_Te_i
                
                log_Te = np.append(log_Te, log_Te_i)  

        # turn performance mode on or off (meaning to save or to delete the data from memory)
        if performance_mode and (self._log_temperatures is None):
            self._log_temperatures = log_Te
        elif (not performance_mode) and (self._log_temperatures is not None):
            self._log_temperatures = None
        return log_Te
        
    def absolute_magnitudes(self, filters=None, realistic_remnants=True, performance_mode=True):
        """Gives the absolute magnitudes of the stars using isochrone files and the initial masses.
        A list of filters can be specified; None will result in all available magnitudes.
        Realistic remnants can be emulated by using black body spectra.
        performance_mode: save the data returned by this function in memory for faster access later.
        """
        if filters is None:
            filters = self.mag_names
        
        if self._absolute_magnitudes is not None:
            abs_mag = self._absolute_magnitudes[:, utils.get_filter_mask(filters)]
            if (len(filters) == 1):
                abs_mag = abs_mag.flatten()  # correct for 2D array
        else:
            abs_mag = np.empty((0,) + (len(filters),)*(len(filters) != 1))
            index = np.cumsum(np.append([0], self.gen_n_stars))  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial'])
                iso_mag = utils.stellar_isochrone(age, self.metal[i], columns=filters).T
                M_init_i = self.M_init[index[i]:index[i+1]]
                # arg fill_value: return 30 --> L of less than 10**-9 (for stars heavier than available)
                interper = spi.interp1d(iso_M_ini, iso_mag, bounds_error=False, fill_value=30, axis=0)
                mag_i = interper(M_init_i)
                
                if realistic_remnants:
                    remnants_i = remnants_single_pop(M_init_i, age, self.metal[i])
                    remnant_time = form.remnant_time(M_init_i[remnants_i], age, self.metal[i])
                    r_M_cur_i = form.remnant_mass(M_init_i[remnants_i], self.metal[i])
                    r_R_cur_i = form.remnant_radius(r_M_cur_i)
                    r_Te_i = form.remnant_temperature(r_M_cur_i, r_R_cur_i, remnant_time)
                    mag_i[remnants_i] = form.bb_magnitude(r_Te_i, r_R_cur_i, filters)
                
                abs_mag = np.append(abs_mag, mag_i, axis=0)

        # turn performance mode on or off (meaning to save or to delete the data from memory)
        if performance_mode and (self._absolute_magnitudes is None):
            # to avoid confusing behavior and nontrivial code, save all of the magnitudes
            if (filters != self.mag_names):
                all_abs_mag = self.absolute_magnitudes(filters=None, realistic_remnants=realistic_remnants,
                                                       performance_mode=False)
            else:
                all_abs_mag = abs_mag
            self._absolute_magnitudes = all_abs_mag
        elif (not performance_mode) and (self._absolute_magnitudes is not None):
            self._absolute_magnitudes = None
        return abs_mag
    
    def apparent_magnitudes(self, distance, filters=None, extinction=0, redshift=None, realistic_remnants=True,
                            performance_mode=True):
        """Computes the apparent magnitude from the absolute magnitude and the individual
        distances (in pc). Needs the luminosity distance to the astronomical object (in pc).
        A list of filters can be specified; None will result in all available magnitudes.
        Redshift can be roughly included (uses black body spectra) by giving a redshift value.
        """
        if filters is None:
            filters = self.mag_names

        if (distance > 100*np.abs(np.min(self.coords[:, 2]))):  # distance 'much larger' than individual variations
            # approximate the distances to each star using the z-coordinate
            true_dist = distance - self.coords[:, 2]
        else:
            # calculate the distances for each star properly
            true_dist = form.distance_3d(self.coords, np.array([0, 0, distance]))

        true_dist = true_dist.reshape((len(true_dist),) + (1,)*(len(filters) > 1))  # fix dimension for broadcast

        if self._absolute_magnitudes:
            abs_mag = self._absolute_magnitudes[:, utils.get_filter_mask(filters)]
            if (len(filters) == 1):
                abs_mag = abs_mag.flatten()  # correct for 2D array
        else:
            # add redshift (rough approach)
            if redshift is not None:
                filter_means = utils.open_photometric_data(columns=['mean'], filters=filters)
                shifted_filters = (1 + redshift)*filter_means
                R_cur = self.stellar_radii(realistic_remnants=True)
                T_eff = 10**self.log_temperatures(realistic_remnants=True)
                abs_mag = form.bb_magnitude(T_eff, R_cur, filters, filter_means=shifted_filters)
            else:
                abs_mag = self.absolute_magnitudes(filters=filters, realistic_remnants=realistic_remnants,
                                                   performance_mode=performance_mode)

        app_mag = form.apparent_magnitude(abs_mag, true_dist, ext=extinction)  # true_dist in pc!
        return app_mag
        
    def spectral_types(self, realistic_remnants=True, performance_mode=True):
        """Gives the spectral types (as indices, to conserve memory) for the stars and
        the corresponding spectral type names.
        Uses isochrone files and the given initial masses of the stars.
        performance_mode: save the data returned by this function in memory for faster access later.
        """
        if self._spectral_types is not None:
            spec_indices = self._spectral_types
            spec_names = self._spec_names
        else:
            log_T_eff = self.log_temperatures(realistic_remnants=realistic_remnants)
            log_L = self.log_luminosities(realistic_remnants=realistic_remnants)
            log_M_cur = np.log10(self.current_masses(realistic_remnants=realistic_remnants))
            # assign spectra to the stars
            spec_indices, spec_names = find_spectral_type(log_T_eff, log_L, log_M_cur)
            
            # if this is run for the first time, save the spectral names
            if self._spec_names is None:
                self._spec_names = spec_names

        # turn performance mode on or off (meaning to save or to delete the data from memory)
        if performance_mode and (self._spectral_types is None):
            self._spectral_types = spec_indices
        elif (not performance_mode) and (self._spectral_types is not None):
            self._spectral_types = None
        return spec_indices, spec_names
    
    def remnants(self, performance_mode=True):
        """Gives the indices of the positions of remnants (not white dwarfs) as a boolean array. 
        (WD should be handled with the right isochrone files, but NS/BHs are not)
        performance_mode: save the data returned by this function in memory for faster access later.
        """
        if self._remnants is not None:
            remnants = self._remnants
        else:
            remnants = np.array([], dtype=bool)
            index = np.cumsum(np.append([0], self.gen_n_stars))  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                iso_M_ini = utils.stellar_isochrone(age, self.metal[i], columns=['M_initial'])
                max_mass = np.max(iso_M_ini)  # maximum initial mass in isoc file
                remnants_i = (self.M_init[index[i]:index[i+1]] > max_mass)
                remnants = np.append(remnants, remnants_i)

        # turn performance mode on or off (meaning to save or to delete the data from memory)
        if performance_mode and (self._remnants is None):
            self._remnants = remnants
        elif (not performance_mode) and (self._remnants is not None):
            self._remnants = None
        return remnants

    def total_initial_mass(self):
        """Returns the total initial mass in stars (in Msun)."""
        return np.sum(self.M_init)

    def total_current_mass(self):
        """Returns the total current mass in stars (in Msun)."""
        return np.sum(self.current_masses())
    
    def total_luminosity(self):
        """Returns log of the total luminosity from stars (in Lsun)."""
        # remnants don't add much, so leave them out to save some computation time
        return np.log10(np.sum(10**self.log_luminosities(realistic_remnants=False)))
        
    def coords_arcsec(self, distance):
        """Returns coordinates converted to arcseconds (from pc). Needs the angular distance
        to the object (in pc).
        """
        return conv.parsec_to_arcsec(self.coords, distance)
        
    def orbital_radii(self, distance=None, unit='pc', spher=False):
        """Returns the radial coordinate of the stars (spherical or cylindrical) in pc/as."""
        if spher:
            radii = form.distance_3d(self.coords)
        else:
            radii = form.distance_2d(self.coords)
        
        if (unit == 'as'):
            # convert to arcsec if wanted
            radii = conv.parsec_to_arcsec(radii, distance)
        
        return radii
    
    def orbital_velocities(self):
        """"""
        # todo: make this
    
    def half_mass_radius(self, spher=False):
        """Returns the (spherical or cylindrical) half mass radius in pc/as."""
        M_cur = self.current_masses()
        tot_mass = np.sum(M_cur)  # do this locally, to avoid unnecessary overhead (we already have M_cur anyway)
        
        if spher:
            r_star = form.distance_3d(self.coords)
        else:
            r_star = form.distance_2d(self.coords)
            
        indices = np.argsort(r_star)
        r_sorted = r_star[indices]
        mass_sorted = M_cur[indices]  # masses sorted by radius
        mass_sum = np.cumsum(mass_sorted)  # cumulative sum of sorted masses
        hmr = r_sorted[mass_sum <= tot_mass/2][-1]  # 2D/3D radius at half the mass
        return hmr
        
    def half_lum_radius(self, spher=False):
        """Returns the (spherical or cylindrical) half luminosity radius in pc/as."""
        lum = 10**self.log_luminosities(realistic_remnants=False)
        tot_lum = np.sum(lum)  # do this locally, to avoid unnecessary overhead (we already have lum anyway)
        
        if spher:
            r_star = form.distance_3d(self.coords)
        else:
            r_star = form.distance_2d(self.coords)
            
        indices = np.argsort(r_star)
        r_sorted = r_star[indices]
        lum_sorted = lum[indices]  # luminosities sorted by radius
        lum_sum = np.cumsum(lum_sorted)  # cumulative sum of sorted luminosities
        hlr = r_sorted[lum_sum <= tot_lum/2][-1]  # 2D/3D radius at half the luminosity
        return hlr

    def compact_parameters(self, mag_limit=None, d_lum=None, extinction=None):
        """Reduces the total amount of stars generated by decreasing the range of stellar initial masses."""
        # check if compact mode is on
        fraction_generated = np.ones_like(self.n_stars, dtype=float)  # set to ones initially
        mass_limit = np.copy(self.imf_param)  # set to imf params initially

        if not mag_limit:
            mag_limit = default_mag_lim  # if not specified, use default
        # todo: take away for loop (requires isochrones)
        for i, pop_num in enumerate(self.n_stars):
            if (self.compact_mode == 'mag'):
                mass_limit[i] = magnitude_limited(self.ages[i], self.metal[i], mag_lim=mag_limit,
                                                  distance=d_lum, ext=extinction, filter_name='Ks')
            else:
                mass_limit[i] = number_limited(self.n_stars, self.ages[i], self.metal[i], imf=self.imf_param[i])

            if (mass_limit[i, 1] > self.imf_param[i, 1]):
                # don't increase the upper limit!
                mass_limit[i, 1] = self.imf_param[i, 1]

            fraction_generated[i] = form.mass_fraction_from_limits(mass_limit[i], imf=self.imf_param[i])

        if np.any(mass_limit[:, 0] > mass_limit[:, 1]):
            # lower limit > upper limit!
            raise RuntimeError('compacting failed, mass limit raised above upper mass.')
        elif np.any(self.n_stars * fraction_generated < 10):
            # don't want too few stars
            raise RuntimeError('Population compacted to <10, try not compacting or '
                               'generating a higher number of stars.')
        return mass_limit, fraction_generated


class Gas():
    """"""
    def __init__(self, stuff):
        pass
    
    def __repr__(self):
        pass
    
    def __str__(self):
        pass
    
    def __add__(self):
        pass
    
    def add_inclination(self, incl):
        """Put the object at an inclination w.r.t. the observer, in radians. 
        The given angle is measured from the x-axis towards the z-axis (which is the l.o.s.).
        Can be specified for each cloud separately or for all at once.
        This is additive (given angles stack with previous ones).
        """
    
    def add_ellipticity(self, axes):
        """Give the object axes (cartesian) different relative proportions. 
        The given axes (a,b,c) are taken as relative sizes, the volume of the object is conserved.
        Can be specified for each cloud separately or for all at once.
        This is additive (given axis sizes stack with previous ones).
        """
    
    
class Dust():
    """"""
    def __init__(self, stuff):
        pass
    
    def __repr__(self):
        pass
    
    def __str__(self):
        pass
    
    def __add__(self):
        pass
    
    def add_inclination(self, incl):
        """Put the object at an inclination w.r.t. the observer, in radians. 
        The given angle is measured from the x-axis towards the z-axis (which is the l.o.s.).
        Can be specified for each cloud separately or for all at once.
        This is additive (given angles stack with previous ones).
        """
    
    def add_ellipticity(self, axes):
        """Give the object axes (cartesian) different relative proportions. 
        The given axes (a,b,c) are taken as relative sizes, the volume of the object is conserved.
        Can be specified for each cloud separately or for all at once.
        This is additive (given axis sizes stack with previous ones).
        """


class AstronomicalObject():
    """Base class for astronomical objects like clusters and galaxies.
    Contains the basic information and functionality. It is also a composite of the different
    component classes: Stars, Gas and Dust.
    
    Notes
    -----
    takes in all the kwargs necessary for the component classes (Stars, Gas and Dust)


    Arguments
    ---------
    distance
    d_type: distance type [l for luminosity, z for redshift]
    extinct: extinction between source and observer
    **kwargs: the kwargs are used in initiation of the component classes (see Stars, Gas and Dust)


    Attributes
    ----------
    redshift: redshift for the object
    d_lum: luminosity distance to the object (in pc)
    d_ang: angular distance for the object (in pc)
    AddFieldStars
    AddBackGround
    AddNGS
    SaveTo
    LoadFrom

    """
    def __init__(self, distance=10, d_type='l', extinct=0, **kwargs):
        # set some distance measures
        self.d_type = d_type
        if (self.d_type == 'z'):
            self.redshift = distance
            self.d_lum = form.d_luminosity(self.redshift)
        else:
            self.d_lum = distance
            self.redshift = form.d_luminosity_to_redshift(self.d_lum)

        self.d_ang = form.d_angular(self.redshift)
        self.extinction = extinct
        
        # initialise the component classes (pop the right kwargs per object)
        stars_args = [k for k, v in inspect.signature(Stars).parameters.items()]
        stars_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in stars_args}
        if stars_dict:
            # the stellar component
            self.stars = Stars(**stars_dict)
            self.stars.generate_stars()
        
        gas_args = [k for k, v in inspect.signature(Gas).parameters.items()]
        gas_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in gas_args}
        if gas_dict:
            # the gaseous component
            self.gas = Gas(**gas_dict)
        
        dust_args = [k for k, v in inspect.signature(Dust).parameters.items()]
        dust_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in dust_args}
        if dust_dict:
            # the dusty component
            self.dust = Dust(**dust_dict)
        
        super().__init__(**kwargs)
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

    def add_field_stars(self, n=10, fov=53, sky_coords=None):
        """Adds (Milky Way) field stars to the object in the given f.o.v (in as).
        [WIP] Nothing is done with sky_coords at the moment.
        """
        # self.field_stars = np.array([pos_x, pos_y, mag, filt, spec_types])
        # todo: WIP add_field_stars
        # add like a query to Gaia data, 
        # default: some random distribution in the shape of the fov
        fov_rad = fov*as_rad
        phi = dist.uniform(n=n, min_val=0, max_val=2*np.pi, power=1)
        theta = dist.angle_theta(n=n)
        radius = dist.uniform(n=n, min_val=10, max_val=10**3, power=3)  # uniform between 10 pc and a kpc
        coords = conv.spher_to_cart(radius, theta, phi)
        distance = self.d_lum - np.abs(coords[:,3])
        
        mag = dist.uniform(n=n, min_val=5, max_val=20, power=1)  # placeholder for magnitudes
        filt = np.full_like(mag, 'V', dtype='U5')  # placeholder for filter
        spec_types = np.full_like(mag, 'G0V', dtype='U5')  # placeholder for spectral types
        self.field_stars = np.array([coords[:,0], coords[:,1], mag, filt, spec_types])
        return
        
    def add_back_ground(self, n=10, fov=53):
        """Adds additional structures to the background, like more clusters or galaxies. 
        These will be unresolved. Give the imaging f.o.v (in as).
        """
        # self.back_ground = 
        # todo: WIP add_back_ground
        return
        
    def add_ngs(self, mag=13, filter_name='V', pos_x=None, pos_y=None, spec_types='G0V'):
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
            pos_x = np.array([7.1])  # x position in as
            pos_y = np.array([10.6])  # y position in as
        else:
            # assume we are using MCAO now. generate random positions within patrol field
            angle = dist.angle_phi(n=len(mag))
            radius = dist.uniform(n=len(mag), min_val=46, max_val=90, power=2)  # radius of patrol field in as
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
        # the guide stars can be in a different filter than the observations
        filters = np.full_like(mag, filter_name, dtype='U5')
        
        self.natural_guide_stars = [pos_x, pos_y, mag, filters, spec_types]
        return

    def plot_stars_2d(self, title='Scatter', xlabel='x (pc)', ylabel='y (pc)', axes='xy', colour='blue',
                      filter_name='V', theme='dark1', show=True):
        """Make a plot of the object positions in two dimensions
        Set colour to 'temperature' for a temperature representation.
        Set filter_name to None to avoid markers scaling in size with magnitude.
        Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but
            save-able dark plot, 'fits' for a plot that resembles a .fits image,
            and None for normal light colours.
        """
        if (filter_name is not None):
            mags = self.stars.apparent_magnitudes(self.d_lum, filters=filter_name)
        else:
            mags = None

        if (colour == 'temperature'):
            temps = 10**self.stars.log_temperatures()
        else:
            temps = None

        vis.scatter_2d(self.stars.coords, title=title, xlabel=xlabel, ylabel=ylabel, axes=axes, colour=colour,
                       T_eff=temps, mag=mags, theme=theme, show=show)
        return

    def plot_stars_3d(self, title='Scatter', xlabel='x (pc)', ylabel='y (pc)', zlabel='z (pc)', colour='blue',
                      filter_name='V', theme='dark1', show=True):
        """Make a plot of the object positions in three dimensions.
        Set colour to 'temperature' for a temperature representation.
        Set filter_name to None to avoid markers scaling in size with magnitude.
        Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but
            save-able dark plot, and None for normal light colours.
        """
        if (filter_name is not None):
            mags = self.stars.apparent_magnitudes(self.d_lum, filters=filter_name)
        else:
            mags = None

        if (colour == 'temperature'):
            temps = 10**self.stars.log_temperatures()
        else:
            temps = None

        vis.scatter_3d(self.stars.coords, title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                       colour=colour, T_eff=temps, mag=mags, theme=theme, show=show)
        return

    def plot_hrd(self, title='hr_diagram', colour='temperature', theme='dark1', show=True):
        """Make a plot of stars in an HR diagram.
        Set colour to 'temperature' for a temperature representation.
        Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but
            save-able dark plot, and None for normal light colours.
        Remnants are not plotted.
        """
        r_mask = np.invert(self.stars.remnants())
        temps = 10**self.stars.log_temperatures()
        lums = self.stars.log_luminosities()

        vis.hr_diagram(T_eff=temps, log_lum=lums, title=title, xlabel='Temperature (K)',
                       ylabel=r'Luminosity log($L/L_\odot$)', colour=colour, theme=theme, mask=r_mask, show=show)
        return

    def plot_cmd(self, x='B-V', y='V', title='cm_diagram', colour='blue', theme=None, show=True):
        """Make a plot of the stars in a cm_diagram
        Set x and y to the colour and magnitude to be used (x needs format 'A-B')
        Set colour to 'temperature' for a temperature representation.
        Remnants are not plotted.
        """
        x_filters = x.split('-')
        mag_A = self.stars.apparent_magnitudes(self.d_lum, filters=x_filters[0])
        mag_B = self.stars.apparent_magnitudes(self.d_lum, filters=x_filters[1])
        c_mag = mag_A - mag_B

        if (y == x_filters[0]):
            mag = mag_A
        elif (y == x_filters[1]):
            mag = mag_B
        else:
            mag = self.stars.apparent_magnitudes(self.d_lum, filters=y)

        r_mask = np.invert(self.stars.remnants())
        vis.cm_diagram(c_mag, mag, title=title, xlabel=x, ylabel=y, colour=colour, T_eff=None, theme=theme,
                       adapt_axes=True, mask=r_mask, show=show)
        return

    def quick_image(self, filter_name='V'):
        """"A quick look at what the image might look like."""
        # todo: use scipy.signal.convolve2d

    def save_to(self, filename):
        """Saves the class to a file."""
        if (filename[-4:] != '.pkl'):
            filename += '.pkl'
        
        if os.path.isdir('objects'):
            with open(os.path.join('objects', filename), 'wb') as output:
                pickle.dump(self, output, -1)
        else:
            # if for some reason the 'objects' folder isn't there
            with open(filename, 'wb') as output:
                pickle.dump(self, output, -1)
        return

    @staticmethod
    def load_from(filename):
        """Loads the class from a file."""
        if (filename[-4:] != '.pkl'):
            filename += '.pkl'
        
        if os.path.isdir('objects'):
            with open(os.path.join('objects', filename), 'rb') as input_file:
                data = pickle.load(input_file)
        else:
            # if for some reason the 'objects' folder isn't there
            with open(filename, 'rb') as input_file:
                data = pickle.load(input_file)
        return data


class StarCluster(AstronomicalObject):
    """For generating star clusters."""    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        return


class EllipticalGalaxy(AstronomicalObject):
    """For generating elliptical galaxies."""    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        return


class SpiralGalaxy(AstronomicalObject):
    """For generating spiral galaxies."""    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        return


def gen_spherical(n_stars, dist_types=default_rdist, kwargs_list=None):
    """Make a spherical distribution of stars using the given radial distribution type.
    Takes additional parameters for the r-distribution function (i.e. scale length s).

    n_stars: int or array of int
    dist_types: str or array of str
    kwargs_list: dict or array of dict

    out:
        2D array of float, shape [3, sum(n_stars)]
    """
    # check if the dist type exists (add the r to the end for radial version, if not already there)
    n_stars = np.atleast_1d(n_stars)
    dist_types = np.atleast_1d(dist_types)
    dist_list = list(set(fnmatch.filter(dir(dist), '*_r')))  # list of available distributions

    n_total = np.sum(n_stars)
    index = np.cumsum(np.append([0], n_stars))  # indices defining the different populations
    r_dist = np.zeros(n_total)  # the radial distribution
    for i, dist_type in enumerate(dist_types):
        if (dist_type[-2:] != '_r'):
            dist_types[i] += '_r'
    
        if (dist_type not in dist_list):
            warnings.warn(f'Specified distribution type \'{dist_type}\' does not exist. '
                          + f'Using default (={default_rdist})', UserWarning)
            dist_types = default_rdist + '_r'
    
        # check if right parameters given
        sig = inspect.signature(getattr(dist, dist_type))  # parameters of the dist function (includes n)
        key_dict = kwargs_list[i].copy()
        for key in key_dict:
            if key not in sig.parameters:
                warnings.warn(('Wrong keyword given in distribution parameters. Deleted entry.\n '
                               f'  {key} = {kwargs_list[i].pop(key, None)}'), UserWarning)
    
        r_dist[index[i]:index[i+1]] = getattr(dist, dist_type)(n=n_stars[i], **kwargs_list[i])
    phi_dist = dist.angle_phi(n_total)  # distribution for angle with x axis
    theta_dist = dist.angle_theta(n_total)  # distribution for angle with z axis
    return conv.spher_to_cart(r_dist, theta_dist, phi_dist)


def gen_spiral(n_stars):
    """Make a spiral galaxy."""
    # todo: add this
    

def gen_spiral_arms():
    """Make spiral arms."""
    

def gen_irregular(n_stars):
    """Make an irregular galaxy"""
    

def gen_star_masses(n_stars=0, M_tot=0, imf=None):
    """Generate masses using the Initial Mass Function. 
    Either number of stars or total mass should be given.
    imf defines the lower and upper bound to the masses generated in the IMF.
    Note: total generated mass will differ (slightly) from requested mass.

    n_stars: int or array-like of int
    M_tot: int or array-like of int
    imf: 1D or 2D array of float (in pairs of 2), optional

    out:
        array of float
    """
    n_stars = np.atleast_1d(n_stars)
    M_tot = np.atleast_1d(M_tot)
    if imf is None:
        length = max(len(n_stars), len(M_tot))
        imf = np.full([length, len(default_imf_par)], default_imf_par)
    else:
        imf = np.atleast_2d(imf)

    # check input (N_stars or M_tot?)
    if np.all(n_stars == 0) & np.all(M_tot == 0):
        raise ValueError('Input mass and number of stars cannot be zero simultaneously.')
    elif np.all(n_stars == 0):
        # a total mass is given, estimate the number of stars to generate
        n_stars = conv.m_tot_to_n_stars(M_tot, imf)
        
    # assign initial masses using IMF
    M_init = dist.kroupa_imf(n_stars, imf)
    return M_init


def star_form_history(max_age, min_age=1, sfr='exp', Z=0.014, tau=1e10):
    """Finds the relative number of stars to give each (logarithmic) age step
    up to a maximum given age (starting from a minimum age if desired).
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

    # determine if logarithm or not
    if np.all(max_age > 12):
        max_age = np.log10(max_age)
    if np.all(min_age > 12):
        min_age = np.log10(min_age)
    
    rel_num = np.array([])
    log_ages_used = np.array([])
    for i in range(len(max_age)):
        log_ages = np.unique(utils.open_isochrones_file(Z[i], columns=['log_age']))  # available ages
        uni_log_ages = np.unique(log_ages)
        # log t's to use (between min/max ages)
        log_ages_used_i = uni_log_ages[(uni_log_ages <= max_age[i]) & (uni_log_ages >= min_age[i])]
        ages_used = 10**log_ages_used_i  # age of each SSP
    
        if (sfr[i] == 'exp'):
            # relative star formation rates
            psi = np.exp((ages_used - 10**min_age[i]) / tau[i])
        elif(sfr[i] == 'lin-exp'):
            t0 = 10**np.max(uni_log_ages)  # represents the start of time
            # relative star formation rates
            psi = ((t0 - ages_used - 10**min_age[i]) / tau[i]*np.exp((ages_used - 10**min_age[i]) / tau[i]))
        else:
            # for when None is given
            log_ages_used_i = max_age[i]
            psi = 1

        # relative number of stars in each generation
        rel_num = np.append(rel_num, psi/np.sum(psi))
        log_ages_used = np.append(log_ages_used, log_ages_used_i)
    return rel_num, log_ages_used


def find_spectral_type(T_eff, lum, mass):
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
        spec_tbl_names = np.array(file.readline()[:-1].split('\t'))  # read in the column names
    spec_tbl_names = {name: i for i, name in enumerate(spec_tbl_names[1:])}  # make a dict for ease of use
    
    spec_tbl_types = np.loadtxt(os.path.join('tables', 'all_stars.txt'), dtype=str, usecols=[0], unpack=True)
    spec_tbl = np.loadtxt(os.path.join('tables', 'all_stars.txt'), dtype=float, usecols=[1,2,3,4,5,6,7,8], unpack=True)

    # basic spectral letters (increasing temperature)
    spec_letter = ['M', 'K', 'G', 'F', 'A', 'B', 'O']
    
    # collect the relevant spectral types 
    # (C, R, N, W, S and most D are disregarded, 
    # includes DC - WD with no strong spectral features, also leaves out sub-dwarfs)
    T_tbl, L_tbl, M_tbl, type_selection = [], [], [], []
    for i, type in enumerate(spec_tbl_types):
        if ((type[0] in spec_letter) | (type[:2] == 'DC')) & (type[2:] != 'VI'):
            type_selection.append(type)
            T_tbl.append(np.log10(spec_tbl[spec_tbl_names['Temp'], i]))
            # convert to L from Mbol (to avoid L=0 in tbl)
            L_tbl.append(0.4*(M_bol_sun - spec_tbl[spec_tbl_names['Mbol'], i]))
            M_tbl.append(np.log10(spec_tbl[spec_tbl_names['Mass'], i]))
    
    type_selection = np.array(type_selection)
    T_tbl = np.array(T_tbl)
    L_tbl = np.array(L_tbl)
    M_tbl = np.array(M_tbl)
    
    data_points = np.column_stack([T_eff, lum, mass])
    tbl_grid = np.column_stack([T_tbl, L_tbl, M_tbl])  # 3d data grid with points (T, L, M)
    tbl_tree = sps.cKDTree(tbl_grid)  # K-Dimensional lookup Tree
    dists, indices = tbl_tree.query(data_points, k=1)  # the distances to and indices of the closest points

    # correct for missing neighbours
    indices[dists == np.inf] = np.where(type_selection == 'DC9')[0][0]
    
    # WD are taken care of this way, but what to do with NS/BH? 
    # (now they would probably get WD spectra)
    return indices, type_selection


def number_limited(n, age, Z, imf=None):
    """Retrieves the lower mass limit for which the number of stars does not exceed 10**7. 
    Will also give an upper mass limit based on the values in the isochrone.
    The intended number of generated stars, age and metallicity are needed.
    """
    if imf is None:
        imf = default_imf_par

    # fraction of the total number of stars to generate
    fraction = np.clip(limiting_number / n, 0, 1)

    # get the isochrone values and find the highest value in the isochrone
    M_ini = utils.stellar_isochrone(age, Z, columns=['M_initial'])
    mass_lim_high = M_ini[-1]
    
    mass_lim_low = form.mass_limit_from_fraction(fraction, M_max=mass_lim_high, imf=imf)
    
    return mass_lim_low, mass_lim_high


def magnitude_limited(age, Z, mag_lim=default_mag_lim, distance=10, ext=0, filter_name='Ks'):
    """Retrieves the lower mass limit for which the given magnitude threshold is met.
    Will also give an upper mass limit based on the values in the isochrone.
    Works only for resolved stars. 
    If light is integrated along the line of sight (crowded fields), 
    then don't use this method! Resulting images would not be accurate!
    distance, age, metallicity and extinction of the population of stars are needed.
    A filter must be specified in which the given limiting magnitude is measured.
    """
    # get the isochrone values
    iso_M_ini = utils.stellar_isochrone(age, Z, columns=['M_initial'])
    mag_vals = utils.stellar_isochrone(age, Z, columns=[filter_name])

    # calculate the limiting absolute magnitude and take all mag_vals below the limit
    abs_mag_lim = form.absolute_magnitude(mag_lim, distance, ext=ext)
    mask = (mag_vals < abs_mag_lim + 0.1)
    if not mask.any():
        # if limit too high (too low in mag) then it will break
        mask = (mag_vals == np.min(mag_vals))
        warnings.warn(f'Compacting will not work, distance ({distance} pc) too large.', RuntimeWarning)

    # now take the lowest mass where mask==True and the highest mass in the isochrone
    mass_lim_low = np.min(iso_M_ini[mask])
    mass_lim_high = np.max(iso_M_ini)
    return mass_lim_low, mass_lim_high


def remnants_single_pop(M_init, age, Z):
    """Gives the positions of the remnants in a single population (as a boolean mask)."""
    iso_M_ini = utils.stellar_isochrone(age, Z, columns=['M_initial'])
    # maximum initial mass in isoc file
    max_mass = np.max(iso_M_ini)
    return (M_init > max_mass)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


