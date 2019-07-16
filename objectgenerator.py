# Luc IJspeert
# Part of starpopsim: (astronomical) object generator and class
##
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
from inspect import signature

import utils
import distributions as dist
import conversions as conv
import formulas as form
import visualizer as vis


# global constants
M_bol_sun = 4.74                # bolometric magnitude of the sun
as_rad = np.pi/648000           # arcseconds to radians


# global defaults
rdist_default = 'Normal'        # see distributions module for a full list of options
imf_defaults = [0.08, 150]      # lower bound, upper bound on mass
default_mag_lim = 32            # magnitude limit for compact mode
limiting_number = 10**7         # used in compact mode as maximum number of stars


class AstObject:
    """Generates the astronomical object and contains all the information about the object.
    Also functions that can be performed on the object are defined here.
    This is the base class for all other objects.
    """
    structure = 'astronomical object'                                                               # type of object
    
    def __init__(self, N_stars=0, M_tot_init=0, age=None, metal=None, rel_num=None, 
                 distance=10, d_type='l', extinct=0, sf_hist=None, imf_par=None, 
                 compact=False, cp_mode='num', mag_lim=None, 
                 ):
        
        if (age is None):
            age = []                                                                                # something with mutable defaults
        if (metal is None):
            metal = []
        if (rel_num is None):
            rel_num = [1]
        if (imf_par is None):
            imf_par = imf_defaults
        if (sf_hist is None):
            sf_hist = ['none']
        
        self.N_stars = N_stars                                                                      # number of stars
        self.M_tot_init = M_tot_init                                                                # total initial mass in Msun
        self.ages = age                                                                             # ages of the populations (=max age if SFH is used)
        self.metal = metal                                                                          # metallicity (of each population)
        self.rel_number = rel_num                                                                   # relative number of stars in each population (equal if left empty)
        
        self.d_type = d_type                                                                        # distance type [l for luminosity, z for redshift]
        #todo: add actual redshifts
        if (self.d_type == 'z'):
            self.redshift = distance                                                                # redshift for the object
            self.d_lum = form.DLuminosity(self.redshift)
        else:
            self.d_lum = distance                                                                   # distance to the object (in pc)
            self.redshift = form.DLToRedshift(self.d_lum)
        
        self.d_ang = form.DAngular(self.redshift)
        
        # optional parameters
        self.imf_param = imf_par                                                                    # lower bound, knee position, upper bound for the IMF masses
        self.sfhist = sf_hist                                                                       # star formation history type
        self.extinction = extinct                                                                   # extinction between source and observer
        
        # properties that are derived/generated
        self.pop_number = np.array([])                                                              # number of stars in each population
        self.coords = np.empty([0,3])                                                               # spatial coordinates
        self.M_init = np.array([])                                                                  # the masses of the stars
        self.M_diff = 0                                                                             # mass difference between given and generated (total) mass (if given)
        
        self.spec_names = np.array([])                                                              # corresponding spectral type names
        self.mag_names = np.array([])                                                               # names of the filters corresponding to the magnitudes
        
        # compact mode parameters
        self.compact = compact  	                                                                # (compact mode) if True, generates only high mass stars based on limiting mag
        self.compact_mode = cp_mode                                                                 # (compact mode) mode of compacting. num=number limited, mag=magnitude limited
        self.mag_limit = mag_lim                                                                    # (compact mode) limiting magnitude, used only for compact mode
        self.fraction_generated = 1                                                                 # (compact mode) part of the total number of stars that has actually been generated for each population
        self.gen_imf_param = np.array([])                                                           # (compact mode) the imf parameters to use for each population (limits imposed by compacting)
        self.gen_pop_number = np.array([])                                                          # (compact mode) actually generated number of stars per population
        
        self.CheckInput()                                                                           # check user input
        # the stars are not yet generated. This can be done manually.
        # It is done automatically in the subclasses.
        return
        
    def CheckInput(self):
        """Checks the given input for compatibility."""
        # check metallicity and age and make sure they are arrays
        if hasattr(self.ages, '__len__'):
            self.ages = np.array(self.ages)
        else:
            self.ages = np.array([self.ages])
            
        if hasattr(self.metal, '__len__'):
            self.metal = np.array(self.metal)
        else:
            self.metal = np.array([self.metal])
        
        # much used qtt's
        num_ages = len(self.ages)
        num_metal = len(self.metal)
        
        # check (ages and metallicity) for empty input and compatible length
        if (num_ages == 0):
            raise ValueError('objectgenerator//CheckInput: No age was defined.')
        elif (num_metal == 0):
            raise ValueError('objectgenerator//CheckInput: No metallicity was defined.')
        elif (num_ages != num_metal):                                                               # make sure they have the right length
            if (num_ages == 1):
                self.ages = self.ages[0]*np.ones(num_metal)
            elif (num_metal == 1):
                self.metal = self.metal[0]*np.ones(num_ages)
            else:
                warnings.warn(('objectgenerator//CheckInput: age and metallicity have '
                               'incompatible length {0} and {1}. Discarding excess.'
                               ).format(num_ages, num_metal), SyntaxWarning)
                new_len = min(num_ages, num_metal)
                self.ages = self.ages[:new_len]
                self.metal = self.metal[:new_len]
                
            num_ages = len(self.ages)                                                               # update length
            num_metal = len(self.metal)                                                             # update length
        num_pop = num_ages                                                                          # define number of stellar populations
        
        # check input for rel_num [must go before usage of num_pop, after ages and metallicity]
        if hasattr(self.rel_number, '__len__'):
            self.rel_number = np.array(self.rel_number)
        else:
            self.rel_number = np.ones(num_pop)                                                      # any single number will result in equal amounts
        
        relnum_len = len(self.rel_number)
        
        if ((relnum_len > num_pop) & (num_pop != 1)):
            warnings.warn(('objectgenerator//CheckInput: too many relative numbers given. '
                           'Discarding excess.'), SyntaxWarning)
            self.rel_number = self.rel_number[:num_pop]
        elif ((relnum_len > num_pop) & (num_pop == 1)):
            self.ages = np.array([self.ages[0] for i in range(relnum_len)])                         # in this case, add populations of the same age
            self.metal = np.array([self.metal[0] for i in range(relnum_len)])                       #  and metallicity
            num_ages = len(self.ages)                                                               # update length
            num_metal = len(self.metal)                                                             # update length
            num_pop = num_ages                                                                      # [very important] update number of populations
        elif (relnum_len != num_pop):
            self.rel_number = np.ones(num_pop)
        
        rel_frac = self.rel_number/np.sum(self.rel_number)                                          # fraction of the total in each population
        
        # check format of imf_param
        if hasattr(self.imf_param, '__len__'):
            self.imf_param = np.array(self.imf_param)
        else:
            warnings.warn(('objectgenerator//CheckInput: incorrect input type for imf_par, '
                           'using default (={0}).').format(imf_defaults), SyntaxWarning)
            self.imf_param = np.array([imf_defaults for i in range(num_pop)]) 
        
        imf_shape = np.shape(self.imf_param)
        imf_par_len = len(imf_defaults)                                                             # how long one set of imf pars is
        
        if (len(imf_shape) == 1):
            if (imf_shape[0] == imf_par_len):
                self.imf_param = np.array([self.imf_param for i in range(num_pop)])                 # make it a 2D array using same imf for all populations
            elif (imf_shape[0]%imf_par_len == 0):
                self.imf_param = np.reshape(self.imf_param, 
                                            [imf_shape[0]//imf_par_len, imf_par_len])               # make it a 2D array
            else:
                raise ValueError('objectgenerator//CheckInput: Wrong number of arguments for '
                                 'imf_par, must be multiple of {0}.'.format(imf_par_len))
        
        imf_shape = np.shape(self.imf_param)                                                        # update shape
        
        if (imf_shape[0] > num_pop):
            warnings.warn(('objectgenerator//CheckInput: Too many arguments for imf_par. '
                           'Discarding excess.'), SyntaxWarning)
            self.imf_param = self.imf_param[0:num_pop]
        elif (imf_shape[0] == 1) & (num_pop > 1):
            self.imf_param = np.full([num_pop, imf_par_len], self.imf_param[0])                     # fill up imf_par  
        elif (imf_shape[0] < num_pop):
            filler = [imf_defaults for i in range(num_pop - imf_shape[0]//imf_par_len)]
            self.imf_param = np.append(self.imf_param, filler, axis=0)                              # fill missing imf_par with default        
        
        # check the minimum available mass in isoc file [must go after imf_param check]
        max_M_L = 0                                                                                 # maximum lowest mass (to use in IMF)
        for i in range(num_pop):
            M_ini = utils.OpenIsochrone(self.ages[i], self.metal[i], columns='mini')
            max_M_L = max(max_M_L, np.min(M_ini))
        
        imf_max_M_L = np.array([self.imf_param[:,0], np.full(num_pop, max_M_L)])
        self.imf_param[:,0] =  np.max(imf_max_M_L, axis=0)                                          # check against user input (if that was higher, use that instead)
        
        # check input: N_stars or M_tot_init? --> need N_stars [must go after min mass check]
        if (self.N_stars == 0) & (self.M_tot_init == 0):
            raise ValueError('objectgenerator//CheckInput: Input mass and number of stars '
                             'cannot be zero simultaniously. Using N_stars=1000')
        elif (self.N_stars == 0):                                                                   # a total mass is given
            pop_num = conv.MtotToNstars(self.M_tot_init*rel_frac, imf=self.imf_param)
            self.N_stars = np.sum(pop_num)                                                          # estimate of the number of stars to generate
        else:
            pop_num = np.rint(rel_frac*self.N_stars).astype(int)                                    # rounded off number
            self.N_stars = np.rint(np.sum(pop_num)).astype(int)                                     # make sure N_stars is int and rounded off
        
        self.pop_number = utils.FixTotal(self.N_stars, pop_num)                                     # make sure the population numbers add up to N_total
        
        # check the SFH
        if isinstance(self.sfhist, str):
            self.sfhist = np.array([self.sfhist])
        elif hasattr(self.sfhist, '__len__'):
            self.sfhist = np.array(self.sfhist)
        
        sfh_len = len(self.sfhist)
        if ((sfh_len == 1) & (num_pop != 1)):
            self.sfhist = np.full(num_pop, self.sfhist[0])
        elif (sfh_len < num_pop):
            raise ValueError('objectgenerator//CheckInput: too few sfh types given.')
        elif (sfh_len > num_pop):
            warnings.warn(('objectgenerator//CheckInput: too many sfh types given. '
                           'Discarding excess.'), SyntaxWarning)
            self.sfhist = self.sfhist[:num_pop]
        
        # check if compact mode is on
        self.fraction_generated = np.ones_like(self.pop_number, dtype=float)                        # set to ones initially
        mass_limit = np.copy(self.imf_param[:])                                                     # set to imf params initially
        
        if self.compact:
            if (self.mag_limit is None):
                self.mag_limit = default_mag_lim                                                    # if not specified, use default
            
            for i, pop_num in enumerate(self.pop_number):
                if (self.compact_mode == 'mag'):
                    mass_limit[i] = MagnitudeLimited(self.ages[i], self.metal[i], 
                                                     mag_lim=self.mag_limit, d=self.d_lum, 
                                                     ext=self.extinction, filter='Ks')
                else:
                    mass_limit[i] = NumberLimited(self.N_stars, self.ages[i], 
                                                  self.metal[i], imf=self.imf_param[i])
                
                if (mass_limit[i, 1] > self.imf_param[i, 1]):
                    mass_limit[i, 1] = self.imf_param[i, 1]                                         # don't increase the upper limit!
                    
                if (mass_limit[i, 0] > mass_limit[i, 1]):                                       
                    raise RuntimeError('objectgenerator//GenerateStars: compacting failed, '
                                        'mass limit raised above upper mass.')                      # lower limit > upper limit!
                
                self.fraction_generated[i] = form.MassFraction(mass_limit[i], 
                                                               imf=self.imf_param[i])
        
            if np.any(self.pop_number*self.fraction_generated < 10):                                # don't want too few stars
                raise RuntimeError('objectgenerator//GenerateStars: population compacted to <10, '
                                   'try not compacting or generating a higher number of stars.')
        
        # assign the right values for generation
        if self.compact:
            self.gen_pop_number = np.rint(self.pop_number*self.fraction_generated).astype(int)
            self.gen_imf_param = mass_limit
        else:
            self.gen_pop_number = self.pop_number
            self.gen_imf_param = self.imf_param
        
        return
    
    def CheckRadialDistribution(self):
        """Check the radial distribution input."""
        # check if the dist type(s) exists and get the function signatures
        num_pop = len(self.pop_number)
        dist_list = list(set(fnmatch.filter(dir(dist), '*_r')))
        
        if isinstance(self.r_dist_type, str):
            self.r_dist_type = [self.r_dist_type]                                                   # make sure it is a list of str
        
        # check number of dists    
        n_r_dists = len(self.r_dist_type)
        if ((n_r_dists == 1) & (num_pop > 1)):
            self.r_dist_type.extend([self.r_dist_type[0] for i in range(num_pop - n_r_dists)])      # duplicate
            n_r_dists = len(self.r_dist_type)                                                       # update the number
        elif (n_r_dists < num_pop):                                                                 
            self.r_dist_type.extend([rdist_default for i in range(num_pop - n_r_dists)])            # fill up with default
            n_r_dists = len(self.r_dist_type)                                                       # update the number
            
        key_list = []
        val_list = []
        r_dist_n_par = []
        
        for i in range(n_r_dists):
            if (self.r_dist_type[i][-2:] != '_r'):
                self.r_dist_type[i] += '_r'                                                         # add the r to the end for radial version
            
            if (self.r_dist_type[i] not in dist_list):
                warnings.warn(('objectgenerator//CheckInput: Specified distribution <{0}> type '
                               'does not exist. Using default (={1})'
                               ).format(self.r_dist_type[i], rdist_default), SyntaxWarning)
                self.r_dist_type[i] = rdist_default
                 
            sig = inspect.signature(eval('dist.' + self.r_dist_type[i]))
            key_list.append([k for k, v in sig.parameters.items() if k is not 'n'])                 # add the signature keywords to a list
            val_list.append([v.default for k, v in sig.parameters.items() if k is not 'n'])         # add the signature defaults to a list
            r_dist_n_par.append(len(key_list[i]))                                                   # the number of parameters for each function
            
        # check if dist parameters are correctly specified
        if isinstance(self.r_dist_param, dict):                                                     # if just one dict, make a list of (one) dict
            self.r_dist_param = [self.r_dist_param]
        elif not hasattr(self.r_dist_param, '__len__'):                                             # if just one parameter is given, also make a list of (one) dict
            self.r_dist_param = [{key_list[0][0]: self.r_dist_param}]
        elif isinstance(self.r_dist_param, list):
            param_shape = np.shape(self.r_dist_param)
            if np.all([isinstance(item, (int, float)) for item in self.r_dist_param]):              # if a 1D list of parameters is given, fill a 2D list that has the correct form
                temp_par_list = [[]]
                track_index = np.cumsum(r_dist_n_par)
                j = 0
                
                for i in range(param_shape[0]):
                    if np.any(i == track_index):
                        temp_par_list.append([self.r_dist_param[i]])                                # append a new sublist
                        j +=1                                                                       # keep track of number of sublists
                    else:
                        temp_par_list[j].append(self.r_dist_param[i])                               # append to current sublist
                
                self.r_dist_param = temp_par_list
            elif np.all([isinstance(item, (int, float)) for item in self.r_dist_param]):            # if list with numbers combined, do not want that, fix!
                temp_par_list = []
                
                for i in range(param_shape[0]):
                    if isinstance(self.r_dist_param[i], (int, float)):
                        temp_par_list.append([self.r_dist_param[i]])
                    else:
                        temp_par_list.append(self.r_dist_param[i])
                
                self.r_dist_param = temp_par_list
            
            param_shape = np.shape(self.r_dist_param)                                               # recalculate it
            
            if (np.all([isinstance(item, list) for item in self.r_dist_param])):                    # if a 2D list is given (or made above), check further compatibility
                if (param_shape[0] > n_r_dists):
                    warnings.warn(('objectgenerator//CheckInput: Too many radial distribution '
                                   'parameters given. Discarding excess.'), SyntaxWarning)
                    self.r_dist_param = self.r_dist_param[0:n_r_dists]
                elif (param_shape[0] < n_r_dists):                                                  # fill up missing length with defaults
                    filler = [[val_list[param_shape[0] + i]] 
                              for i in range(n_r_dists - param_shape[0])]
                    self.r_dist_param += filler
                
                for i, param in enumerate(self.r_dist_param):
                    if (len(param) < r_dist_n_par[i]):
                        self.r_dist_param[i].extend([item for item in val_list[i][len(param):]])    # not enough parameters for a particular distribution
                
                
                temp_par_dict_list = []                                                             # now it is ready finally for making a dict out of it
                for i in range(n_r_dists):
                    temp_par_dict_list.append({key_list[i][k]: self.r_dist_param[i][k] 
                                               for k in range(r_dist_n_par[i])})
                    
                self.r_dist_param = temp_par_dict_list
                    
        else:
            raise TypeError('objectgenerator//CheckInput: Incompatible data type for rdistpar')     # it is something else... burn it with fire!
            
        for i, param_dict in enumerate(self.r_dist_param):
            if not bool(param_dict):                                                                # if dict empty, fill with defaults
                self.r_dist_param[i] = {key_list[i][k]: val_list[i][k] 
                                        for k in range(r_dist_n_par[i])}
        
        n_r_param = len(self.r_dist_param)
        if (n_r_param < num_pop):                                                                   # check parameter dict number
            self.r_dist_param.extend([{key_list[i][k]: val_list[i][k] 
                                       for k in range(r_dist_n_par[i])} 
                                       for i in range(n_r_param, num_pop)])
            n_r_param = len(self.r_dist_type)
        
        return
    
    def GenerateStars(self):
        """Generate the masses and positions of the stars."""
        # generate the positions, masses   
        for i, pop_num in enumerate(self.gen_pop_number):
            coords_i = Spherical(pop_num)
            M_init_i, M_diff_i = StarMasses(pop_num, 0, imf=self.gen_imf_param[i])
            self.coords = np.append(self.coords, coords_i, axis=0)                           
            self.M_init = np.append(self.M_init, M_init_i)
            self.M_diff += M_diff_i                                                                 # StarMasses already gives diff in Mass (=estimate since no mass was given)
        
        # if only N_stars was given, set M_tot_init to the total generated mass
        mass_generated = np.sum(self.M_init)
        if (self.M_tot_init == 0):
            self.M_tot_init = mass_generated
        else:
            self.M_diff = mass_generated - self.M_tot_init                                          # if negative: too little mass generated, else too much
            self.M_tot_init = mass_generated                                                        # set to actual initial mass
        
        # the filter names of the corresponding magnitudes
        self.mag_names = utils.OpenIsochrone(self.ages[0], self.metal[0], columns='filters')
        
        return
        
    def AddFieldStars(self, n=10, fov=53, sky_coords=None):
        """Adds (Milky Way) field stars to the object in the given f.o.v (in as).
        [WIP] Nothing is done with sky_coords at the moment.
        """
        # self.field_stars = np.array([pos_x, pos_y, mag, filt, spec_types])
        # todo: WIP
        # add like a query to Gaia data, 
        # default: some random distribution in the shape of the fov
        fov_rad = fov*as_rad
        phi = dist.Uniform(n=n, min=0, max=2*np.pi, power=1)
        theta = dist.AngleTheta(n=n)
        radius = dist.Uniform(n=n, min=10, max=10**3, power=3)                                      # uniform between 10 pc and a kpc
        coords = conv.SpherToCart(radius, theta, phi)
        distance = self.d_lum - np.abs(coords[:,3])
        
        mag = dist.Uniform(n=n, min=5, max=20, power=1)                                             # placeholder for magnitudes
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
        
    def AddNGS(self, mag=[13], filter='V', pos_x=None, pos_y=None, spec_types='G0V'):
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
            angle = dist.AnglePhi(n=len(mag))
            radius = dist.Uniform(n=len(mag), min=46, max=90, power=2)                              # radius of patrol field in as
            pos_x_y = conv.PolToCart(radius, angle)
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
        
    def AddInclination(self, incl):
        """Put the object at an inclination w.r.t. the observer, in radians. 
        The given angle is measured from the x-axis towards the z-axis (which is the l.o.s.).
        Can be specified for each population separately or for all at once.
        This is additive (given angles stack with previous ones).
        """
        # check the input, make sure it is an array of the right size
        num_pop = len(self.pop_number)
        
        if hasattr(incl, '__len__'):
            incl = np.array(incl)
            incl_len = len(incl)
            
            if (incl_len == 1):
                incl = np.full(num_pop, incl[0])
            elif (incl_len < num_pop):
                raise ValueError('objectgenerator//AddInclination: too few values given.')
            elif (incl_len > num_pop):
                warnings.warn(('objectgenerator//AddInclination: too many values given. '
                               'Discarding excess.'), SyntaxWarning)
                incl = incl[:num_pop]
        else:
            incl = np.full(num_pop, incl)
        
        # check for existing inclination
        if hasattr(self, 'inclination'):
            self.inclination += incl
        else:
            self.inclination = incl
            
        if np.any(self.inclination > 2*np.pi):
            warnings.warn(('objectgenerator//AddInclination: inclination angle over 2pi detected, '
                           'make sure to use radians!'))
            self.inclination = np.mod(self.inclination, 2*np.pi)
        
        # rotate the XZ plane (x axis towards positive z)
        if np.all(self.inclination == 0):
            return                                                                                  # don't need to do anything
        elif (len(np.unique(self.inclination)) == 1):
            self.coords = conv.RotateXZ(self.coords, self.inclination[0])
        else:
            index = np.cumsum(np.append([0], self.gen_pop_number))                                  # indices defining the different populations
            for i, incl_i in enumerate(self.inclination):
                ind_1, i_2 = index[i], index[i+1]
                self.coords[ind_1:ind_2] = conv.RotateXZ(self.coords[ind_1:ind_2], incl_i)
        
        # if hasattr(self, 'gas')
        #todo: gas
        # add something to rotate gas and dust
        # (only when one single value was given (or when they are all the same?))
        return
        
    def AddEllipticity(self, axes):
        """Give the object axes (cartesian) different relative proportions. 
        The given axes (a,b,c) are taken as relative sizes, the volume of the object is conserved.
        Can be specified for each population separately or for all at once.
        This is additive (given axis sizes stack with previous ones).
        """
        # check the input, make sure it is an array of the right size
        num_pop = len(self.pop_number)
        
        if hasattr(axes, '__len__'):
            axes = np.array(axes)
        else:
            axes = np.ones([num_pop, 3])                                                            # any single number will result in unitary scaling
            
        axes_shape = np.shape(axes)
        if (len(axes_shape) == 1): 
            if (axes_shape[0] == 3):
                axes = np.full([num_pop, 3], axes)                                                  # if 1D make 2D, using same axes for all populations
            elif (axes_shape[0]%3 == 0):
                axes = np.reshape(axes, [axes_shape[0]//3, 3])                                      # make it a 2D array
            else:
                raise ValueError('objectgenerator//AddEllipticity: wrong number of arguments '
                               'for axes, must be multiple of 3.')
        
        axes_shape = np.shape(axes)                                                                 # update shape
                                       
        if (axes_shape[0] > num_pop):
            warnings.warn(('objectgenerator//CheckEllipseAxes: too many arguments for axes. '
                           'Discarding excess.'), SyntaxWarning)
            axes = axes[0:num_pop]
        elif (axes_shape[0] == 1) & (num_pop > 1):
            axes = np.full([num_pop, 3], axes[0])                                                   # duplicate to right shape  
        elif (axes_shape[0] < num_pop):
            filler = np.ones([num_pop - axes_shape[0], 3])
            axes = np.append(axes, filler, axis=0)                                                  # fill up with defaults
            
        axes_shape = np.shape(axes)                                                                 # update shape again
        
        # check for existing ellipse axes
        if hasattr(self, 'ellipse_axes'):
            self.ellipse_axes *= axes
        else:
            self.ellipse_axes = axes
        
        if (len(np.unique(self.ellipse_axes)) == 1):
            return                                                                                  # no relative change is made
        elif (axes_shape[0] == 1):
            self.coords = self.coords*axes/np.prod(axes)**(1/3)                                     # convert to ellipsoid (keeps volume conserved)
        else:
            index = np.cumsum(np.append([0], self.gen_pop_number))                                  # indices defining the different populations
            for i, axes_i in enumerate(self.ellipse_axes):
                ind_1, ind_2 = index[i], index[i+1]
                self.coords[ind_1:ind_2] = self.coords[ind_1:ind_2]*axes_i/np.prod(axes_i)**(1/3)
        
        # if hasattr(self, 'gas')
        #todo: gas
        # add something to deform gas and dust
        # (only when one single value was given (or when they are all the same?))
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
            index = np.cumsum(np.append([0], self.gen_pop_number))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_M_act = utils.OpenIsochrone(age, self.metal[i], columns='mcur')      # get the isochrone values
                M_init_i = self.M_init[index[i]:index[i+1]]
                M_cur_i = np.interp(M_init_i, iso_M_ini, iso_M_act, right=0)                        # (right) return 0 for stars heavier than boundary in isoc file (dead stars) [may change later in the program]
                
                # Isochrones assign wildly wrong properties to remnants (i.e. M_cur=0).
                # The following functions give estimates for remnant masses
                if realistic_remnants:
                    remnants_i = RemnantsSinglePop(M_init_i, age, self.metal[i])
                    r_M_cur_i = form.RemnantMass(M_init_i[remnants_i], self.metal[i])               # approx. remnant masses (depend on Z)
                    
                    M_cur_i[remnants_i] = r_M_cur_i
                M_cur = np.append(M_cur, M_cur_i)  
        
        return M_cur
        
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
            index = np.cumsum(np.append([0], self.gen_pop_number))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_log_L = utils.OpenIsochrone(age, self.metal[i], columns='lum')       # get the isochrone values
                M_init_i = self.M_init[index[i]:index[i+1]]
                log_L_i = np.interp(M_init_i, iso_M_ini, iso_log_L, right=-9)                       # (right) return -9 --> L = 10**-9 Lsun     [subject to change]
                
                # Isochrones assign wildly wrong properties to remnants (i.e. log_L=-9).
                # The following functions give estimates for remnant luminosities
                if realistic_remnants:
                    remnants_i = RemnantsSinglePop(M_init_i, age, self.metal[i])
                    
                    if (age <= 12):                                                                 # determine if stellar age in logarithm or not
                        lin_age = 10**age                                                           # if so, go back to linear
                    else:
                        lin_age = age
                    
                    lifetime = form.MSLifetime(M_init_i[remnants_i])                                # estimated MS time of the stars
                    remnant_time = lin_age - lifetime                                               # approx. time that the remnant had to cool
                    mask = (remnant_time < 0)                                                       # overestimated
                    remnant_time[mask] = -remnant_time[mask]/10                                     # hotfix
                    #TODO: remnant time is bad estimator (becomes negative)
                    
                    r_M_cur_i = form.RemnantMass(M_init_i[remnants_i], self.metal[i])               # approx. remnant masses
                    r_radii = form.RemnantRadius(r_M_cur_i)                                         # approx. remnant radii
                    r_Te_i = form.RemnantTeff(r_M_cur_i, r_radii, remnant_time)                     # approx. remnant temperatures
                    r_log_L_i = np.log10(form.BBLuminosity(r_radii, r_Te_i))                        # remnant luminosity := BB radiation
                    
                    log_L_i[remnants_i] = r_log_L_i
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
            index = np.cumsum(np.append([0], self.gen_pop_number))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_log_Te = utils.OpenIsochrone(age, self.metal[i], columns='temp')     # get the isochrone values
                M_init_i = self.M_init[index[i]:index[i+1]]
                log_Te_i = np.interp(M_init_i, iso_M_ini, iso_log_Te, right=1)                      # (right) return 1 --> Te = 10 K            [subject to change]
                
                # Isochrones assign wildly wrong properties to remnants (i.e. Te=10).
                # The following functions give estimates for remnant temperatures
                if realistic_remnants:
                    remnants_i = RemnantsSinglePop(M_init_i, age, self.metal[i])
                    
                    if (age <= 12):                                                                 # determine if stellar age in logarithm or not
                        lin_age = 10**age                                                           # if so, go back to linear
                    else:
                        lin_age = age
                    
                    lifetime = form.MSLifetime(M_init_i[remnants_i])                                # estimated MS time of the stars
                    remnant_time = lin_age - lifetime                                               # approx. time that the remnant had to cool
                    mask = (remnant_time < 0)                                                       # overestimated
                    remnant_time[mask] = -remnant_time[mask]/10                                     # hotfix
                    #TODO: remnant time is bad estimator (becomes negative)
                    
                    r_M_cur_i = form.RemnantMass(M_init_i[remnants_i], self.metal[i])               # approx. remnant masses
                    r_radii = form.RemnantRadius(r_M_cur_i)                                         # approx. remnant radii
                    r_log_Te_i = np.log10(form.RemnantTeff(r_M_cur_i, r_radii, remnant_time))       # approx. remnant temperatures
                    
                    log_Te_i[remnants_i] = r_log_Te_i
                log_Te = np.append(log_Te, log_Te_i)  
        
        return log_Te
        
    def AbsoluteMagnitudes(self, filter='all'):
        """Gives the absolute magnitudes of the stars.
        Uses isochrone files and the given initial masses of the stars.
        Stars should not have a lower initial mass than the lowest mass in the isochrone file.
        """
        if hasattr(self, 'absolute_magnitudes'):
            abs_mag = self.absolute_magnitudes
        else:
            if (filter == 'all'):
                num_mags = len(self.mag_names)
                mask = np.full_like(self.mag_names, True, dtype=bool)
            else:
                num_mags = 1
                mask = (self.mag_names == filter)
                
            abs_mag = np.empty([num_mags, 0])
            index = np.cumsum(np.append([0], self.gen_pop_number))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                # use the isochrone files to interpolate properties
                iso_M_ini, iso_mag = utils.OpenIsochrone(age, self.metal[i], columns='mag')         # get the isochrone values
                M_init_i = self.M_init[index[i]:index[i+1]]                                         # select the masses of one population
                
                mag_i = np.zeros([num_mags, len(M_init_i)])
                for j, mag in enumerate(iso_mag[mask]):
                    mag_i[j] = np.interp(M_init_i, iso_M_ini, mag, left=30, right=30)               # (right) return 30 --> L of less than 10**-9
                
                # Isochrones assign wildly wrong properties to remnants (i.e. mag=30).
                # The following functions give estimates for remnant magnitudes
                remnants_i = RemnantsSinglePop(M_init_i, age, self.metal[i])
                r_mag_i = 11          # ~typical WD
                #TODO: add better remnant magnitudes
                
                mag_i[:, remnants_i] = r_mag_i
                abs_mag = np.append(abs_mag, mag_i, axis=1)
                
            if (filter != 'all'):
                abs_mag = abs_mag[0]                                                                # correct for 2D array
        
        return abs_mag
    
    def ApparentMagnitudes(self, filter='all'):
        """Compute the apparent magnitude from the absolute magnitude 
        and the individual distances (in pc!). 
        The filter can be specified. 'all' will give all the filters.
        """
        if hasattr(self, 'apparent_magnitudes'):
            app_mag = self.apparent_magnitudes
        else:
            if (filter == 'all'):
                num_mags = len(self.mag_names)
                mask = np.full_like(self.mag_names, True, dtype=bool)
            else:
                num_mags = 1
                mask = (self.mag_names == filter)
            
            delta_d = -self.coords[:,2]                                                             # delta distances relative to distance (taken to be minus the z-coordinate)
            if (self.d_lum > 100*np.abs(np.min(delta_d))):                                          # distance 'much larger' than individual variations
                true_dist = self.d_lum + delta_d                                                    # approximate the individual distance to each star with delta_d
            else:
                delta_d = form.Distance(self.coords, np.array([0, 0, self.d_lum]))                  # distances to each star
                true_dist = delta_d                                                                 # distance is now properly calculated for each star
            
            dimension_2 = np.sum(np.rint(self.pop_number*self.fraction_generated)).astype(int)
            true_dist = np.tile(true_dist, num_mags).reshape(num_mags, dimension_2)
            
            abs_mag = self.AbsoluteMagnitudes(filter=filter)
            if (filter != 'all'):
                true_dist = true_dist[0]                                                            # correct for 2D array
            
            app_mag = form.ApparentMag(abs_mag, true_dist, ext=self.extinction)                     # true_dist in pc!
        
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
            
            spec_indices, spec_names = FindSpectralType(log_T_eff, log_L, log_M_cur)                # assign spectra to the stars
            
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
            index = np.cumsum(np.append([0], self.gen_pop_number))                                  # indices defining the different populations
            
            for i, age in enumerate(self.ages):
                iso_M_ini = utils.OpenIsochrone(age, self.metal[i], columns='mini')
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
        
    def CoordsArcsec(self):
        """Returns coordinates converted to arcseconds (from pc)."""
        return conv.ParsecToArcsec(self.coords, self.d_ang)
        
    def Radii(self, unit='pc', spher=False):
        """Returns the radial coordinate of the stars (spherical or cylindrical) in pc/as."""
        if spher:
            radii = form.Distance(self.coords)
        else:
            radii = form.Distance2D(self.coords)
        
        if (unit == 'as'):                                                                          # convert to arcsec if wanted
            radii = conv.ParsecToArcsec(radii, self.d_ang)
        
        return radii
    
    def HalfMassRadius(self, unit='pc', spher=False):
        """Returns the (spherical or cylindrical) half mass radius in pc/as."""
        M_cur = self.CurrentMasses()
        tot_mass = np.sum(M_cur)                                                                    # do this locally, to avoid unnecesairy overhead
        
        if spher:
            r_star = form.Distance(self.coords)                                                     # spherical radii of the stars
        else:
            r_star = form.Distance2D(self.coords)                                                   # cylindrical radii of the stars
            
        indices = np.argsort(r_star)                                                                # indices that sort the radii
        r_sorted = r_star[indices]                                                                  # sorted radii
        mass_sorted = M_cur[indices]                                                                # masses sorted for radius
        mass_sum = np.cumsum(mass_sorted)                                                           # cumulative sum of sorted masses
        hmr = np.max(r_sorted[mass_sum <= tot_mass/2])                                              # 2D/3D radius at half the mass
        
        if (unit == 'as'):                                                                          # convert to arcsec if wanted
            hmr = conv.ParsecToArcsec(hmr, self.d_ang)
            
        return hmr
        
    def HalfLumRadius(self, unit='pc', spher=False):
        """Returns the (spherical or cylindrical) half luminosity radius in pc/as."""
        lum = 10**self.LogLuminosities(realistic_remnants=False)
        
        tot_lum = np.sum(lum)                                                                       # do this locally, to avoid unnecesairy overhead
        
        if spher:
            r_star = form.Distance(self.coords)                                                     # spherical radii of the stars
        else:
            r_star = form.Distance2D(self.coords)                                                   # cylindrical radii of the stars
            
        indices = np.argsort(r_star)                                                                # indices that sort the radii
        r_sorted = r_star[indices]                                                                  # sorted radii
        lum_sorted = lum[indices]                                                                   # luminosities sorted for radius
        lum_sum = np.cumsum(lum_sorted)                                                             # cumulative sum of sorted luminosities
        hlr = np.max(r_sorted[lum_sum <= tot_lum/2])                                                # 2D/3D radius at half the luminosity
        
        if (unit == 'as'):                                                                          # convert to arcsec if wanted
            hlr = conv.ParsecToArcsec(hlr, self.d_ang)
            
        return hlr
        
    def Plot2D(self, title='Scatter', xlabel='x', ylabel='y', axes='xy', colour='blue', 
               filter='V', theme='dark1'):
        """Make a plot of the object positions in two dimensions
        Set colour to 'temperature' for a temperature representation.
        Set filter to None to avoid markers scaling in size with magnitude.
        Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but 
            saveable dark plot, 'fits' for a plot that resembles a .fits image,
            and None for normal light colours.
        """
        if (filter is not None):
            mags = self.ApparentMagnitudes(filter=filter)
        else:
            mags = None
        
        if (colour == 'temperature'):
            temps = 10**self.LogTemperatures()
        else:
            temps = None
        
        vis.Scatter2D(self.coords, title=title, xlabel=xlabel, ylabel=ylabel,
                      axes=axes, colour=colour, T_eff=temps, mag=mags, theme=theme)
        return
        
    def Plot3D(self, title='Scatter', xlabel='x', ylabel='y', axes='xy', colour='blue', 
               filter='V', theme='dark1'):
        """Make a plot of the object positions in three dimensions.
        Set colour to 'temperature' for a temperature representation.
        Set filter to None to avoid markers scaling in size with magnitude.
        Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but 
            saveable dark plot, and None for normal light colours.
        """
        if (filter is not None):
            mags = self.ApparentMagnitudes(filter=filter)
        else:
            mags = None
        
        if (colour == 'temperature'):
            temps = 10**self.LogTemperatures()
        else:
            temps = None
        
        vis.Scatter3D(self.coords, title=title, xlabel=xlabel, ylabel=ylabel,
                      axes=axes, colour=colour, T_eff=temps, mag=mags, theme=theme)
        return
        
    def PlotHRD(self, title='HRD', colour='temperature', theme='dark1'):
        """Make a plot of stars in an HR diagram.
        Set colour to 'temperature' for a temperature representation.
        Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but 
            saveable dark plot, and None for normal light colours.
        """
        r_mask = np.invert(self.Remnants())
        temps = 10**self.LogTemperatures()
        lums = self.LogLuminosities()
        
        vis.HRD(T_eff=temps, log_Lum=lums, title=title, xlabel='Temperature (K)', 
                ylabel=r'Luminosity log($L/L_\odot$)', colour=colour, theme=theme, mask=r_mask)
        return
        
    def PlotCMD(self, x='B-V', y='V', title='CMD', colour='blue', theme='dark1'):
        """Make a plot of the stars in a CMD
        Set x and y to the colour and magnitude to be used (x needs format 'A-B')
        Set colour to 'temperature' for a temperature representation.
        """
        x_filters = x.split('-')
        mag_A = self.ApparentMagnitudes(filter=x_filters[0])
        mag_B = self.ApparentMagnitudes(filter=x_filters[1])
        c_mag = mag_A - mag_B
        
        if (y == x_filters[0]):
            mag = mag_A
        elif (y == x_filters[1]):
            mag = mag_B
        else:
            mag = self.ApparentMagnitudes(filter=y)
        
        vis.CMD(c_mag, mag, title='CMD', xlabel=x, ylabel=y, 
                colour='blue', T_eff=None, theme=None, adapt_axes=True, mask=None)
        return
        
    def PerformanceMode(self, full=False, turn_off=False):
        """Sacrifices memory usage for performance during the simulation of images or other tasks.
        The full set of variables like luminosity and temperature can be stored or only
            a selection needed for imaging, by setting full to True or False.
        If for some reason this mode has to be turned off (delete stored data), set turn_off=True.
        """
        if (not hasattr(self, 'current_masses') & (full == True)):
            self.current_masses = self.CurrentMasses(realistic_remnants=True)
        elif (turn_off & (full == True)):
            del self.current_masses
            
        if (not hasattr(self, 'log_luminosities') & (full == True)):
            self.log_luminosities = self.LogLuminosities(realistic_remnants=True)
        elif (turn_off & (full == True)):
            del self.log_luminosities
            
        if (not hasattr(self, 'log_temperatures') & (full == True)):
            self.log_temperatures = self.LogTemperatures(realistic_remnants=True)
        elif (turn_off & (full == True)):
            del self.log_temperatures
        
        if (not hasattr(self, 'absolute_magnitudes') & (full == True)):
            self.absolute_magnitudes = self.AbsoluteMagnitudes()
        elif (turn_off & (full == True)):
            del self.absolute_magnitudes
        
        if not hasattr(self, 'apparent_magniutdes'):
            self.apparent_magniutdes = self.ApparentMagnitudes()
        elif turn_off:
            del self.apparent_magniutdes
        
        if not hasattr(self, 'spectral_types'):
            self.spectral_types, names = self.SpectralTypes(realistic_remnants=True)
        elif turn_off:
            del self.spectral_types
        
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


class StarCluster(AstObject):
    """For generating star clusters."""
    structure = 'star cluster'                                                                      # type of object
    
    def __init__(self, N_stars=0, M_tot_init=0, age=None, metal=None, rel_num=None, distance=10, 
                 d_type='l', imf_par=None, sf_hist=None, extinct=0, r_dist=None, r_dist_par=None, 
                 **kwargs 
                 ):
        
        super().__init__(N_stars=N_stars, M_tot_init=M_tot_init, age=age, metal=metal, 
                         rel_num=rel_num, distance=distance, d_type=d_type, extinct=extinct, 
                         sf_hist=sf_hist, imf_par=imf_par, **kwargs 
                         )
        
        if (r_dist is None):
            r_dist = [rdist_default]
        if (r_dist_par is None):
            r_dist_par = {}
        
        # object specific parameters
        self.r_dist_type = r_dist                                                                   # type of radial distribution
        self.r_dist_param = r_dist_par                                                              # the further spatial distribution parameters (dictionary)
        
        self.CheckRadialDistribution()
        self.GenerateStars()                                                                        # actually generate the stars
        return
        
    def GenerateStars(self):
        """Generate the masses and positions of the stars for the cluster."""
        # generate the positions, masses   
        for i, pop_num in enumerate(self.gen_pop_number):
            coords_i = Spherical(pop_num, dist_type=self.r_dist_type[i], **self.r_dist_param[i])
            M_init_i, M_diff_i = StarMasses(pop_num, 0, imf=self.gen_imf_param[i])
            self.coords = np.append(self.coords, coords_i, axis=0)                           
            self.M_init = np.append(self.M_init, M_init_i)
            self.M_diff += M_diff_i                                                                 # StarMasses already gives diff in Mass (=estimate since no mass was given)
        
        # if only N_stars was given, set M_tot_init to the total generated mass
        mass_generated = np.sum(self.M_init)
        if (self.M_tot_init == 0):
            self.M_tot_init = mass_generated
        else:
            self.M_diff = mass_generated - self.M_tot_init                                          # if negative: too little mass generated, else too much
            self.M_tot_init = mass_generated                                                        # set to actual initial mass
        
        # the filter names of the corresponding magnitudes
        self.mag_names = utils.OpenIsochrone(self.ages[0], self.metal[0], columns='filters')
        
        return


class EllipticalGalaxy(AstObject):
    """For generating elliptical galaxies."""
    structure = 'elliptical galaxy'                                                                 # type of object
    
    def __init__(self, N_stars=0, M_tot_init=0, age=None, metal=None, rel_num=None, distance=1000, 
                 d_type='l', imf_par=None, sf_hist=None, extinct=0, incl=None, r_dist=None, 
                 r_dist_par=None, ellipse_axes=None, **kwargs
                 ):
        
        super().__init__(N_stars=N_stars, M_tot_init=M_tot_init, age=age, metal=metal, 
                         rel_num=rel_num, distance=distance, d_type=d_type, extinct=extinct, 
                         sf_hist=sf_hist, imf_par=imf_par, **kwargs
                         )
        
        if (incl is None):
            incl = [0]
        if (r_dist is None):
            r_dist = [rdist_default]
        if (r_dist_par is None):
            r_dist_par = {}
        if (ellipse_axes is None):
            ellipse_axes = [1, 1, 1]
        
        # object specific parameters
        self.r_dist_type = r_dist                                                                   # type of radial distribution
        self.r_dist_param = r_dist_par                                                              # the further spatial distribution parameters (dictionary)
        
        self.CheckRadialDistribution()
        self.GenerateStars()                                                                        # actually generate the stars
        self.AddEllipticity(ellipse_axes)
        self.AddInclination(incl)
        
        # make subclass of StarCluster?
        return

    def GenerateStars(self):
        """Generate the masses and positions of the stars for the elliptical galaxy."""
        # generate the positions, masses   
        for i, pop_num in enumerate(self.gen_pop_number):
            coords_i = Spherical(pop_num, dist_type=self.r_dist_type[i], **self.r_dist_param[i])
            coords_i = Ellipsify(coords_i, self.ellipse_axes[i])
            M_init_i, M_diff_i = StarMasses(pop_num, 0, imf=self.gen_imf_param[i])
            self.coords = np.append(self.coords, coords_i, axis=0)                           
            self.M_init = np.append(self.M_init, M_init_i)
            self.M_diff += M_diff_i                                                                 # StarMasses already gives diff in Mass (=estimate since no mass was given)
        
        # if only N_stars was given, set M_tot_init to the total generated mass
        mass_generated = np.sum(self.M_init)
        if (self.M_tot_init == 0):
            self.M_tot_init = mass_generated
        else:
            self.M_diff = mass_generated - self.M_tot_init                                          # if negative: too little mass generated, else too much
            self.M_tot_init = mass_generated                                                        # set to actual initial mass
        
        # the filter names of the corresponding magnitudes
        self.mag_names = utils.OpenIsochrone(self.ages[0], self.metal[0], columns='filters')
        
        return


class SpiralGalaxy(AstObject):
    """For generating spiral galaxies."""
    structure = 'spiral galaxy'                                                                     # type of object
    
    def __init__(self, N_stars=0, M_tot_init=0, age=None, metal=None, rel_num=None, distance=1000, 
                 d_type='l', imf_par=None, sf_hist=None, extinct=0, incl=None, spiral_arms=0, 
                 spiral_bulge=0, spiral_bar=0, **kwargs
                 ):
        
        super().__init__(N_stars=N_stars, M_tot_init=M_tot_init, age=age, metal=metal, 
                         rel_num=rel_num, distance=distance, d_type=d_type, extinct=extinct, 
                         sf_hist=sf_hist, imf_par=imf_par, **kwargs
                         )
        
        if (incl is None):
            incl = [0]
        
        # object specific parameters
        self.spiral_arms = spiral_arms                                                              # number of spiral arms
        self.spiral_bulge = spiral_bulge                                                            # relative proportion of central bulge
        self.spiral_bar = spiral_bar                                                                # relative proportion of central bar
        
        self.GenerateStars()                                                                        # actually generate the stars
        self.AddInclination(incl)
        return
        
    def GenerateStars(self):
        """Generate the masses and positions of the stars for the spiral galaxy."""
        # generate the positions, masses   
        for i, pop_num in enumerate(self.gen_pop_number):
            coords_i = Spiral(pop_num)
            M_init_i, M_diff_i = StarMasses(pop_num, 0, imf=self.gen_imf_param[i])
            self.coords = np.append(self.coords, coords_i, axis=0)                           
            self.M_init = np.append(self.M_init, M_init_i)
            self.M_diff += M_diff_i                                                                 # StarMasses already gives diff in Mass (=estimate since no mass was given)
        
        # if only N_stars was given, set M_tot_init to the total generated mass
        mass_generated = np.sum(self.M_init)
        if (self.M_tot_init == 0):
            self.M_tot_init = mass_generated
        else:
            self.M_diff = mass_generated - self.M_tot_init                                          # if negative: too little mass generated, else too much
            self.M_tot_init = mass_generated                                                        # set to actual initial mass
        
        # the filter names of the corresponding magnitudes
        self.mag_names = utils.OpenIsochrone(self.ages[0], self.metal[0], columns='filters')
        
        return


def Spherical(N_stars, dist_type=rdist_default, axes=None, **param):
    """Make a spherical distribution of stars using the given radial distribution type.
    Takes additional parameters for the r-distribution function (i.e. scale length s).
    """
    # check if the dist type exists
    if (dist_type[-2:] != '_r'):
        dist_type += '_r'                                                                           # add the r to the end for radial version
        
    dist_list = list(set(fnmatch.filter(dir(dist), '*_r')))
                        
    if (dist_type not in dist_list):
        warnings.warn(('objectgenerator//Spherical: Specified distribution type does not exist. '
                       'Using default (={})').format(rdist_default), SyntaxWarning)
        dist_type = rdist_default + '_r'

    # check if right parameters given    
    sig = signature(eval('dist.' + dist_type))                                                      # parameters of the dist function (includes n)
    dict = param                                                                                    # need a copy for popping in iteration
    for key in dict:
        if key not in sig.parameters:
            warnings.warn(('objectgenerator//Spherical: Wrong keyword given in distribution '
                           'parameters. Deleted entry.\n    {0} = {1}'
                           ).format(key, param.pop(key, None)), SyntaxWarning)
    
    r_dist = eval('dist.' + dist_type)(n=N_stars, **param)                                          # the radial distribution
    phi_dist = dist.AnglePhi(N_stars)                                                               # dist for angle with x axis
    theta_dist = dist.AngleTheta(N_stars)                                                           # dist for angle with z axis
    
    return conv.SpherToCart(r_dist, theta_dist, phi_dist).transpose()


def Spiral(N_stars):
    """Make a spiral galaxy."""
    #todo: add this
    

def SpiralArms():
    """Make spiral arms."""
    

def Irregular(N_stars):
    """Make an irregular galaxy"""
    

def StarMasses(N_stars=0, M_tot=0, imf=[0.08, 150]):
    """Generate masses using the Initial Mass Function. 
    Either number of stars or total mass should be given.
    imf defines the lower and upper bound to the masses generated in the IMF.
    Also gives the difference between the total generated mass and the input mass 
        (or estimated mass when using N_stars).
    """
    # check input (N_stars or M_tot?)
    if (N_stars == 0) & (M_tot == 0):
        warnings.warn(('objectgenerator//StarMasses: Input mass and number of stars '
                       'cannot be zero simultaniously. Using N_stars=10'), SyntaxWarning)
        N_stars = 10
    elif (N_stars == 0):                                                                            # a total mass is given
        N_stars = conv.MtotToNstars(M_tot, imf)                                                     # estimate the number of stars to generate
        
    # mass
    M_init = dist.KroupaIMF(N_stars, imf)                                                           # assign initial masses using IMF
    M_tot_gen = np.sum(M_init)                                                                      # total generated mass (will differ from input mass, if given)
    
    if (M_tot != 0):
        M_diff = M_tot_gen - M_tot
    else:
        M_tot_est = conv.NstarsToMtot(N_stars, imf)
        M_diff = M_tot_gen - M_tot_est                                                              # will give the difference to the estimated total mass for n stars
        
    return M_init, M_diff


def StarFormHistory(max_age, log_t, sfr='exp', t0=1e10, tau_star=1e15):
    """Finds the relative number of stars to give a certain age up to maximum given age, 
    using a certain Star Formation Rate.
    log_t provides the available times in the isochrone file.
    """
    uni_log_t = np.unique(log_t)
    log_t_use = uni_log_t[uni_log_t <= max_age]                                                     # log t's to use 
    
    if (sfr == 'exp'):
        t_use = 10**log_t_use                                                                       # Age of each SSP
        tau = t0 - t_use                                                                            # Time since t0
        psi = np.exp(-tau/tau_star)                                                                 # Star formation rates (relative)
        rel_num = psi/np.sum(psi)                                                                   # relative number in each generation
    #TODO: make this work + also return the ages for each number in rel_num
    return rel_num  


def FindSpectralType(T_eff, Lum, Mass):
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
    
    #TODO: WD are taken care of this way, but what to do with NS/BH? 
    # (now they would probably get WD spectra)
    
    return indices, type_selection


def NumberLimited(N, age, Z, imf=imf_defaults):
    """Retrieves the lower mass limit for which the number of stars does not exceed 10**7. 
    Will also give an upper mass limit based on the values in the isochrone.
    The intended number of generated stars, age and metallicity are needed.
    """
    fraction = np.clip(limiting_number/N, 0, 1)                                                     # fraction of the total number of stars to generate
    
    M_ini = utils.OpenIsochrone(age, Z, columns='mini')                                             # get the isochrone values
    mass_lim_high = M_ini[-1]                                                                       # highest value in the isochrone
    
    mass_lim_low = form.MassLimit(fraction, M_max=mass_lim_high, imf=imf)
    
    return mass_lim_low, mass_lim_high


def MagnitudeLimited(age, Z, mag_lim=default_mag_lim, d=10, ext=0, filter='Ks'):
    """Retrieves the lower mass limit for which the given magnitude threshold is met. 
    Will also give an upper mass limit based on the values in the isochrone.
    Works only for resolved stars. 
    If light is integrated along the line of sight (crowded fields), 
    then don't use this method! Resulting images would not be accurate!
    distance, age, metallicity and extinction of the population of stars are needed.
    A filter must be specified in which the given limiting magnitude is measured.
    """
    M_ini, mag_vals = utils.OpenIsochrone(age, Z, columns='mag')                                    # get the isochrone values
    mag_vals = mag_vals[self.mag_names == filter][0]                                                # select the right filter ([0] needed to reduce to 1D array)

    abs_mag_lim = form.AbsoluteMag(mag_lim, d, ext=ext)                                             # calculate the limiting absolute magnitude
    mask = (mag_vals < abs_mag_lim + 0.1)                                                           # take all mag_vals below the limit
    if not mask.any():
        mask = (mag_vals == np.min(mag_vals))                                                       # if limit too high (too low in mag) then it will break
        warnings.warn(('objectgenerator//MagnitudeLimited: compacting will not work, '
                       'distance too large.'), RuntimeWarning)

    mass_lim_low = M_ini[mask][0]                                                                   # the lowest mass where mask==True
    
    mass_lim_high = M_ini[-1]                                                                       # highest value in the isochrone
    
    return mass_lim_low, mass_lim_high


def RemnantsSinglePop(M_init, age, Z):
    """Gives the positions of the remnants in a single population (as a boolean mask)."""
    iso_M_ini = utils.OpenIsochrone(age, Z, columns='mini')
    max_mass = np.max(iso_M_ini)                                                                    # maximum initial mass in isoc file
    return (M_init > max_mass)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


