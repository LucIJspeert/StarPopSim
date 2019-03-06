# Luc IJspeert
# Part of smoc: (astronomical) object generator and class
##
import os
import inspect
import fnmatch
import pickle
import warnings

import numpy as np
import scipy.spatial as sps

from inspect import signature

import distributions as dist
import conversions as conv
import formulas as form

'''This module defines the Astronomical Object class that holds all the information for the astr. object.
It contains all the checks performed on the user input.
It also defines the functions that can be performed on the astr. object.
As well as the functions to make (parts of) the astr. object,
like the properties of the individual objects (usually stars).
Several options for the distribution of the objects are available; 
personal distributions might be added to the distribution module.
'''

# global defaults
rdist_default = 'Normal'                                                                            # see distributions module for a full list of options
default_mag_lim = 32                                                                                # magnitude limit found by testing in Ks with exp=1800s
default_lum_fraction = 0.9999                                                                       # depending on (log)age, the following fractions might work better: <6: 0.99, ~7: 0.999, >8: 0.9999
imf_defaults = [0.08, 150]                                                                          # lower bound, upper bound on mass


class AstObject:
    '''Contains all the information about a generated astronomical object.
    Also functions that can be performed on the object are defined here.
    '''
    def __init__(self, 
                struct='ellipsoid', 
                N_obj=0, 
                M_tot_init=0, 
                age=None, 
                metal=None, 
                rel_num=None, 
                distance=0,
                d_type='l',
                imf_par=None,
                sf_hist=None,
                extinct=0,
                incl=None,
                r_dist=None,
                r_dist_par=None,
                ellipse_axes=None,
                spiral_arms=0,
                spiral_bulge=0,
                spiral_bar=0,
                compact=False,
                cp_mode='num',
                mag_lim=None,
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
        if (incl is None):
            incl = [0]
        if (r_dist is None):
            r_dist = [rdist_default]
        if (r_dist_par is None):
            r_dist_par = {}
        if (ellipse_axes is None):
            ellipse_axes = [1, 1, 1]
        
        self.structure = struct                                                                     # type of object
        self.N_obj = N_obj                                                                          # number of objects
        self.M_tot_init = M_tot_init                                                                # total initial mass in Msol
        self.ages = age                                                                             # ages of the populations (=max age if SFH is used)
        self.metal = metal                                                                          # metallicity (of each population)
        self.rel_number = rel_num                                                                   # relative number of objects in each population (equal if left empty)
        
        self.d_type = d_type                                                                        # distance type [l for luminosity, z for redshift]

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
        self.inclination = incl                                                                     # inclination angle (rad) - rotation of object's x-axis towards z-axis (l.o.s.)
        self.r_dist_type = r_dist                                                                   # (ellipse) type of radial distribution
        self.r_dist_param = r_dist_par                                                              # (ellipse) the further spatial distribution parameters (dictionary)
        self.ellipse_axes = ellipse_axes                                                            # (ellipse) axes scales used for elliptical
        self.spiral_arms = spiral_arms                                                              # (spiral) number of spiral arms
        self.spiral_bulge = spiral_bulge                                                            # (spiral) relative proportion of central bulge
        self.spiral_bar = spiral_bar                                                                # (spiral) relative proportion of central bar
        
        # properties that are derived
        self.pop_number = np.array([])                                                              # number of objects in each population
        self.coords = np.empty([0,3])                                                               # spatial coordinates
        self.M_init = np.array([])                                                                  # the masses of the objects
        self.M_diff = 0                                                                             # mass difference between given and generated (total) mass (if given)
        
        # from an isoc
        self.M_cur = np.array([])                                                                   # current object masses (at given age) in Msol
        self.spec_types = np.array([])                                                              # (indices of) spectral type of the stars
        self.spec_names = []                                                                        # corresponding spectral type names
        self.remnant = np.array([], dtype=bool)                                                     # keep track if the star has died (==no longer in isoc file) (bool)
        self.log_L = np.array([])                                                                   # logarithm of luminosities in Lsol
        self.log_Te = np.array([])                                                                  # effective temperature of the objects in K
        self.log_g = np.array([])                                                                   # log of the surface gravity (in cgs!)
        self.abs_mag = np.empty([8,0])                                                              # absolute magnitudes (in various filters! [U, B, V, R, I, J, H, K])
        self.mag_names = np.array([])                                                               # names of the filters corresponding to the magnitudes
        
        self.compact = compact  	                                                                # (compact mode) if True, generates only high mass stars based on limiting mag
        self.compact_mode = cp_mode                                                                 # (compact mode) mode of compacting. num=number limited, mag=magnitude limited
        self.mag_limit = mag_lim                                                                    # (compact mode) limiting magnitude, used only for compact mode
        self.fraction_generated = 1                                                                 # (compact mode) part of the total number of stars that has actually been generated for each population
        self.mass_limit = np.array([])                                                              # (compact mode) mass limits imposed by compacting (for each population)
        
        self.CheckInput()                                                                           # check user input
        self.GenerateObj()                                                                          # actually generate the objects                                                                        
        return
        
    def CheckInput(self):
        '''Checks the given input for compatibility.'''
        # check metallicity and age
        if not isinstance(self.ages, (tuple, list, np.ndarray)):
            raise ValueError('AstObjectClass//CheckInput: wrong input type for the age given.')
        else:
            self.ages = np.array(self.ages)                                                         # make sure it is an array
        if not isinstance(self.metal, (tuple, list, np.ndarray)):
            raise ValueError('AstObjectClass//CheckInput: wrong input type for the metallicity given.')
        else:
            self.metal = np.array(self.metal)                                                       # make sure it is an array
        
        # much used qtt's
        num_ages = len(self.ages)
        num_metal = len(self.metal)
        
        # check for empty input (ages and metallicity)
        if (num_ages == 0) | (num_metal == 0):
            raise ValueError('AstObjectClass//CheckInput: No age and/or metallicity was defined.')
        elif (num_ages != num_metal):                                                               # make sure they have the right length
            if (num_ages == 1):
                self.ages = self.ages[0]*np.ones(num_metal)
            elif (num_metal == 1):
                self.metal = self.metal[0]*np.ones(num_ages)
            else:
                warnings.warn('AstObjectClass//CheckInput: age and metallicity have incompatible length {0} and {1}. Discarding excess.'.format(num_ages, num_metal), SyntaxWarning)
                new_len = min(num_ages, num_metal)
                self.ages = self.ages[:new_len]
                self.metal = self.metal[:new_len]
                
            num_ages = len(self.ages)                                                               # update length
            num_metal = len(self.metal)                                                             # update length
        
        num_pop = num_ages                                                                          # define number of populations
        
        # check the input for rel_num [must go before usage of num_pop, after ages and metallicity]
        if isinstance(self.rel_number, (int, float)):
            self.rel_number = np.ones(num_pop)
        elif not isinstance(self.rel_number, (tuple, list, np.ndarray)):
            warnings.warn('AstObjectClass//CheckInput: relative number input not understood. Using ones.', SyntaxWarning)
            self.rel_number = np.ones(num_pop)
            
        relnum_len = len(self.rel_number)
        if ((relnum_len == 1) & (num_pop != 1)):
            self.rel_number = self.rel_number[0]*np.ones(num_pop)
        elif (relnum_len < num_pop):
            self.rel_number = np.append(self.rel_number, [1 for i in range(num_pop - relnum_len)])
        elif ((relnum_len > num_pop) & (num_pop != 1)):
            warnings.warn('AstObjectClass//CheckInput: too many relative numbers given. Discarding excess.', SyntaxWarning)
            self.rel_number = self.rel_number[:num_pop]
        elif ((relnum_len > num_pop) & (num_pop == 1)):
            self.ages = np.array([self.ages[0] for i in range(relnum_len)])                         # in this case, add populations of the same age
            self.metal = np.array([self.metal[0] for i in range(relnum_len)])                       #  and metallicity
            num_ages = len(self.ages)                                                               # update length
            num_metal = len(self.metal)                                                             # update length
            num_pop = num_ages                                                                      # [very important] update number of populations
        
        rel_frac = self.rel_number/np.sum(self.rel_number)                                          # fraction of the total in each population
        
        # check format of imf_param
        if isinstance(self.imf_param, (tuple, list)):
            self.imf_param = np.array(self.imf_param)                                               # make it an array
        elif not isinstance(self.imf_param, (np.ndarray)):
            warnings.warn('AstObjectClass//CheckInput: Data type for imf_par not understood, using default (={0}).'.format(imf_defaults), SyntaxWarning)
            self.imf_param = np.array([imf_defaults for i in range(num_pop)]) 
        
        imf_shape = np.shape(self.imf_param)
        imf_par_len = len(imf_defaults)
        if (len(imf_shape) == 1):
            if (imf_shape[0] == imf_par_len):
                self.imf_param = np.array([self.imf_param for i in range(num_pop)])                 # make it a 2D array using same imf for all populations
            elif (imf_shape[0]%imf_par_len == 0):
                self.imf_param = np.reshape(self.imf_param, [int(imf_shape[0]/imf_par_len), imf_par_len])      # make it a 2D array
            else:
                warnings.warn('AstObjectClass//CheckInput: Wrong number of arguments for imf_par, using default (={0}).'.format(imf_defaults), SyntaxWarning)
                self.imf_param = np.array([imf_defaults for i in range(num_pop)])
        elif (len(imf_shape) != 2):
            warnings.warn('AstObjectClass//CheckInput: Wrong dimension for imf_par, using default (={0}).'.format(imf_defaults), SyntaxWarning)
            self.imf_param = np.array([imf_defaults for i in range(num_pop)])
        
        imf_shape = np.shape(self.imf_param)                                                        # update shape
        if (imf_shape[0] > num_pop):
            warnings.warn('AstObjectClass//CheckInput: Too many arguments for imf_par. Discarding excess.', SyntaxWarning)
            self.imf_param = self.imf_param[0:num_pop]
        elif (imf_shape[0] < num_pop):
            filler = [imf_defaults for i in range(num_pop - int(imf_shape[0]/imf_par_len))]
            self.imf_param = np.append(self.imf_param, filler, axis=0)                              # fill missing imf_par with default        
        
        # check the minimum available mass in isoc file [must go after imf_param check]
        max_M_L = 0
        for Z in self.metal:
            file_name = os.path.join('tables', 'isoc_Z{1:1.{0}f}.dat'.format(-int(np.floor(np.log10(Z)))+1, Z))
            if os.path.isfile(file_name):                                                           # check wether file for Z exists
                var_names, column_names = np.loadtxt(os.path.join('tables', 'column_names.dat'), dtype=str, unpack=True)
                with open(file_name) as file:
                    for line in file:
                        if line.startswith('#'):
                            header = np.array(line.replace('#', '').split())                        # find the column names in the isoc file
                        else:
                            break
                col = np.where(header == column_names[var_names == 'M_initial'])[0][0]
                M_ini = np.loadtxt(file_name, usecols=(col), unpack=True)
            else:
                raise ValueError('AstObjectClass//CheckInput: Input metallicity (Z={0}) does not correspond to an existing file.'.format(Z))
            max_M_L = max(max_M_L, np.min(M_ini))
        
        imf_max_M_L = np.array([self.imf_param[:,0], [max_M_L for i in range(len(self.imf_param[:,0]))]])  # check against user input (if that was higher, use it)
        self.imf_param[:,0] =  np.max(imf_max_M_L, axis=0)                                          # maximum lowest mass (to use in IMF)
        
        # check input: N_obj or M_tot_init? --> need N_obj [must go after min mass check]
        if (self.N_obj == 0) & (self.M_tot_init == 0):
            warnings.warn('AstObjectClass//CheckInput: Input mass and number of objects cannot be zero simultaniously. Using N_obj=1000', SyntaxWarning)
            self.N_obj = 1000
            pop_num = np.rint(rel_frac*self.N_obj).astype(int)                                      # rounded off number
        elif (self.N_obj == 0):                                                                     # a total mass is given
            pop_num = np.array([conv.MtotToNobj(self.M_tot_init*rel_frac[i], imf=self.imf_param[i]) for i in range(num_pop)])
            self.N_obj = np.sum(pop_num)                                                            # estimate of the number of objects to generate
        else:
            pop_num = np.rint(rel_frac*self.N_obj).astype(int)                                      # rounded off number
            self.N_obj = np.rint(np.sum(pop_num)).astype(int)                                       # make sure N_obj is int and rounded off
        
        # check if the population numbers add up to N total and save correct ones
        self.pop_number = FixTotal(self.N_obj, pop_num)
        
        # check the inclination(s), make sure it is an array of the right size
        if isinstance(self.inclination, (int, float)):
            self.inclination = np.array([self.inclination for i in range(num_pop)]) 
        elif isinstance(self.inclination, (tuple, list, np.ndarray)):
            incl_len = len(self.inclination)
            if (incl_len == 1):
                self.inclination = np.array([self.inclination[0] for i in range(num_pop)])
            elif (incl_len < num_pop):
                self.inclination = np.append(self.inclination, [0 for i in range(num_pop - incl_len)])
            elif (incl_len > num_pop):
                warnings.warn('AstObjectClass//CheckInput: too many inclination angles given. Discarding excess.', SyntaxWarning)
                self.inclination = np.array(self.inclination[:num_pop])
            else:
                self.inclination = np.array(self.inclination)
        else:
            raise ValueError('AstObjectClass//CheckInput: incompatible inclination given. Use a number, list, or an array of numbers.')
                
        if np.any(self.inclination > 2*np.pi):
            warnings.warn('AstObjectClass//CheckInput: inclination angle over 2pi detected, make sure to use radians!', SyntaxWarning)
        
        # check if the dist type(s) exists and get the function signatures
        dist_list = list(set(fnmatch.filter(dir(dist), '*_r')))
        
        if not isinstance(self.r_dist_type, list):
            self.r_dist_type = [self.r_dist_type]                                                   # make sure it is a list
            
        n_r_dists = len(self.r_dist_type)
        if (n_r_dists < num_pop):                                                                   # check number of dists
            self.r_dist_type.extend([rdist_default for i in range(num_pop - n_r_dists)])
            n_r_dists = len(self.r_dist_type)                                                       # update the number
            
        key_list = []
        val_list = []
        r_dist_n_par = []
        
        for i in range(n_r_dists):
            if (self.r_dist_type[i][-2:] != '_r'):
                self.r_dist_type[i] += '_r'                                                         # add the r to the end for radial version
            
            if (self.r_dist_type[i] not in dist_list):
                warnings.warn('AstObjectClass//CheckInput: Specified distribution <{0}> type does not exist. Using default (=Normal_r)'.format(self.r_dist_type[i]), SyntaxWarning)
                self.r_dist_type[i] = 'Normal_r'
                 
            sig = inspect.signature(eval('dist.' + self.r_dist_type[i]))
            key_list.append([k for k, v in sig.parameters.items() if k is not 'n'])                 # add the signature keywords to a list
            val_list.append([v.default for k, v in sig.parameters.items() if k is not 'n'])         # add the signature defaults to a list
            r_dist_n_par.append(len(key_list[i]))                                                   # the number of parameters for each function
            
        # check if dist parameters are correctly specified
        if isinstance(self.r_dist_param, dict):                                                     # if just one dict, make a list of (one) dict
            self.r_dist_param = [self.r_dist_param]
        elif isinstance(self.r_dist_param, (int, float)):                                           # if just one parameter is given, also make a list of (one) dict
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
                    warnings.warn('AstObjectClass//CheckInput: Too many radial distribution parameters given. Discarding excess.', SyntaxWarning)
                    self.r_dist_param = self.r_dist_param[0:n_r_dists]
                elif (param_shape[0] < n_r_dists):                                                  # fill up missing length with defaults
                    filler = [[val_list[param_shape[0] + i]] for i in range(n_r_dists - param_shape[0])]
                    self.r_dist_param += filler
                
                for i, param in enumerate(self.r_dist_param):
                    if (len(param) < r_dist_n_par[i]):
                        self.r_dist_param[i].extend([item for item in val_list[i][len(param):]])    # not enough parameters for a particular distribution
                
                
                temp_par_dict_list = []                                                             # now it is ready finally for making a dict out of it
                for i in range(n_r_dists):
                    temp_par_dict_list.append({key_list[i][k]: self.r_dist_param[i][k] for k in range(r_dist_n_par[i])})
                    
                self.r_dist_param = temp_par_dict_list
                    
        else:
            raise TypeError('AstObjectClass//CheckInput: Incompatible data type for rdistpar')      # it is something else... burn it with fire!
            
        for i, param_dict in enumerate(self.r_dist_param):
            if not bool(param_dict):                                                                # if dict empty, fill with defaults
                self.r_dist_param[i] = {key_list[i][k]: val_list[i][k] for k in range(r_dist_n_par[i])}
        
        n_r_param = len(self.r_dist_param)
        if (n_r_param < num_pop):                                                                   # check parameter dict number
            self.r_dist_param.extend([{key_list[i][k]: val_list[i][k] for k in range(r_dist_n_par[i])} for i in range(n_r_param, num_pop)])
            n_r_param = len(self.r_dist_type)                                                       # update the number
        
        # check the ellipse axes
        if isinstance(self.ellipse_axes, (int, float)):
            self.ellipse_axes = np.ones([num_pop, 3])                                               # any single number will result in unitary scaling
        elif isinstance(self.ellipse_axes, (tuple, list, np.ndarray)):
            self.ellipse_axes = np.array(self.ellipse_axes)                                         # make sure it is an array
        elif not isinstance(self.ellipse_axes, (np.ndarray)):
            warnings.warn('AstObjectClass//CheckInput: Incompatible input for ellipse_axes. Using [1,1,1].', SyntaxWarning)
            self.ellipse_axes = np.ones([num_pop, 3])
            
        axes_shape = np.shape(self.ellipse_axes)
        if (len(axes_shape) == 1): 
            if (axes_shape[0] == 3):
                self.ellipse_axes = np.array([self.ellipse_axes for i in range(num_pop)])           # if 1D make 2D, using same axes for all populations
            elif (axes_shape[0]%3 == 0):
                self.ellipse_axes = np.reshape(self.ellipse_axes, [int(axes_shape[0]/3),3])         # make it a 2D array
            else:
                warnings.warn('AstObjectClass//CheckInput: Wrong number of arguments for ellipse_axes. Using [1,1,1].', SyntaxWarning)
                self.ellipse_axes = np.ones([num_pop, 3])
        elif (len(axes_shape) != 2):
            warnings.warn('AstObjectClass//CheckInput: Wrong dimension for ellipse_axes. Using [1,1,1].', SyntaxWarning)
            self.ellipse_axes = np.ones([num_pop, 3])
            
        axes_shape = np.shape(self.ellipse_axes)                                                    # update shape                                      
        if (axes_shape[0] > num_pop):
            warnings.warn('AstObjectClass//CheckInput: Got too many arguments for ellipse_axes. Discarding excess.', SyntaxWarning)
            self.ellipse_axes = self.ellipse_axes[0:num_pop]
        elif (axes_shape[0] < num_pop):
            filler = np.ones([num_pop - axes_shape[0], 3])
            self.ellipse_axes = np.append(self.ellipse_axes, filler, axis=0)
        
        # check the SFH
        if not isinstance(self.sfhist, list):
            self.sfhist = [self.sfhist]                                                             # make sure it is a list
        
        sfh_len = len(self.sfhist)
        if ((sfh_len == 1) & (num_pop != 1)):
            self.sfhist = [self.sfhist[0] for i in range(num_pop)]
        elif (sfh_len < num_pop):
            self.sfhist.extend(['none' for i in range(num_pop - sfh_len)])
        elif (sfh_len > num_pop):
            warnings.warn('AstObjectClass//CheckInput: too many sfh types given. Discarding excess.', SyntaxWarning)
            self.sfhist = self.sfhist[:num_pop]
            
        return
        
    def GenerateObj(self):
        '''Generate the objects and their properties.'''
        # check if compact mode is on
        self.fraction_generated = np.ones_like(self.pop_number, dtype=float)                        # set to ones initially
        self.mass_limit = np.copy(self.imf_param[:])                                                # set to imf params initially
        
        if self.compact:
            if (self.mag_limit is None):
                self.mag_limit = default_mag_lim                                                    # if not specified, use default
            
            for i, pop_num in enumerate(self.pop_number):
                if (self.compact_mode == 'mag'):
                    self.mass_limit[i] = MagnitudeLimited(self.ages[i], self.metal[i], mag_lim=self.mag_limit,
                                                    d=self.d_lum, ext=self.extinction, filter='Ks')
                else:
                    self.mass_limit[i] = NumberLimited(self.N_obj, self.ages[i], self.metal[i], imf=self.imf_param[i])
                
                if (self.mass_limit[i, 1] > self.imf_param[i, 1]):
                    self.mass_limit[i, 1] = self.imf_param[i, 1]                                    # don't increase the upper limit!
                
                self.fraction_generated[i] = form.MassFraction(self.mass_limit[i], imf=self.imf_param[i])
        
            if np.any(self.pop_number*self.fraction_generated < 10):                                # don't want too few stars
                raise RuntimeError('AstObjectClass//GenerateObj: population compacted to <10, try not compacting or generating a higher number of stars.', SyntaxWarning)
        
        # assign the right values for generation
        if self.compact:
            gen_pop_number = np.rint(self.pop_number*self.fraction_generated).astype(int)
            gen_imf_param = self.mass_limit
        else:
            gen_pop_number = self.pop_number
            gen_imf_param = self.imf_param
        
        # generate the positions, masses   
        if (self.structure in ['ellipsoid', 'elliptical']):
            for i, pop_num in enumerate(gen_pop_number):
                coords_i = Ellipsoid(pop_num, dist_type=self.r_dist_type[i], axes=self.ellipse_axes[i], **self.r_dist_param[i])
                if (self.inclination[i] != 0):
                    coords_i = conv.RotateXZ(coords_i, self.inclination[i])                         # rotate the XZ plane (x axis towards positive z)
                M_init_i, M_diff_i = ObjMasses(pop_num, 0, imf=gen_imf_param[i])
                self.coords = np.append(self.coords, coords_i, axis=0)                           
                self.M_init = np.append(self.M_init, M_init_i)
                self.M_diff += M_diff_i                                                             # ObjMasses already gives diff in Mass (=estimate since no mass was given)
        elif (self.structure == 'Spiral'):
            pass
        
        # if only N_obj was given, set M_tot_init to the total generated mass
        mass_generated = np.sum(self.M_init)
        if (self.M_tot_init == 0):
            self.M_tot_init = mass_generated
        else:
            self.M_diff = mass_generated - self.M_tot_init                                          # if negative: too little mass generated, else too much
            self.M_tot_init = mass_generated                                                        # set to actual initial mass
        
        # use the isochrone files to extrapolate properties
        for i, age in enumerate(self.ages):
            index = np.cumsum(np.append([0], gen_pop_number))
            M_cur_i, log_L_i, log_Te_i, log_g_i, abs_mag_i, mag_names = IsocProps(self.M_init[index[i]:index[i+1]], age, self.metal[i])
            
            # IsocProps assigns wildly wrong properties to remnants (i.e. M_cur=0). to get better properties, also run RemnantProps
            remnants_i = np.array([M_cur_i == 0])
            self.remnant = np.append(self.remnant, remnants_i)                                      # keep track of which stars are remnants
            # r_M_cur_i, r_log_L_i, r_log_Te_i, r_abs_mag_i = RemnantProps(self.M_init[index[i]:index[i+1]], age, self.metal[i], remnants=remnants_i)
            # 
            # M_cur_i[remnants_i] = r_M_cur_i
            # log_L_i[remnants_i] = r_log_L_i
            # log_Te_i[remnants_i] = r_log_Te_i
            # abs_mag_i[:, remnants_i] = r_abs_mag_i
            #TODO: remnant properties not implemented/activated yet
            
            self.M_cur = np.append(self.M_cur, M_cur_i)  
            self.log_L = np.append(self.log_L, log_L_i)
            self.log_Te = np.append(self.log_Te, log_Te_i)
            self.log_g = np.append(self.log_g, log_g_i)
            self.abs_mag = np.append(self.abs_mag, abs_mag_i, axis=1)
            
        
        self.mag_names = mag_names                                                                  # the filter names of the corresponding magnitudes
        
        # lastly, assign spectral types to the stars
        spec_indices, spec_names = FindSpectralType(self.log_Te, self.log_L, np.log10(self.M_cur))  # assign spectra to the stars
        self.spec_types = spec_indices                                                              # only save the indices to save memory
        self.spec_names = spec_names
        
        return
    
    def MtotCurrent(self):
        '''Returns the total current mass in Msun.'''
        return np.sum(self.M_cur)
    
    def TotalLuminosity(self):
        '''Returns log of the total luminosity in Lsun.'''
        return np.log10(np.sum(10**self.log_L))
        
    def ObjRadii(self, spher=True):
        '''Returns the radii of the objects (spherical or cylindrical) in pc.'''
        if spher:
            radii = form.Distance(self.coords)
        else:
            radii = form.Distance2D(self.coords)
        return radii
        
    def ApparentMagnitude(self, filter_name='all'):
        '''Compute the apparent magnitude from the absolute magnitude and th individual distances (in pc!). 
        The filter_name can be specified. 'all' will give all the filters.
        '''
        if (filter_name == 'all'):
            num_mags = len(self.mag_names)
            mask = np.full_like(self.mag_names, True, dtype=bool)
        else:
            num_mags = 1
            mask = (self.mag_names == filter_name)
        
        delta_d = -self.coords[:,2]                                                                 # delta distances relative to distance (taken to be minus the z-coordinate)
        if (self.d_lum > 100*np.abs(np.min(delta_d))):                                              # distance 'much larger' than individual variations
            true_dist = self.d_lum + delta_d                                                        # approximate the individual distance to each object with delta_d
        else:
            delta_d = form.Distance(self.coords, np.array([0, 0, self.d_lum]))                      # distances to each object
            true_dist = delta_d                                                                     # distance is now properly calculated for each object
        
        true_dist = np.tile(true_dist, num_mags).reshape(num_mags, np.sum(np.rint(self.pop_number*self.fraction_generated)).astype(int))
        
        return self.abs_mag[mask] + 5*np.log10((true_dist)/10) + self.extinction                    # true_dist in pc!
    
    def HalfMassRadius(self, spher=False):
        '''Returns the (spherical or cylindrical) half mass radius in pc.'''
        tot_mass = self.MtotCurrent()
        if spher:
            r_obj = form.Distance(self.coords)                                                      # spherical radii of the objects
        else:
            r_obj = form.Distance2D(self.coords)                                                    # cylindrical radii of the objects
            
        indices = np.argsort(r_obj)                                                                 # indices that sort the radii
        r_sorted = r_obj[indices]                                                                   # sorted radii
        mass_sorted = self.M_cur[indices]                                                           # masses sorted for radius
        mass_sum = np.cumsum(mass_sorted)                                                           # cumulative sum of sorted masses
        return np.max(r_sorted[mass_sum <= tot_mass/2])                                             # 2D/3D radius at half the mass
        
    def HalfLumRadius(self, spher=False):
        '''Returns the (spherical or cylindrical) half luminosity radius in pc. '''
        tot_lum = 10**self.TotalLuminosity()
        if spher:
            r_obj = form.Distance(self.coords)                                                      # spherical radii of the objects
        else:
            r_obj = form.Distance2D(self.coords)                                                    # cylindrical radii of the objects
            
        indices = np.argsort(r_obj)                                                                 # indices that sort the radii
        r_sorted = r_obj[indices]                                                                   # sorted radii
        lum_sorted = 10**self.log_L[indices]                                                        # luminosities sorted for radius
        lum_sum = np.cumsum(lum_sorted)                                                             # cumulative sum of sorted luminosities
        return np.max(r_sorted[lum_sum <= tot_lum/2])                                               # 2D/3D radius at half the luminosity
        
    def SaveTo(self, filename):
        '''Saves the class to a file.'''
        if (filename[-4:] != '.pkl'):
            filename += '.pkl'
        
        if os.path.isdir('objects'):
            with open(os.path.join('objects', filename), 'wb') as output:
                pickle.dump(self, output, -1)
        else:                                                                                       # if for some reason the pickle folder isn't there
            with open(filename, 'wb') as output:
                pickle.dump(self, output, -1)
        return
        
    def LoadFrom(filename):
        '''Loads the class from a file.'''
        if (filename[-4:] != '.pkl'):
            filename += '.pkl'
        
        if os.path.isdir('objects'):
            with open(os.path.join('objects', filename), 'rb') as input:
                data = pickle.load(input)
        else:                                                                                       # if for some reason the pickle folder isn't there
            with open(filename, 'rb') as input:
                data = pickle.load(input)
        return data
        
# non-class-functions below

def Ellipsoid(N_obj, dist_type='Normal_r', axes=None, **param):
    '''Make a spherical distribution of objects using the given 1d radial distribution type.
    If axes-scales are given [a*x, b*y, c*z], then shape is elliptical instead of spherical.
    Takes additional parameters for the r-distribution function.
    '''
    # check if the dist type exists
    if (dist_type[-2:] != '_r'):
        dist_type += '_r'                                                                           # add the r to the end for radial version
        
    dist_list = list(set(fnmatch.filter(dir(dist), '*_r')))
                        
    if (dist_type not in dist_list):
        warnings.warn('ObjectGen//Ellipsoid: Specified distribution type does not exist. Using default (=Normal_r)', SyntaxWarning)
        dist_type = 'Normal_r'

    # check if right parameters given    
    sig = signature(eval('dist.' + dist_type))                                                      # parameters of the dist function (includes n)
    dict = param                                                                                    # need a copy for popping in iteration
    for key in dict:
        if key not in sig.parameters:
            warnings.warn('ObjectGen//Ellipsoid: Wrong keyword given in distribution parameters. Deleted entry.' + 
                            '\n    {0} = {1}'.format(key, param.pop(key, None)), SyntaxWarning)
    
    r_dist = eval('dist.' + dist_type)(n=N_obj, **param)                                            # the radial distribution
    phi_dist = dist.AnglePhi(N_obj)                                                                 # dist for angle with x axis
    theta_dist = dist.AngleTheta(N_obj)                                                             # dist for angle with z axis
    
    xyz_dist = conv.SpherToCart(r_dist, theta_dist, phi_dist).transpose()
    
    if (axes is None):
        pass
    elif (len(axes) == 3):
        axes = np.array(axes)                                                                       # in case list is given
        xyz_dist = xyz_dist*axes/np.prod(axes)**(1/3)                                               # convert to ellipsoid (axes are relative sizes)
    else:
        warnings.warn('ObjectGen//Ellipsoid: Expecting 0 or 3 axis scales [a,b,c] for [x,y,z], {0} were given. Using [1,1,1]'.format(len(axes)), SyntaxWarning)
    
    return xyz_dist

def Spiral(N_obj):
    '''Make a spiral galaxy.'''
    
def SpiralArms():
    '''Make spiral arms.'''
    
def Irregular(N_obj):
    '''Make an irregular galaxy'''
    
def ObjMasses(N_obj=0, M_tot=0, imf=[0.08, 0.5, 150]):
    '''Generate masses using the Initial Mass Function. Either number of objects or total mass should be given.
    mass defines the lower and upper bound to the masses as well as the position of the 'knee' in the IMF.
    Also gives the difference between the total generated mass and the input mass (or estimated mass when using N_obj).
    '''
    # check input (N_obj or M_tot?)
    if (N_obj == 0) & (M_tot == 0):
        warnings.warn('ObjectGen//ObjMasses: Input mass and number of objects cannot be zero simultaniously. Using N_obj=10', SyntaxWarning)
        N_obj = 10
    elif (N_obj == 0):                                                                              # a total mass is given
        N_obj = conv.MtotToNobj(M_tot, imf)                                                         # estimate the number of objects to generate
        
    # mass
    M_init = dist.invCIMF(N_obj, imf)                                                               # assign initial masses using IMF
    M_tot_gen = np.sum(M_init)                                                                      # total generated mass (will differ from input mass, if given)
    
    if (M_tot != 0):
        M_diff = M_tot_gen - M_tot
    else:
        M_tot_est = conv.NobjToMtot(N_obj, imf)
        M_diff = M_tot_gen - M_tot_est                                                              # will give the difference to the estimated total mass for n objects
        
    return M_init, M_diff

def IsocProps(M_init, age, Z):
    '''IsochroneProperties: Assigns the other properties to the stellar objects using isochrone files and 
    the given initial masses of the objects.
    The objects should not have a lower initial mass than the lowest mass in the isochrone file.
    '''
    M_ini, M_act, log_L, log_Te, log_g, mag, mag_names = OpenIsochrone(age, Z, columns='all')       # get the isochrone values
    mag_num = len(mag_names)
    
    # interpolate the mass at log_age to get values for the other parameters
    M_act_obj = np.interp(M_init, M_ini, M_act, right=0)                                            # return 0 for stars heavier than boundary in isoc file (dead stars) [may change later in the program]
    log_L_obj = np.interp(M_init, M_ini, log_L, right=-9)                                           # return -9 --> L = 10**-9 Lsun     [subject to change]
    log_Te_obj = np.interp(M_init, M_ini, log_Te, right=1)                                          # return 1 --> Te = 10 K            [subject to change]
    log_g_obj = np.interp(M_init, M_ini, log_g, right=5)                                            # return 5 --> g = 10**5 cm/s^2     [subject to change]
    # [note] assigning the right properties to remnants has been moved to the function RemnantProps
    
    mag_obj = np.zeros([mag_num, len(M_init)])
    for i in range(mag_num):
        mag_obj[i] = np.interp(M_init, M_ini, mag[i], left=30, right=30)                            # return 30 --> L of less than 10**-9
    
    return M_act_obj, log_L_obj, log_Te_obj, log_g_obj, mag_obj, mag_names
    
def RemnantProps(M_init, age, Z, remnants='all'):
    '''Assigns properties to the stellar remnants based on their initial mass (and maybe age).
    These objects are the ones that have been assigned a current mass of 0 in the above function (IsocProps)
    '''
    if (remnants == 'all'):
        remnants = np.array([True for i in len(M_init)])                                            # if no mask given, all are assumed to be remnants
    
    # stellar age
    if (age <= 12):                                                                                 # determine if logarithm or not
        age = 10**age                                                                               # if so, go back to linear
    
    lifetime = form.MSLifetime(M_init)                                                              # estimated MS time of the stars
    remnant_time = age - lifetime                                                                   # approx. time that the remnant had to cool
    
    r_M_obj_act = form.RemnantMass(M_init[remnants], Z)                                             # approx. remnant masses
    
    r_radii = form.RemnantRadius(r_M_obj_act)                                                       # approx. remnant radii
    r_Te_obj = form.RemnantTeff(r_M_obj_act, r_radii, remnant_time)                                 # approx. remnant temperatures
    
    r_logTe_obj = np.log10(r_Te_obj)
    r_logL_obj = np.log10(BBLuminosity(r_radii, r_Te_obj))                                          # remnant luminosity == BB radiation
    
    r_logg_obj = form.RadiusToGravity(r_radii, r_M_obj_act)                                         # log g is just a conversion
    r_mag_obj = 11          # WD
    #TODO: add better properties (and check the HRD for pos of WD)
    return r_M_obj_act, r_logL_obj, r_logTe_obj, r_mag_obj

def StarFormHistory(max_age, log_t, sfr='exp', t0=1e10, tau_star=1e15):
    '''Finds the relative number of stars to give a certain age up to maximum given age, using a certain SFR.
    log_t provides the available times in the isochrone file.'''
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
    '''Finds the spectral type of a star from its properties using a table.
    T_eff: effective temperature (K)
    Lum: logarithm of the luminosity in Lsun
    Mass: log of the mass in Msun
    '''
    #   Stellar_Type    Mass        Luminosity	Radius       Temp    B-V	Mv	BC(Temp) Mbol
    #   None            Mstar/Msun  Lstar/Lsun	Rstar/Rsun   K       Color	Mag	Corr	 Mag
    
    with open(os.path.join('tables', 'all_stars.txt')) as file:
        spec_tbl_names  = np.array(file.readline()[:-1].split('\t'))                                # read in the column names
    spec_tbl_names = {name: i for i, name in enumerate(spec_tbl_names[1:])}                         # make a dict for ease of use
    
    spec_tbl_types = np.loadtxt(os.path.join('tables', 'all_stars.txt'), dtype=str, usecols=[0], unpack=True)
    spec_tbl = np.loadtxt(os.path.join('tables', 'all_stars.txt'), dtype=float, usecols=[1,2,3,4,5,6,7,8], unpack=True)
    
    M_bol_sun = 4.74
    spec_letter = ['M', 'K', 'G', 'F', 'A', 'B', 'O']                                               # basic spectral letters (increasing temperature)
    
    # collect the relevant spectral types 
    # (C, R, N, W, S and most D are disregarded, include DC - WD with no strong spectral features, also leave out subdwarfs)
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
    
    # spec_types = type_selection[indices]                                                            # the spectral types of the stars
    
    #TODO: WD are taken care of this way, but what to do with NS/BH? (now they would probably get WD spectra)
    
    return indices, type_selection
    
def NumberLimited(N, age, Z, imf=imf_defaults):
    '''Retrieves the lower mass limit for which the number of stars does not exceed 10**7. Will also
    give an upper mass limit based on the values in the isochrone.
    The intended number of generated stars, age and metallicity are needed.
    '''
    fraction = np.clip(10**7/N, 0, 1)                                                               # fraction of the total number of stars to generate
    
    M_ini, mag_vals, mag_names = OpenIsochrone(age, Z, columns='mag')                               # get the isochrone values
    mass_lim_high = M_ini[-1]                                                                       # highest value in the isochrone
    
    mass_lim_low = form.MassLimit(fraction, M_max=mass_lim_high, imf=imf)
    
    return mass_lim_low, mass_lim_high
    
def MagnitudeLimited(age, Z, mag_lim=default_mag_lim, d=10, ext=0, filter='Ks'):
    '''Retrieves the lower mass limit for which the given magnitude threshold is met. Will also give
    an upper mass limit based on the values in the isochrone.
    Works only for resolved stars. If light is integrated along the line of sight (crowded fields), 
    then don't use this method! Resulting images would not be accurate!
    distance, age, metallicity and extinction of the population of stars are needed.
    A filter must be specified in which the given limiting magnitude is measured.
    '''
    M_ini, mag_vals, mag_names = OpenIsochrone(age, Z, columns='mag')                               # get the isochrone values
    mag_vals = mag_vals[mag_names == filter][0]                                                     # select the right filter ([0] needed to reduce to 1D array)

    abs_mag_lim = form.AbsoluteMag(mag_lim, d, ext=ext)                                             # calculate the limiting absolute magnitude
    mask = (mag_vals < abs_mag_lim + 0.1)                                                           # take all mag_vals below the limit
    if not mask.any():
        mask = (mag_vals == np.min(mag_vals))                                                       # if limit too high (too low in mag) then it will break
        warnings.warn('ObjectGen//MagnitudeLimited: compacting will not work, distance too large.')

    mass_lim_low = M_ini[mask][0]                                                                   # the lowest mass where mask==True
    
    mass_lim_high = M_ini[-1]                                                                       # highest value in the isochrone
    
    return mass_lim_low, mass_lim_high
    
def OpenIsochrone(age, Z, columns='all'):
    '''Opens the isochrone file and gives the relevant columns.
    columns can be: 'all', 'mag'
    age can be either linear or logarithmic input.
    '''
    # opening the file (actual opening lateron)
    file_name = os.path.join('tables', 'isoc_Z{1:1.{0}f}.dat'.format(-int(np.floor(np.log10(Z)))+1, Z))
    if not os.path.isfile(file_name):
        raise FileNotFoundError('ObjectGen//OpenIsochrone: file {0} not found. Try a different metallicity.'.format(file_name))
    
    # names to use in the code, and a mapping to the isoc file column names 
    code_names = np.array(['log_age', 'M_initial', 'M_actual', 'log_L', 'log_Te', 'log_g', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks'])
    mag_names = code_names[6:]                                                                      # names of filters for later reference
    mag_num = len(mag_names)
    var_names, column_names = np.loadtxt(os.path.join('tables', 'column_names.dat'), dtype=str, unpack=True)
    
    if (len(code_names) != len(var_names)):
        raise SyntaxError('ObjectGen//OpenIsochrone: file \'column_names.dat\' has incorrect names specified. Use: {0}'.format(', '.join(code_names)))
    elif np.any(code_names != var_names):
        raise SyntaxError('ObjectGen//OpenIsochrone: file \'column_names.dat\' has incorrect names specified. Use: {0}'.format(', '.join(code_names)))
        
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
    
    # load the right columns (1/2)
    if (columns == 'all'):
        log_t, M_ini, M_act, log_L, log_Te, log_g = np.loadtxt(file_name, usecols=var_cols, unpack=True)
        mag = np.loadtxt(file_name, usecols=mag_cols, unpack=True)
    elif (columns == 'mag'):
        log_t, M_ini = np.loadtxt(file_name, usecols=var_cols[:2], unpack=True)
        mag = np.loadtxt(file_name, usecols=mag_cols, unpack=True)
    elif (columns == 'lum'):
        log_t, M_ini = np.loadtxt(file_name, usecols=var_cols[:2], unpack=True)
        log_L = np.loadtxt(file_name, usecols=(col_dict[name_dict['log_L']]), unpack=True)
    else:
        raise ValueError('ObjectGen//OpenIsochrone: wrong argument for \'columns\' given.')
    
    # stellar age
    if (age <= 12):                                                                                 # determine if logarithm or not
        log_age = age                                                                               # define log_age
    else:
        log_age = np.log10(age)                                                                     # calculate log_age (assuming it is not already log)
        
    log_t_min = np.min(log_t)                                                                       # minimum available age                                                
    log_t_max = np.max(log_t)                                                                       # maximum available age
    uni_log_t = np.unique(log_t)                                                                    # unique array of ages
        
    lim_min = (log_age < log_t_min - 0.01)                         
    lim_max = (log_age > log_t_max + 0.01)
    if lim_min or lim_max:                                                                          # check the age limits
        raise Warning('ObjectGen//OpenIsochrone: Specified age exceeds limit for Z={0}. Using limit value (log_age={1}).'
                        .format(Z, lim_min*log_t_min + lim_max*log_t_max))
        log_age = lim_min*log_t_min + lim_max*log_t_max
    
    t_steps = uni_log_t[1:] - uni_log_t[:-1]                                                        # determine the age steps in the isoc files (step sizes may vary)
    a = np.log10((10**(t_steps) + 1)/2)                                                             # half the age step in logarithms
    b = t_steps - a                                                                                 # a is downward step, b upward
    a = np.insert(a, 0, 0.01)                                                                       # need step down for first t value
    b = np.append(b, 0.01)                                                                          # need step up for last t value
    log_closest = uni_log_t[(uni_log_t > log_age - a) & (uni_log_t <= log_age + b)]                 # the closest available age (to given one)
    where_t = np.where(log_t == log_closest)
    
    # return the right columns (2/2)
    M_ini = M_ini[where_t]                                                                          # M_ini is always returned
    if (columns == 'all'):
        M_act = M_act[where_t]
        log_L = log_L[where_t]
        log_Te = log_Te[where_t]
        log_g = log_g[where_t]
        mag = np.array([m[0] for m in mag[:, where_t]])                                             # weird conversion needed because normal slicing would result in deeper array
        return M_ini, M_act, log_L, log_Te, log_g, mag, mag_names
    elif (columns == 'mag'):
        mag = np.array([m[0] for m in mag[:, where_t]])                                             # weird conversion needed because normal slicing would result in deeper array
        return M_ini, mag, mag_names
    elif (columns == 'lum'):
        log_L = log_L[where_t]
        return M_ini, log_L
    
def FixTotal(tot, nums):
    '''Check if nums add up to total and fixes it.''' 
    i = 0
    while (np.sum(nums) != tot):                                                          # assure number conservation
        i += 1
        sum_nums = np.sum(nums)
        if (sum_nums > tot):
            nums[-np.mod(i, len(nums))] -= 1
        elif (sum_nums < tot):
            nums[-np.mod(i, len(nums))] += 1
            
    return nums

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


