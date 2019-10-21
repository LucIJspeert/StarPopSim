# Luc IJspeert
# Part of starpopsim: utility functions
##
"""Utility functions are thrown over here."""
import os
import warnings
import numpy as np

import fnmatch
import inspect
import distributions as dist


# global defaults
prop_names = np.array(['log_age', 'M_initial', 'M_current', 'log_L', 'log_Te', 'log_g', 'phase'])   # names of stellar properties
default_isoc_file = 'isoc_Z0.014.dat'                                                               # for reference (make sure this one exists!)
default_imf_par = [0.08, 150]   # M_sun     lower bound, upper bound on mass
default_rdist = 'Normal'        # see distributions module for a full list of options


def OpenIsochronesFile(Z, columns=None):
    """Opens the isochrones file and gives the right columns.
    columns: list of column names (see code_names), None will give all columns.
    """
    # check the file name (actual opening lateron)
    file_name = ('isoc_Z{1:1.{0}f}.dat').format(-int(np.floor(np.log10(Z)))+1, Z)
    file_name = os.path.join('tables', file_name)

    if not os.path.isfile(file_name):                                                               # check wether file for Z exists
        file_name = ('isoc_Z{1:1.{0}f}.dat').format(-int(np.floor(np.log10(Z))), Z)                 # try one digit less
        file_name = os.path.join('tables', file_name)
        if not os.path.isfile(file_name):
            raise FileNotFoundError(('objectgenerator//OpenIsochronesFile: file {0} not found. '
                                    'Try a different metallicity.').format(file_name))

    # mapping the names used in the code to the isoc file column names
    alt_filter_names = SupportedFilters(alt_names=True)
    full_filter_names = SupportedFilters(alt_names=False)
    code_names, isoc_names = np.loadtxt(os.path.join('tables', 'column_names.dat'),
                                        dtype=str, unpack=True)
    all_code_names = np.append(code_names, alt_filter_names)

    name_dict = {code: iso for code, iso in zip(alt_filter_names, full_filter_names)}
    name_dict.update({iso: iso for iso in full_filter_names})                                       # can also use the full names
    name_dict.update({code: iso for code, iso in zip(code_names, isoc_names)})
    name_dict.update({iso: iso for iso in isoc_names})                                              # can use isoc names directly

    with open(file_name) as file:
        for line in file:
            if line.startswith('#'):
                header = np.array(line.replace('#', '').split())                                    # find the column names in the isoc file
            else:
                break

    col_dict = {name: col for col, name in enumerate(header)}

    if columns is None:
        cols_to_use = [col_dict[name_dict[name]] for name in all_code_names]
    else:
        cols_to_use = [col_dict[name_dict[name]] for name in columns]                               # use the given column names

    data = np.loadtxt(file_name, usecols=cols_to_use, unpack=True)
    return data


def OpenPhotometricData(columns=None, filters=None):
    """Gives the data in the file photometric_filters.dat as a structured array.
    Columns can be specified if less of the data is wanted (array of strings).
    Filters can be specified to get only those rows (array of strings)
    """
    file_name = os.path.join('tables', 'photometric_filters.dat')
    column_types = [('name', 'U16'),
                    ('alt_name', 'U4'),
                    ('mean', 'f4'),
                    ('width', 'f4'),
                    ('solar_mag', 'f4'),
                    ('zp_flux', 'f4')]
    reduce = False

    # select the columns
    if columns is not None:
        use_i = [i for i, item in enumerate(column_types) if (item[0] in columns)]
        used_types = [column_types[i] for i in use_i]
        if (len(columns) == 1):
            reduce = True
            single_column = columns[0]
        columns = use_i
    else:
        used_types = column_types

    phot_dat = np.loadtxt(file_name, dtype=used_types, usecols=columns)

    # select the filters
    if filters is not None:
        filter_names = np.loadtxt(file_name, dtype=column_types[:2], usecols=[0,1])
        mask_filters = np.sum([((filter_names['name'] == name) | (filter_names['alt_name'] == name))
                               for name in filters], dtype=bool, axis=0)
        phot_dat = phot_dat[mask_filters]

    # some default conversions
    if ('mean' in np.array(used_types)[:, 0]):
        phot_dat['mean'] = phot_dat['mean']*1e-9                                                    # convert to m
    if ('width' in np.array(used_types)[:, 0]):
        phot_dat['width'] = phot_dat['width']*1e-9                                                  # convert to m
    if ('zp_flux' in np.array(used_types)[:, 0]):
        phot_dat['zp_flux'] = phot_dat['zp_flux']*1e7                                               # convert to W/m^3
    if reduce:
        phot_dat = phot_dat[single_column]                                                          # get rid of the array structure
    return phot_dat


def SelectAge(age, Z):
    """Selects the timestep in the isochrone closest to the given age (lin or log years)."""
    log_t = OpenIsochronesFile(Z, columns=['log_age'])

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
        warnings.warn(('objectgenerator//OpenIsochrone: Specified age exceeds limit for this'
                       'isochrones file. Using limit value (log_age={1}).').format(Z,
                       lim_min*log_t_min + lim_max*log_t_max), RuntimeWarning)
        log_age = lim_min*log_t_min + lim_max*log_t_max

    t_steps = uni_log_t[1:] - uni_log_t[:-1]                                                        # determine the age steps in the isoc files (step sizes may vary)
    a = np.log10((10**(t_steps) + 1)/2)                                                             # half the age step in logarithms
    b = t_steps - a                                                                                 # a is downward step, b upward
    a = np.insert(a, 0, 0.01)                                                                       # need step down for first t value
    b = np.append(b, 0.01)                                                                          # need step up for last t value
    log_closest = uni_log_t[(uni_log_t > log_age - a) & (uni_log_t <= log_age + b)]                 # the closest available age (to given one)
    return np.where(log_t == log_closest)[0]


def StellarIsochrone(age, Z, columns=None):
    """Gives the isochrone data for a specified age and metallicity (Z).
    columns: list of column names (see code_names), None will give all columns.
    """
    data = OpenIsochronesFile(Z, columns=columns)
    where_t = SelectAge(age, Z)

    if (len(np.shape(data)) == 1):
        data = data[where_t]
    else:
        data = data[:, where_t]
    return data


def SupportedFilters(alt_names=True):
    """Returns the supported filter names corresponding to the (default set of) magnitudes.
    alt_names=False makes the function return the full filter names.
    """
    file_name = os.path.join('tables', default_isoc_file)
    with open(file_name) as file:
        for line in file:
            if line.startswith('#'):
                column_names = np.array(line.replace('#', '').split())                              # find the column names in the isoc file
            else:
                break

    filter_data = OpenPhotometricData(columns=['name', 'alt_name'], filters=None)
    full_name_set = [name in column_names for name in filter_data['name']]
    alt_name_set = [name in column_names for name in filter_data['alt_name']]
    supported_filters = [a or b for (a, b) in zip(full_name_set, alt_name_set)]
    ordered_set = filter_data['name'][supported_filters]
    if alt_names:
        with_alt = [(name != '-') for name in filter_data['alt_name'][supported_filters]]            # filters with alt name
        ordered_set[with_alt] = filter_data['alt_name'][supported_filters][with_alt]
    return ordered_set


def GetFilterMask(filters):
    """Makes a mask for the data to return the right magnitude set(s)."""
    alt_names = SupportedFilters(alt_names=True)
    full_names = SupportedFilters(alt_names=False)
    if filters is None:
        filters = alt_names
    elif isinstance(filters, str):
        filters = [filters]
    mask_filters = [((full_names == name) | (alt_names == name)) for name in filters]
    return np.sum(mask_filters, dtype=bool, axis=0)


def FixTotal(nums, tot):
    """Check if nums add up to total and fixes it."""
    i = 0
    while (np.sum(nums) != tot):                                                                    # assure number conservation
        i += 1
        sum_nums = np.sum(nums)
        if (sum_nums > tot):
            nums[-np.mod(i, len(nums))] -= 1
        elif (sum_nums < tot):
            nums[-np.mod(i, len(nums))] += 1

    return nums


def IsFloat(value, integer=False):
    """Just a little thing to check input for being integer or float."""
    if integer:                                                                                     # check for int instead
        try:
            int(value)
            return True
        except:
            return False

    if isinstance(value, str):                                                                      # check strings for *'s
        value = value.replace('*10**', 'e')
        value = value.replace('10**', '1e')

    try:
        float(value)
        return True
    except:
        return False


def WhileAsk(question, options, add_opt=None, function='', check='str', help_arg=''):
    """Asks a question and checks input in a while loop.
    question: string containing the question to ask
    options: list of options separated by </>; can be an empty string
    add_opt: list of additional options that are not printed (i.e. to define abbreviations)
    function: the name of the calling function, for use in Help
    check: type of answer to check for; can be <str>, <float> or <int>
    args: passed to Help function
    """
    if (add_opt is None):
        add_opt = []

    a = 0
    b = 1
    for i,char in enumerate(options):
        if (char == '['):
            a = i
        elif (char == ']'):
            b = i

    default_val = options[a+1:b]                                                                    # default option is between the [brackets]

    if (options == ''):
        ans = input(question + ': ')                                                                # ask the question
    else:
        ans = input(question + ' ' + options + ': ')                                                # ask the question

    if (check == 'str'):
        option_list = options.lower().replace('[', '').replace(']', '').split('/')                  # lower the case, default is between [], options separated by /
        option_list += [item.lower() for item in add_opt]                                           # add the additional options

        if ((options == '') & len(option_list) == 1):
            while ((ans == '') | (' ' in ans)):
                ans = CheckAnswer(ans, question, options, default_val, function, help_arg)          # check the answer (this gives the possibility for open questions)
        else:
            while (ans.lower() not in option_list):
                ans = CheckAnswer(ans, question, options, default_val, function, help_arg)          # check the answer

    elif (check == 'float'):
        while not IsFloat(ans):
            ans = CheckAnswer(ans, question, options, default_val, function, help_arg)              # check the answer

    elif (check == 'int'):
        while not IsFloat(ans, integer=True):
            ans = CheckAnswer(ans, question, options, default_val, function, help_arg)              # check the answer

    return ans


def CheckAnswer(ans, question, options, default, function, *args):
    """Helper function of WhileAsk."""
    if ((ans == '') & (default != '')):
        ans = default
    elif (ans in ['help', 'h']):
        AskHelp(function, *args)                                                                    # call help function
        if (options == ''):
            ans = input(question + ': ')
        else:
            ans = input(question + ' ' + options + ': ')
    elif (ans.lower() in ['quit', 'q']):
        raise KeyboardInterrupt #SystemExit                                                         # exit if wanted
    else:
        if (options == ''):
            ans = input(question + ': ')
        else:
            ans = input(question + ' ' + options + ': ')

    return ans


def CheckInput(self):
    """Checks the given input for compatibility."""# todo: make this work over here (name 'castinput'?)
    # # check metallicity and age and make sure they are arrays
    # if hasattr(self.ages, '__len__'):
    #     self.ages = np.array(self.ages)
    # else:
    #     self.ages = np.array([self.ages])
    #     
    # if hasattr(self.metal, '__len__'):
    #     self.metal = np.array(self.metal)
    # else:
    #     self.metal = np.array([self.metal])
    # 
    # # much used qtt's
    # num_ages = len(self.ages)
    # num_metal = len(self.metal)
    # 
    # # check (ages and metallicity) for empty input and compatible length
    # if (num_ages == 0):
    #     raise ValueError('objectgenerator//CheckInput: No age was defined.')
    # elif (num_metal == 0):
    #     raise ValueError('objectgenerator//CheckInput: No metallicity was defined.')
    # elif (num_ages != num_metal):                                                               # make sure they have the right length
    #     if (num_ages == 1):
    #         self.ages = self.ages[0]*np.ones(num_metal)
    #     elif (num_metal == 1):
    #         self.metal = self.metal[0]*np.ones(num_ages)
    #     else:
    #         warnings.warn(('objectgenerator//CheckInput: age and metallicity have '
    #                         'incompatible length {0} and {1}. Discarding excess.'
    #                         ).format(num_ages, num_metal), SyntaxWarning)
    #         new_len = min(num_ages, num_metal)
    #         self.ages = self.ages[:new_len]
    #         self.metal = self.metal[:new_len]
    #         
    #     num_ages = len(self.ages)                                                               # update length
    #     num_metal = len(self.metal)                                                             # update length
    # num_pop = num_ages                                                                          # define number of stellar populations
    
    # # check input for rel_num [must go before usage of num_pop, after ages and metallicity]
    # if hasattr(self.rel_number, '__len__'):
    #     self.rel_number = np.array(self.rel_number)
    # else:
    #     self.rel_number = np.ones(num_pop)                                                      # any single number will result in equal amounts
    # 
    # relnum_len = len(self.rel_number)
    # 
    # if ((relnum_len > num_pop) & (num_pop != 1)):
    #     warnings.warn(('objectgenerator//CheckInput: too many relative numbers given. '
    #                     'Discarding excess.'), SyntaxWarning)
    #     self.rel_number = self.rel_number[:num_pop]
    # elif ((relnum_len > num_pop) & (num_pop == 1)):
    #     self.ages = np.array([self.ages[0] for i in range(relnum_len)])                         # in this case, add populations of the same age
    #     self.metal = np.array([self.metal[0] for i in range(relnum_len)])                       #  and metallicity
    #     num_ages = len(self.ages)                                                               # update length
    #     num_metal = len(self.metal)                                                             # update length
    #     num_pop = num_ages                                                                      # [very important] update number of populations
    # elif (relnum_len != num_pop):
    #     self.rel_number = np.ones(num_pop)
    # 
    # rel_frac = self.rel_number/np.sum(self.rel_number)                                          # fraction of the total in each population
    
    # # check format of imf_param
    # if hasattr(self.imf_param, '__len__'):
    #     self.imf_param = np.array(self.imf_param)
    # else:
    #     warnings.warn(('objectgenerator//CheckInput: incorrect input type for imf_par, '
    #                     'using default (={0}).').format(default_imf_par), SyntaxWarning)
    #     self.imf_param = np.array([default_imf_par for i in range(num_pop)]) 
    # 
    # imf_shape = np.shape(self.imf_param)
    # imf_par_len = len(default_imf_par)                                                             # how long one set of imf pars is
    # 
    # if (len(imf_shape) == 1):
    #     if (imf_shape[0] == imf_par_len):
    #         self.imf_param = np.array([self.imf_param for i in range(num_pop)])                 # make it a 2D array using same imf for all populations
    #     elif (imf_shape[0]%imf_par_len == 0):
    #         self.imf_param = np.reshape(self.imf_param, 
    #                                     [imf_shape[0]//imf_par_len, imf_par_len])               # make it a 2D array
    #     else:
    #         raise ValueError('objectgenerator//CheckInput: Wrong number of arguments for '
    #                             'imf_par, must be multiple of {0}.'.format(imf_par_len))
    # 
    # imf_shape = np.shape(self.imf_param)                                                        # update shape
    # 
    # if (imf_shape[0] > num_pop):
    #     warnings.warn(('objectgenerator//CheckInput: Too many arguments for imf_par. '
    #                     'Discarding excess.'), SyntaxWarning)
    #     self.imf_param = self.imf_param[0:num_pop]
    # elif (imf_shape[0] == 1) & (num_pop > 1):
    #     self.imf_param = np.full([num_pop, imf_par_len], self.imf_param[0])                     # fill up imf_par  
    # elif (imf_shape[0] < num_pop):
    #     filler = [default_imf_par for i in range(num_pop - imf_shape[0]//imf_par_len)]
    #     self.imf_param = np.append(self.imf_param, filler, axis=0)                              # fill missing imf_par with default        
    
    # # check the minimum available mass in isoc file [must go after imf_param check]
    # max_M_L = 0                                                                                 # maximum lowest mass (to use in IMF)
    # for i in range(num_pop):
    #     M_ini = utils.StellarIsochrone(self.ages[i], self.metal[i], columns=['M_initial'])
    #     max_M_L = max(max_M_L, np.min(M_ini))
    # 
    # imf_max_M_L = np.array([self.imf_param[:,0], np.full(num_pop, max_M_L)])
    # self.imf_param[:,0] =  np.max(imf_max_M_L, axis=0)                                          # check against user input (if that was higher, use that instead)
    
    # # check input: N_stars or M_tot_init? --> need N_stars [must go after min mass check]
    # if (self.N_stars == 0) & (self.M_tot_init == 0):
    #     raise ValueError('objectgenerator//CheckInput: Input mass and number of stars '
    #                         'cannot be zero simultaniously. Using N_stars=1000')
    # elif (self.N_stars == 0):                                                                   # a total mass is given
    #     pop_num = conv.MtotToNstars(self.M_tot_init*rel_frac, imf=self.imf_param)
    #     self.N_stars = np.sum(pop_num)                                                          # estimate of the number of stars to generate
    # else:
    #     pop_num = np.rint(rel_frac*self.N_stars).astype(int)                                    # rounded off number
    #     self.N_stars = np.rint(np.sum(pop_num)).astype(int)                                     # make sure N_stars is int and rounded off
    # 
    # self.pop_number = utils.FixTotal(self.N_stars, pop_num)                                     # make sure the population numbers add up to N_total
    
    # # check the SFH
    # if isinstance(self.sfhist, str):
    #     self.sfhist = np.array([self.sfhist])
    # elif hasattr(self.sfhist, '__len__'):
    #     self.sfhist = np.array(self.sfhist)
    # 
    # sfh_len = len(self.sfhist)
    # if ((sfh_len == 1) & (num_pop != 1)):
    #     self.sfhist = np.full(num_pop, self.sfhist[0])
    # elif (sfh_len < num_pop):
    #     raise ValueError('objectgenerator//CheckInput: too few sfh types given.')
    # elif (sfh_len > num_pop):
    #     warnings.warn(('objectgenerator//CheckInput: too many sfh types given. '
    #                     'Discarding excess.'), SyntaxWarning)
    #     self.sfhist = self.sfhist[:num_pop]
    
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
    
    return

def CheckRadialDistribution(self):
    """Check the radial distribution input."""
    # # check if the dist type(s) exists and get the function signatures
    # num_pop = len(self.pop_number)
    # dist_list = list(set(fnmatch.filter(dir(dist), '*_r')))
    # 
    # if isinstance(self.r_dist_types, str):
    #     self.r_dist_types = [self.r_dist_types]                                                   # make sure it is a list of str
    # 
    # # check number of dists    
    # n_r_dists = len(self.r_dist_types)
    # if ((n_r_dists == 1) & (num_pop > 1)):
    #     self.r_dist_types.extend([self.r_dist_types[0] for i in range(num_pop - n_r_dists)])      # duplicate
    #     n_r_dists = len(self.r_dist_types)                                                       # update the number
    # elif (n_r_dists < num_pop):                                                                 
    #     self.r_dist_types.extend([default_rdist for i in range(num_pop - n_r_dists)])            # fill up with default
    #     n_r_dists = len(self.r_dist_types)                                                       # update the number
    #     
    # key_list = []
    # val_list = []
    # r_dist_n_par = []
    # 
    # for i in range(n_r_dists):
    #     if (self.r_dist_types[i][-2:] != '_r'):
    #         self.r_dist_types[i] += '_r'                                                         # add the r to the end for radial version
    #     
    #     if (self.r_dist_types[i] not in dist_list):
    #         warnings.warn(('objectgenerator//CheckInput: Specified distribution <{0}> type '
    #                         'does not exist. Using default (={1})'
    #                         ).format(self.r_dist_types[i], default_rdist), SyntaxWarning)
    #         self.r_dist_types[i] = default_rdist
    #             
    #     sig = inspect.signature(eval('dist.' + self.r_dist_types[i]))
    #     key_list.append([k for k, v in sig.parameters.items() if k is not 'n'])                 # add the signature keywords to a list
    #     val_list.append([v.default for k, v in sig.parameters.items() if k is not 'n'])         # add the signature defaults to a list
    #     r_dist_n_par.append(len(key_list[i]))                                                   # the number of parameters for each function
        
    # # check if dist parameters are correctly specified
    # if isinstance(self.r_dist_param, dict):                                                     # if just one dict, make a list of (one) dict
    #     self.r_dist_param = [self.r_dist_param]
    # elif not hasattr(self.r_dist_param, '__len__'):                                             # if just one parameter is given, also make a list of (one) dict
    #     self.r_dist_param = [{key_list[0][0]: self.r_dist_param}]
    # elif isinstance(self.r_dist_param, list):
    #     param_shape = np.shape(self.r_dist_param)
    #     if np.all([isinstance(item, (int, float)) for item in self.r_dist_param]):              # if a 1D list of parameters is given, fill a 2D list that has the correct form
    #         temp_par_list = [[]]
    #         track_index = np.cumsum(r_dist_n_par)
    #         j = 0
    #         
    #         for i in range(param_shape[0]):
    #             if np.any(i == track_index):
    #                 temp_par_list.append([self.r_dist_param[i]])                                # append a new sublist
    #                 j +=1                                                                       # keep track of number of sublists
    #             else:
    #                 temp_par_list[j].append(self.r_dist_param[i])                               # append to current sublist
    #         
    #         self.r_dist_param = temp_par_list
    #     elif np.all([isinstance(item, (int, float)) for item in self.r_dist_param]):            # if list with numbers combined, do not want that, fix!
    #         temp_par_list = []
    #         
    #         for i in range(param_shape[0]):
    #             if isinstance(self.r_dist_param[i], (int, float)):
    #                 temp_par_list.append([self.r_dist_param[i]])
    #             else:
    #                 temp_par_list.append(self.r_dist_param[i])
    #         
    #         self.r_dist_param = temp_par_list
    #     
    #     param_shape = np.shape(self.r_dist_param)                                               # recalculate it
    #     
    #     if (np.all([isinstance(item, list) for item in self.r_dist_param])):                    # if a 2D list is given (or made above), check further compatibility
    #         if (param_shape[0] > n_r_dists):
    #             warnings.warn(('objectgenerator//CheckInput: Too many radial distribution '
    #                             'parameters given. Discarding excess.'), SyntaxWarning)
    #             self.r_dist_param = self.r_dist_param[0:n_r_dists]
    #         elif (param_shape[0] < n_r_dists):                                                  # fill up missing length with defaults
    #             filler = [[val_list[param_shape[0] + i]] 
    #                         for i in range(n_r_dists - param_shape[0])]
    #             self.r_dist_param += filler
    #         
    #         for i, param in enumerate(self.r_dist_param):
    #             if (len(param) < r_dist_n_par[i]):
    #                 self.r_dist_param[i].extend([item for item in val_list[i][len(param):]])    # not enough parameters for a particular distribution
    #         
    #         
    #         temp_par_dict_list = []                                                             # now it is ready finally for making a dict out of it
    #         for i in range(n_r_dists):
    #             temp_par_dict_list.append({key_list[i][k]: self.r_dist_param[i][k] 
    #                                         for k in range(r_dist_n_par[i])})
    #             
    #         self.r_dist_param = temp_par_dict_list
    #             
    # else:
    #     raise TypeError('objectgenerator//CheckInput: Incompatible data type for rdistpar')     # it is something else... burn it with fire!
    #     
    # for i, param_dict in enumerate(self.r_dist_param):
    #     if not bool(param_dict):                                                                # if dict empty, fill with defaults
    #         self.r_dist_param[i] = {key_list[i][k]: val_list[i][k] 
    #                                 for k in range(r_dist_n_par[i])}
    # 
    # n_r_param = len(self.r_dist_param)
    # if (n_r_param < num_pop):                                                                   # check parameter dict number
    #     self.r_dist_param.extend([{key_list[i][k]: val_list[i][k] 
    #                                 for k in range(r_dist_n_par[i])} 
    #                                 for i in range(n_r_param, num_pop)])
    #     n_r_param = len(self.r_dist_types)
    
    return


def NumberOfPopulations(ages, metal, rel_num):
    """Figures out the intended number of populations from three input parameters."""
    if hasattr(ages, '__len__'):
        len_ages = len(ages)
    else:
        len_ages = 1
    
    if hasattr(metal, '__len__'):
        len_metal = len(metal)
    else:
        len_metal = 1
        
    if hasattr(rel_num, '__len__'):
        len_rel_num = len(rel_num)
    else:
        len_rel_num = 1
    
    len_array = np.array([len_ages, len_metal, len_rel_num])
    n_pop = max(len_array)
    if (sum(len_array == n_pop) < (3 - sum(len_array == 1))):
        warnings.warn(('utils//NumberOfPopulations: Input of ages, metallicityies or relative '
                       'number did not match 1 or {0}. Unexpected behaviour might '
                       'occur'.format(n_pop)), SyntaxWarning)
    return n_pop


def CastSimpleArray(arr, length, fill_value='last', warning=None):
    """Cast input for a 1D array into the right format.
    Needs the input as well as the intended lenght of the array.
    Optionally a fill value can be given to fill up missing values, default is last value in arr.
        a warning message can be supplied for when too many values are given.
    """
    if hasattr(arr, '__len__'):
        arr = np.array(arr)
    else:
        arr = np.array([arr])
    
    if (fill_value == 'last'):
        fill_value = arr[-1]                                                                        # fill with last value by default
    
    len_arr = len(arr)
    if ((length > 1) & (len_arr == 1)):
        arr = np.full(length, arr[0])
    elif (len_arr < length):
        arr = np.append(arr, np.full(length - len_arr, fill_value))                                 # extend length
    elif (len_arr > length):
        if warning:
            warnings.warn(warning, SyntaxWarning)
        arr = arr[:length]                                                                          # reduce length
    return arr


def CastAges(ages, n_pop):
    """Cast input for ages into the right format."""
    if (not ages):
        raise ValueError('utils//CastAges: No age was defined.')
    ages = CastSimpleArray(ages, n_pop, fill_value='last')
    return ages


def CastMetallicities(metal, n_pop):
    """Cast input for metalicities into the right format."""
    if (not metal):
        raise ValueError('utils//CastMetallicities: No metallicity was defined.')
    metal = CastSimpleArray(metal, n_pop, fill_value='last')
    return metal


def CastRelNumber(rel_num, n_pop):
    """Cast input for rel_num into the right format."""
    if (not rel_num):
        rel_num = np.ones(n_pop)
    rel_num = CastSimpleArray(rel_num, n_pop, fill_value='last')
    rel_frac = rel_num/np.sum(rel_num)                                                              # fraction of the total in each population
    return rel_frac


def CastIMFParameters(imf_par, n_pop, fill_value=default_imf_par):
    """Cast input for IMF parameters into the right format."""
    if not imf_par:
        imf_par = np.full([n_pop, len(fill_value)], fill_value)
    elif hasattr(imf_par, '__len__'):
        imf_par = np.array(imf_par)
    else:
        warnings.warn(('utils//CastIMFParameters: incorrect input type for imf_par, '
                        'using default (={0}).').format(default_imf_par), SyntaxWarning)
        imf_par = np.full([n_pop, len(fill_value)], fill_value)                                     # cannot be interpreted
    
    shape_imf_par = np.shape(imf_par)
    default_len = len(fill_value)
    if ((len(shape_imf_par) == 1) & (shape_imf_par[0] == default_len)):
        imf_par = np.full([n_pop, default_len], imf_par)                                            # make it a 2D array using same imf for all populations
    elif ((len(shape_imf_par) == 1) & (shape_imf_par[0]//default_len == n_pop)):
        imf_par = imf_par.reshape([n_pop, default_len])                                             # make it a 2D array
    elif ((len(shape_imf_par) == 1) & (shape_imf_par[0]%default_len == 0)):
        imf_par = imf_par.reshape([shape_imf_par[0]//default_len, default_len])                     # make it a 2D array
        extension = np.full([n_pop - shape_imf_par[0]//default_len, default_len], imf_par[-1])
        imf_par = np.append(imf_par, extension, axis=0)                                             # extend length
    elif (len(shape_imf_par) == 1):
        warnings.warn(('utils//CastIMFParameters: incorrect input for imf_par, '
                        'using default (={0}).').format(fill_value), SyntaxWarning)
        imf_par = np.full([n_pop, len(fill_value)], fill_value)                                     # cannot be interpreted
        # otherwise (below): assume a 2D shape with correct inner axis length 
    elif ((n_pop > 1) & (shape_imf_par[0] == 1)):
        imf_par = np.full([n_pop, default_len], imf_par[0])
    elif (shape_imf_par[0] < n_pop):
        extension = np.full([n_pop - shape_imf_par[0], default_len], imf_par[-1])
        imf_par = np.append(imf_par, extension, axis=0)                                             # extend length
    elif (shape_imf_par[0] > n_pop):
        warnings.warn(('utils//CastIMFParameters: Too many arguments for imf_par. '
                        'Excess discarded.'), SyntaxWarning)
        imf_par = imf_par[:n_pop]                                                                   # reduce length
    return imf_par


def CheckLowestIMFMass(imf_par, ages, metal):
    """Check the minimum available mass in isoc file"""
    # note: must go after imf_par cast (and ages, metal cast)
    max_lower_mass = np.copy(imf_par[:, 0])                                                          # maximum lowest mass (to use in IMF)
    for i in range(len(ages)):
        M_ini = StellarIsochrone(ages[i], metal[i], columns=['M_initial'])
        max_lower_mass[i] = max(max_lower_mass[i], np.min(M_ini))                                   # check against user input (if that was higher, use that instead)
    imf_par[:, 0] = max_lower_mass
    return imf_par


def CheckNStars(N_stars, M_tot_init, rel_frac, imf_par):
    """Check if we got a value for number of stars, or calculate it."""
    if (N_stars == 0) & (M_tot_init == 0):
        raise ValueError('objectgenerator//CheckInput: Input mass and number of stars '
                         'cannot be zero simultaniously.')
    elif (N_stars == 0):
        pop_num = conv.MtotToNstars(M_tot_init*rel_frac, imf=imf_param)
        self.N_stars = np.sum(pop_num)                                                              # estimate of the number of stars to generate
    return N_stars


def CastSFHistory(sfhist, n_pop):
    """Cast input for sf-history into the right format."""
    if not sfhist:
        sfhist = np.full(n_pop, None)
    sfhist = CastSimpleArray(sfhist, n_pop, fill_value=None, warning='utils//CastSFHistory: '
                             'too many sfh types given. Excess discarded.')
    return sfhist


def CastInclination(incl, n_pop):
    """Cast input for inclination into the right format."""
    if not incl:
        incl = np.zeros(n_pop)
    incl = CastSimpleArray(incl, n_pop, fill_value=0, warning='utils//CastInclination: too many '
                           'incl values given. Excess discarded.')
    return incl


def CastRadialDistType(r_dist, n_pop):
    """Cast input for radial distribution type into the right format."""
    if not r_dist:
        r_dist = np.full(n_pop, default_rdist)
    r_dist = CastSimpleArray(r_dist, n_pop, fill_value=default_rdist, 
                             warning='utils//CastRadialDistType: too many radial distribution '
                             'types given. Excess discarded.')
    return r_dist
    

def CheckRadialDistType(r_dist):
    """Check if the dist types exist."""
    dist_list = list(set(fnmatch.filter(dir(dist), '*_r')))    
    for i in range(len(r_dist)):
        if (not r_dist[i].endswith('_r')):
            r_dist[i] += '_r'                                                                       # add the r to the end for radial version
        
        if (r_dist[i] not in dist_list):
            warnings.warn(('utils//CheckRadialDistType: Specified distribution <{0}> type does '
                           'not exist. Using default (=<{1}>)').format(r_dist[i], default_rdist), 
                           SyntaxWarning)
            r_dist[i] = default_rdist
    return r_dist


def CastRadialDistParam(r_dist_par, r_dist, n_pop):
    """Cast the radial distribution parameters into the right format.
    Uses parameter names 'n' and 's' in function signatures.
    """
    # get the function signatures for the given distributions
    func_sigs = [inspect.signature(eval('dist.' + type)) for type in r_dist]
    
    # cast to right form and check if dist parameters are correctly specified
    if not r_dist_par:
        r_dist_par = np.array([{k: v.default for k, v in sig.parameters.items() if k is not 'n'} 
                               for sig in func_sigs])                                               # if none given, fill dictionaries with defaults
    elif not hasattr(r_dist_par, '__len__'):
        r_dist_par = np.array([{k: r_dist_par for k, v in sig.parameters.items() if k is not 'n'} 
                               for sig in func_sigs])                                               # if just one parameter is given, make an array of dict
    elif isinstance(r_dist_par, dict):
        r_dist_par = np.full(n_pop, r_dist_par)                                                     # if just one dict, make an array of them
        
    len_param = len(r_dist_par)
    if ((n_pop > 1) & (len_param == 1)):
        r_dist_par = np.full(n_pop, r_dist_par[0])
    elif (len_param < n_pop):
        r_dist_par = np.append(r_dist_par, np.full(n_pop - len_param, r_dist_par[-1]))              # extend length
    elif (len_param > n_pop):
        warnings.warn('utils//CastRadialDistParam: too many radial distribution parameters '
                      'given. Excess discarded.', SyntaxWarning)
        r_dist_par = r_dist_par[:n_pop]                                                             # reduce length  
        
    # some final typecasting
    if np.all([isinstance(item, (int, float)) for item in r_dist_par]):
        r_dist_par = np.array([{'s': par} for par in r_dist_par])                                   # if 1D array of parameters given, fill dictionaries accordingly
    elif np.all([(hasattr(item, '__len__') & (not isinstance(item, dict))) for item in r_dist_par]):
        temp_arr = np.array([])
        for item, sig in zip(r_dist_par, func_sigs):
            keys = [k for k, v in sig.parameters.items() if k is not 'n']
            temp_arr = np.append(temp_arr, {keys[i]: par for i, par in enumerate(item)})
        r_dist_par = temp_arr                                                                       # if a 2D array/list is given, make an array of dict out of 
    
    return r_dist_par


def CastEllipseAxes(axes, n_pop):
    """Cast input for radial distribution type into the right format."""
    if hasattr(axes, '__len__'):
        axes = np.array(axes)
    else:
        axes = np.ones([n_pop, 3])
    
    shape_axes = np.shape(axes)
    if ((len(shape_axes) == 1) & (len(axes)%3 == 0)):
        axes = axes.reshape(len(axes)//3, 3)
    elif (len(shape_axes) == 1):
        raise ValueError('objectgenerator//AddEllipticity: wrong number of arguments '
                         'for axes, must be multiple of 3.')
    
    shape_axes = np.shape(axes)
    if ((n_pop > 1) & (shape_axes[0] == 1)):
        axes = np.full([n_pop, 3], axes[0])
    elif (shape_axes[0] < n_pop):
        axes = np.append(axes, np.full(n_pop - shape_axes[0], axes[-1]), axis=0)                    # extend length
    elif (shape_axes[0] > n_pop):
        warnings.warn(('utils//CastEllipseAxes: too many arguments for axes. '
                        'Excess discarded.'), SyntaxWarning)
        axes = axes[:n_pop]                                                                         # reduce length
    axes = axes.astype(float)
    return axes
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









