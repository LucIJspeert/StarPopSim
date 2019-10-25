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


def NumberOfPopulations(N, M_tot, ages, metal):
    """Figures out the intended number of populations from four input parameters."""
    if hasattr(N, '__len__'):
        len_N = len(N)
    else:
        len_N = 1
    
    if hasattr(M_tot, '__len__'):
        len_M_tot = len(M_tot)
    else:
        len_M_tot = 1
    
    if hasattr(ages, '__len__'):
        len_ages = len(ages)
    else:
        len_ages = 1
    
    if hasattr(metal, '__len__'):
        len_metal = len(metal)
    else:
        len_metal = 1
    
    len_array = np.array([len_N, len_M, len_ages, len_metal])
    n_pop = max(len_array)
    if (sum(len_array == n_pop) < (len(len_array) - sum(len_array == 1))):
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


def CastMTotal(M_tot, n_pop):
    """Cast input for total initial mass into the right format."""
    if hasattr(M_tot, '__len__'):
        M_tot = np.array(M_tot)
    else:
        M_tot = np.array([M_tot])
    
    len_M_tot = len(M_tot)
    if ((n_pop > 1) & (len_M_tot == 1)):
        M_tot = np.full(n_pop, M_tot[0])/n_pop                                                      # the mass is divided equally among the populations
    elif (len_M_tot < n_pop):
        M_tot = np.append(M_tot, np.full(n_pop - len_M_tot, M_tot[-1]))                             # extend length (dividing mass among pops)
        M_tot[len_M_tot:] /= (n_pop - len_M_tot)
    elif (len_M_tot > n_pop):
        warnings.warn('utils//CastMTotal: too many values received for M_tot_init', SyntaxWarning)
        M_tot = M_tot[:n_pop]                                                                       # reduce length
    return M_tot


def CheckNStars(N_stars, M_tot, n_pop, imf_par):
    """Check if we got values for number of stars, or calculate them."""
    if ((N_stars == 0) & np.all(M_tot == 0)):
        raise ValueError('objectgenerator//CheckInput: Input mass and number of stars '
                         'cannot be zero simultaniously.')
    elif (N_stars == 0):
        N_stars = conv.MtotToNstars(M_tot, imf=imf_par)                                             # estimate of the number of stars to generate
    else:
        if hasattr(N_stars, '__len__'):
            N_stars = np.array(N_stars)
        else:
            N_stars = np.array([N_stars])
        
        len_N_stars = len(N_stars)
        if ((n_pop > 1) & (len_N_stars == 1)):
            N_stars = np.full(n_pop, N_stars[0])/n_pop                                              # the stars are divided equally among the populations
        elif (len_N_stars < n_pop):
            N_stars = np.append(N_stars, np.full(n_pop - len_N_stars, N_stars[-1]))                 # extend length (dividing stars among pops)
            N_stars[len_N_stars:] /= (n_pop - len_N_stars)
        elif (len_N_stars > n_pop):
            warnings.warn('utils//CheckNStars: too many values received for N_stars', SyntaxWarning)
            N_stars = N_stars[:n_pop]                                                               # reduce length
    
    N_stars = FixTotal(np.rint(N_stars).astype(int), np.sum(N_stars))                               # make sure it is int, and adds up nicely  
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
    r_dist = r_dist.astype(object)                                                                  # to make sure we can increase the string length
    for i in range(len(r_dist)):
        if (not r_dist[i].endswith('_r')):
            r_dist[i] = r_dist[i] + '_r'                                                            # add the r to the end for radial version
        
        if (r_dist[i] not in dist_list):
            warnings.warn(('utils//CheckRadialDistType: Specified distribution <{0}> type does '
                           'not exist. Using default (=<{1}>)').format(r_dist[i], default_rdist), 
                           SyntaxWarning)
            r_dist[i] = default_rdist + '_r'
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









