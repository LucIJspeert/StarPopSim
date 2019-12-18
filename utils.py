"""Utility functions are thrown over here."""
import os
import warnings
import numpy as np

import fnmatch
import inspect
import distributions as dist
import conversions as conv


# global defaults
# names of stellar properties:
prop_names = np.array(['log_age', 'M_initial', 'M_current', 'log_L', 'log_Te', 'log_g', 'phase'])
# for reference (make sure this one exists!)
default_isoc_file = 'isoc_Z0.014.dat'
default_imf_par = [0.08, 150]   # M_sun     lower bound, upper bound on mass
default_rdist = 'normal'        # see distributions module for a full list of options


def open_isochrones_file(Z, columns=None):
    """Opens the isochrones file and gives the right columns.
    columns: list of column names (see code_names), None will give all columns.
    """
    # check the file name (actual opening lateron)
    decimals = -int(np.floor(np.log10(Z))) + 1
    file_name = f'isoc_Z{Z:1.{decimals}f}.dat'
    file_name = os.path.join('tables', file_name)

    # check wether file for Z exists
    if not os.path.isfile(file_name):
        # try one digit less
        file_name = f'isoc_Z{Z:1.{decimals - 1}f}.dat'
        file_name = os.path.join('tables', file_name)
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f'File {file_name} not found. Try a different metallicity.')

    # mapping the names used in the code to the isoc file column names
    alt_filter_names = get_supported_filters(alt_names=True)
    full_filter_names = get_supported_filters(alt_names=False)
    code_names, isoc_names = np.loadtxt(os.path.join('tables', 'column_names.dat'), dtype=str, unpack=True)
    all_code_names = np.append(code_names, alt_filter_names)

    # define the filter names that can be used in terms of the isochrone filter names
    name_dict = {code: iso for code, iso in zip(alt_filter_names, full_filter_names)}
    name_dict.update({iso: iso for iso in full_filter_names})
    name_dict.update({code: iso for code, iso in zip(code_names, isoc_names)})
    name_dict.update({iso: iso for iso in isoc_names})

    # find the column names in the isoc file
    with open(file_name) as file:
        for line in file:
            if line.startswith('#'):
                header = np.array(line.replace('#', '').split())
            else:
                break

    col_dict = {name: col for col, name in enumerate(header)}

    if columns is None:
        cols_to_use = [col_dict[name_dict[name]] for name in all_code_names]
    else:
        # use the given column names
        cols_to_use = [col_dict[name_dict[name]] for name in columns]

    data = np.loadtxt(file_name, usecols=cols_to_use, unpack=True)
    return data


def open_photometric_data(columns=None, filters=None):
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
    else:
        use_i = None
        used_types = column_types

    phot_dat = np.loadtxt(file_name, dtype=used_types, usecols=use_i)

    # select the filters
    if filters is not None:
        filter_names = np.loadtxt(file_name, dtype=column_types[:2], usecols=[0, 1])
        mask_filters = np.sum([((filter_names['name'] == name) | (filter_names['alt_name'] == name))
                               for name in filters], dtype=bool, axis=0)
        phot_dat = phot_dat[mask_filters]

    # some default conversions
    if ('mean' in np.array(used_types)[:, 0]):
        phot_dat['mean'] = phot_dat['mean']*1e-9  # convert to m
    if ('width' in np.array(used_types)[:, 0]):
        phot_dat['width'] = phot_dat['width']*1e-9  # convert to m
    if ('zp_flux' in np.array(used_types)[:, 0]):
        phot_dat['zp_flux'] = phot_dat['zp_flux']*1e7  # convert to W/m^3
    if reduce:
        phot_dat = phot_dat[columns[0]]  # get rid of the array structure
    return phot_dat


def select_age(age, Z):
    """Selects the timestep in the isochrone closest to the given age (lin or log years)."""
    log_t = open_isochrones_file(Z, columns=['log_age'])

    # determine if logarithm or not (assumes everything <= 12 is a logarithm)
    if (age <= 12):
        log_age = age
    else:
        log_age = np.log10(age)

    log_t_min = np.min(log_t)  # minimum available age
    log_t_max = np.max(log_t)  # maximum available age
    uni_log_t = np.unique(log_t)  # unique array of ages

    lim_min = (log_age < log_t_min - 0.01)
    lim_max = (log_age > log_t_max + 0.01)
    # check if the age limits are exceeded
    if lim_min or lim_max:
        log_age = lim_min*log_t_min + lim_max*log_t_max
        warnings.warn(f'Specified age exceeds limit for isochrones file with Z={Z}. Using limit value '
                      f'(log_age={log_age}).', RuntimeWarning)

    # determine the closest available age (to the given one)
    t_steps = uni_log_t[1:] - uni_log_t[:-1]  # determine the age steps in the isoc files (step sizes may vary)
    a = np.log10((10**(t_steps) + 1)/2)  # half the age step in logarithms
    b = t_steps - a  # a is downward step, b upward
    a = np.insert(a, 0, 0.01)  # need one extra step down for first t value
    b = np.append(b, 0.01)  # need one extra step up for last t value
    log_closest = uni_log_t[(uni_log_t > log_age - a) & (uni_log_t <= log_age + b)]
    return np.where(log_t == log_closest)[0]


def stellar_isochrone(age, Z, columns=None):
    """Gives the isochrone data for a specified age and metallicity (Z).
    columns: list of column names (see code_names), None will give all columns.
    """
    data = open_isochrones_file(Z, columns=columns)
    where_t = select_age(age, Z)

    if (np.ndim(data) == 1):
        data = data[where_t]
    else:
        data = data[:, where_t]
    return data


def get_supported_filters(alt_names=True):
    """Returns the supported filter names corresponding to the (default set of) magnitudes.
    alt_names=False makes the function return the full filter names.
    """
    # find the column names in the isoc file
    file_name = os.path.join('tables', default_isoc_file)
    with open(file_name) as file:
        for line in file:
            if line.startswith('#'):
                column_names = np.array(line.replace('#', '').split())
            else:
                break

    filter_data = open_photometric_data(columns=['name', 'alt_name'], filters=None)
    full_name_set = [name in column_names for name in filter_data['name']]
    alt_name_set = [name in column_names for name in filter_data['alt_name']]
    supported_filters = [a or b for (a, b) in zip(full_name_set, alt_name_set)]
    ordered_set = filter_data['name'][supported_filters]
    if alt_names:
        # replace full names with alternatives where provided
        with_alt = [(name != '-') for name in filter_data['alt_name'][supported_filters]]
        ordered_set[with_alt] = filter_data['alt_name'][supported_filters][with_alt]
    return ordered_set


def get_filter_mask(filters):
    """Makes a mask for the data to return the right magnitude set(s)."""
    alt_names = get_supported_filters(alt_names=True)
    full_names = get_supported_filters(alt_names=False)
    if filters is None:
        filters = alt_names
    elif isinstance(filters, str):
        filters = [filters]
    mask_filters = [((full_names == name) | (alt_names == name)) for name in filters]
    return np.sum(mask_filters, dtype=bool, axis=0)


def fix_total(nums, tot):
    """Check if nums add up to total and fixes it."""
    i = 0
    while (np.sum(nums) != tot):
        i += 1
        sum_nums = np.sum(nums)
        if (sum_nums > tot):
            nums[-np.mod(i, len(nums))] -= 1
        elif (sum_nums < tot):
            nums[-np.mod(i, len(nums))] += 1

    return nums


def is_float(value, integer=False):
    """Just a little thing to check input (string) for being integer or float.
    :rtype: bool
    """
    if integer:
        # check for int instead
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False

    if isinstance(value, str):
        # check strings for *'s
        value = value.replace('*10**', 'e')
        value = value.replace('10**', '1e')

    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def while_ask(question, options, add_opt=None, function='', check='str', help_arg=''):
    """Asks a question and checks input in a while loop.
    :param question: string containing the question to ask
    :param options: list of options separated by </>; can be an empty string
    :param add_opt: list of additional options that are not printed (i.e. to define abbreviations)
    :param function: the name of the calling function, for use in Help
    :param check: type of answer to check for; can be <str>, <float> or <int>
    :param help_arg: passed to Help function
    :return: the answer to the question, as either <str>, <float> or <int>
    """
    if (add_opt is None):
        add_opt = []

    a = 0
    b = 1
    for i, char in enumerate(options):
        if (char == '['):
            a = i
        elif (char == ']'):
            b = i

    default_val = options[a+1:b]  # default option is between the [brackets]

    # ask the question
    if (options == ''):
        ans = input(question + ': ')
    else:
        ans = input(question + ' ' + options + ': ')

    # check the answer (this gives the possibility for open questions)
    if (check == 'str'):
        # lower the case, default is between [], options separated by /
        option_list = options.lower().replace('[', '').replace(']', '').split('/')
        # add the additional options
        option_list += [item.lower() for item in add_opt]

        if ((options == '') & len(option_list) == 1):
            while ((ans == '') | (' ' in ans)):
                ans = check_answer(ans, question, options, default_val, function, help_arg)
        else:
            while (ans.lower() not in option_list):
                ans = check_answer(ans, question, options, default_val, function, help_arg)

    elif (check == 'float'):
        while not is_float(ans):
            ans = check_answer(ans, question, options, default_val, function, help_arg)

    elif (check == 'int'):
        while not is_float(ans, integer=True):
            ans = check_answer(ans, question, options, default_val, function, help_arg)

    return ans


def ask_help(function, add_args):
    """Shows the help on a particular function."""
    # todo: [this is obviously not the thing to put here... remove later]
    if (function == 'StructureType'):
        print(' |  [WIP] Only clusters can be made at the moment.')
    elif (function == 'PopulationAmount'):
        print(' |  Amount of stellar populations to generate '
              '(each having different age/metallicity).')
        print(' |  If making a spiral galaxy, this is the amount of populations '
              'in the disk only (separate from the bulge and bar).')
    elif (function == 'PopAges'):
        print(' |  Values above 100 are taken to be in years. '
              'Values below 11 are taken to be log(years).')
    elif (function == 'PopMetallicity'):
        print(' |  Check the isochrone files for available metallicities.')
    elif (function == 'OptionalParam'):
        print(' |  Change parameters like the mass boundaries for the IMF, '
              'radial distribution type and more.')
    elif (function == 'IMFParam'):
        print(' |  The Kroupa IMF above 0.08 solar mass is used by default.')
        print(' |  This uses a low mass slope of -1.35 and a high mass slope of -2.35 '
              '(as Salpeter IMF).')
        print(' |  The lower bound is 0.08 M_sol, position of the knee is 0.5 M_sol, '
              'upper bound is at 150 M_sol.')
    elif (function == 'SFHType'):
        print(' |  A fixed (given) age is used by default.')
        print(' |  A period of star formation can be included: this effectively gives '
              'the stars a certain age distribution.')
        print(' |  The Star Formation History type is just the form of this distribution.')
        print(' |  The given ages will be used as maximum ages in this case. Choose from: '
              '[{0}]'.format(add_args))
    elif (function == 'RadialDistribution'):
        print(' |  A normal (gaussian) radial distribution is used by default.')
        print(' |  For more information on the available radial distributions, '
              'see the documentation.')
        print(' |  Radial distributions can be added to the module <distributions> '
              '(use the right format!).')
        print(' |  List of available distributions: [{0}]'.format(add_args))
    elif (function == 'RadialParameters'):
        print(' |  Parameters for the function: {0}'.format(add_args[0]))
        print(' |  The parameters and their default values are: {0}'.format(add_args[1]))
    elif (function == 'EllipseAxes'):
        print(' |  The axes are scaled relatively to eachother so that the volume '
              'of the ellipsiod stays constant.')
    elif (function == 'SaveFileName'):
        print(' |  The default savename is astobj_default_save. Enter a name without '
              'file extention (.pkl)')
    else:
        print(' |  No help available for this function at this stage.')

    return


def check_answer(ans, question, options, default, function, *args):
    """Helper function of while_ask.
    :rtype: string
    """
    if ((ans == '') & (default != '')):
        ans = default
    elif (ans in ['help', 'h']):
        ask_help(function, *args)  # call help function
        if (options == ''):
            ans = input(question + ': ')
        else:
            ans = input(question + ' ' + options + ': ')
    elif (ans.lower() in ['quit', 'q']):
        raise KeyboardInterrupt  # SystemExit  # exit if wanted
    else:
        if (options == ''):
            ans = input(question + ': ')
        else:
            ans = input(question + ' ' + options + ': ')

    return ans


def check_number_of_populations(N, M_tot, ages, metal):
    """Figures out the intended number of populations from four input parameters.
    :rtype: int
    """
    if hasattr(N, '__len__'):
        len_N = len(N)
    else:
        len_N = 1
    
    if hasattr(M_tot, '__len__'):
        len_M = len(M_tot)
    else:
        len_M = 1
    
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
        warnings.warn(f'Input of ages, metallicityies or relative number did not match 1 or {n_pop}. '
                      'Unexpected behaviour might occur', SyntaxWarning)
    return n_pop


def cast_simple_array(arr, length, fill_value='last', warning=None):
    """Cast input for a 1D array into the right format.
    Needs the input as well as the intended lenght of the array.
    Optionally a fill value can be given to fill up missing values, default is last value in arr.
        a warning message can be supplied for when too many values are given.
    :rtype: np.ndarray
    """
    if hasattr(arr, '__len__'):
        arr = np.array(arr)
    else:
        arr = np.array([arr])
    
    if (fill_value == 'last'):
        # fill with last value by default
        fill_value = arr[-1]

    # correct the array length
    len_arr = len(arr)
    if ((length > 1) & (len_arr == 1)):
        arr = np.full(length, arr[0])
    elif (len_arr < length):
        arr = np.append(arr, np.full(length - len_arr, fill_value))
    elif (len_arr > length):
        if warning:
            warnings.warn(warning, SyntaxWarning)
        arr = arr[:length]
    return arr


def cast_ages(ages, n_pop):
    """Cast input for ages into the right format.
    :rtype: np.ndarray
    """
    if (not ages):
        raise ValueError('No age was defined.')
    ages = cast_simple_array(ages, n_pop, fill_value='last')
    return ages


def cast_metallicities(metal, n_pop):
    """Cast input for metalicities into the right format.
    :rtype: np.ndarray
    """
    if (not metal):
        raise ValueError('No metallicity was defined.')
    metal = cast_simple_array(metal, n_pop, fill_value='last')
    return metal


def cast_imf_parameters(imf_par, n_pop, fill_value='default'):
    """Cast input for IMF parameters into the right format.
    :rtype: np.ndarray
    """
    if (fill_value == 'default'):
        fill_value = default_imf_par

    if not imf_par:
        imf_par = np.full([n_pop, len(fill_value)], fill_value)
    elif hasattr(imf_par, '__len__'):
        imf_par = np.array(imf_par)
    else:
        warnings.warn(f'Incorrect input type for imf_par; using default (={default_imf_par}).', SyntaxWarning)
        imf_par = np.full([n_pop, len(fill_value)], fill_value)
    
    shape_imf_par = np.shape(imf_par)
    default_len = len(fill_value)
    # make it a 2D array with right length
    if ((len(shape_imf_par) == 1) & (shape_imf_par[0] == default_len)):
        imf_par = np.full([n_pop, default_len], imf_par)
    elif ((len(shape_imf_par) == 1) & (shape_imf_par[0]//default_len == n_pop)):
        imf_par = imf_par.reshape([n_pop, default_len])
    elif ((len(shape_imf_par) == 1) & (shape_imf_par[0]%default_len == 0)):
        imf_par = imf_par.reshape([shape_imf_par[0]//default_len, default_len])
        extension = np.full([n_pop - shape_imf_par[0]//default_len, default_len], imf_par[-1])
        imf_par = np.append(imf_par, extension, axis=0)
    elif (len(shape_imf_par) == 1):
        warnings.warn(f'Incorrect input for imf_par; using default (={fill_value}).', SyntaxWarning)
        imf_par = np.full([n_pop, len(fill_value)], fill_value)
        # otherwise (below): assume a 2D shape with correct inner axis length 
    elif ((n_pop > 1) & (shape_imf_par[0] == 1)):
        imf_par = np.full([n_pop, default_len], imf_par[0])
    elif (shape_imf_par[0] < n_pop):
        extension = np.full([n_pop - shape_imf_par[0], default_len], imf_par[-1])
        imf_par = np.append(imf_par, extension, axis=0)
    elif (shape_imf_par[0] > n_pop):
        warnings.warn('Too many arguments for imf_par. Excess discarded.', SyntaxWarning)
        imf_par = imf_par[:n_pop]
    return imf_par


def check_lowest_imf_mass(imf_par, ages, metal):
    """Check the minimum available mass per population in isoc file.
    :rtype: np.ndarray
    """
    # note: must go after imf_par cast (and ages, metal cast)
    max_lower_mass = np.copy(imf_par[:, 0])  # maximum lowest mass (to use in IMF)
    for i in range(len(ages)):
        M_ini = stellar_isochrone(ages[i], metal[i], columns=['M_initial'])
        # check against user input (if that was higher, use that instead)
        max_lower_mass[i] = max(max_lower_mass[i], np.min(M_ini))
    imf_par[:, 0] = max_lower_mass
    return imf_par


def cast_m_total(M_tot, n_pop):
    """Cast input for total initial mass per population into the right format.
    :rtype: np.ndarray
    """
    if hasattr(M_tot, '__len__'):
        M_tot = np.array(M_tot)
    else:
        M_tot = np.array([M_tot])
    
    len_M_tot = len(M_tot)
    if ((n_pop > 1) & (len_M_tot == 1)):
        # the mass is divided equally among the populations
        M_tot = np.full(n_pop, M_tot[0])/n_pop
    elif (len_M_tot < n_pop):
        # extend length (dividing mass among pops)
        M_tot = np.append(M_tot, np.full(n_pop - len_M_tot, M_tot[-1]))
        M_tot[len_M_tot:] /= (n_pop - len_M_tot)
    elif (len_M_tot > n_pop):
        warnings.warn('Too many values received for M_tot_init. Discarded excess', SyntaxWarning)
        M_tot = M_tot[:n_pop]
    return M_tot


def check_and_cast_n_stars(n_stars, M_tot, n_pop, imf_par):
    """Check if we got values for number of stars (per population), or calculate them.
    :rtype: np.ndarray
    """
    if (np.all(n_stars == 0) & np.all(M_tot == 0)):
        raise ValueError('Input mass and number of stars cannot be zero simultaneously.')
    elif np.all(n_stars == 0):
        # estimate of the number of stars to generate
        n_stars = conv.m_tot_to_n_stars(M_tot, imf=imf_par)
    else:
        if hasattr(n_stars, '__len__'):
            n_stars = np.array(n_stars)
        else:
            n_stars = np.array([n_stars])
        
        len_N_stars = len(n_stars)
        if ((n_pop > 1) & (len_N_stars == 1)):
            # the stars are divided equally among the populations
            n_stars = np.full(n_pop, n_stars[0])/n_pop
        elif (len_N_stars < n_pop):
            # extend length (dividing stars among pops)
            n_stars = np.append(n_stars, np.full(n_pop - len_N_stars, n_stars[-1]))
            n_stars[len_N_stars:] /= (n_pop - len_N_stars)
        elif (len_N_stars > n_pop):
            warnings.warn('utils//check_n_stars: too many values received for N_stars', SyntaxWarning)
            n_stars = n_stars[:n_pop]

    # make sure they are int, and add up nicely
    n_stars = fix_total(np.rint(n_stars).astype(int), np.sum(n_stars))
    return n_stars


def cast_sfhistory(sfhist, n_pop):
    """Cast input for sf-history into the right format.
    :rtype: np.ndarray
    """
    if not sfhist:
        sfhist = np.full(n_pop, None)
    sfhist = cast_simple_array(sfhist, n_pop, fill_value=None, warning='Too many sfh types given. Excess discarded.')
    return sfhist


def cast_inclination(incl, n_pop):
    """Cast input for inclination into the right format.
    :rtype: np.ndarray
    """
    if not incl:
        incl = np.zeros(n_pop)
    incl = cast_simple_array(incl, n_pop, fill_value=0, warning='Too many incl values given. Excess discarded.')
    return incl


def cast_radial_dist_type(r_dist, n_pop):
    """Cast input for radial distribution type into the right format.
    :rtype: np.ndarray
    """
    if not r_dist:
        r_dist = np.full(n_pop, default_rdist)
    r_dist = cast_simple_array(r_dist, n_pop, fill_value=default_rdist,
                               warning='Too many radial distribution types given. Excess discarded.')
    return r_dist
    

def check_radial_dist_type(r_dist):
    """Check if the dist types exist.
    :rtype: np.ndarray
    """
    dist_list = list(set(fnmatch.filter(dir(dist), '*_r')))
    # to make sure we can increase the string length
    r_dist = r_dist.astype(object)
    for i in range(len(r_dist)):
        if (not r_dist[i].endswith('_r')):
            # add the r to the end for radial version of profile
            r_dist[i] = r_dist[i] + '_r'
        
        if (r_dist[i] not in dist_list):
            warnings.warn(f'Specified distribution <{r_dist[i]}> type does not exist. Using default '
                          f'(=<{default_rdist}>)', SyntaxWarning)
            r_dist[i] = default_rdist + '_r'*(not default_rdist.endswith('_r'))
    return r_dist


def cast_radial_dist_param(r_dist_par, r_dist, n_pop):
    """Cast the radial distribution parameters into the right format.
    Uses parameter names 'n' and 's' in function signatures.
    :rtype: np.ndarray
    """
    # get the function signatures for the given distributions
    func_sigs = [inspect.signature(getattr(dist, dist_type)) for dist_type in r_dist]
    
    # cast to right form and check if dist parameters are correctly specified
    if not r_dist_par:
        r_dist_par = np.array([{k: v.default for k, v in sig.parameters.items() if k is not 'n'} for sig in func_sigs])
    elif not hasattr(r_dist_par, '__len__'):
        # if just one parameter is given, make an array of dict
        r_dist_par = np.array([{k: r_dist_par for k, v in sig.parameters.items() if k is not 'n'} for sig in func_sigs])
    elif isinstance(r_dist_par, dict):
        # if just one dict, make an array of the same ones
        r_dist_par = np.full(n_pop, r_dist_par)
        
    len_param = len(r_dist_par)
    if ((n_pop > 1) & (len_param == 1)):
        r_dist_par = np.full(n_pop, r_dist_par[0])
    elif (len_param < n_pop):
        r_dist_par = np.append(r_dist_par, np.full(n_pop - len_param, r_dist_par[-1]))
    elif (len_param > n_pop):
        warnings.warn('Too many radial distribution parameters given. Excess discarded.', SyntaxWarning)
        r_dist_par = r_dist_par[:n_pop]
        
    # some final typecasting
    if np.all([isinstance(item, (int, float)) for item in r_dist_par]):
        # if 1D array of parameters given, fill dictionaries accordingly
        r_dist_par = np.array([{'s': par} for par in r_dist_par])
    elif np.all([(hasattr(item, '__len__') & (not isinstance(item, dict))) for item in r_dist_par]):
        temp_arr = np.array([])
        for item, sig in zip(r_dist_par, func_sigs):
            keys = [k for k, v in sig.parameters.items() if k is not 'n']
            temp_arr = np.append(temp_arr, {keys[i]: par for i, par in enumerate(item)})
        # if a 2D array/list is given, make an array of dict out of
        r_dist_par = temp_arr
    
    return r_dist_par


def cast_ellipse_axes(axes, n_pop):
    """Cast input for ellipse axes into the right format.
    :rtype: np.ndarray
    """
    if hasattr(axes, '__len__'):
        axes = np.array(axes)
    else:
        axes = np.ones([n_pop, 3])
    
    shape_axes = np.shape(axes)
    if ((len(shape_axes) == 1) & (len(axes) % 3 == 0)):
        axes = axes.reshape(len(axes) // 3, 3)
    elif ((len(shape_axes) == 1) & (shape_axes[0] == 1)):
        axes = np.zeros([n_pop, 3])
    elif (len(shape_axes) == 1):
        raise ValueError('Wrong number of arguments for axes, must be multiple of 3.')
    
    shape_axes = np.shape(axes)
    if ((n_pop > 1) & (shape_axes[0] == 1)):
        axes = np.full([n_pop, 3], axes[0])
    elif (shape_axes[0] < n_pop):
        axes = np.append(axes, np.full(n_pop - shape_axes[0], np.ones(3)), axis=0)
    elif (shape_axes[0] > n_pop):
        warnings.warn('Too many arguments for axes. Excess discarded.', SyntaxWarning)
        axes = axes[:n_pop]
    axes = axes.astype(float)
    return axes


def cast_translation(translation, n_pop):
    """Cast input for coordinate translation into the right format.
    :rtype: np.ndarray
    """
    # if hasattr(translation, '__len__'):
    #     translation = np.array(translation)
    # else:
    #     translation = np.array([translation])
    translation = np.atleast_1d(translation)
    # todo: could use 2d ... requires rebuilding

    shape_trans = np.shape(translation)
    if ((len(shape_trans) == 1) & (len(translation) % 3 == 0)):
        translation = translation.reshape(len(translation) // 3, 3)
    elif ((len(shape_trans) == 1) & (shape_trans[0] == 1)):
        translation = np.zeros([n_pop, 3])
    elif (len(shape_trans) == 1):
        raise ValueError('Wrong number of arguments for translation, must be multiple of 3.')

    shape_trans = np.shape(translation)
    if ((n_pop > 1) & (shape_trans[0] == 1)):
        translation = np.full([n_pop, 3], translation[0])
    elif (shape_trans[0] < n_pop):
        translation = np.append(translation, np.full(n_pop - shape_trans[0], np.zeros(3)), axis=0)
    elif (shape_trans[0] > n_pop):
        warnings.warn('Too many arguments for translation. Excess discarded.', SyntaxWarning)
        translation = translation[:n_pop]
    return translation

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









