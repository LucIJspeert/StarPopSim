"""Utility functions are thrown over here."""
import os
import warnings
import numpy as np
import scipy.interpolate as spi

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


class isochrone_data():
    """Store isochrone files in a useful data frame."""
    def __init__(self, n_stars, ages, metal):
        self.n_stars = n_stars  # list of ages per population
        self.ages = ages  # list of ages
        self.metal = metal  # list of metallicities
        self.pop_isoc_map = {}  # dictionary mapping population index to isoc length
        # open the relevant files and store the data
        self.col_name_map = get_isoc_col_names()  # go from code names to isoc column names
        n_cols = len(self.col_name_map)
        self.isoc_data = np.empty([0, n_cols])  # isochrone data sheet (no cube for file length reasons)
        for i, Z in enumerate(self.metal):
            isoc_i = open_isochrones_file(Z).T  # turn back from unpacked shape
            self.pop_isoc_map.update({i: len(isoc_i)})
            self.isoc_data = np.vstack([self.isoc_data, isoc_i])
        return

    def isochrone_mask(self, index, age=True):
        """Returns a boolean mask that masks out all but one isochrone file in the isochrone data.
        The input is the index of the wanted population and whether to select out its age.
        """
        # handy array of indices of the populations in each isochrone
        isoc_index = np.cumsum(np.append([0], list(self.pop_isoc_map.values())))
        mask = np.zeros(isoc_index[-1], dtype=bool)
        if age:
            age_mask = select_age(self.ages[index], self.metal[index])
        else:
            age_mask = np.ones(self.pop_isoc_map[index], dtype=bool)
        mask[isoc_index[index]:isoc_index[index + 1]] = age_mask
        return mask

    def population_mask(self, index):
        """Returns a boolean mask that masks out all but one population in the list of stars.
        The input is the index of the wanted population.
        """
        # indices defining the different populations in the total list of stars
        star_index = np.cumsum(np.append([0], self.n_stars))
        mask = np.zeros(star_index[-1], dtype=bool)
        mask[star_index[index]:star_index[index + 1]] = True
        return mask

    def get_columns(self, columns):
        """Gives a set of full columns of the isoc data sheet, given a list of column names."""
        col_index_map = {name: i for i, name in enumerate(self.col_name_map.keys())}  # map code names to column index
        col_index = [col_index_map[col] for col in columns]
        if (len(col_index) == 1):
            data_columns = self.isoc_data[:, col_index[0]]
        else:
            data_columns = np.transpose(self.isoc_data[:, col_index])
        return data_columns

    def max_isoc_masses(self):
        """Gives the maximum initial mass in the isochrones per population."""
        n_pop = len(self.n_stars)
        all_masses = self.get_columns(['M_initial'])
        max_mass = np.zeros(n_pop)
        for i in range(n_pop):
            isoc_mask = self.isochrone_mask(i, age=True)
            max_mass[i] = np.max(all_masses[isoc_mask])
        return max_mass

    def interpolate(self, column, M_init, left=None, right=None, conversion=None):
        """Gives the requested property of the stars by interpolating the isochrone grid.
        Only works for a single data column at a time (give name as string).
        The conversion option gives the ability to convert the given column to a different
            parameter, by supplying a lambda expression.
        """
        n_pop = len(self.n_stars)
        iso_M_ini, iso_qtt = self.get_columns(['M_initial', column])
        if conversion is not None:
            pars = inspect.signature(conversion).parameters.keys()
            # see if we need to fill in just the qtt or also mass
            if (len(pars) == 1):
                iso_qtt = conversion(iso_qtt)
            else:
                iso_qtt = conversion(iso_qtt, iso_M_ini)

        qtt = np.empty(np.sum(self.n_stars))
        for i in range(n_pop):
            isoc_mask = self.isochrone_mask(i, age=True)
            pop_mask = self.population_mask(i)
            iso_M_ini_i = iso_M_ini[isoc_mask]
            iso_qtt_i = iso_qtt[isoc_mask]
            qtt[pop_mask] = np.interp(M_init[pop_mask], iso_M_ini_i, iso_qtt_i, left=left, right=right)
        return qtt

    def interpolate_1d(self, columns, M_init, fill_value=None):
        """Gives the requested properties of the stars by interpolating the isochrone grid.
        Only works for a set of columns at a time (list of strings).
        """
        n_pop = len(self.n_stars)
        iso_M_ini = self.get_columns(['M_initial'])
        iso_qtt = self.get_columns(columns)
        qtts = np.empty([len(columns), np.sum(self.n_stars)])
        for i in range(n_pop):
            isoc_mask = self.isochrone_mask(i, age=True)
            pop_mask = self.population_mask(i)
            iso_M_ini_i = iso_M_ini[isoc_mask]
            iso_qtt_i = iso_qtt[:, isoc_mask]
            interper = spi.interp1d(iso_M_ini_i, iso_qtt_i, bounds_error=False, fill_value=fill_value, axis=1)
            qtts[:, pop_mask] = interper(M_init[pop_mask])
        return qtts


def open_isochrones_file(Z, columns=None):
    """Opens the isochrones file and gives the right columns.
    columns: list of column names (see code_names), None will give all columns.
    """
    # check the file name (actual opening later-on)
    decimals = -int(np.floor(np.log10(Z))) + 1
    file_name = f'isoc_Z{Z:1.{decimals}f}.dat'
    file_name = os.path.join('tables', file_name)

    # check whether file for Z exists
    if not os.path.isfile(file_name):
        # try one digit less
        file_name = f'isoc_Z{Z:1.{decimals - 1}f}.dat'
        file_name = os.path.join('tables', file_name)
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f'File {file_name} not found. Try a different metallicity.')

    # get the right column name mapping
    name_dict = get_isoc_col_names()
    all_code_names = name_dict.keys()

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
    Filters can be specified to get only those rows (array of strings).
    Note: array structure is removed if only one column is wanted.
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
        phot_dat['mean'] = phot_dat['mean'] * 1e-9  # convert to m
    if ('width' in np.array(used_types)[:, 0]):
        phot_dat['width'] = phot_dat['width'] * 1e-9  # convert to m
    if ('zp_flux' in np.array(used_types)[:, 0]):
        phot_dat['zp_flux'] = phot_dat['zp_flux'] * 1e7  # convert to W/m^3
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

    uni_log_t = np.unique(log_t)  # unique array of ages
    log_t_min = np.min(uni_log_t)  # minimum available age
    log_t_max = np.max(uni_log_t)  # maximum available age

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
    log_closest = uni_log_t[(uni_log_t - a < log_age) & (uni_log_t + b >= log_age)]
    return log_t == log_closest


def stellar_isochrone(age, Z, columns=None):
    """Gives the isochrone data for a specified age and metallicity (Z).
    columns: list of column names (see code_names), None will give all columns.
    """
    data = open_isochrones_file(Z, columns=columns)
    t_indices = select_age(age, Z)
    data = data[..., t_indices]
    return data


def stellar_track(mass, Z):
    """Gives a stellar evolution track for the given mass and metallicity (Z).
    columns: list of column names (see code_names), None will give all columns.
    """
    # todo: code idea... make stellar track with interpolation


def get_isoc_col_names():
    """Returns a dictionary of all usable (in code) column names to the
    appropriate column names in the isochrone files.
    """
    # mapping the names used in the code to the isoc file column names
    alt_filter_names = get_supported_filters(alt_names=True)
    full_filter_names = get_supported_filters(alt_names=False)
    code_names, isoc_names = np.loadtxt(os.path.join('tables', 'column_names.dat'), dtype=str, unpack=True)

    # define the filter names that can be used in terms of the isochrone filter names
    name_dict = {code: iso for code, iso in zip(alt_filter_names, full_filter_names)}
    name_dict.update({iso: iso for iso in full_filter_names})
    name_dict.update({code: iso for code, iso in zip(code_names, isoc_names)})
    name_dict.update({iso: iso for iso in isoc_names})
    return name_dict


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
        warnings.warn(f'Input of ages, metallicities or relative number did not match 1 or {n_pop}. '
                      'Unexpected behaviour might occur', UserWarning)
    return n_pop


def cast_simple_array(arr, length, fill_value='last', warning=None, error=None):
    """Cast input for a 1D array into the right format.
    Needs the input as well as the intended lenght of the array.
    Optionally a fill value can be given to fill up missing values, default is last value in arr.
        a warning message can be supplied for when too many values are given.
    An error message can be given to display when arr is empty.

    :rtype: np.ndarray
    """
    arr = np.atleast_1d(arr)
    len_arr = len(arr)

    if (fill_value == 'last'):
        if (len_arr == 0):
            raise ValueError(error)  # cannot resolve this conflict internally
        fill_value = arr[-1]  # fill with last value by default

    # correct the array length
    if ((length > 1) & (len_arr == 1)):
        arr = np.full(length, arr[0])
    elif (len_arr < length):
        arr = np.append(arr, np.full(length - len_arr, fill_value))
    elif (len_arr > length):
        if warning:
            # if no warning given, arr is shortened silently
            warnings.warn(warning, UserWarning)
        arr = arr[:length]
    return arr


def cast_imf_parameters(imf_par, n_pop, fill_value='default'):
    """Cast input for IMF parameters into the right format.
    :rtype: np.ndarray
    """
    if (fill_value == 'default'):
        fill_value = default_imf_par

    if imf_par is None:
        imf_par = np.full([n_pop, len(fill_value)], fill_value)
    elif hasattr(imf_par, '__len__'):
        imf_par = np.array(imf_par)
    else:
        warnings.warn(f'Incorrect input type for imf_par; using default (={default_imf_par}).', UserWarning)
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
        warnings.warn(f'Incorrect input for imf_par; using default (={fill_value}).', UserWarning)
        imf_par = np.full([n_pop, len(fill_value)], fill_value)
        # otherwise (below): assume a 2D shape with correct inner axis length 
    elif ((n_pop > 1) & (shape_imf_par[0] == 1)):
        imf_par = np.full([n_pop, default_len], imf_par[0])
    elif (shape_imf_par[0] < n_pop):
        extension = np.full([n_pop - shape_imf_par[0], default_len], imf_par[-1])
        imf_par = np.append(imf_par, extension, axis=0)
    elif (shape_imf_par[0] > n_pop):
        warnings.warn('Too many arguments for imf_par. Excess discarded.', UserWarning)
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


def cast_and_check_n_stars(n_stars, M_tot, n_pop, imf_par):
    """Check if we got values for number of stars (per population), or calculate them from M_tot.
    :rtype: np.ndarray
    """
    # first cast M_tot
    M_tot = np.atleast_1d(M_tot)

    len_M_tot = len(M_tot)
    if ((n_pop > 1) & (len_M_tot == 1)):
        # the mass is divided equally among the populations
        M_tot = np.full(n_pop, M_tot[0]) / n_pop
    elif (len_M_tot < n_pop):
        # extend length (dividing mass among pops)
        M_tot = np.append(M_tot, np.full(n_pop - len_M_tot, M_tot[-1]))
        M_tot[len_M_tot:] /= (n_pop - len_M_tot)
    elif (len_M_tot > n_pop):
        warnings.warn('Too many values received for M_tot_init. Discarded excess', UserWarning)
        M_tot = M_tot[:n_pop]

    # now get on with n_stars
    if (np.all(n_stars == 0) & np.all(M_tot == 0)):
        raise ValueError('Input mass and number of stars cannot be zero simultaneously.')
    elif np.all(n_stars == 0):
        # estimate of the number of stars to generate
        n_stars = conv.m_tot_to_n_stars(M_tot, imf=imf_par)
    else:
        n_stars = np.atleast_1d(n_stars)
        
        len_N_stars = len(n_stars)
        if ((n_pop > 1) & (len_N_stars == 1)):
            # the stars are divided equally among the populations
            n_stars = np.full(n_pop, n_stars[0])/n_pop
        elif (len_N_stars < n_pop):
            # extend length (dividing stars among pops)
            n_stars = np.append(n_stars, np.full(n_pop - len_N_stars, n_stars[-1]))
            n_stars[len_N_stars:] /= (n_pop - len_N_stars)
        elif (len_N_stars > n_pop):
            warnings.warn('utils//check_n_stars: too many values received for N_stars', UserWarning)
            n_stars = n_stars[:n_pop]

    # make sure they are int, and add up nicely
    n_stars = fix_total(np.rint(n_stars).astype(int), np.sum(n_stars))
    return n_stars
    

def check_radial_dist_type(r_dist):
    """Check if the dist types exist.
    :rtype: np.ndarray
    """
    dist_list = list(set(fnmatch.filter(dir(dist), '*_r')))
    # to make sure we can increase the string length
    r_dist = r_dist.astype(object)
    for i in range(len(r_dist)):
        if not r_dist[i]:
            r_dist[i] = default_rdist  # if none given, use default
        if (not r_dist[i].endswith('_r')):
            # add the r to the end for radial version of profile
            r_dist[i] = r_dist[i] + '_r'
        
        if (r_dist[i] not in dist_list):
            warnings.warn(f'Specified distribution <{r_dist[i]}> type does not exist. Using default '
                          f'(=<{default_rdist}>)', UserWarning)
            r_dist[i] = default_rdist + '_r' * (not default_rdist.endswith('_r'))
    return r_dist


def cast_radial_dist_param(r_dist_par, r_dist, n_pop):
    """Cast the radial distribution parameters into the right format.
    Uses parameter names 'n' and 's' in function signatures.
    :rtype: np.ndarray
    """
    # get the function signatures for the given distributions
    func_sigs = [inspect.signature(getattr(dist, dist_type)) for dist_type in r_dist]
    
    # cast to right form and check if dist parameters are correctly specified
    if r_dist_par is None:
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
        warnings.warn('Too many radial distribution parameters given. Excess discarded.', UserWarning)
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
    if ((axes.ndim == 1) & (len(axes) % 3 == 0)):
        axes = axes.reshape(len(axes) // 3, 3)
    elif ((axes.ndim == 1) & (shape_axes[0] == 1)):
        axes = np.zeros([n_pop, 3])
    elif (axes.ndim == 1):
        raise ValueError('Wrong number of arguments for axes, must be multiple of 3.')
    
    shape_axes = np.shape(axes)
    if ((n_pop > 1) & (shape_axes[0] == 1)):
        axes = np.full([n_pop, 3], axes[0])
    elif (shape_axes[0] < n_pop):
        axes = np.append(axes, np.full(n_pop - shape_axes[0], np.ones(3)), axis=0)
    elif (shape_axes[0] > n_pop):
        warnings.warn('Too many arguments for axes. Excess discarded.', UserWarning)
        axes = axes[:n_pop]
    axes = axes.astype(float)
    return axes


def cast_translation(translation, n_pop):
    """Cast input for coordinate translation into the right format.
    :rtype: np.ndarray
    """
    translation = np.atleast_1d(translation)
    # todo: could use 2d ... requires rebuilding

    shape_trans = np.shape(translation)
    if ((translation.ndim == 1) & (len(translation) % 3 == 0)):
        translation = translation.reshape(len(translation) // 3, 3)
    elif ((translation.ndim == 1) & (shape_trans[0] == 1)):
        translation = np.zeros([n_pop, 3])
    elif (translation.ndim == 1):
        raise ValueError('Wrong number of arguments for translation, must be multiple of 3.')

    shape_trans = np.shape(translation)
    if ((n_pop > 1) & (shape_trans[0] == 1)):
        translation = np.full([n_pop, 3], translation[0])
    elif (shape_trans[0] < n_pop):
        translation = np.append(translation, np.full(n_pop - shape_trans[0], np.zeros(3)), axis=0)
    elif (shape_trans[0] > n_pop):
        warnings.warn('Too many arguments for translation. Excess discarded.', UserWarning)
        translation = translation[:n_pop]
    return translation

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









