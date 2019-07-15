# Luc IJspeert
# Part of starpopsim: Utils
##
"""Utility functions are thrown over here."""
import os
import warnings
import numpy as np


def OpenIsochrone(age, Z, columns='all'):
    """Opens the isochrone file and gives the relevant columns.
    columns can be: 'all', 'mini', 'mcur', 'lum', 'temp', 'mag' (and 'filters')
    age can be either linear or logarithmic input.
    """
    # opening the file (actual opening lateron)
    file_name = os.path.join('tables', ('isoc_Z{1:1.{0}f}.dat'
                                        ).format(-int(np.floor(np.log10(Z)))+1, Z))
    if not os.path.isfile(file_name):                                                               # check wether file for Z exists
        file_name = os.path.join('tables', ('isoc_Z{1:1.{0}f}.dat'
                                            ).format(-int(np.floor(np.log10(Z))), Z))               # try one digit less
        if not os.path.isfile(file_name):
            raise FileNotFoundError(('objectgenerator//OpenIsochrone: file {0} not found. '
                                    'Try a different metallicity.').format(file_name))
    #todo: perhaps think of a more elegant way to do this
    # names to use in the code, and a mapping to the isoc file column names
    code_names = np.array(['log_age', 'M_initial', 'M_actual', 'log_L', 'log_Te', 'log_g',
                           'U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks'])
    mag_names = code_names[6:]                                                                      # names of filters for later reference
    mag_num = len(mag_names)
    var_names, column_names = np.loadtxt(os.path.join('tables', 'column_names.dat'),
                                         dtype=str, unpack=True)

    if ((len(code_names) != len(var_names)) | np.any(code_names != var_names)):
        raise SyntaxError(('objectgenerator//OpenIsochrone: file "column_names.dat" has '
                           'incorrect names specified. Use: {0}').format(', '.join(code_names)))

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

    # load the right columns (part 1 of 2)
    if (columns == 'all'):
        log_t, M_ini, M_act, log_L, log_Te = np.loadtxt(file_name, usecols=var_cols, unpack=True)
        mag = np.loadtxt(file_name, usecols=mag_cols, unpack=True)
    elif (columns == 'mcur'):
        log_t, M_ini = np.loadtxt(file_name, usecols=var_cols[:2], unpack=True)
        M_act = np.loadtxt(file_name, usecols=(col_dict[name_dict['M_actual']]), unpack=True)
    elif (columns == 'lum'):
        log_t, M_ini = np.loadtxt(file_name, usecols=var_cols[:2], unpack=True)
        log_L = np.loadtxt(file_name, usecols=(col_dict[name_dict['log_L']]), unpack=True)
    elif (columns == 'temp'):
        log_t, M_ini = np.loadtxt(file_name, usecols=var_cols[:2], unpack=True)
        log_Te = np.loadtxt(file_name, usecols=(col_dict[name_dict['log_Te']]), unpack=True)
    elif (columns == 'mag'):
        log_t, M_ini = np.loadtxt(file_name, usecols=var_cols[:2], unpack=True)
        mag = np.loadtxt(file_name, usecols=mag_cols, unpack=True)
    elif (columns == 'filters'):
        # just return the filter names
        return mag_names
    else:
        # either (columns == 'mini') or a wrong parameter is given
        log_t, M_ini = np.loadtxt(file_name, usecols=var_cols[:2], unpack=True)

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
        warnings.warn(('objectgenerator//OpenIsochrone: Specified age exceeds limit for Z={0}. '
                       'Using limit value (log_age={1}).').format(Z, lim_min*log_t_min
                       + lim_max*log_t_max), RuntimeWarning)
        log_age = lim_min*log_t_min + lim_max*log_t_max

    t_steps = uni_log_t[1:] - uni_log_t[:-1]                                                        # determine the age steps in the isoc files (step sizes may vary)
    a = np.log10((10**(t_steps) + 1)/2)                                                             # half the age step in logarithms
    b = t_steps - a                                                                                 # a is downward step, b upward
    a = np.insert(a, 0, 0.01)                                                                       # need step down for first t value
    b = np.append(b, 0.01)                                                                          # need step up for last t value
    log_closest = uni_log_t[(uni_log_t > log_age - a) & (uni_log_t <= log_age + b)]                 # the closest available age (to given one)
    where_t = np.where(log_t == log_closest)

    # return the right columns (part 2 of 2)
    M_ini = M_ini[where_t]                                                                          # M_ini is always returned (exept with 'filters')
    if (columns == 'all'):
        M_act = M_act[where_t]
        log_L = log_L[where_t]
        log_Te = log_Te[where_t]
        mag = np.array([m[0] for m in mag[:, where_t]])                                             # weird conversion needed because normal slicing would result in deeper array
        return M_ini, M_act, log_L, log_Te, mag
    elif (columns == 'mcur'):
        M_act = M_act[where_t]
        return M_ini, M_act
    elif (columns == 'lum'):
        log_L = log_L[where_t]
        return M_ini, log_L
    elif (columns == 'temp'):
        log_Te = log_Te[where_t]
        return M_ini, log_Te
    elif (columns == 'mag'):
        mag = np.array([m[0] for m in mag[:, where_t]])                                             # weird conversion needed because normal slicing would result in deeper array
        return M_ini, mag
    else:
        # (columns == 'mini') or wrong argument given
        return M_ini


def StellarTrack(Z, M_ini):
    """Opens the isochrone file and gives ... for one star.
    """
    # opening the file (actual opening lateron)
    file_name = os.path.join('tables', ('isoc_Z{1:1.{0}f}.dat'
                                        ).format(-int(np.floor(np.log10(Z)))+1, Z))
    if not os.path.isfile(file_name):                                                               # check wether file for Z exists
        file_name = os.path.join('tables', ('isoc_Z{1:1.{0}f}.dat'
                                            ).format(-int(np.floor(np.log10(Z))), Z))               # try one digit less
        if not os.path.isfile(file_name):
            raise FileNotFoundError(('objectgenerator//OpenIsochrone: file {0} not found. '
                                    'Try a different metallicity.').format(file_name))

    # names to use in the code, and a mapping to the isoc file column names
    code_names = np.array(['log_age', 'M_initial', 'M_actual', 'log_L', 'log_Te', 'log_g',
                           'U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks'])
    mag_names = code_names[6:]                                                                      # names of filters for later reference
    mag_num = len(mag_names)
    var_names, column_names = np.loadtxt(os.path.join('tables', 'column_names.dat'),
                                         dtype=str, unpack=True)

    if ((len(code_names) != len(var_names)) | np.any(code_names != var_names)):
        raise SyntaxError(('objectgenerator//OpenIsochrone: file "column_names.dat" has '
                           'incorrect names specified. Use: {0}').format(', '.join(code_names)))

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

    # load the right columns (part 1 of 2)
    log_t, M_ini, M_act, log_L, log_Te = np.loadtxt(file_name, usecols=var_cols, unpack=True)


    log_t_min = np.min(log_t)                                                                       # minimum available age
    log_t_max = np.max(log_t)                                                                       # maximum available age
    uni_log_t = np.unique(log_t)                                                                    # unique array of ages

    lim_min = (log_age < log_t_min - 0.01)
    lim_max = (log_age > log_t_max + 0.01)
    if lim_min or lim_max:                                                                          # check the age limits
        warnings.warn(('objectgenerator//OpenIsochrone: Specified age exceeds limit for Z={0}. '
                       'Using limit value (log_age={1}).').format(Z, lim_min*log_t_min
                       + lim_max*log_t_max), RuntimeWarning)
        log_age = lim_min*log_t_min + lim_max*log_t_max

    t_steps = uni_log_t[1:] - uni_log_t[:-1]                                                        # determine the age steps in the isoc files (step sizes may vary)
    a = np.log10((10**(t_steps) + 1)/2)                                                             # half the age step in logarithms
    b = t_steps - a                                                                                 # a is downward step, b upward
    a = np.insert(a, 0, 0.01)                                                                       # need step down for first t value
    b = np.append(b, 0.01)                                                                          # need step up for last t value
    log_closest = uni_log_t[(uni_log_t > log_age - a) & (uni_log_t <= log_age + b)]                 # the closest available age (to given one)
    where_t = np.where(log_t == log_closest)
    return


def FixTotal(tot, nums):
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
































