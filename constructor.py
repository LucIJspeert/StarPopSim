# Luc IJspeert
# Part of starpopsim: program used for constructing an astronomical object
##
"""This module provides the argsparse for user input for making an astronomical object 
as well as an interactive function (DynamicConstruct) that leads the user through all the options.
"""
import argparse
import fnmatch
import inspect
import numpy as np

import distributions as dist
import objectgenerator as obg


# global defaults
default_object_file_name = 'astobj_default_save'

    
def DynamicConstruct():
    """Dynamically give parameters for construction via this interactive function."""
    print('--------------------------------------------------------------------------------')
    print('The program will now ask for the parameters to construct your object.')
    print('Type <quit> to exit this function. Type <help> for additional information.')
    print('--------------------------------------------------------------------------------')
    
    struct = StructureType()                                                                        # get the structure type
    
    if (struct.startswith('premade')):
        # do different things for pre-built objects
        pass
    else:
        N, M = NumAndMass()                                                                         # get the number or mass
        
        pop_n = PopulationAmount()                                                                  # get the amount of populations
        ages = PopAges(pop_n)                                                                       # get the age for each population
        Z = PopMetallicity(pop_n)                                                                   # get the metallicity for each population
        relN = PopRelativeN(pop_n)                                                                  # get the relative number in each population
        
        D_z, D_type = Distance()                                                                    # get the distance to the centre of the object (plus type of distance measurement)
        
        print('')
        opt = OptionalParam()                                                                       # does the user want optional parameters
        print('')
        
        if opt:
            dflt = False
        else:
            dflt = True
        
        IMF = IMFParam(pop_n, default=dflt)                                                         # get the IMF parameters
        SFH = SFHType(pop_n, default=dflt)                                                          # get the SFH type
        A = Extinction(default=dflt)                                                                # get the extinction for this source
        i = Inclination(struct, pop_n, default=dflt)                                                # get the inclination (x axis towards z axis = l.o.s.)
        
        rdist = RadialDistribution(struct, pop_n, default=dflt)                                     # get the radial distribution of stars
        rdistpar = RadialParameters(struct, rdist, default=dflt)                                    # get the parameters for said distribution
        axes = EllipseAxes(struct, pop_n, default=dflt)                                             # get the relative axes scales

        arms = SpiralArms(struct, default=dflt)                                                     # get the amount of spiral arms
        bulge = SpiralBulge(struct, default=dflt)                                                   # get the relative size of the bulge (to the disk)
        bar = SpiralBar(struct, default=dflt)                                                       # get the relative size of the bar (to the disk)
                
        if (struct == 'spiral'):                                                                    # (need +1 population for each spiral component)
            if (bulge > 0):
                bulge_props = SpiralProps(bulge, 'bulge')                                           # get age, z and relN for the bulge
                ages.append(bulge_props[0])                                                         # so these lists contain bulge and bar at the end, resp. (pop_n is still the same)
                Z.append(bulge_props[1])
                relN.append(bulge_props[2])
            elif (bar > 0):
                bar_props = SpiralProps(bar, 'bar')                                                 # get age, z and relN for the bar
                ages.append(bar_props[0])
                Z.append(bar_props[1])
                relN.append(bar_props[2])
                
    #TODO: add compact, cp mode and mag lim to questions
    
    savename = SaveFileName()
    
    astobj = obg.AstObject(struct=struct, 
                           N_stars=N, 
                           M_tot_init=M, 
                           age=ages, 
                           metal=Z, 
                           rel_num=relN, 
                           distance=D_z,
                           d_type=D_type,
                           imf_par=IMF,
                           sf_hist=SFH,
                           extinct=A,
                           incl=i,
                           r_dist=rdist,
                           r_dist_par=rdistpar,
                           ellipse_axes=axes,
                           spiral_arms=arms,
                           spiral_bulge=bulge,
                           spiral_bar=bar,
                           compact=False,
                           cp_mode='num',
                           mag_lim=None,
                           )
    
    astobj.SaveTo(savename + '.pkl')                                                                # save the object
    
    command = OneLineCommand(astobj, N, M, IMF, savename)
    print('')
    print('To make this astronomical object quickly, the following command can be used:')
    print(command)
    print('')
    
    return astobj
    
def isfloat(value, integer=False):
    """Just a little thing to check input."""
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
            
    default_val = options[a+1:b]                                                                        # default option is between the [brackets]
    
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
        while not isfloat(ans):
            ans = CheckAnswer(ans, question, options, default_val, function, help_arg)              # check the answer
            
    elif (check == 'int'):
        while not isfloat(ans, integer=True):
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
        raise KeyboardInterrupt #SystemExit                                                                            # exit if wanted
    else:
        if (options == ''):
            ans = input(question + ': ')
        else:
            ans = input(question + ' ' + options + ': ') 
        
    return ans
        
def AskHelp(function, add_args):
    """Shows the help on a particular function."""
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
    
def StructureType():
    """Asks what structure type you want."""
    cluster = ['cluster', 'c']                                                                      # cluster synonyms
    galaxy = ['galaxy', 'g']                                                                        # galaxy synonyms
    
    clu_gal = WhileAsk('Do you want a', '[cluster]/galaxy', add_opt=['c', 'g'], 
                        function='StructureType')
        
    if (clu_gal.lower() in cluster):
        clu_gal = 'cluster'
        structlist = ['ellipsoid', 'premade-globular', 'premade-open']                              # define the cluster options
    elif (clu_gal.lower() in galaxy):
        clu_gal = 'galaxy'
        structlist = ['elliptical', 'spiral', 'premade-elliptical', 'premade-spiral']               # define the galaxy options
        
    print(' |  Available structures: {0}'.format(', '.join(structlist)))
    
    struct = WhileAsk(' |  Kind of structure to generate', '[{0}]'.format(structlist[0]), 
                      add_opt=structlist, function='StructureType')
    
    print(' |  Selected: {0} -> {1}'.format(clu_gal, struct))
    return struct

def NumAndMass():
    """Asks how many stars or mass you want."""
    NorM = WhileAsk('Use number of stars or total mass', '[N]/M', function='NumAndMass')
    
    N = 0
    M = 0
    
    if (NorM.lower() in ['n']):
        Nstr = WhileAsk(' |  Number of stars to generate', '[1000]', function='NumAndMass', 
                        check='float')

        Nstr = Nstr.replace('*10**', 'e')
        Nstr = Nstr.replace('10**', '1e')
        N = int(float(Nstr))
    elif (NorM.lower() == 'm'):
        Mstr = WhileAsk(' |  Total mass in stars to generate', '[1000]', function='NumAndMass', 
                        check='float')

        Mstr = Mstr.replace('*10**', 'e')
        Mstr = Mstr.replace('10**', '1e')
        M = int(float(Mstr))
        
    return N, M

def PopulationAmount():
    """Asks how many stellar populations you want."""
    pop_str = WhileAsk('Amount of stellar populations', '[1]', function='PopulationAmount', 
                       check='int')
    return int(pop_str)
    
def PopAges(pop_n):
    """Asks what population ages you want."""
    default_val = '9.65'
    
    if (pop_n == 1):
        same_age = 'y'
    else:
        same_age = WhileAsk('Use the same age for all populations?', 'y/[n]', 
                            function='PopAges', add_opt=['yes', 'no'])
    
    if (same_age.lower() in ['y', 'yes']):
        age_str = WhileAsk(' |  Give the age', '[{0}]'.format(default_val), 
                           function='PopAges', check='float')
        age_list = [float(age_str) for i in range(pop_n)]
    elif (same_age.lower() in ['n', 'no']):
        age_list = []
        for i in range(pop_n):
            age_str = WhileAsk(' |  Age for population {0}'.format(i+1), 
                               '[{0}]'.format(default_val), function='PopAges', check='float')
            age_list.append(float(age_str))
    
    for i in range(pop_n):    
        if ((age_list[i] > 11) & (age_list[i] < 10**2)):
            age_list[i] = float(default_val)
            print(' |  Warning: input less than 100 years; using {0}'.format(age_list[i]))
    
    return age_list

def PopMetallicity(pop_n):
    """Asks what metallicities you want."""
    default_val = '0.014'
    
    if (pop_n == 1):
        same_z = 'y'
    else:
        same_z = WhileAsk('Use the same metallicity for all populations?', 'y/[n]', 
                          function='PopMetallicity', add_opt=['yes', 'no'])
    
    if (same_z.lower() in ['y', 'yes']):
        z_str = WhileAsk(' |  Give the metallicity', '[{0}]'.format(default_val), 
                         function='PopMetallicity', check='float')
        z_list = [float(z_str) for i in range(pop_n)]
    elif (same_z.lower() in ['n', 'no']):
        z_list = []
        for i in range(pop_n):
            z_str = WhileAsk(' |  Metallicity for population {0}'.format(i+1), 
                             '[{0}]'.format(default_val), function='PopMetallicity', check='float')
            z_list.append(float(z_str))
    
    for i in range(pop_n):    
        if (z_list[i] > 1):
            if isfloat(z_list[i], integer=True):
                z_list[i] = float('0.0' + str(z_list[i]))
            else:
                z_list[i] = float(default_val)
            print(' |  Warning: input larger than 1; using {0}'.format(z_list[i]))
    
    return z_list
    
def PopRelativeN(pop_n):
    """Asks what relative (population) numbers you want."""
    if (pop_n == 1):
        rel_list = [1]
    else:
        same_rel = WhileAsk('Use equal numbers of stars in all populations?', '[y]/n', 
                            function='PopRelativeN', add_opt=['yes', 'no'])
        
        if (same_rel.lower() in ['n', 'no']):
            print(' |  Give the relative number of stars in each population')
            
            rel_list = []
            for i in range(pop_n):
                rel_str = WhileAsk(' |  Relative number for population {0}'.format(i+1), '[1]', 
                                   function='PopRelativeN', check='float')
                rel_list.append(float(rel_str))
        elif (same_rel.lower() in ['y', 'yes']):
            rel_list = [1 for i in range(pop_n)]
        
    return rel_list

def Distance():
    """Asks what distance to the object you want."""
    l_opt = ['l', 'lum', 'luminosity']
    z_opt = ['z', 'r', 'red', 'redshift']
    options = l_opt + z_opt
    
    d_type = WhileAsk('Use luminosity based distance or redshift?', '[l]/z', add_opt=options, 
                      function='Distance')
    
    if (d_type in l_opt):
        D_str = WhileAsk('Distance to the object in pc', '[100]', function='Distance', 
                         check='float')
        D_type = 'l'
        val = float(D_str)
    elif (d_type in z_opt):
        z_str = WhileAsk('Redshift of the object', '[0.1]', function='Distance', check='float')
        D_type = 'z'
        val = float(z_str)
    
    return val, D_type
    
def OptionalParam():
    """Asks if optional parameters are used."""
    change = WhileAsk('Do you want to use optional parameters?', 'y/[n]/help', 
                      function='OptionalParam')
    if (change.lower() in ['yes', 'y']):
        opt = True
    else:
        opt = False
    return opt
    
def IMFParam(pop_n, default=False):
    """Asks to change the IMF parameters"""
    if default:
        change = 'n'
    else:
        change = WhileAsk('Do you want to change the IMF parameters?', 'y/[n]', 
                          add_opt=['yes', 'no'], function='IMFParam')
            
    if (change.lower() in ['y', 'yes']):
        if (pop_n > 1):
            same_imf = WhileAsk(' |  Use the same IMF parameters for all populations?', 
                                '[y]/n', function='IMFParam', add_opt=['yes', 'no'])
        else:
            same_imf = 'y'
        
        if (same_imf.lower() in ['n', 'no']):
            param_list = []
            for i in range(pop_n):
                print(' |  IMF parameters for population {0}:'.format(i+1))
                low = WhileAsk((' |  Lower bound to use (in M_sol) for population {0}'
                                ).format(i+1), '[0.08]', function='IMFParam', check='float')
                mid = WhileAsk((' |  Knee position to use (in M_sol) for population {0}'
                                ).format(i+1), '[0.5]', function='IMFParam', check='float')
                upp = WhileAsk((' |  Upper bound to use (in M_sol) for population {0}'
                                ).format(i+1), '[150]', function='IMFParam', check='float')    
                param_list.append([float(low), float(mid), float(upp)])
        elif (same_sfh.lower() in ['y', 'yes']):
            low = WhileAsk(' |  Lower bound to use (in M_sol)', '[0.08]', function='IMFParam', 
                           check='float')
            mid = WhileAsk(' |  Knee position to use (in M_sol)', '[0.5]', function='IMFParam', 
                           check='float')
            upp = WhileAsk(' |  Upper bound to use (in M_sol)', '[150]', function='IMFParam', 
                           check='float')    
            param_list = [[float(low), float(mid), float(upp)]]
                
    elif (change.lower() in ['n', 'no']):
        param_list = [[0.08, 0.5, 150.] for i in range(pop_n)]
    
    return param_list
    
def SFHType(pop_n, default=False):
    """Asks what SFH type to use (if at all)."""
    sfh_types = ['none', 'exp']                                                                     # list of available SFH types
    joined_sfhlist = ', '.join(sfh_types)
    
    if default:
        change = 'n'
    else:
        change = WhileAsk('Do you want to change the SFH type?', 'y/[n]', add_opt=['yes', 'no'], 
                          function='SFHType', help_arg=joined_sfhlist)
            
    if (change.lower() in ['y', 'yes']):
        if (pop_n > 1):
            same_sfh = WhileAsk(' |  Use the same SFH for all populations?', '[y]/n', 
                                function='SFHType', add_opt=['yes', 'no'])
        else:
            same_sfh = 'y'
        
        print(' |  Types to choose from: {0}'.format(', '.join(sfh_types)))
        if (same_sfh.lower() in ['n', 'no']):
            sfh_list = []
            for i in range(pop_n):
                sfhtype = WhileAsk(' |  SFH type to use for population {0}'.format(i+1), '[none]', 
                                   add_opt=sfh_types, function='SFHType', help_arg=joined_sfhlist)
                sfh_list.append(sfhtype)
        elif (same_sfh.lower() in ['y', 'yes']):
                sfhtype = WhileAsk(' |  SFH type to use', '[none]', add_opt=sfh_types, 
                                   function='SFHType', help_arg=joined_sfhlist)
                sfh_list = [sfhtype for i in range(pop_n)]
                
    elif (change.lower() in ['n', 'no']):
        sfh_list = ['none' for i in range(pop_n)]
        
    return sfh_list

def Extinction(default=False):
    """Asks what the extinction is for the source."""
    def_val = '0'
    
    if default:
        change = 'n'
    else:
        change = WhileAsk('Do you want to change the extiction?', 'y/[n]', add_opt=['yes', 'no'], 
                          function='Extinction')
    
    if (change.lower() in ['n', 'no']):
        A_str = def_val
    elif (change.lower() in ['y', 'yes']):
        A_str = WhileAsk(' |  Extinction magnitude', '[{0}]'.format(def_val), 
                         function='Extinction', check='float')                                                                      
        
    return abs(float(A_str))                                                                        # make sure it is positive

def Inclination(struct, pop_n, default=False):
    """Asks what the inclination of the object is."""
    def_val = '0'
    
    if default:
        change = 'n'
    else:
        change = WhileAsk('Do you want to change the inclination?', 'y/[n]', 
                          add_opt=['yes', 'no'], function='Inclination')
    
    if (change.lower() in ['n', 'no']):
        incl_arr = np.array([float(def_val) for i in range(pop_n)])
    elif (change.lower() in ['y', 'yes']):
        if ((pop_n > 1) & (struct in ['ellipsoid', 'elliptical'])):
            same_incl = WhileAsk(' |  Use the same inclination for all populations?', '[y]/n', 
                                 function='Inclination', add_opt=['yes', 'no'])
        else:
            same_incl = 'y'                                                                         # if we have a spiral, only use one inclination
        
        incl_arr = np.zeros(pop_n)
        
        if (same_incl.lower() in ['n', 'no']):
            for i in range(pop_n):
                incl = WhileAsk(' |  Inclination angle in radians for population {0}'.format(i+1), 
                                '[{0}]'.format(def_val), function='Inclination', check='float')
                incl_arr[i] = float(incl)
        elif (same_incl.lower() in ['y', 'yes']):
            incl = WhileAsk(' |  Inclination angle in radians', '[{0}]'.format(def_val), 
                            function='Inclination', check='float')
            incl_arr[:] = float(incl)
    
    return incl_arr
    
def RadialDistribution(struct, pop_n, default=False):
    """Asks what radial distribution to use."""
    def_val = 'Normal'
    
    if default:
        change = 'n'
    else:
        change = WhileAsk('Do you want to change the radial distribution?', 'y/[n]', 
                          add_opt=['yes', 'no'], function='RadialDistribution')
    
    dist_list_r = list(set(fnmatch.filter(dir(dist), '*_r')))                                       # get all available radial distributions
    dist_list = [item.replace('_r', '') for item in dist_list_r]                                    # format without trailing _r
    dist_list_lwr = [item.lower() for item in dist_list]                                            # format with all lower case for comparison
        
    if (change.lower() in ['n', 'no']):
        dist_type = np.array(dist_list_r)[np.array(dist_list_lwr) == def_val.lower()]               # get back the right format
        dist_type_list = [dist_type[0] for i in range(pop_n)]
    elif (change.lower() in ['y', 'yes']):
        if (pop_n > 1):
            same_dist = WhileAsk(' |  Use the same radial distribution for all populations?', 
                                 '[y]/n', function='RadialDistribution', add_opt=['yes', 'no'])
        else:
            same_dist = 'y' 
            
        short_list = ['norm', 'exp', 'pearson', 'cauchy', 'king']                                   # abbreviations that can be used
        joined_dist_list = ', '.join(dist_list)
        
        print(' |  Available radial distributions for the stars in this object:')
        print(' |  {0}'.format(joined_dist_list))
        
        if (same_dist.lower() in ['n', 'no']):
            dist_type_list = []
            for i in range(pop_n):
                dist_type = WhileAsk(' |  Radial distribution for population {0}'.format(i+1), 
                                     '[{0}]'.format(def_val), add_opt=dist_list_lwr + short_list, 
                                     function='RadialDistribution', help_arg=joined_dist_list)
            
                if (dist_type.lower() in short_list):
                    for item in dist_list_lwr:
                        if (dist_type.lower() in item):
                            dist_type = item                                                        # get the full name from the abbreviation
                            
                dist_type = np.array(dist_list_r)[np.array(dist_list_lwr) == dist_type.lower()]     # get back the right format
                dist_type_list.append(dist_type[0])
        elif (same_dist.lower() in ['y', 'yes']):
            dist_type = WhileAsk(' |  Radial distribution', '[{0}]'.format(def_val), 
                                 add_opt=dist_list_lwr + short_list, 
                                 function='RadialDistribution', help_arg=joined_dist_list)
        
            if (dist_type.lower() in short_list):
                for item in dist_list_lwr:
                    if (dist_type.lower() in item):
                        dist_type = item                                                            # get the full name from the abbreviation
                        
            dist_type = np.array(dist_list_r)[np.array(dist_list_lwr) == dist_type.lower()]         # get back the right format
            dist_type_list = [dist_type[0] for i in range(pop_n)]
            
    
    return dist_type_list
    
def RadialParameters(struct, dist_type_list, default=False):
    """Asks what parameters to use for the radial distribution."""
    if default:
        change = 'n'
    else:
        change = WhileAsk('Do you want to change the radial distribution parameters?', '[y]/n', 
                          function='RadialParameters', add_opt=['yes', 'no'])
        
    if ((dist_type_list[1:] == dist_type_list[:-1]) 
            & (change.lower() in ['y', 'yes']) 
            & (len(dist_type_list) != 1)):
        same_par = WhileAsk(' |  Use the same radial parameters for all populations?', '[y]/n', 
                            function='RadialParameters', add_opt=['yes', 'no'])
    elif (len(dist_type_list) == 1):
        same_par = 'y'
    else:
        same_par = 'n'
    
    if (same_par.lower() in ['n', 'no']):
        dict_list = []
        for i, dist_type in enumerate(dist_type_list):
            sig = inspect.signature(eval('dist.' + dist_type))                                              # parameters of the dist function (includes n)
            joined_par = ', '.join(['{0}={1}'.format(k, v.default) 
                                    for k, v in sig.parameters.items() if k is not 'n'])
            
            if (change.lower() in ['n', 'no']):
                par_dict = {k: v.default for k, v in sig.parameters.items() if k is not 'n'}
            elif (change.lower() in ['y', 'yes']):
                print(' |  Radial parameters for population {0}:'.format(i+1))
                par_dict = {}
                for k, v in sig.parameters.items():                                                         # loop through needed parameters (except n)
                    if (k != 'n'):
                        if ((dist_type == 'KingGlobular_r') & (k == 'R')):                                  # catch the default (None) in KingGlobular_r
                            value = WhileAsk(' |  Value for parameter {0}'.format(k), 
                                             '[{0}]'.format(30*par_dict['s']), 
                                             function='RadialParameters', check='float', 
                                             help_arg=[dist_type, joined_par])
                        else:
                            value = WhileAsk(' |  Value for parameter {0}'.format(k), 
                                             '[{0}]'.format(v.default), 
                                             function='RadialParameters', check='float', 
                                             help_arg=[dist_type, joined_par])
                        
                        par_dict[k] = float(value)
                        
            dict_list.append(par_dict)
    elif (same_par.lower() in ['y', 'yes']):
        dist_type = dist_type_list[0]
        sig = inspect.signature(eval('dist.' + dist_type))                                              # parameters of the dist function (includes n)
        joined_par = ', '.join(['{0}={1}'.format(k, v.default) 
                                for k, v in sig.parameters.items() if k is not 'n'])
        
        if (change.lower() in ['n', 'no']):
            par_dict = {k: v.default for k, v in sig.parameters.items() if k is not 'n'}
        elif (change.lower() in ['y', 'yes']):
            par_dict = {}
            for k, v in sig.parameters.items():                                                         # loop through needed parameters (except n)
                if (k != 'n'):
                    if ((dist_type == 'KingGlobular_r') & (k == 'R')):                                  # catch the default (None) in KingGlobular_r
                        value = WhileAsk(' |  Value for parameter {0}'.format(k), 
                                         '[{0}]'.format(30*par_dict['s']), 
                                        function='RadialParameters', check='float', 
                                        help_arg=[dist_type, joined_par])
                    else:
                        value = WhileAsk(' |  Value for parameter {0}'.format(k), 
                                         '[{0}]'.format(v.default), 
                                         function='RadialParameters', check='float', 
                                         help_arg=[dist_type, joined_par])
                    
                    par_dict[k] = float(value)
        dict_list = [par_dict]            
    
    return dict_list
    
def EllipseAxes(struct, pop_n, default=False):
    """Asks what the relative axes scales must be."""
    if (default | (struct == 'spiral')):
        axes_arr = np.ones([pop_n, 3])
    elif (struct in ['ellipsoid', 'elliptical']):
        change = WhileAsk('Do you want to change the axes scaling (default=[1,1,1])?', 
                          'y/[n]', function='EllipseAxes', add_opt=['yes', 'no'])
        
        if ((pop_n > 1) & (change.lower() in ['y', 'yes'])):
            same_axes = WhileAsk(' |  Use same axes for all populations?', 
                                 '[y]/n', function='EllipseAxes', add_opt=['yes', 'no'])
        else:
            same_axes = 'y'
        
        axes_arr = np.ones([pop_n, 3])
        
        if ((change.lower() in ['y', 'yes']) & (same_axes.lower() in ['n', 'no'])):
            for i in range(pop_n):
                print(' |  Scaling of the axes for population {0}:'.format(i+1))
                for j, axis in enumerate(['x', 'y', 'z']):
                    scale = WhileAsk(' |  Relative axis scale for {0}'.format(axis), '[1]', 
                                     function='EllipseAxes', check='float')
                    axes_arr[i,j] = float(scale)
        elif ((change.lower() in ['y', 'yes']) & (same_axes.lower() in ['y', 'yes'])):
            for j, axis in enumerate(['x', 'y', 'z']):
                scale = WhileAsk(' |  Relative axis scale for {0}'.format(axis), '[1]', 
                                 function='EllipseAxes', check='float')
                axes_arr[:,j] = float(scale)
    
    return axes_arr
    
def SpiralArms(struct, default=False):
    """Asks what the number of spiral arms must be."""
    if (default & (struct == 'spiral')):
        arms = 2
    elif (default | (struct in ['ellipsoid', 'elliptical'])):
        arms = 0
    elif (struct == 'spiral'):
        arms_str = WhileAsk('Amount of spiral arms', '[2]', function='SpiralArms', check='int')
        
        arms = int(arms_str)
        if (arms < 1):
            print(' |  Invalid input value, need at least one spiral arm. Using 1.')
            arms = 1
    
    return arms
    
def SpiralBulge(struct, default=False):
    """Asks what the relative size (to the disk) of the bulge must be."""
    if (default & (struct == 'spiral')):
        bulge = 0.1
    elif (default | (struct in ['ellipsoid', 'elliptical'])):
        bulge = 0.0
    elif (struct == 'spiral'):
        bulge_str = WhileAsk('Relative size of the bulge', '[0.1]', function='SpiralBulge', 
                             check='float')
        
        bulge = float(bulge_str)
        bulge = np.clip(bulge, 0.0, 1.0)
    
    return bulge
    
def SpiralBar(struct, default=False):
    """Asks what the relative size (to the disk) of the bar must be."""
    if (default & (struct == 'spiral')):
        bar = 0.2
    elif (default | (struct in ['ellipsoid', 'elliptical'])):
        bar = 0.0
    elif (struct == 'spiral'):
        bar_str = WhileAsk('Relative size of the central bar', '[0.2]', function='SpiralBar', 
                           check='float')
        
        bar = float(bar_str)
        bar = np.clip(bar, 0.0, 1.0)
        
    return bar
    
def SpiralProps(bulge, comp):
    """Asks what the properties of the bulge or bar must be."""
    if (comp not in ['bulge', 'bar']):
        return 10.0, 0.008, 1
    
    age_str = WhileAsk('Give the age of the {0} (in yr or log(yr))'.format(comp), '[10.0]', 
                       function='SpiralProps', check='float')
    age = float(age_str)
    
    z_str = WhileAsk('Give the metallicity of the {0}'.format(comp), '[0.008]', 
                     function='SpiralProps', check='float')
    z = float(z_str)
    
    relN_str = WhileAsk('Give the relative number of stars in the {0}'.format(comp), '[1]', 
                        function='SpiralProps', check='float')
    relN = float(relN_str)
    
    return age, z, relN
    
def SaveFileName():
    """Asks if you want to change the filename."""
    file_name = default_object_file_name
    
    print('The object will be saved under the name ' + file_name)
    change = WhileAsk('Save object with a different filename?', 'y/[n]', 
                      add_opt=['yes', 'no'], function='SaveFileName')
    
    if (change.lower() in ['y', 'yes']):
        file_name = WhileAsk('Give the new name', '', function='SaveFileName')
        
    return file_name

def OneLineCommand(astobj, N, M, IMF, savename):
    """Gives the command line format to get the generated object."""
    command = 'python3 constructor.py'
    command += ' -struct ' + astobj.structure
    if (N != 0):
        command += ' -N ' + str(astobj.N_obj)
    if (M != 0):
        command += ' -M ' + str(astobj.M_tot_init)
    command += ' -ages ' + ' '.join(str(item) for item in astobj.ages)
    command += ' -z ' + ' '.join(str(item) for item in astobj.metal)
    if np.any(astobj.rel_number != 1):
        command += ' -relN ' + ' '.join(str(item) for item in astobj.rel_number)
    if (astobj.d_type == 'z'):
        command += ' -D ' + str(astobj.redshift)
    else:
        command += ' -D ' + str(astobj.d_lum)
    if (astobj.d_type != 'l'):
        command += ' -Dtype ' + str(astobj.d_type)
    if np.any(np.array(IMF) != [0.08, 0.5, 150.]):
        command += ' -IMF ' + ' '.join(str(item) for item in astobj.imf_param.flatten())
    if np.any(np.array(astobj.sfhist) != 'none'):
        command += ' -SFH ' + ' '.join(str(item) for item in astobj.sfhist)
    if (astobj.extinction != 0):
        command += ' -A ' + str(astobj.extinction)
    if (astobj.inclination != 0).any():
        command += ' -i ' + ' '.join(str(item) for item in astobj.inclination)
    if np.any(np.array(astobj.r_dist_type) != 'Normal_r'):
        command += ' -rdist ' + ' '.join(str(item) for item in astobj.r_dist_type)
    if np.array([param != {'s': 1.0} for param in astobj.r_dist_param]).any():
        command += ' -rdistpar '
        for param_dict in astobj.r_dist_param:
            command += ' '.join(str(item) for item in param_dict.values())
    if np.any(astobj.ellipse_axes != [1, 1, 1]):
        command += ' -axes ' + ' '.join(str(item) for item in astobj.ellipse_axes.flatten())
    if (astobj.spiral_arms != 0):
        command += ' -arms ' + str(astobj.spiral_arms)
    if (astobj.spiral_bulge != 0):
        command += ' -bulge ' + str(astobj.spiral_bulge)
    if (astobj.spiral_bar != 0):
        command += ' -bar ' + str(astobj.spiral_bar)
    if (astobj.compact != False):
        command += ' -compact '
        if (astobj.compact_mode != 'num'):
            command += ' -cp_mode ' + str(astobj.compact_mode)
            if (astobj.mag_limit is not None):
                command += ' -limit ' + str(astobj.mag_limit)
    if (savename != default_object_file_name):
        command += ' -save ' + savename
    
    return command

## read in arguments from cmd line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct an astronomical object from scratch.')
    
    parser.add_argument('-struct', type=str, required=False, default='ellipsoid',
                        choices=['ellipsoid','spiral', 'elliptical'],
                        help='type of structure to create') 
    
    parser.add_argument('-N', type=int, required=False, default=0,
                        help='the total number of stars generated')
                        
    parser.add_argument('-M', type=int, required=False, default=0,
                        help='the total mass in generated stars')
                    
    parser.add_argument('-ages', type=float, nargs='+', required=False, default=[9],
                        help='age(s) of the stellar population(s), logarithmic or linear.')
                        
    parser.add_argument('-Z', type=float, nargs='+', required=False, default=[0.019], 
                        help='metallicity(/ies) of the stellar population(s)')
                        
    parser.add_argument('-relN', type=float, nargs='+', required=False, default=[1],
                        help='relative number of stars in each population')
    
    parser.add_argument('-D', type=float, required=False, default=1e3,
                        help='distance to center of object in pc')   
    
    # optional arguments
    parser.add_argument('-Dtype', type=str, required=False, default='l',
                        choices=['l', 'z'],
                        help='type of distance; luminosity distance or redshift.')   
                        
    parser.add_argument('-IMF', type=float, nargs='+', required=False, default=[0.08, 0.5, 150],
                        help='lower bound, knee position, upper bound for the IMF masses')
                        
    parser.add_argument('-SFH', type=str, nargs='+', required=False, default=['none'],
                        help='Star Formation History type')
                        
    parser.add_argument('-A', type=float, required=False, default=0.,
                        help='extinction between source and observer')
                        
    parser.add_argument('-i', type=float, nargs='+', required=False, default=[0.],
                        help='inclination angle (rotation of object\'s '
                        'x-axis towards z-axis (=l.o.s.))')
                        
    parser.add_argument('-rdist', type=str, nargs='+', required=False, default=['Normal'],
                        choices=['Exponential','Normal','SquaredCauchy','PearsonVII',
                        'KingGlobular'], help='(ellipse) type of radial distribution')
                        
    parser.add_argument('-rdistpar', type=float, nargs='+', required=False, default=[1.0],
                        help='(ellipse) radial distribution parameters (s, R)')
                        
    parser.add_argument('-axes', type=float, nargs='+', required=False, default=[1,1,1],
                        help='(ellipse) relative scales of the x,y,z axes')
                        
    parser.add_argument('-arms', type=int, required=False, default=0,
                        help='(spiral) number of spiral arms')
                        
    parser.add_argument('-bulge', type=float, required=False, default=0.,
                        help='(spiral) relative proportion of central bulge')
                        
    parser.add_argument('-bar', type=float, required=False, default=0.,
                        help='(spiral) relative proportion of central bar')
    
    # additional functions
    parser.add_argument('-inter', action='store_true', required=False,
                        help='to use the interactive constructor function')
                        
    parser.add_argument('-compact', action='store_true', required=False,
                        help='only keep bright stars above a magnitude limit')
                        
    parser.add_argument('-cp_mode', type=str, required=False, default='num',
                        choices=['num', 'mag'],
                        help='compacting mode')
                        
    parser.add_argument('-limit', type=float, required=False, default=None,
                        help='magnitude limit to use for compacting')
    
    parser.add_argument('-save', type=str, required=False, default=default_object_file_name,
                        help='file to save object to')
    
    args = parser.parse_args()

    # execute the main function
    if args.inter:
        astobj = DynamicConstruct()
    else:
        astobj = obg.AstObject(struct=args.struct, 
                               N_stars=args.N, 
                               M_tot_init=args.M, 
                               age=args.ages, 
                               metal=args.Z, 
                               rel_num=args.relN, 
                               distance=args.D,
                               d_type=args.Dtype,
                               imf_par=args.IMF,
                               sf_hist=args.SFH,
                               extinct=args.A,
                               incl=args.i,
                               r_dist=args.rdist,
                               r_dist_par=args.rdistpar,
                               ellipse_axes=args.axes,
                               spiral_arms=args.arms,
                               spiral_bulge=args.bulge,
                               spiral_bar=args.bar,
                               compact=args.compact,
                               cp_mode=args.cp_mode,
                               mag_lim=args.limit,
                               )
        
        astobj.SaveTo(args.save)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    