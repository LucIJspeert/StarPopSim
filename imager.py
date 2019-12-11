"""This module provides a function (Image) that takes user input for making an astronomical image
as well as an interactive function (DynamicImage) that leads the user through all the options.
"""
#todo: [warning: due to changes in <obg> some functions this module will not work as intended]
import argparse
import os
import numpy as np

import utils
import fitshandler as fh
import imagegenerator as img
import objectgenerator as obg


# defaults
default_image_file_name = 'image_default_save'
default_object_file_name = 'astobj_default_save'
default_exp = 1800
default_filter = 'K'
default_fov = 'wide'
default_chip = 'centre'
default_ao = 'scao'


def Image(astobj, exp_time, ndit, filter_name, view_mode, chip_mode, ao_mode, savename):
    """Image the given object."""
    if isinstance(astobj, str):
        astobj = obg.AstObject.load_from(astobj)
    
    src = img.MakeSource(astobj, filter=filter_name)
    
    astimage = img.MakeImage(src, 
                             exposure=exp_time, 
                             NDIT=ndit, 
                             view=view_mode, 
                             chip=chip_mode, 
                             filter=filter_name, 
                             ao_mode=ao_mode, 
                             filename=savename)
    
    return astimage


def DynamicImage(astobj=None, name=None):
    """Dynamically give parameters for imaging via this interactive function."""
    print('--------------------------------------------------------------------------------')
    print('The program will now ask for the parameters to image your object.')
    print('Type <quit> to exit this function. Type <help> for additional information.')
    print('--------------------------------------------------------------------------------')
    
    if ((astobj is None) | (isinstance(astobj, str))):
        astobj, astobj_str = LoadAstObj(str(astobj))
    else:
        if (name is None):
            astobj_str = default_object_file_name                                                   # no name was given for the preloaded obj, assume default
        else:
            astobj_str = str(name)
    
    exp_time = ExposureTime()
    ndit = NIterations()
    filter = Filter(astobj.mag_names)
    view_mode = FieldOfView()
    chip_mode = ChipConfig()
    ao_mode = AdaptiveOptics()
    
    savename = SaveFileName()
    
    print('\n  Working...\n')
    
    astimage = Image(astobj, exp_time, ndit, filter, view_mode, chip_mode, ao_mode, savename)
    
    print('')
    
    command = OneLineCommand(astobj_str, exp_time, ndit, filter, view_mode, chip_mode, 
                             ao_mode, savename)
    
    print('To make this astronomical image quickly, the following command can be used:')
    if (command == 'python imager.py'):
        print(' |  [Note: all default values were used, the command is very short!]')
    print(command)
    
    ShowImage(astimage)
    
    return astimage


def AskHelp(function, add_args):
    """Shows the help on a particular function."""
    if (function == 'Filter'):
        print(' |  Filters must be in the intersection of the filter set in the isochrone files')
        print(' |  and the set of the imaging program (SimCADO).')
        print(' |  The available filters are: {0}.'.format(', '.join(add_args)))
    elif (function == 'FieldOfView'):
        print(' |  The setting "wide" gives a fov of 53 arcsec with 4 mas/pixel,')
        print(' |  "zoom" gives a fov of 16 arcsec having 1.5 mas/pixel.')
    elif (function == 'ChipConfig'):
        print(' |  Choose "full" to use all 9 of the 4k chips on the detector,')
        print(' |  "centre" to use only the centre chip (4096x4096 pixels) and')
        print(' |  "small" for the middle 1024x1024 pixels')
    elif (function == 'AdaptiveOptics'):
        print(' |  Selects what PSF file is used. Choose from: scao, ltao, '
              '(mcao not available yet)')
    else:
        print(' |  No help available for this function at this stage.')
        
    return


def LoadAstObj(astobj_name=''):
    """Asks what astobj you want to load."""
    # make a list of available files (can only pick available files in this way)
    fnames_cur_folder = [f for f in os.listdir() if os.path.isfile(f)]
    if os.path.isdir('objects'):                                                                     # make sure the folder is there
        fnames_pkl_folder = [f for f in os.listdir('objects') 
                             if os.path.isfile(os.path.join('objects', f))]
        fnames_available = fnames_pkl_folder + fnames_cur_folder
    else:
        fnames_available = fnames_cur_folder
        
    fnames_list = [f for f in fnames_available if (f[-4:] == '.pkl')]
    
    if (len(fnames_list) == 0):
        raise OSError('imager//LoadAstObj: no astronomical object files available: '
                      'first make an object.')
    
    fnames_list += [f.replace('.pkl', '') for f in fnames_list]
    
    if ((astobj_name != '') & (astobj_name in fnames_list)):
        file_name = astobj_name
    else:
        if default_object_file_name in fnames_list:
            default = '[' + default_object_file_name + ']'
        else:
            default = ''
        
        file_name = utils.while_ask('Which astronomical object file do you want to load?',
                             '{}'.format(default), add_opt=fnames_list, function='LoadAstObj')
    
    astobj = obg.AstObject.load_from(file_name)
    return astobj, file_name


def ExposureTime():
    """Asks how long the exposure time must be."""
    exp_str = utils.while_ask('Exposure time for the observation (seconds)',
                       '[{}]'.format(default_exp), function='ExposureTime', check='int')
    return int(exp_str)


def NIterations():
    """Asks how many iterations to make."""
    ndit_str = utils.while_ask('Number of exposures to make:', '[1]', function='NIterations', check='int')
    
    if (ndit_str != '1'):
        print(' |  [Note: the number of iterations doesn\'t seem to do anything '
              'at the moment*, except make readout slower.]')
        print(' |  [*except when you changed some default to True in the config file of SimCADO]')
        
    return int(ndit_str)


def Filter(filters):
    """Asks what astronomical filter to use."""
    filter_default = '/'.join(filters)
    filter_default = filter_default.replace(default_filter, '[' + default_filter + ']')
    filter = utils.while_ask('Filter to use:', filter_default, function='Filter', help_arg=filters)
    return filter


def FieldOfView():
    """Asks what view mode the telescope should be set to."""
    options = 'wide/zoom'.replace(default_fov, '[' + default_fov + ']')
    
    fov = utils.while_ask('View mode:', options, add_opt=['w', 'z'], function='FieldOfView')
    
    if (fov.lower() == 'w'):
        fov = 'wide'
    elif (fov.lower() == 'z'):
        fov = 'zoom'
    
    return fov


def ChipConfig():
    """Asks what detector layout to use."""
    options = 'full/centre/small'.replace(default_chip, '[' + default_chip + ']')
    
    chip = utils.while_ask('Detector layout:', options, add_opt=['f', 'c', 's'], function='ChipConfig')
    
    if (chip.lower() == 'f'):
        chip = 'full'
    elif (chip.lower() == 'c'):
        chip = 'centre'
    elif (chip.lower() == 's'):
        chip = 'small'
    
    return chip


def AdaptiveOptics():
    """Asks what AO mode to use."""
    options = 'scao/mcao/ltao'.replace(default_ao, '[' + default_ao + ']')
    
    ao = utils.while_ask('Adaptive optics mode:', options, add_opt=['sc', 'mc', 'lt'],
                         function='AdaptiveOptics')
    
    if (ao.lower() == 'sc'):
        ao = 'scao'
    elif (ao.lower() == 'mc'):
        ao = 'mcao'
    elif (ao.lower() == 'lt'):
        ao = 'ltao'
    
    return ao


def SaveFileName():
    """Asks if you want to change the filename."""
    file_name = default_image_file_name
    
    print('The image will be saved under the name ' + file_name)
    change = utils.while_ask(' |  Save image with a different filename?', 'y/[n]',
                             add_opt=['yes', 'no'], function='SaveFileName')
    
    if (change.lower() in ['y', 'yes']):
        file_name = utils.while_ask(' |  Give the new name', '', function='SaveFileName')
        
    return file_name


def ShowImage(image_name):
    """Asks if you want the image displayed."""
    show = utils.while_ask('Display the image?', '[y]/n', add_opt=['yes', 'no'], function='ShowImage')
    
    if (show in ['y', 'yes']):
        fh.plot_fits(image_name)
    
    return


def OneLineCommand(astobj_str, exp, ndit, filter, fov, chip, ao, savename):
    """Gives the command line format to get the generated object."""
    command = 'python imager.py'
    if (astobj_str != default_object_file_name):
        command += ' -astobj ' + astobj_str
    if (exp != default_exp):
        command += ' -exp ' + str(exp)
    if (ndit != 1):
        command += ' -ndit ' + str(ndit)
    if (filter != default_filter):
        command += ' -filter ' + filter
    if (fov != default_fov):
        command += ' -fov ' + fov
    if (chip != default_chip):
        command += ' -chip ' + chip
    if (ao != default_ao):
        command += ' -ao ' + ao
    if (savename != default_image_file_name):
        command += ' -save ' + savename
        
    return command


## read in arguments from cmd line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct an astronomical object from scratch.')
    
    parser.add_argument('-astobj', type=str, required=False, default=default_object_file_name,
                        help='the name of the file of the astronomical object to image.') 
    
    parser.add_argument('-exp', type=int, required=False, default=default_exp,
                        help='the exposure time in seconds for the image.')
                        
    parser.add_argument('-ndit', type=int, required=False, default=1,
                        help='the number of exposures to take.')
                    
    parser.add_argument('-filter', type=str, required=False, default=default_filter,
                        help='astronomical filter to use.')
                        
    parser.add_argument('-fov', type=str, required=False, default=default_fov,
                        choices=['wide','zoom'],
                        help='field of view mode of the telescope.')
                        
    parser.add_argument('-chip', type=str, required=False, default=default_chip,
                        choices=['full','centre','small'],
                        help='detector configuration to use.')
    
    parser.add_argument('-ao', type=str, required=False, default=default_ao,
                        choices=['scao','mcao','ltao'],
                        help='adaptive optics mode to use.')    
    
    parser.add_argument('-inter', type=bool, required=False, default=False,
                        help='set True to use the interactive imager function')
    
    parser.add_argument('-save', type=str, required=False, default=default_image_file_name,
                        help='file to save object to')
    
    args = parser.parse_args()

    # execute the main function
    if args.inter:
        astimage = DynamicImage(args.astobj)
    else:
        astimage = Image(args.astobj, 
                         args.exp, 
                         args.ndit, 
                         args.filter,
                         args.fov,
                         args.chip,
                         args.ao,
                         args.save)
    