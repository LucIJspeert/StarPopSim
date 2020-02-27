"""This module provides a parser that takes user input for making an astronomical image
"""
import argparse
import utils
import objectgenerator as obg
import imagegenerator as img


# defaults
default_image_file_name = 'image_default_save'
default_object_file_name = 'astobj_default_save'
default_exp = 1800  # in s
default_filter = 'K'
default_fov = 'wide'
default_chip = 'centre'
default_ao = 'scao'


def Image(astobj, exp_time, ndit, filter_name, view_mode, chip_mode, ao_mode, savename):
    """Image the given object with the given settings."""
    if isinstance(astobj, str):
        astobj = obg.AstronomicalObject.load_from(astobj)
    
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


# read in arguments from cmd line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an image of an astronomical object.')
    
    parser.add_argument('-fname', type=str, required=False, default=default_object_file_name,
                        help='the name of the file of the astronomical object to image.') 
    
    parser.add_argument('-exp', type=int, required=False, default=default_exp,
                        help='the exposure time in seconds for the image.')
                        
    parser.add_argument('-ndit', type=int, required=False, default=1,
                        help='the number of exposures to take.')

    filters = list(utils.get_supported_filters(alt_names=True))
    parser.add_argument('-filter', type=str, required=False, default=default_filter,
                        choices=filters, help='astronomical filter to use.')
                        
    parser.add_argument('-fov', type=str, required=False, default=default_fov,
                        choices=['wide', 'zoom'],
                        help='field of view mode of the telescope.')
                        
    parser.add_argument('-chip', type=str, required=False, default=default_chip,
                        choices=['full', 'centre', 'small'],
                        help='detector configuration to use.')
    
    parser.add_argument('-ao', type=str, required=False, default=default_ao,
                        choices=['scao', 'mcao', 'ltao'],
                        help='adaptive optics mode to use.')    

    parser.add_argument('-save', type=str, required=False, default=default_image_file_name,
                        help='file to save object to')
    
    args = parser.parse_args()
    # execute the main function
    image = Image(args.fname, args.exp, args.ndit, args.filter, args.fov, args.chip, args.ao, args.save)