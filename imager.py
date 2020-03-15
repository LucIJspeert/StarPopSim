"""This module provides the command line argument parser
that takes user input for making an astronomical image
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


# read in arguments from cmd line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an image of an astronomical object.')
    
    parser.add_argument('-astobj', '-obj', type=str, required=False, default=default_object_file_name,
                        help='The name of the file of the astronomical object to image.')
    
    parser.add_argument('-exp_time', '-exp', type=int, required=False, default=default_exp,
                        help='The exp_time time in seconds for the image.')
                        
    parser.add_argument('-ndit', type=int, required=False, default=1,
                        help='The number of exposures to take.')

    filters = list(utils.get_supported_filters(alt_names=True))
    parser.add_argument('-filter', type=str, required=False, default=default_filter, choices=filters,
                        help='Astronomical filter to use.')
                        
    parser.add_argument('-fov', type=str, required=False, default=default_fov, choices=['wide', 'zoom'],
                        help='field of fov mode of the telescope.')
                        
    parser.add_argument('-chip', type=str, required=False, default=default_chip,
                        choices=['full', 'centre', 'small'],
                        help='detector configuration to use.')
    
    parser.add_argument('-ao_mode', '-ao', type=str, required=False, default=default_ao,
                        choices=['scao', 'mcao', 'ltao'],
                        help='Adaptive optics mode to use.')

    parser.add_argument('-file_name', '-fname', type=str, required=False, default=default_image_file_name,
                        help='File name to save the image to.')
    
    kwargs = vars(parser.parse_args())  # convert namespace to dict
    # execute the main function
    astobj = obg.AstronomicalObject.load_from(kwargs.pop('astobj'))
    src = img.MakeSource(astobj, filter=kwargs['filter'])
    image = img.MakeImage(src, **kwargs)
