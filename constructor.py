"""This module provides the command line argument parser
that takes user input for making an astronomical object
"""
import argparse
import fnmatch
import inspect

import objectgenerator as obg


# global defaults
default_rdist = 'normal'        # see distributions module for a full list of options
default_imf_par = [0.08, 150]   # M_sun     lower bound, upper bound on mass
default_object_file_name = 'astobj_default_save'

# read in arguments from cmd line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct an astronomical object from scratch.')
    # todo: add default settings to file
    parser.add_argument('-struct', type=str, required=False, default='AstronomicalObject',
                        choices=['StarCluster', 'EllipticalGalaxy', 'SpiralGalaxy'],
                        help='Type of structure to create. The default is a generic type.')
    # generic arguments
    parser.add_argument('-distance', '-d', type=float, required=False, default=1e3,
                        help='distance_3d to center of object in pc')

    parser.add_argument('-d_type', '-dtype', type=str, required=False, default='l',
                        choices=['l', 'z'],
                        help='Type of distance: luminosity distance or redshift.')

    parser.add_argument('-extinct', '-ext', type=float, required=False, default=0.,
                        help='Extinction between source and observer (measured in magnitudes).')
    # arguments for 'Stars' object
    parser.add_argument('-n_stars', '-n', type=int, nargs='+', required=False, default=[0],
                        help='The number of stars to be generated per stellar population.')
                        
    parser.add_argument('-M_tot_init', '-M', type=int, nargs='+', required=False, default=[0],
                        help='The total initial mass in stars per stellar population.')
                    
    parser.add_argument('-ages', type=float, nargs='+', required=False, default=[9],
                        help='Age per stellar population, logarithmic or linear scale.')
                        
    parser.add_argument('-metal', '-Z', type=float, nargs='+', required=False, default=[0.019],
                        help='Metallicity per stellar population.')

    parser.add_argument('-imf_par', '-imf', type=float, nargs='+', required=False, default=default_imf_par,
                        help='Lower bound and upper bound for the IMF star masses, '
                             'in pars of two per stellar population.')
                        
    parser.add_argument('-sfh', type=str, nargs='+', required=False, default=['none'],
                        help='Star formation history type per stellar population.')

    parser.add_argument('-min_ages', type=float, nargs='+', required=False, default=None,
                        help='Minimum ages for when sfh is used, per stellar population.')

    parser.add_argument('-tau_sfh', type=float, nargs='+', required=False, default=None,
                        help='Characteristic timescales for when sfh is used, per stellar population.')

    parser.add_argument('-origin', type=float, nargs='+', required=False, default=[0., 0., 0.],
                        help='Position of the origin of each stellar population in cartesian space.')

    parser.add_argument('-incl', '-inclination', type=float, nargs='+', required=False, default=[0.],
                        help='Inclination angle per stellar population '
                             '(rotation of object\'s x-axis towards z-axis (=l.o.s.)).')
                        
    parser.add_argument('-r_dist', '-rdist', type=str, nargs='+', required=False, default=[default_rdist],
                        choices=['exponential', 'normal', 'squared_cauchy', 'pearson_vii', 'king_globular'],
                        help='Type of radial distribution to use per stellar population.')
                        
    parser.add_argument('-r_dist_par', '-rdistpar', type=float, nargs='+', required=False, default=[1.0],
                        help='Radial distribution parameters (s, R). Specify in sets per stellar population.')
                        
    parser.add_argument('-ellipse_axes', '-axes', type=float, nargs='+', required=False, default=[1., 1., 1.],
                        help='Relative scales of the (x, y, z) axes per stellar population.')
                        
    parser.add_argument('-spiral_arms', '-arms', type=int, required=False, default=0,
                        help='Number of spiral arms (per stellar population).')
                        
    parser.add_argument('-spiral_bulge', '-bulge', type=float, required=False, default=0.,
                        help='Relative proportion of the central bulge (per stellar population).')
                        
    parser.add_argument('-spiral_bar', '-bar', type=float, required=False, default=0.,
                        help='Relative proportion of the central bar (per stellar population).')

    parser.add_argument('-compact_mode', '-cp_mode', type=str, required=False, default=None,
                        choices=['num', 'mag'],
                        help='Compacting mode for limiting the number of stars generated.')
                        
    # parser.add_argument('-limit', type=float, required=False, default=None,
    #                     help='magnitude limit to use in compacting')
    # arguments for 'Gas' object
    # arguments for 'Dust' object
    # additional arguments
    parser.add_argument('-file_name', '-fname', type=str, required=False, default=default_object_file_name,
                        help='file to save object to')
    
    kwargs = vars(parser.parse_args())  # convert namespace to dict
    file_name = kwargs.pop('file_name')
    # Construct an astronomical object with the given settings.
    astobj = getattr(obg, kwargs.pop('struct'))(**kwargs)
    astobj.save_to(file_name)  # save the object
