"""This module provides the argsparse for user input for making an astronomical object
as well as an interactive function (DynamicConstruct) that leads the user through all the options.
"""
#todo: [warning: due to changes in <obg> this module is no longer up to dat and will not work as intended]
import argparse
import fnmatch
import inspect
import numpy as np

import utils
import distributions as dist
import objectgenerator as obg


# global defaults
default_object_file_name = 'astobj_default_save'

    
def Construct(struct, n_stars, M_tot_init, age, metal, rel_num, distance, d_type, imf_par, sf_hist, extinct, incl,
              r_dist, r_dist_par, ellipse_axes, spiral_arms, spiral_bulge, spiral_bar, compact, cp_mode, mag_lim, save):
    """Construct an astronomical object with the given settings."""
    # todo: fix this to do the right thing
    N, M = NumAndMass()                                                                         # get the number or mass

    pop_n = PopulationAmount()                                                                  # get the amount of populations
    ages = PopAges(pop_n)                                                                       # get the age for each population
    Z = PopMetallicity(pop_n)                                                                   # get the metallicity for each population
    relN = PopRelativeN(pop_n)                                                                  # get the relative number in each population

    D_z, D_type = Distance()                                                                    # get the distance_3d to the centre of the object (plus type of distance_3d measurement)

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

    astobj = obg.AstObject(N_stars=N,
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
    astobj.save_to(save + '.pkl')  # save the object
    return astobj


# read in arguments from cmd line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct an astronomical object from scratch.')
    # todo: add default settings to file
    parser.add_argument('-struct', type=str, required=False, default='ellipsoid',
                        choices=['StarCluster', 'EllipticalGalaxy', 'SpiralGalaxy'],
                        help='type of structure to create')
    
    parser.add_argument('-n', type=int, nargs='+', required=False, default=0,
                        help='the total number of stars generated')
                        
    parser.add_argument('-M', type=int, nargs='+', required=False, default=0,
                        help='the total mass in generated stars')
                    
    parser.add_argument('-ages', type=float, nargs='+', required=False, default=[9],
                        help='age(s) of the stellar population(s), logarithmic or linear.')
                        
    parser.add_argument('-Z', type=float, nargs='+', required=False, default=[0.019], 
                        help='metallicity(/ies) of the stellar population(s)')
                        
    parser.add_argument('-reln', type=float, nargs='+', required=False, default=[1],
                        help='relative number of stars in each population')
    
    parser.add_argument('-d', type=float, required=False, default=1e3,
                        help='distance_3d to center of object in pc')
    
    # optional arguments
    parser.add_argument('-dtype', type=str, required=False, default='l',
                        choices=['l', 'z'],
                        help='type of distance; luminosity distance or redshift.')
                        
    parser.add_argument('-imf', type=float, nargs='+', required=False, default=[0.08, 0.5, 150],
                        help='lower bound, knee position, upper bound for the IMF masses')
                        
    parser.add_argument('-sfh', type=str, nargs='+', required=False, default=['none'],
                        help='star formation history type')
                        
    parser.add_argument('-ext', type=float, required=False, default=0.,
                        help='extinction between source and observer')
                        
    parser.add_argument('-incl', type=float, nargs='+', required=False, default=[0.],
                        help='inclination angle (rotation of object\'s '
                        'x-axis towards z-axis (=l.o.s.))')
                        
    parser.add_argument('-rdist', type=str, nargs='+', required=False, default=['normal'],
                        choices=['exponential', 'normal', 'squared_cauchy', 'pearson_vii',
                        'king_globular'], help='(ellipse) type of radial distribution')
                        
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
    parser.add_argument('-cp_mode', type=str, required=False, default=None,
                        choices=['num', 'mag'], help='compacting mode')
                        
    parser.add_argument('-limit', type=float, required=False, default=None,
                        help='magnitude limit to use for compacting')
    
    parser.add_argument('-save', type=str, required=False, default=default_object_file_name,
                        help='file to save object to')
    
    args = parser.parse_args()

    # execute the main function
    astobj = Construct(struct=args.struct,
                       n_stars=args.n,
                       M_tot_init=args.M,
                       age=args.ages,
                       metal=args.Z,
                       rel_num=args.reln,
                       distance=args.d,
                       d_type=args.dtype,
                       imf_par=args.imf,
                       sf_hist=args.sfh,
                       extinct=args.ext,
                       incl=args.incl,
                       r_dist=args.rdist,
                       r_dist_par=args.rdistpar,
                       ellipse_axes=args.axes,
                       spiral_arms=args.arms,
                       spiral_bulge=args.bulge,
                       spiral_bar=args.bar,
                       cp_mode=args.cp_mode,
                       mag_lim=args.limit,
                       save=args.save
                       )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    