# Luc IJspeert
# Part of starpopsim: (basic) PSF photometry
##
"""This module contains the analysis functions for fits files. 
It is likely biased towards the purpose of the main program.
It is assumed that the fits data is in a folder called 'images', 
positioned within the working directory.
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import astropy as apy
import astropy.modeling as asm
import photutils as phu


# global defaults
default_picture_file_name = 'picture_default_save'


def BuildEPSF(filter='Ks'):
    """Builds the effective PSF used for the photometry.
    Currently uses the SCAO PSF from SimCADO. 
    """
    src = sim.source.star(mag=19, filter_name=filter, spec_type='M0V')
    image = img.MakeImage(src, exposure=1800, NDIT=1, view='wide', 
                          chip='centre', filter=filter, ao_mode='scao')
                          # PSF_AnisoCADO_SCAO_FVPSF_4mas_EsoMedian_20190328.fits
                          
    peaks_tbl = phu.find_peaks(image, threshold=135000., box_size=11)
    peaks_tbl['peak_value'].info.format = '%.8g'  # for consistent table output
    # make sure the positions are correct (use the exact ones)
    peaks_tbl['x_peak'] = src.x_pix
    peaks_tbl['y_peak'] = src.y_pix
    positions = (peaks_tbl['x_peak'], peaks_tbl['y_peak'])
    apertures = phu.CircularAperture(positions, r=5.)
    # extract cutouts of the stars using the extract_stars() function
    stars_tbl = apta.Table()
    stars_tbl['x'] = peaks_tbl['x_peak']
    stars_tbl['y'] = peaks_tbl['y_peak']
    mean_val, median_val, std_val = apy.stats.sigma_clipped_stats(img_data, sigma=2.)
    img_data -= median_val                                                                          # subtract background
    nddata = apy.nddata.NDData(data=img_data)
    stars = phu.psf.extract_stars(nddata, stars_tbl, size=170)
    # build the epsf
    epsf_builder = phu.EPSFBuilder(oversampling=4, maxiters=5, progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)
    # save the epsf
    with open(os.path.join('objects', 'epsf-scao-fv.pkl'), 'wb') as output:
        pickle.dump(epsf, output, -1)






















