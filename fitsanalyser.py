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
import scipy.spatial as sps
import matplotlib.pyplot as plt
import astropy as apy
import astropy.modeling as asm
import photutils as phu


# global constants
pix_scale = 4*10**-3                             # arcseconds per pixel


# global defaults
default_picture_file_name = 'picture_default_save'


def PixelToArcsec(x, filter):
    #todo: idea: automate centering with photutils
    if (filter == 'Ks'):
        filter_corr = 1.0 # 0.55
    elif (filter == 'H'):
        filter_corr = 1.4 # 0.95
    elif (filter == 'J'):
        filter_corr = 0.6 # 0.2
    return (x - 2048 - filter_corr)*pix_scale
    

def ArcsecToPixel(x, filter):
    if (filter == 'Ks'):
        filter_corr = 1.0 # 0.55
    elif (filter == 'H'):
        filter_corr = 1.4 # 0.95
    elif (filter == 'J'):
        filter_corr = 0.6 # 0.2
    return x/pix_scale + 2048 + filter_corr


def FindStarMatch(x, y, mag, x_ref, y_ref, mag_ref):
    """Matches stars to a reference set of coordinates and magnitudes.
    Expects arrays as input. Outputs distances, reference indices and outlier mask.
    For outlier detection it is assumed coordinates are in arcseconds
    """
    mag_mask = (mag_ref < np.max(mag) + 2)                                                          # preselect mag_ref below max(mag) + 2
    mag_ind = np.where(mag_mask)[0]
    
    # find a set of closest points based on x,y
    data_points = np.column_stack([x, y])
    tbl_grid = np.column_stack([x_ref[mag_mask], y_ref[mag_mask]])
    tbl_tree = sps.cKDTree(tbl_grid)                                                                # K-Dimensional lookup Tree
    dists_set, ind_set = tbl_tree.query(data_points, k=8)
    ind_set = mag_ind[ind_set]                                                                      # convert back to original indexing
    
    # determine best match with x,y,mag
    dists = np.zeros_like(x)
    ind_final = np.zeros_like(x, dtype=int)
    
    for i, ind_s in enumerate(ind_set):
        data_point = np.column_stack([x[i], y[i], mag[i]])
        tbl_grid = np.column_stack([x_ref[ind_s], y_ref[ind_s], mag_ref[ind_s]])
        tbl_tree = sps.cKDTree(tbl_grid)
        dists[i], ind_final[i] =  tbl_tree.query(data_point, k=1)
        ind_final[i] = ind_s[ind_final[i]]                                                          # convert back to original indexing
        
    # determine the bad matches (more than 1.5 pixel radial distance_3d)
    outliers = (((x_ref[ind_final] - x)**2 + (y_ref[ind_final] - y)**2)**(1/2) > 1.5*pix_scale)
    return dists, ind_final, outliers
    
    
def FindCoordMatch(x, y, x_ref, y_ref):
    """Matches (x,y) coordinates to a reference set by closest distance_3d.
    Expects arrays as input. Outputs distances, reference indices and outlier mask.
    """
    data_points = np.column_stack([x, y])
    tbl_grid = np.column_stack([x_ref, y_ref])
    tbl_tree = sps.cKDTree(tbl_grid)                                                                # K-Dimensional lookup Tree
    dists, ind = tbl_tree.query(data_points, k=1)
    outliers = (dists > 1.5*pix_scale)
    return dists, ind, outliers


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


def DoPhotometry(img_data, filter='Ks', show=False):
    with open(os.path.join('objects', 'epsf-scao-{0}-m18.pkl'.format(filter)), 'rb') as input:
        epsf = pickle.load(input)
    
    # define optimal values
    filter_names = np.array(['I', 'J', 'H', 'Ks'])
    param_values = np.array([[20, 4.8, 0.35], 
                             [30, 7.2, 0.33],
                             [30, 6.1, 0.34],
                             [20, 8.2, 0.31]
                             ])
    
    # do photometry
    sigma_to_fwhm = apy.stats.gaussian_sigma_to_fwhm
    bkgrms = phu.background.MADStdBackgroundRMS()
    std = bkgrms.calc_background_rms(data=img_data)
    th_val = param_values[filter_names == filter][0][0]
    sigma_psf = param_values[filter_names == filter][0][1]
    sl_val = param_values[filter_names == filter][0][2]
    
    photometry = phu.psf.DAOPhotPSFPhotometry(threshold=th_val*std, 
                                            fwhm=sigma_psf*sigma_to_fwhm, 
                                            sharplo=sl_val, sharphi=20.0, 
                                            roundlo=-10.0, roundhi=10.0, 
                                            crit_separation=2.0*sigma_psf*sigma_to_fwhm, 
                                            psf_model=epsf, 
                                            fitter=asm.fitting.LevMarLSQFitter(), 
                                            fitshape=(191,191), niters=5, 
                                            aperture_radius=sigma_psf*sigma_to_fwhm
                                            )
    result_tab = photometry(image=img_data)
    residual_image = photometry.get_residual_image()
    
    # throw away negative fluxes and uncertainties of zero
    result_tab_redux = result_tab[(result_tab['flux_fit'] > 0) & (result_tab['flux_0'] > 0)]
    result_tab_redux = result_tab_redux[np.invert((result_tab_redux['flux_unc'] == 0) & (result_tab_redux['iter_detected'] != 1))]
    
    if show:
        # show photometry results
        positions = (result_tab_redux['x_fit'], result_tab_redux['y_fit'])
        apertures = phu.CircularAperture(positions, r=5.)
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_data, cmap='viridis', aspect=1, interpolation='nearest', origin='upper')
        plt.title('Simulated data')
        plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04)
        
        plt.subplot(1, 2, 2)
        plt.imshow(residual_image, cmap='viridis', aspect=1, interpolation='nearest', origin='upper')
        apertures.plot(color='blue', lw=1.5)
        plt.title('Residual Image')
        plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04)
        plt.show()

    return result_tab, result_tab_redux


















