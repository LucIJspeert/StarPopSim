"""This module generates an image or collection thereof, in the usual fits format.
The software package SimCADO is used for the optical train.
The source is made with the provided astronomical object.
-> a future project could be to write this part out as well
"""
import os
import numpy as np

import simcado as sim


# global constants
default_image_file_name = 'image_default_save'
rad_as = 648000/np.pi               # rad to arcsec


def MakeSource(astobj, filter='V'):
    """Makes a SimCADO Source object from an AstObj.
    filter determines what magnitudes are used (corresponding to that filter).
    """
    coords_as = astobj.CoordsArcsec()
    x_as = coords_as[:, 0]
    y_as = coords_as[:, 1]
    
    magnitudes = astobj.ApparentMagnitudes(filter=filter)
    spec_i, spec_names = astobj.SpectralTypes()
    
    src = sim.source.stars(mags=magnitudes, x=x_as, y=y_as, filter=filter, 
                           spec_types=spec_names[spec_i])
    
    # add the guide stars (if generated)                        
    if hasattr(astobj, 'natural_guide_stars'):
        x_as, y_as, magnitudes, ngs_filter, spec_types = astobj.natural_guide_stars
        ngs_filter = np.unique(ngs_filter)[0]                                                       # reduce to 1 filter
        src += sim.source.stars(mags=magnitudes, x=x_as, y=y_as, filter=ngs_filter, 
                                spec_types=spec_types)
                                
    # add the foreground stars (if generated)                        
    # if hasattr(astobj, 'field_stars'):
    #     x_as, y_as, magnitudes, filters, spec_types = astobj.field_stars
    #     filter = np.unique(filters)[0]                                                              # reduce to 1 filter
    #     src += sim.source.stars(mags=magnitudes, x=x_as, y=y_as, filter=filter, 
    #                             spec_types=spec_types)
    
    return src


def MakeImage(src, exposure=60, NDIT=1, view='wide', chip='centre', filter='V', ao_mode='scao', 
              filename=default_image_file_name, internals=None, return_int=False):
    """Make the image with SimCADO.
    exposure = time in seconds, NDIT = number of exposures taken.
    view = mode = ['wide', 'zoom']: fov 53 arcsec (4 mas/pixel) or 16 (1.5 mas/pixel)
    chip = detector_layout = ['small', 'centre', 'full']: 
        1024x1024 pix, one whole detector (4096x4096 pix) or full array of 9 detectors
    filter = the filter used in the 'observation'
    ao_mode = PSF file used [scao, ltao, (mcao not available yet)]
    """
    savename = os.path.join('images', filename)
    if (savename[-5:] != '.fits'):
        savename += '.fits'
    
    if internals is not None:
        cmd, opt, fpa = internals
    else:
        cmd, opt, fpa = None, None, None
    
    image_int = sim.run(src, 
                        filename=savename, 
                        mode=view, 
                        detector_layout=chip, 
                        filter_name=filter, 
                        SCOPE_PSF_FILE=ao_mode, 
                        OBS_EXPTIME=exposure, 
                        OBS_NDIT=NDIT,
                        cmds=cmd,
                        opt_train=opt, 
                        fpa=fpa,
                        return_internals=return_int,
                        FPA_LINEARITY_CURVE='FPA_linearity.dat'
                        )
    
    if return_int:
        image, internals = image_int
        return image, internals
    else:
        image = image_int
        return image














    