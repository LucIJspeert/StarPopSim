# Luc IJspeert
# Part of smoc: (astronomical) image generator (uses SimCADO)
##
import os
import numpy as np

import simcado as sim

'''This module generates an image or collection thereof, in the usual fits format.
The software package SimCADO is used for the optical train.
The source is made with the provided astronomical object.
-> a future project could be to write this part out as well
'''

def MakeSource(astobj, filter='V'):
    '''Makes a SimCADO Source object from an AstObj.
    filter determines what magnitudes are used (corresponding to that filter).
    '''
    x_as = list(np.arctan(astobj.coords[:, 0]/astobj.d_ang)*648000/np.pi)                           # original coordinates assumed to be in pc
    y_as = list(np.arctan(astobj.coords[:, 1]/astobj.d_ang)*648000/np.pi)                           # the  *648000/np.pi  is rad to as
    
    magnitudes = astobj.ApparentMagnitude(filter_name=filter)[0]
    
    src = sim.source.stars(mags=magnitudes, 
                            x=x_as, 
                            y=y_as, 
                            filter_name=filter, 
                            spec_types=astobj.spec_names[astobj.spec_types])
    
    return src
    
def MakeImage(src, exposure=60, NDIT=1, view='wide', chip='centre', filter='V', ao_mode='scao', filename='image_default_save'):
    '''Make the image with SimCADO.
    exposure = time in seconds, NDIT = number of exposures taken.
    view = mode = ['wide', 'zoom']: fov 53 arcsec (4 mas/pixel) or 16 (1.5 mas/pixel)
    chip = detector_layout = ['small', 'centre', 'full']: 
        1024x1024 pix, one whole detector (4096x4096 pix) or full array of 9 detectors
    filter = the filter used in the 'observation'
    ao_mode = PSF file used [scao, ltao, (mcao not available yet)]
    '''
    savename = os.path.join('images', filename)
    if (savename[-5:] != '.fits'):
        savename += '.fits'
    
    image = sim.run(src, 
                    filename=savename, 
                    mode=view, 
                    detector_layout=chip, 
                    filter_name=filter, 
                    SCOPE_PSF_FILE=ao_mode, 
                    OBS_EXPTIME=exposure, 
                    OBS_NDIT=NDIT)
    
    return image















    