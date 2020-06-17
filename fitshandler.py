"""This module contains the interactions with fits files.
It is likely biased towards the purpose of the main program.
It is assumed that the fits data is in a folder called 'images', 
positioned within the working directory.
"""
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.visualization import astropy_mpl_style


# global defaults
default_picture_file_name = 'picture_default_save'


def print_info(filename):
    """Shows the info of the HDUlist"""
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
        
    # clean way to open file (closes it automagically)
    with fits.open(os.path.join('images', filename)) as hdul:
        hdul.info()
    return

    
def print_keys(filename, index=0):
    """Shows the keywords for the cards. Optional arg: HDUlist index."""
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
    
    with fits.open(os.path.join('images', filename)) as hdul:
        hdr = hdul[index].header
        print(list(hdr.keys()))
    return

    
def print_hdr(filename, index=0, hdr_range=None):
    """Prints the header. Optional args: HDUlist index, header range."""
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'

    if not hdr_range:
        hdr_range = [0, -1]
    
    with fits.open(os.path.join('images', filename)) as hdul:
        hdr = hdul[index].header[hdr_range[0]:hdr_range[1]]
        print(repr(hdr), '\n')
    return

    
def print_card(filename, keyword, index=0, card_index=None):
    """Prints card: keyword (str or int). Optional arg: HDUlist index, card index."""
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
    
    with fits.open(os.path.join('images', filename)) as hdul:
        if (card_index is None):
            crd = (str.upper(keyword) + ' = ' + str(hdul[index].header[keyword]) 
                   + '       / ' + hdul[index].header.comments[keyword])
        else:
            # for history or comment cards
            crd = str.upper(keyword) + ' = ' + str(hdul[index].header[keyword][card_index])
        print(crd)
    return


def print_data(filename, index=0):
    """Prints the data. Optional arg: HDUlist index."""
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
        
    with fits.open(os.path.join('images', filename)) as hdul:
        print(hdul[index].data)
    return
    
    
def change_hdr(filename, keyword, value, comment='', index=0):
    """Adds/updates card 'keyword' (str) in the current file. 
    Input: 'value' (str or number) and optionally 'comment' (str). 
    Optional arg: HDUlist index.
    """
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
    
    with fits.open(os.path.join('images', filename), mode='update') as hdul:
        hdul[index].header.set(keyword, value, comment)
    return


def change_data(filename, input_data, index=0):
    """Changes (and saves) the data in the current fits file. 
    Optional arg: HDUlist index.
    """
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
    
    with fits.open(os.path.join('images', filename), mode='update') as hdul:
        hdul[index].data = input_data
    return


def get_card_value(filename, keyword, index=0):
    """Returns the value of card 'keyword' (str). Returns 0 if value is a string. 
    Optional arg: HDUlist index.
    """
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
    
    with fits.open(os.path.join('images', filename)) as hdul:
        value = hdul[index].header[keyword]
        if isinstance(value, str):
            warnings.warn('Card value is a string.')
            value = 0
    
    return value


def get_data(filename, index=0):
    """Returns the requested data. [NOTE: data[1, 4] gives pixel value at x=5, y=2.] 
    Optional arg: HDUlist index.
    """
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
            
    with fits.open(os.path.join('images', filename)) as hdul:
        return hdul[index].data


def new_hdr(keywords, values, comments=None):
    """Returns a new header object. 
    Inputs are lists of the keywords (str), values (str or number) and optional comments (str).
    """
    if (len(keywords) != len(values)):
        raise ValueError('Must enter as much values as keywords.')
    elif ((not hasattr(keywords, '__len__')) | (not hasattr(values, '__len__'))):
        raise ValueError('Arguments have length.')

    if not comments:
        comments = ['' for i in range(len(keywords))]
    elif (len(comments) != len(keywords)):
        raise ValueError('Must enter as much comments as keywords.')

    hdr = fits.Header()
    for i in range(len(keywords)):
        hdr.set(keywords[i], values[i], comments[i])
    return hdr


def new_fits(filename, input_data, input_header=None):
    """Saves the input_data to a new file 'file_name'.
    Optional arg: input_header (header object)
    """
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
    
    fits.writeto(os.path.join('images', filename), input_data, header=input_header)
    return


def add_to_fits(filename, input_data, input_header=None):
    """Appends the header/data to fits file if 'file_name' exists, creates one if not.
    Optional arg: input_header (header object).
    """
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
    
    fits.append(os.path.join('images', filename), input_data, header=input_header)
    return


def plot_fits(filename, index=0, colours='gray', scale='lin', grid=False, chip='single', show=True):
    """Displays the image in a fits file. Optional args: HDUlist index, colours.
    Can also take image objects directly.
    scale can be set to 'lin', 'sqrt', and 'log'
    chip='single': plots single data array at given index.
        ='full': expects data in index 1-9 and combines it.
    """
    if isinstance(filename, str):
        if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
            filename += '.fits'
            
        if (chip == 'single'):
            image_data = get_data(filename, index)
        elif (chip == 'full'):
            image_data = [get_data(filename, i + 1) for i in range(9)]
        else:
            raise ValueError('Chip configuration not recognised.')
    else:
        if (chip == 'single'):
            image_data = filename[index].data
        elif (chip == 'full'):
            image_data = [filename[i+1].data for i in range(9)]
        else:
            raise ValueError('Chip configuration not recognised.')
            
    if (chip == 'full'):
        image_data_r1 = np.concatenate(image_data[6], image_data[7],
                                       image_data[8], axis=1)
        image_data_r2 = np.concatenate(image_data[5], image_data[4],
                                       image_data[3], axis=1)
        image_data_r3 = np.concatenate(image_data[0], image_data[1],
                                       image_data[2], axis=1)
        # stitch all chips together
        image_data = np.concatenate(image_data_r1, image_data_r2, image_data_r3, axis=0)
            
    if (scale == 'log'):
        image_data = np.log10(image_data - np.min(image_data))
    elif (scale == 'sqrt'):
        image_data = (image_data - np.min(image_data))**(1/2)
    
    # use nice plot parameters
    plt.style.use(astropy_mpl_style)
    
    fig, ax = plt.subplots(figsize=[12.0, 12.0])
    ax.grid(grid)
    cax = ax.imshow(image_data, cmap=colours)
    fig.colorbar(cax)
    plt.tight_layout()
    if show:
        plt.show()
    return


def save_fits_plot(filename, index=0, colours='gray', scale='lin', grid=False, chip='single'):
    """Saves the plotted image in a fits file. Optional args: HDUlist index, colours.
    Can also take image objects directly.
    scale can be set to 'lin', 'sqrt', and 'log'
    chip='single': plots single data array at given index.
        ='full': expects data in index 1-9 and combines it.
    """
    if isinstance(filename, str):
        if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
            filename += '.fits'
            
        if (chip == 'single'):
            image_data = get_data(filename, index)
        elif (chip == 'full'):
            image_data = [get_data(filename, i + 1) for i in range(9)]
        else:
            raise ValueError('fitshandler//save_fits_plot: chip configuration not recognised.')
    else:
        if (chip == 'single'):
            image_data = filename[index].data
        elif (chip == 'full'):
            image_data = [filename[i+1].data for i in range(9)]
        else:
            raise ValueError('fitshandler//save_fits_plot: chip configuration not recognised.')
            
    if (chip == 'full'):
        image_data_r1 = np.concatenate(image_data[6], image_data[7],
                                       image_data[8], axis=1)
        image_data_r2 = np.concatenate(image_data[5], image_data[4],
                                       image_data[3], axis=1)
        image_data_r3 = np.concatenate(image_data[0], image_data[1],
                                       image_data[2], axis=1)
        
        image_data = np.concatenate(image_data_r1, image_data_r2, image_data_r3, axis=0)
    
    if (scale == 'log'):
        image_data = np.log10(image_data - np.min(image_data))
    elif (scale == 'sqrt'):
        image_data = (image_data - np.min(image_data))**(1/2)
    
    # use nice plot parameters
    plt.style.use(astropy_mpl_style)
    
    fig, ax = plt.subplots(figsize=[12.0, 12.0])
    ax.grid(grid)
    cax = ax.imshow(image_data, cmap=colours)
    fig.colorbar(cax)
    
    if isinstance(filename, str):
        name = os.path.join('images', filename.replace('.fits', '.png').replace('.fit', '.png'))
    else:
        name = os.path.join('images', default_picture_file_name + '.png')
    
    plt.savefig(name, bbox_inches='tight', dpi=600)
    plt.close()
    return
    
    
#todo: make add and subtract functions 
# (does this have added value?)






















