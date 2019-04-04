# Luc IJspeert
# Part of smoc: fits file handler
##
"""This module contains the interactions with fits files. 
It is likely biased towards the purpose of the main program.
It is assumed that the fits data is in a folder called 'images', 
positioned within the working directory.
"""
import os

import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.visualization import astropy_mpl_style


# global defaults
default_picture_file_name = 'picture_default_save'


def ShowInfo(filename):
    """Shows the info of the HDUlist"""
    # clean way to open file (closes it automagically)
    with fits.open(os.path.join('images', filename)) as hdul:
        hdul.info()
    return

    
def ShowKeys(filename, index=0):
    """Shows the keywords for the cards. Optional arg: HDUlist index."""
    with fits.open(os.path.join('images', filename)) as hdul:
        hdr = hdul[index].header
        print(list(hdr.keys()))
    return

    
def PrintHdr(filename, index=0, range=[0, -1]):
    """Prints the header. Optional args: HDUlist index, header range."""
    with fits.open(os.path.join('images', filename)) as hdul:
        hdr = hdul[index].header[range[0]:range[1]]
        print(repr(hdr), '\n')
    return

    
def PrintCard(filename, keyword, index=0, card_index=None):
    """Prints card: keyword (str or int). Optional arg: HDUlist index, card index."""
    with fits.open(os.path.join('images', filename)) as hdul:
        if (card_index is None):
            crd = (str.upper(keyword) + ' = ' + str(hdul[index].header[keyword]) 
                   + '       / ' + hdul[index].header.comments[keyword])
        else:
            #for history or comment cards
            crd = str.upper(keyword) + ' = ' + str(hdul[index].header[keyword][card_index])
        print(crd)
    return


def PrintData(filename, index=0):
    """Prints the data. Optional arg: HDUlist index."""
    with fits.open(os.path.join('images', filename)) as hdul:
        print(hdul[index].data)
    return
    
    
def ChangeHdr(filename, keyword, value, comment='', index=0):
    """Adds/updates card 'keyword' (str) in the current file. 
    Input: 'value' (str or number) and optionally 'comment' (str). 
    Optional arg: HDUlist index.
    """
    with fits.open(os.path.join('images', filename), mode='update') as hdul:
        hdul[index].header.set(keyword, value, comment)
    return


def ChangeData(filename, input_data, index=0):
    """Changes (and saves) the data in the current fits file. 
    Optional arg: HDUlist index.
    """
    open_mode = ['readonly', 'update']
    with fits.open(os.path.join('images', filename), mode='update') as hdul:
        hdul[index].data = input_data
    return


def GetCardValue(filename, keyword, index=0):
    """Returns the value of card 'keyword' (str). Returns 0 if value is a string. 
    Optional arg: HDUlist index.
    """
    with fits.open(os.path.join('images', filename)) as hdul:
        value = hdul[index].header[keyword]
        if isinstance(value, str):
            raise ValueError('fitshandler//GetCardValue: Card value is a string.')
        else:
            return value


def GetData(filename, index=0):
    """Returns the requested data. [NOTE: data[1, 4] gives pixel value at x=5, y=2.] 
    Optional arg: HDUlist index.
    """
    if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
        filename += '.fits'
            
    with fits.open(os.path.join('images', filename)) as hdul:
        return hdul[index].data


def NewHdr(keywords, values, comments=''):
    """Returns a new header object. 
    Inputs are lists of the keywords (str), values (str or number) and optional comments (str).
    """
    hdr = fits.Header()
    if (len(keywords) != len(values)):
        raise ValueError('fitshandler//NewHdr: Must enter as much values as keywords')
    elif (type(keywords) != list) | (type(values) != list):
        raise ValueError('fitshandler//NewHdr: Arguments must be lists')
    else:
        if (type(comments) != list):
            comments = ['' for i in range(len(keywords))]
        elif (len(comments) != len(keywords)):
            raise ValueError('fitshandler//NewHdr: Must enter as much comments as keywords')
        for i in range(len(keywords)):
            hdr.set(keywords[i], values[i], comments[i])
        return hdr


def NewFits(filename, input_data, input_header=None):
    """Saves the input_data to a new file 'filename'. 
    Optional arg: input_header (header object)
    """
    fits.writeto(os.path.join('images', filename), input_data, header=input_header)
    return


def AddToFits(filename, input_data, input_header=None):
    """Appends the header/data to fits file if 'filename' exists, creates one if not. 
    Optional arg: input_header (header object).
    """
    fits.append(os.path.join('images', filename), input_data, header=input_header)
    return


def PlotFits(filename, index=0, colours='gray', grid=True):
    """Displays the image in a fits file. Optional args: HDUlist index, colours.
    Can also take image objects directly.
    """
    if isinstance(filename, str):
        if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
            filename += '.fits'
    
        image_data = GetData(filename, index)
    else:
        image_data = filename[index].data
        
    # use nice plot parameters
    plt.style.use(astropy_mpl_style)
    
    fig, ax = plt.subplots(figsize=[12.0, 12.0])
    cax = ax.imshow(image_data, cmap=colours)
    ax.grid(grid)
    cbar = fig.colorbar(cax)
    plt.show()
    return


def SaveFitsPlot(filename, index=0, colours='gray', grid=True):
    """Saves the plotted image in a fits file. Optional args: HDUlist index, colours.
    Can also take image objects directly.
    """
    if isinstance(filename, str):
        if ((filename[-5:] != '.fits') & (filename[-4:] != '.fit')):
            filename += '.fits'
    
        image_data = GetData(filename, index)
    else:
        image_data = filename[index].data
        
    # use nice plot parameters
    plt.style.use(astropy_mpl_style)
    
    fig, ax = plt.subplots(figsize=[12.0, 12.0])
    cax = ax.imshow(image_data, cmap=colours)
    ax.grid(grid)
    cbar = fig.colorbar(cax)
    
    if isinstance(filename, str):
        name = os.path.join('images', filename.replace('.fits', '.png').replace('.fit', '.png'))
    else:
        name = os.path.join('images', default_picture_file_name + '.png')
    
    plt.savefig(name, bbox_inches='tight', dpi=600)
    return























