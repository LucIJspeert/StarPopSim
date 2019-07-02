# Luc IJspeert
# Part of starpopsim: Visualizer
##
"""Visualize data by making images/plots in 2D or 3D. 
Just for convenience really. And nice plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from mpl_toolkits.mplot3d import Axes3D

import conversions as conv


def Scatter2D(coords, title='Scatter', xlabel='x', ylabel='y',
              axes='xy', colour='blue', T_eff=None, mag=None, theme=None):
    """Plots a 2D scatter of a 3D object (array) along given axes [xy, yz, zx]. 
    Giving effective temperatures makes the marker colour represent temperature.
    Giving magnitudes makes the marker size and alpha scale with brightness.
    Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but 
        saveable dark plot, 'fits' for a plot that resembles a .fits image,
        and None for normal light colours.
    """
    # determine which axes to put on the horizontal and vertical axis of the plot
    axes_list = np.array([[0, 1], [1, 2], [2, 0]])
    hor_axis, vert_axis = axes_list[np.array(['xy', 'yz', 'zx']) == axes][0]
    n_obj = len(coords[:,hor_axis])
    
    # colours can be made to match the temperature (using T_eff)
    if (T_eff is not None):
        colour = conv.TemperatureToRGB(T_eff).transpose()                                           # T_eff array of temps of the objects
        colour[T_eff <= 10] = [0.2, 0.2, 0.2]                                                       # dead stars
        colour = mcol.to_rgba_array(colour, 0.5)                                                    # add alpha
    elif (colour == 'blue'):
        colour = np.array([mcol.to_rgba('tab:blue', 0.5)])
    else:
        colour = np.array([mcol.to_rgba(colour, 0.5)])
    
    # marker sizes and transparancy scaling with mag
    if (mag is not None):
        m_max = np.max(mag)
        sizes = (0.3 + (m_max - mag)/1.5)**2                                                        # formula for representation of magnitude
        if (theme == 'fits'):
            alpha = (m_max - mag)**3
            alpha = alpha/np.max(alpha)                                                             # scale alpha with mag
            colour = np.tile((1,1,1), (len(alpha), 1))                                              # set colours to white
            colour = mcol.to_rgba_array(colour, alpha)
    else:
        sizes = 20                                                                                  # default size
    
    fig, ax = plt.subplots(figsize=[7.0, 5.5])
    ax.scatter(coords[:,hor_axis], coords[:,vert_axis], marker='.', 
               linewidths=0.0, c=colour, s=sizes)
    
    # take the maximum distance from the origin as axis scale
    if (np.shape(coords[0]) == (2,)):
        axis_size = max(max(coords[:,0]), max(coords[:,1]))                                       # two dimensional data
    else:
        axis_size = max(max(coords[:,0]), max(coords[:,1]), max(coords[:,2]))
    ax.set_xlim(-axis_size, axis_size) 
    ax.set_ylim(-axis_size, axis_size)
    ax.set(aspect='equal', adjustable='datalim')
    
    # The further setup (custom dark theme)
    if (theme == 'dark1'):
        # fancy dark theme
        c_grey = '0.6'
        c_grey2 = '0.4'
        fig.patch.set_color('black')
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color(c_grey2)
        ax.spines['top'].set_color(c_grey2) 
        ax.spines['right'].set_color(c_grey2)
        ax.spines['left'].set_color(c_grey2)
        ax.tick_params(axis='x', colors=c_grey2)
        ax.tick_params(axis='y', colors=c_grey2)
        ax.title.set_color(c_grey)
        ax.xaxis.label.set_color(c_grey)
        ax.yaxis.label.set_color(c_grey)
    elif (theme == 'dark2'):
        # dark theme for good saving
        c_grey = '0.6'
        c_grey2 = '0.4'
        # fig.patch.set_color('black')
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color(c_grey2)
        ax.spines['top'].set_color(c_grey2) 
        ax.spines['right'].set_color(c_grey2)
        ax.spines['left'].set_color(c_grey2)
        ax.tick_params(axis='x', colors=c_grey2)
        ax.tick_params(axis='y', colors=c_grey2)
        ax.title.set_color(c_grey)
        ax.xaxis.label.set_color(c_grey)
        ax.yaxis.label.set_color(c_grey)
    elif (theme == 'fits'):
        # theme for imitating .fits images
        c_grey = '0.6'
        c_grey2 = '0.4'
        fig.patch.set_color('black')
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color(c_grey2)
        ax.spines['top'].set_color(c_grey2) 
        ax.spines['right'].set_color(c_grey2)
        ax.spines['left'].set_color(c_grey2)
        ax.tick_params(axis='x', colors=c_grey2)
        ax.tick_params(axis='y', colors=c_grey2)
        ax.title.set_color(c_grey)
        ax.xaxis.label.set_color(c_grey)
        ax.yaxis.label.set_color(c_grey)
        ax.invert_yaxis()
    # todo: dark theme 2 needs some work (no difference yet)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show() 
    return


def Scatter3D(coords, title='Scatter', xlabel='x', ylabel='y', zlabel='z', 
              colour='blue', T_eff=None, mag=None, theme=None):
    """Plots a 3D scatter of a 3D object (array).
    colour can be set to 'temperature' to represent the effective temperatures.
    magnitudes can be given to mag to make the marker sizes represent magnitudes.
    Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but 
        saveable dark plot, and None for normal light colours.
    """
    n_obj = len(coords[:, 0])
    
    # colours can be made to match the temperature (using T_eff)
    if (T_eff is not None):
        colour = conv.TemperatureToRGB(T_eff).transpose()                                           # T_eff array of temps of the objects
        colour[T_eff <= 10] = [0.2, 0.2, 0.2]                                                       # dead stars
        colour = mcol.to_rgba_array(colour, 0.5)                                                    # add alpha
    elif (colour == 'blue'):
        colour = mcol.to_rgba('tab:blue', 0.5)
    else:
        colour = mcol.to_rgba(colour, 0.5)
    
    # marker sizes and transparancy scaling with mag
    if (mag is not None):
        m_max = np.max(mag)
        sizes = (0.3 + (m_max - mag)/1.5)**2                                                        # formula for representation of magnitude
        if (theme == 'fits'):
            alpha = (m_max - mag)**3
            alpha = alpha/np.max(alpha)                                                             # scale alpha with mag
            colour = np.tile((1,1,1), (len(alpha), 1))                                              # set colours to white
            colour = mcol.to_rgba_array(colour, alpha)
    else:
        sizes = 20                                                                                  # default size
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=[6.0, 6.0])
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], 
               marker='.', linewidths=0.0, c=colour, s=sizes) 
    
    # take the maximum distance from the origin as axis scale
    axis_size = max(max(coords[:,0]), max(coords[:,1]), max(coords[:,2]))
    ax.set_xlim3d(-axis_size, axis_size) 
    ax.set_ylim3d(-axis_size, axis_size)
    ax.set_zlim3d(-axis_size, axis_size)
    ax.set(aspect='equal', adjustable='box')
    
    # The further setup (custom dark theme)
    if (theme == 'dark1'):
        # fancy dark theme
        c_grey = (0.6, 0.6, 0.6)
        c_grey2 = (0.4, 0.4, 0.4)
        c_black = (0, 0, 0, 0)                                                                      # (R,G,B,A)
        fig.patch.set_color('black')
        ax.set_facecolor(c_black)
        ax.w_xaxis.set_pane_color(c_black)     
        ax.w_yaxis.set_pane_color(c_black)
        ax.w_zaxis.set_pane_color(c_black)
        # find way to darken grid [not doable I think]
        ax.tick_params(axis='x', colors=c_grey2)
        ax.tick_params(axis='y', colors=c_grey2)
        ax.tick_params(axis='z', colors=c_grey2)
        ax.title.set_color(c_grey)
        ax.xaxis.label.set_color(c_grey)
        ax.yaxis.label.set_color(c_grey)
        ax.zaxis.label.set_color(c_grey)
    elif (theme == 'dark2'):
        # dark theme for good saving
        c_grey = (0.6, 0.6, 0.6)
        c_grey2 = (0.4, 0.4, 0.4)
        c_black = (0, 0, 0, 0)
        # fig.patch.set_color('black')
        ax.set_facecolor(c_black)
        ax.w_xaxis.set_pane_color(c_black)     
        ax.w_yaxis.set_pane_color(c_black)
        ax.w_zaxis.set_pane_color(c_black)
        # find way to darken grid [not doable I think]
        ax.tick_params(axis='x', colors=c_grey2)
        ax.tick_params(axis='y', colors=c_grey2)
        ax.tick_params(axis='z', colors=c_grey2)
        ax.title.set_color(c_grey)
        ax.xaxis.label.set_color(c_grey)
        ax.yaxis.label.set_color(c_grey)
        ax.zaxis.label.set_color(c_grey)
    # todo: dark theme 2 needs some work (no difference yet)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.tight_layout()
    plt.show() 
    return


def HRD(T_eff, log_Lum, title='HRD', xlabel='Temperature (K)', 
        ylabel=r'Luminosity log($L/L_\odot$)', colour='temperature', theme=None, mask=None):
    """Plot the Herzsprung Russell Diagram. Use mask to select certain stars.
    colours can be made to match the temperature (default behaviour)
    Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but 
        saveable dark plot, and None for normal light colours.
    """
    # colours can be made to match the temperature (using T_eff)
    if ((T_eff is not None) & (colour == 'temperature')):
        colour = conv.TemperatureToRGB(T_eff).transpose()                                           # T_eff array of temps of the objects
        colour[T_eff <= 10] = [0.2, 0.2, 0.2]                                                       # dead stars
        colour = mcol.to_rgba_array(colour, 0.5)                                                    # add alpha
    elif (colour == 'blue'):
        colour = mcol.to_rgba('tab:blue', 0.5)
    else:
        colour = mcol.to_rgba(colour, 0.5)
     
    # the mask can be used to hide remnants as the screw with the plot area   
    if mask is None:
        mask = np.ones_like(T_eff, dtype=bool)
    
    fig, ax = plt.subplots(figsize=[7.0, 5.5])
    ax.scatter(T_eff[mask], log_Lum[mask], marker='.', linewidths=0.0, c=colour)
    
    if (theme == 'dark1'):
        # fancy dark theme
        c_1 = '0.9'
        c_2 = '0.7'
        c_3 = '0.22'
        c_4 = '0.15'
    elif (theme == 'dark2'):
        # dark theme for good saving
        c_1 = '0.22'
        c_2 = '0.22'
        c_3 = '1.0'
        c_4 = '0.15'
    else:
        # defaults (not actually used)
        c_1 = '0.0'                                                                                 # words
        c_2 = '0.0'                                                                                 # lines
        c_3 = '1.0'                                                                                 # outer rim
        c_4 = '1.0'                                                                                 # inner area
    # todo: dark theme 2 needs some work
    
    if (theme is not None):
        fig.patch.set_color(c_3)
        ax.set_facecolor(c_4)
        ax.spines['bottom'].set_color(c_2)
        ax.spines['top'].set_color(c_2) 
        ax.spines['right'].set_color(c_2)
        ax.spines['left'].set_color(c_2)
        ax.tick_params(axis='x', colors=c_2)
        ax.tick_params(axis='y', colors=c_2)
        ax.title.set_color(c_1)
        ax.xaxis.label.set_color(c_1)
        ax.yaxis.label.set_color(c_1)
    
    ax.set_xlim(40000, 500) 
    ax.set_ylim(-5, 7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show() 
    return


def CMD(c_mag, mag, title='CMD', xlabel='colour', ylabel='magnitude', 
        colour='blue', T_eff=None, theme=None, adapt_axes=True, mask=None):
    """Plot the Colour Magnitude Diagram. Use mask to select certain stars.
    colours can be made to match the temperature (default behaviour)
    Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but 
        saveable dark plot, and None for normal light colours.
    """
    # colours can be made to match the temperature (using T_eff)
    if (T_eff is not None):
        colour = conv.TemperatureToRGB(T_eff).transpose()                                           # T_eff array of temps of the objects
        colour[T_eff <= 10] = [0.2, 0.2, 0.2]                                                       # dead stars
        colour = mcol.to_rgba_array(colour, 0.5)                                                    # add alpha
    elif (colour == 'blue'):
        colour = mcol.to_rgba('tab:blue', 0.5)
    else:
        colour = mcol.to_rgba(colour, 0.5)
        
    # the mask can be used to hide remnants as the screw with the plot area
    if mask is None:
        mask = np.ones_like(c_mag, dtype=bool)
    
    fig, ax = plt.subplots(figsize=[7.0, 5.5])
    ax.scatter(c_mag[mask], mag[mask], c=colour, marker='.', linewidths=0.0)
    
    if adapt_axes:
        max_x = np.max(c_mag[mask])
        min_x = np.min(c_mag[mask])
        max_y = np.max(mag[mask])
        min_y = np.min(mag[mask])
        ax.set_xlim(min_x - 0.25, max_x + 0.25) 
        ax.set_ylim(max_y + 4, min_y - 4)
    else:
        ax.set_xlim(-0.5, 2.0)                                                                      # good for seeing vertical differences
        ax.set_ylim(17.25, -12.75)
    
    if (theme == 'dark1'):
        # fancy dark theme
        c_1 = '0.9'
        c_2 = '0.7'
        c_3 = '0.22'
        c_4 = '0.15'
    elif (theme == 'dark2'):
        # dark theme for good saving
        c_1 = '0.22'
        c_2 = '0.22'
        c_3 = '1.0'
        c_4 = '0.15'
    else:
        # defaults (not actually used)
        c_1 = '0.0'                                                                                 # words
        c_2 = '0.0'                                                                                 # lines
        c_3 = '1.0'                                                                                 # outer rim
        c_4 = '1.0'                                                                                 # inner area
    # todo: dark theme 2 needs some work
    
    if (theme is not None):
        fig.patch.set_color(c_3)
        ax.set_facecolor(c_4)
        ax.spines['bottom'].set_color(c_2)
        ax.spines['top'].set_color(c_2) 
        ax.spines['right'].set_color(c_2)
        ax.spines['left'].set_color(c_2)
        ax.tick_params(axis='x', colors=c_2)
        ax.tick_params(axis='y', colors=c_2)
        ax.title.set_color(c_1)
        ax.xaxis.label.set_color(c_1)
        ax.yaxis.label.set_color(c_1)
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show() 
    return


def DistHist(dist, title='Histogram', xlabel='parameter', ylabel='relative number', 
             step=True, type='linear', labels=[]):
    """Display the histogram for some distribution. Can handle multiple distributions.
    type can be linear, linlog, loglin (x,y) and loglog. Step=False will give a plot instead.
    """
    dim = len(np.shape(dist))                                                                       # see if multiple dists given
    if (dim != 1):
        num_dists = np.shape(dist)[0]                                                               # number of dists
        if (len(labels) == num_dists):
            use_labels = True
        else:
            use_labels = False
    else:
        use_labels = False
    
    def plotfunc(dist):
        hist, bins = np.histogram(dist, bins='auto', density=True)
        
        if (type == 'linlog'):
            hist = np.log10(hist)                                                                   # y axis logarithmic
        elif (type == 'loglin'):
            bins = np.log10(bins)                                                                   # x axis logarithmic
        elif (type == 'loglog'):
            bins = np.log10(bins)                                                                   # x axis logarithmic and
            hist = np.log10(hist)                                                                   # y axis logarithmic   
        
        if use_labels:
            if step:
                ax.step(bins[:-1], hist, label=labels[i])
            else:
                ax.plot(bins[:-1], hist, label=labels[i])
        else:
            if step:
                ax.step(bins[:-1], hist)
            else:
                ax.plot(bins[:-1], hist)
    
    fig, ax = plt.subplots(figsize=[7.0, 5.5])
    
    if (dim == 1):
        plotfunc(dist)
    else:
        for i, dist_i in enumerate(dist):
            plotfunc(dist_i)
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if use_labels: plt.legend()
    plt.tight_layout()
    plt.show() 
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    