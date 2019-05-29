# Luc IJspeert
# Part of starpopsim: Visualizer
##
"""Visualize data by making images/plots in 2D or 3D. 
Just for convenience really. And nice plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import conversions as conv


def Objects2D(objects, title='Scatter', xlabel='x', ylabel='y',
              axes='xy', colour='blue', T_eff=None, mag=None, dark_theme=1):
    """Plots a 2D scatter of a 3D object (array) along given axes [xy, yz, zx].
    colour can be set to 'temperature' to represent the effective temperatures.
    magnitudes can be given to mag to make the marker sizes represent magnitudes.
    set dark_theme to 1 for a fancy dark plot, 2 for a less fancy dark saveable plot 
        and 0 for normal light colours.
    """
    # determine which axes to put on the horizontal and vertical axis of the plot
    axes_list = np.array([[0, 1], [1, 2], [2, 0]])
    hor_axis, vert_axis = axes_list[np.array(['xy', 'yz', 'zx']) == axes][0]
    
    # colours can be made to match the temperature (uses T_eff)
    if (colour == 'temperature'):
        colour = conv.TemperatureToRGB(T_eff).transpose()                                           # T_eff array of temps of the objects
        colour[T_eff == 10] = [0.2, 0.2, 0.2]                                                       # dead stars
    elif (colour == 'blue'):
        colour = 'tab:blue'
    
    if mag is not None:
        m_max = np.max(mag)
        sizes = 30*(0.0 + (m_max - mag)/5.0)**2                                                     # formula for representation of magnitude
        s_max = np.max(sizes)
        print(sizes)
        colour = np.array([[0,0,1] for i in range(len(sizes))])
        print(colour)
        colour = np.append(colour, sizes/s_max, axis=0)
        print(colour)
    else:
        sizes = [20 for i in range(len(objects[:,hor_axis]))]                                       # default size
    
    fig, ax = plt.subplots()
    ax.scatter(objects[:,hor_axis], objects[:,vert_axis], 
               marker='.', linewidths=0.0, c=colour, s=sizes)
    
    # take the maximum distance from the origin as axis scale
    if (np.shape(objects[0]) == (2,)):
        axis_size = max(max(objects[:,0]), max(objects[:,1]))                                       # two dimensional data
    else:
        axis_size = max(max(objects[:,0]), max(objects[:,1]), max(objects[:,2]))
    ax.set_xlim(-axis_size, axis_size) 
    ax.set_ylim(-axis_size, axis_size)
    ax.set(aspect='equal', adjustable='datalim')
    
    # The further setup (custom dark theme)
    if (dark_theme == 1):
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
    elif (dark_theme == 2):
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
    # todo: dark theme 2 needs some work (no difference yet)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show() 
    return


def Objects3D(objects, title='Scatter', xlabel='x', ylabel='y', zlabel='z', 
              colour='blue', T_eff=None, mag=None, dark_theme=1):
    """Plots a 3D scatter of a 3D object (array).
    colour can be set to 'temperature' to represent the effective temperatures.
    magnitudes can be given to mag to make the marker sizes represent magnitudes.
    set dark_theme to 1 for a fancy dark plot, 2 for a less fancy dark saveable plot 
        and 0 for normal light colours.
    """
    # colours can be made to match the temperature (uses T_eff)
    if (colour == 'temperature'):
        colour = conv.TemperatureToRGB(T_eff).transpose()                                           # T_eff array of temps of the objects
        colour[T_eff == 10] = [0.2, 0.2, 0.2]                                                       # dead stars
    elif (colour == 'blue'):
        colour = 'tab:blue'
    
    if mag is not None:
        sizes = 30*(0.5 + (7.0-mag)/3.0)**2                                                         # formula for representation of magnitude
    else:
        sizes = [20 for i in range(len(objects[:,hor_axis]))]                                       # default size
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.scatter(objects[:,0], objects[:,1], objects[:,2], marker='.', linewidths=0.0, c=colour) 
    
    # take the maximum distance from the origin as axis scale
    axis_size = max(max(objects[:,0]), max(objects[:,1]), max(objects[:,2]))
    ax.set_xlim3d(-axis_size, axis_size) 
    ax.set_ylim3d(-axis_size, axis_size)
    ax.set_zlim3d(-axis_size, axis_size)
    ax.set(aspect='equal', adjustable='box')
    
    # The further setup (custom dark theme)
    if (dark_theme == 1):
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
    elif (dark_theme == 2):
        c_grey = (0.6, 0.6, 0.6)
        c_grey2 = (0.4, 0.4, 0.4)
        c_black = (0, 0, 0, 0)
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
    # todo: dark theme 2 needs some work (no difference yet)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show() 
    return


def HRD(T_eff, log_Lum, title='HRD', xlabel='Temperature (K)', 
        ylabel=r'Luminosity log($L/L_\odot$)', colour='temperature', dark_theme=0, mask=None):
    """Plot the Herzsprung Russell Diagram. Use mask to select certain stars.
    colours can be made to match the temperature (default behaviour)
    set dark_theme to 1 for a fancy dark plot, 2 for a less fancy dark saveable plot 
        and 0 for normal light colours.
    """
    if (colour == 'temperature'):
        colour = conv.TemperatureToRGB(T_eff).transpose()
    elif (colour == 'blue'):
        colour = 'tab:blue'
        
    if mask is None:
        mask = [True for i in range(len(T_eff))]
    
    fig, ax = plt.subplots()
    ax.scatter(T_eff[mask], log_Lum[mask], marker='.', linewidths=0.0, c=colour)

    if (dark_theme == 1):
        c_light = '0.9'
        c_grey = '0.7'
        c_dark1 = '0.22'
        c_dark2 = '0.15'
        fig.patch.set_color(c_dark1)
        ax.set_facecolor(c_dark2)
        ax.spines['bottom'].set_color(c_grey)
        ax.spines['top'].set_color(c_grey) 
        ax.spines['right'].set_color(c_grey)
        ax.spines['left'].set_color(c_grey)
        ax.tick_params(axis='x', colors=c_grey)
        ax.tick_params(axis='y', colors=c_grey)
        ax.title.set_color(c_light)
        ax.xaxis.label.set_color(c_light)
        ax.yaxis.label.set_color(c_light)
    elif (dark_theme == 2):
        c_light = '0.9'
        c_grey = '0.7'
        c_dark1 = '0.22'
        c_dark2 = '0.15'
        ax.set_facecolor(c_dark2)
        ax.spines['bottom'].set_color(c_dark1)
        ax.spines['top'].set_color(c_dark1) 
        ax.spines['right'].set_color(c_dark1)
        ax.spines['left'].set_color(c_dark1)
        ax.tick_params(axis='x', colors=c_dark1)
        ax.tick_params(axis='y', colors=c_dark1)
        ax.title.set_color(c_dark1)
        ax.xaxis.label.set_color(c_dark1)
        ax.yaxis.label.set_color(c_dark1)
    # todo: dark theme 2 needs some work
    
    ax.set_xlim(40000, 500) 
    ax.set_ylim(-5, 7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show() 
    return


def CMD(c_mag, mag, title='CMD', xlabel='colour', ylabel='magnitude', 
        colour='blue', T_eff=None, dark_theme=0, adapt_axes=True, mask=None):
    """Plot the Colour Magnitude Diagram. Use mask to select certain stars.
    colours can be made to match the temperature (default behaviour)
    set dark_theme to 1 for a fancy dark plot, 2 for a less fancy dark saveable plot 
        and 0 for normal light colours.
    """
    if (colour == 'temperature'):
        colour = conv.TemperatureToRGB(T_eff).transpose()                                           # T_eff array of temps of the objects
    elif (colour == 'blue'):
        colour = 'tab:blue'
        
    if mask is None:
        mask = [True for i in range(len(c_mag))]
    
    fig, ax = plt.subplots()
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
    
    if (dark_theme == 1):
        c_light = '0.9'
        c_grey = '0.7'
        c_dark1 = '0.22'
        c_dark2 = '0.15'
        fig.patch.set_color(c_dark1)
        ax.set_facecolor(c_dark2)
        ax.spines['bottom'].set_color(c_grey)
        ax.spines['top'].set_color(c_grey) 
        ax.spines['right'].set_color(c_grey)
        ax.spines['left'].set_color(c_grey)
        ax.tick_params(axis='x', colors=c_grey)
        ax.tick_params(axis='y', colors=c_grey)
        ax.title.set_color(c_light)
        ax.xaxis.label.set_color(c_light)
        ax.yaxis.label.set_color(c_light)
    elif (dark_theme == 2):
        c_light = '0.9'
        c_grey = '0.7'
        c_dark1 = '0.22'
        c_dark2 = '0.15'
        ax.set_facecolor(c_dark2)
        ax.spines['bottom'].set_color(c_dark1)
        ax.spines['top'].set_color(c_dark1) 
        ax.spines['right'].set_color(c_dark1)
        ax.spines['left'].set_color(c_dark1)
        ax.tick_params(axis='x', colors=c_dark1)
        ax.tick_params(axis='y', colors=c_dark1)
        ax.title.set_color(c_dark1)
        ax.xaxis.label.set_color(c_dark1)
        ax.yaxis.label.set_color(c_dark1)
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
    
    fig, ax = plt.subplots()
    
    if (dim == 1):
        plotfunc(dist)
    else:
        for i, dist_i in enumerate(dist):
            plotfunc(dist_i)
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if use_labels: plt.legend()
    plt.show() 
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    