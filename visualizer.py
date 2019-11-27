"""Visualize data by making images/plots in 2D or 3D.
Just for convenience really. And nice plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import conversions as conv


def scatter_2d(coords, title='Scatter', xlabel='x', ylabel='y', axes='xy', colour='blue', T_eff=None, mag=None,
               theme=None, show=True):
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

    # colours can be made to match the temperature (using T_eff)
    if (T_eff is not None):
        colour = conv.temperature_to_rgb(T_eff).transpose()
        # make dead stars grey and add alpha
        colour[T_eff <= 10] = [0.2, 0.2, 0.2]
        colour = mcol.to_rgba_array(colour, 0.5)
    elif (theme == 'fits'):
        # set the colours to white
        colour = np.array([mcol.to_rgba((1, 1, 1), 0.2)])
    elif (colour == 'blue'):
        colour = np.array([mcol.to_rgba('tab:blue', 0.5)])
    else:
        colour = np.array([mcol.to_rgba(colour, 0.5)])
    
    # marker sizes and transparancy scaling with mag
    if (mag is not None):
        m_max = np.max(mag)
        # formula for representation of magnitude
        sizes = (0.3 + (m_max - mag)/1.5)**2
        if (theme == 'fits'):
            # scale alpha with mag
            alpha = (m_max - mag)**3
            alpha = alpha/np.max(alpha)
            colour = np.repeat(colour, len(alpha), axis=0)
            colour[:, 3] = alpha
    else:
        sizes = 20  # default size
    
    fig, ax = plt.subplots(figsize=[7.0, 5.5])
    ax.scatter(coords[hor_axis], coords[vert_axis], marker='.', linewidths=0.0, c=colour, s=sizes)
    
    # take the maximum coordinate distance_3d as axis sizes
    axis_size = np.max(np.abs(coords))
    ax.set_xlim(-axis_size, axis_size) 
    ax.set_ylim(-axis_size, axis_size)
    ax.set(aspect='equal', adjustable='datalim')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # The further setup (custom dark theme)
    if (theme == 'dark1'):
        # fancy dark theme
        c_1 = '0.7'
        c_2 = '0.5'
        c_3 = '0.0'
        c_4 = '0.0'
        s_1 = 12
        s_2 = 12
    elif (theme == 'dark2'):
        # dark theme for good saving
        c_1 = '0.2'
        c_2 = '0.3'
        c_3 = '1.0'
        c_4 = '0.0'
        s_1 = 12
        s_2 = 12
    elif (theme == 'fits'):
        # theme for imitating .fits images
        c_1 = '0.6'
        c_2 = '0.4'
        c_3 = '0.0'
        c_4 = '0.0'
        s_1 = 12
        s_2 = 12
    else:
        # defaults (not actually used)
        c_1 = '0.0'                                                                                 # text
        c_2 = '0.0'                                                                                 # lines
        c_3 = '1.0'                                                                                 # outer rim
        c_4 = '1.0'                                                                                 # inner area
        s_1 = 12                                                                                    # title/labels
        s_2 = 12                                                                                    # tick params
    
    if theme:
        # set the colours
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
        # invert y for (0,0) at top left
        if (theme == 'fits'):
            ax.invert_yaxis()
    
    # set the text sizes
    ax.title.set_size(s_1)
    ax.xaxis.label.set_size(s_1)
    ax.yaxis.label.set_size(s_1)
    ax.tick_params(labelsize=s_2)
    
    plt.tight_layout()
    if show:
        plt.show()
    return


def scatter_3d(coords, title='Scatter', xlabel='x', ylabel='y', zlabel='z', colour='blue', T_eff=None, mag=None,
               theme=None, show=True):
    """Plots a 3D scatter of a 3D object (array).
    colour can be set to 'temperature' to represent the effective temperatures.
    magnitudes can be given to mag to make the marker sizes represent magnitudes.
    Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but 
        save-able dark plot, and None for normal light colours.
    """
    # colours can be made to match the temperature (using T_eff)
    if (T_eff is not None):
        colour = conv.temperature_to_rgb(T_eff).transpose()
        # make dead stars grey and add alpha
        colour[T_eff <= 10] = [0.2, 0.2, 0.2]
        colour = mcol.to_rgba_array(colour, 0.5)
    elif (theme == 'fits'):
        # set colours to white
        colour = np.array([mcol.to_rgba((1, 1, 1), 0.2)])
    elif (colour == 'blue'):
        colour = np.array([mcol.to_rgba('tab:blue', 0.5)])
    else:
        colour = np.array([mcol.to_rgba(colour, 0.5)])
    
    # marker sizes and transparancy scaling with mag
    if (mag is not None):
        m_max = np.max(mag)
        # formula for representation of magnitude
        sizes = (0.3 + (m_max - mag)/1.5)**2
        if (theme == 'fits'):
            # scale alpha with mag
            alpha = (m_max - mag)**3
            alpha = alpha/np.max(alpha)
            colour = np.repeat(colour, len(alpha), axis=0)
            colour[:, 3] = alpha
    else:
        sizes = 20  # default size
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=[6.0, 6.0])
    ax.scatter(coords[0], coords[1], coords[2], marker='.', linewidths=0.0, c=colour, s=sizes)
    
    # take the maximum coordinate distance_3d as axis sizes
    axis_size = np.max(np.abs(coords))
    ax.set_xlim3d(-axis_size, axis_size) 
    ax.set_ylim3d(-axis_size, axis_size)
    ax.set_zlim3d(-axis_size, axis_size)
    ax.set(aspect='equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    # The further setup (custom dark theme)
    if (theme == 'dark1'):
        # fancy dark theme
        c_1 = (0.7, 0.7, 0.7)
        c_2 = (0.5, 0.5, 0.5)
        c_3 = (0.0, 0.0, 0.0)
        c_4 = (0.0, 0.0, 0.0)
        s_1 = 12
        s_2 = 12
    elif (theme == 'dark2'):
        # dark theme for good saving
        c_1 = (0.7, 0.7, 0.7)
        c_2 = (0.5, 0.5, 0.5)
        c_3 = (1.0, 1.0, 1.0)
        c_4 = (0.0, 0.0, 0.0)
        s_1 = 12
        s_2 = 12
    elif (theme == 'fits'):
        # theme for imitating .fits images
        c_1 = (0.6, 0.6, 0.6)
        c_2 = (0.4, 0.4, 0.4)
        c_3 = (0.0, 0.0, 0.0)
        c_4 = (0.0, 0.0, 0.0)
        s_1 = 12
        s_2 = 12
    else:
        # defaults (not actually used)
        c_1 = (0.0, 0.0, 0.0)  # text
        c_2 = (0.0, 0.0, 0.0)  # lines
        c_3 = (1.0, 1.0, 1.0)  # outer rim
        c_4 = (1.0, 1.0, 1.0)  # inner area
        s_1 = 12  # title/labels
        s_2 = 12  # tick params
    
    if (theme is not None):
        # set the colours
        fig.patch.set_color(c_3)
        ax.set_facecolor(c_4)
        ax.w_xaxis.set_pane_color(c_4)     
        ax.w_yaxis.set_pane_color(c_4)
        ax.w_zaxis.set_pane_color(c_4)
        ax.tick_params(axis='x', colors=c_2)
        ax.tick_params(axis='y', colors=c_2)
        ax.tick_params(axis='z', colors=c_2)
        ax.title.set_color(c_1)
        ax.xaxis.label.set_color(c_1)
        ax.yaxis.label.set_color(c_1)
        ax.zaxis.label.set_color(c_1)
    
    # set the textsizes
    ax.title.set_size(s_1)
    ax.xaxis.label.set_size(s_1)
    ax.yaxis.label.set_size(s_1)
    ax.tick_params(labelsize=s_2)
    
    plt.tight_layout()
    if show:
        plt.show()
    return


def hr_diagram(T_eff, log_lum, title='hr_diagram', xlabel='Temperature (K)', ylabel=r'Luminosity log($L/L_\odot$)',
               colour='temperature', theme=None, mask=None, show=True):
    """Plot the Herzsprung Russell Diagram. Use mask to select certain stars.
    colours can be made to match the temperature (default behaviour)
    Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but 
        saveable dark plot, and None for normal light colours.
    """
    # the mask can be used to hide remnants as the screw with the plot area   
    if mask is None:
        mask = np.ones_like(T_eff, dtype=bool)
    
    # colours can be made to match the temperature (using T_eff)
    if ((T_eff is not None) & (colour == 'temperature')):
        colour = conv.temperature_to_rgb(T_eff).transpose()
        # make dead stars grey and add alpha
        colour[T_eff <= 10] = [0.2, 0.2, 0.2]
        colour = mcol.to_rgba_array(colour, 0.5)
        colour = colour[mask]
    elif (colour == 'blue'):
        colour = mcol.to_rgba('tab:blue', 0.5)
    else:
        colour = mcol.to_rgba(colour, 0.5)
    
    fig, ax = plt.subplots(figsize=[7.0, 5.5])
    ax.scatter(T_eff[mask], log_lum[mask], c=colour, marker='.', linewidths=0.0)
    ax.set_xlim(40000, 500) 
    ax.set_ylim(-5, 7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if (theme == 'dark1'):
        # fancy dark theme
        c_1 = '0.9'
        c_2 = '0.7'
        c_3 = '0.22'
        c_4 = '0.15'
        s_1 = 12
        s_2 = 12
    elif (theme == 'dark2'):
        # dark theme for good saving
        c_1 = '0.10'
        c_2 = '0.16'
        c_3 = '1.0'
        c_4 = '0.25'
        s_1 = 12
        s_2 = 12
    else:
        # defaults (not actually used)
        c_1 = '0.0'  # text
        c_2 = '0.0'  # lines
        c_3 = '1.0'  # outer rim
        c_4 = '1.0'  # inner area
        s_1 = 12  # title/labels
        s_2 = 12  # tick params
    
    if (theme is not None):
        # set the colours
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
    
    # set the textsizes
    ax.title.set_size(s_1)
    ax.xaxis.label.set_size(s_1)
    ax.yaxis.label.set_size(s_1)
    ax.tick_params(labelsize=s_2)
    
    plt.tight_layout()
    if show:
        plt.show()
    return


def cm_diagram(c_mag, mag, title='cm_diagram', xlabel='colour', ylabel='magnitude', colour='blue', T_eff=None,
               theme=None, adapt_axes=True, mask=None, show=True):
    """Plot the Colour Magnitude Diagram. Use mask to select certain stars.
    colours can be made to match the temperature (default behaviour)
    Set theme to 'dark1' for a fancy dark plot, 'dark2' for a less fancy but 
        saveable dark plot, and None for normal light colours.
    """
    # colours can be made to match the temperature (using T_eff)
    if (T_eff is not None):
        colour = conv.temperature_to_rgb(T_eff).transpose()
        # make dead stars grey and add alpha
        colour[T_eff <= 10] = [0.2, 0.2, 0.2]
        colour = mcol.to_rgba_array(colour, 0.5)
        colour = colour[mask]
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
        # good for seeing vertical differences
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(17.25, -12.75)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if (theme == 'dark1'):
        # fancy dark theme
        c_1 = '0.9'
        c_2 = '0.7'
        c_3 = '0.22'
        c_4 = '0.15'
        s_1 = 12
        s_2 = 12
    elif (theme == 'dark2'):
        # dark theme for good saving
        c_1 = '0.10'
        c_2 = '0.16'
        c_3 = '1.0'
        c_4 = '0.25'
        s_1 = 12
        s_2 = 12
    else:
        # defaults (not actually used)
        c_1 = '0.0'  # text
        c_2 = '0.0'  # lines
        c_3 = '1.0'  # outer rim
        c_4 = '1.0'  # inner area
        s_1 = 12  # title/labels
        s_2 = 12  # tick params
    
    if (theme is not None):
        # set the colours
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
    
    # set the textsizes
    ax.title.set_size(s_1)
    ax.xaxis.label.set_size(s_1)
    ax.yaxis.label.set_size(s_1)
    ax.tick_params(labelsize=s_2)
        
    plt.tight_layout()
    if show:
        plt.show()
    return


def dist_histogram(dist, title='Histogram', xlabel='parameter', ylabel='relative number', step=True, type='linear',
                   labels=None, show=True):
    """Display the histogram for some distribution. Can handle multiple distributions.
    type can be linear, linlog, loglin (x,y) and loglog. Step=False will give a plot instead.
    """
    # see if multiple dists given
    dim = len(np.shape(dist))
    if (dim != 1):
        num_dists = np.shape(dist)[0]
        if not labels:
            labels = []

        if (len(labels) == num_dists):
            use_labels = True
        else:
            use_labels = False
    else:
        use_labels = False
    
    def plotfunc(dist):
        hist, bins = np.histogram(dist, bins='auto', density=True)
        
        if (type == 'linlog'):
            hist = np.log10(hist)  # y axis logarithmic
        elif (type == 'loglin'):
            bins = np.log10(bins)  # x axis logarithmic
        elif (type == 'loglog'):
            bins = np.log10(bins)  # x axis logarithmic and
            hist = np.log10(hist)  # y axis logarithmic
        
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
    if use_labels:
        plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
