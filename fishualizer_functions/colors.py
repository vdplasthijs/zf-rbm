import os
import numpy as np
from zipfile import ZipFile
from PyQt5 import QtCore, QtGui


COLORMAPS_PATH = 'Content/colormaps.zip'


def interp_values(values, stops, n_colors=256):
    """
    Linear interpolation of a given vector of values, taken at given stop points in [0, 1]
    """

    ip_vals = np.zeros(n_colors)
    i_stops = np.intp(stops * (n_colors - 1))
    ix = np.arange(1, n_colors - 1)
    ind_stops = np.searchsorted(i_stops, ix)
    slope = (values[ind_stops] - values[ind_stops - 1]) / (i_stops[ind_stops] - i_stops[ind_stops - 1])
    ip_vals[0] = values[0]
    ip_vals[-1] = values[-1]
    ip_vals[1:-1] = slope * (ix - i_stops[ind_stops - 1]) + values[
        ind_stops - 1]  # dx has to be computed from previous stop

    return ip_vals


def interp_colors(stops, colors, n_colors=256):
    """
    """
    # Everyhing in numpy arrays
    stops = np.array(stops)
    colors = np.array(colors)
    # Ensure the stops are sorted
    stop_order = np.argsort(stops)
    stops = stops[stop_order]
    colors = colors[stop_order, :]

    if stops[0] != 0:
        raise ValueError('First stop should be 0')
    if stops[-1] != 1:
        raise ValueError('Last stop should be 1')
    ip_colors = np.apply_along_axis(interp_values, 0, colors, stops)

    return ip_colors


def normalize(x):
    """
    Normalize a vector to the 0-1 range
    """
    xmin = np.min(x)
    xmax = np.max(x)

    xn = (x - xmin)
    d = xmax - xmin
    if d > 0:
        xn = xn / d
    return xn


def log_normalize(x):
    """
    Log-normalize a vector to the 0-1 range
    """
    xmin = np.min(x)
    xmax = np.max(x)

    xn = np.log(x - xmin + 1)  # Puts the minimum to 1 before taking the log
    xn = xn / np.log(xmax - xmin + 1) # Puts the maximum to 1
    return xn


def linear_cmap(colors, normalizer=normalize):
    """
    Parameters
    ----------

    Return
    ------

    """
    colors = np.array(colors)
    n_colors = colors.shape[0]

    def user_cmap(x, vmin=None, vmax=None):
        """
        x: array of floats
            Data to be color coded
        vmin: float or None
        vmax: float or None
        """

        if vmin is None:
            vmin = np.min(x)
        if vmax is None:
            vmax = np.max(x)
        assert vmin <= vmax, f'vmin should be less than vmax. Currently they are {vmin} and {vmax}'
        # Clip out of range values
        xc = np.clip(x, vmin, vmax)
        # Normalize vmin and vmax between 0 and 1, then bring between 0 and n_colors-1
        xcn = (xc - vmin)
        d = vmax - vmin
        if d > 0:
            xcn = xcn / d
        xcn = xcn * (n_colors - 1)
        # xcn = normalizer(xc) * (n_colors - 1)
        # Convert to int for indexing
        xcn = xcn.astype(int)
        # Select colors according to the data
        x_colors = colors.take(xcn, axis=0)

        return x_colors

    return user_cmap


def get_cmap(cmap):
    """
    Get the colors from a color map whose name is given

    Parameters
    ----------
    cmap: str
     Name (case insensitive) of the color map to load

    Returns
    -------
    colors: Numpy array or None
        Colors or None if colormap is not found

    """
    cmap = cmap.lower()
    with ZipFile(COLORMAPS_PATH, 'r') as cmapfile:
        for f_name in cmapfile.namelist():
            if cmap in f_name.lower():
                e_path = cmapfile.extract(f_name)
                colors = np.loadtxt(e_path, delimiter=',', skiprows=1)
                os.remove(e_path)
                return colors
    return None


def get_all_cmaps():
    """
    Return the names of all color maps available

    Returns
    -------

    """
    with ZipFile(COLORMAPS_PATH, 'r') as cmapfile:
        cmaps = cmapfile.namelist()
        cmaps = [os.path.splitext(n)[0] for n in cmaps]

    return cmaps


def to_qcolor(color):
    ucolor = np.ubyte(color*255)
    return QtGui.QColor(*ucolor)


def get_qt_gradient(colors, p1, p2):
    """
    Creates a QtLinearGradient spanning a line defined by two points p1 and p2
    Assumes regularly spaced colors arranged in a n_colors * n_channels numpy array
    Most of this code is adapted from pyqgraph Colormap.getGradient

    Parameters
    ----------
    colors: Numpy array
    p1: tuple
        coordinates
    p2: tuple
        coordinates

    Returns
    -------

    """
    p1 = QtCore.QPointF(*p1)
    p2 = QtCore.QPointF(*p2)
    stops = np.linspace(0, 1, colors.shape[0])
    qcolors = [to_qcolor(c) for c in colors]
    """Return a QLinearGradient object spanning from QPoints p1 to p2."""
    g = QtGui.QLinearGradient(p1, p2)
    g.setStops(zip(stops, qcolors))

    return g
