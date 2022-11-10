#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

from collections.abc import Iterable
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seislib.colormaps as scm



def add_inset_axis(axis, rect, polar=False):
    """ Adds an axis inset
    
    Parameters
    ----------
    axis : matplotlib AxesSubplot
        Instance of `matplotlib.pyplot.subplot`
        
    rect : list or tuple of shape (4,)
        [bottom, left, width, height] of the inset. Bottom and left 
        coordinates are expressed with respect to `axis`
        
    polar : bool
        If `True`, a polar plot is created. Default is `False`
        
    
    Returns
    -------
    Inset axis
    """
    def axis_to_fig(axis):
        fig = axis.figure
        def transform(coord):
            return fig.transFigure.inverted().transform(
                axis.transAxes.transform(coord))
        return transform
    
    fig = axis.figure
    left, bottom, width, height = rect
    trans = axis_to_fig(axis)
    figleft, figbottom = trans((left, bottom))
    figwidth, figheight = trans([width,height]) - trans([0,0])
    return fig.add_axes([figleft, figbottom, figwidth, figheight], polar=polar)


def lower_threshold_projection(projection, thresh=1e3, **kwargs):
    """ 
    Effective work around to get a higher-resolution curvature of the 
    great-circle paths to be plotted in a given projection. This is useful 
    when plotting the great-circle paths in a relatively small region.
    
    Parameters
    ----------
    projection : class
        Should be one of the cartopy projection classes, e.g., 
        `cartopy.crs.Mercator`
        
    thresh : float
        Smaller values achieve higher resolutions. Default is 1e3
        
    Returns
    ------- 
    Instance of the input (`projection`) class
    
    
    Examples
    --------
    >>> proj = lower_threshold_projection(cartopy.crs.Mercator, 
    ...                                   thresh=1e3)
    
    Note that the `cartopy.crs.Mercator` was not initialized (i.e., 
    there are no brackets after the word `Mercator`)
    """
    class LowerThresholdProjection(projection):
        
        @property
        def threshold(self):
            return thresh
        
    return LowerThresholdProjection(**kwargs)

    
def add_earth_features_GSHHS(ax, scale='auto', oceans_color='aqua', 
                             lands_color='coral', edgecolor='face'):
    """ 
    Adds natural features to a `cartopy.mpl.geoaxes.GeoAxesSubplot`, fetching data
    from the `GSHHS dataset <https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html>`_
    
    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
    
    scale : {'auto', 'coarse', 'low', 'intermediate', 'high', 'full'}
        Scale of the dataset, passed to `cartopy.feature.GSHHSFeature`. 
        Default is 'auto'
        
    oceans_color, lands_color : str
        Color of oceans and continents in the map. They should be valid 
        `matplotlib` colors or be part of `cartopy.cfeature.COLORS`. Defaults 
        are 'aqua' and 'coral'
        
    edgecolor : str
        Color of the boundaries between, e.g., lakes and land. Passed to 
        `cartopy.feature.GSHHSFeature`. Default is 'face' (the boundaries will
        have the same color as the continent)
    """
    if oceans_color == 'water':
        oceans_color = cfeature.COLORS[oceans_color]
    if lands_color == 'land':
        lands_color = cfeature.COLORS[lands_color]
    colors = [lands_color, oceans_color] * 2
    ax.background_patch.set_facecolor(oceans_color)
    for level, color in zip(range(1, 5), colors):
        feature = cfeature.GSHHSFeature(levels=[level], 
                                        scale=scale,
                                        edgecolor=edgecolor, 
                                        facecolor=color)
        ax.add_feature(feature)



def add_earth_features(ax, scale='110m', oceans_color='aqua', 
                       lands_color='coral', edgecolor='k', lands_lw=0.5,
                       oceans_lw=0.5, lakes_lw=0.5):
    """ 
    Adds natural features to a `cartopy.mpl.geoaxes.GeoAxesSubplot`, fetching
    data from the `Natural Earth dataset <http://www.naturalearthdata.com/>`_
    
    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
    
    scale : {'10m', '50m', '110m'}
        Resolution of the Earth features displayed in the figure. Passed to
        `cartopy.feature.NaturalEarthFeature`. Default is '110m'
        
    oceans_color, lands_color : str
        Color of oceans and continents in the map. They should be valid 
        `matplotlib` colors or be part of `cartopy.cfeature.COLORS`. Defaults 
        are 'aqua' and 'coral'
        
    edgecolor : str
        Color of the boundaries between, e.g., lakes and land. Passed to 
        `cartopy.feature.NaturalEarthFeature`. Default is 'k' (black)
        
    lands_lw, oceans_lw, lakes_lw : float
        Linewidths for lands, oceans, and lakes. Arguments passed to
        `GeoAxesSubplot.add_feature`
    """
    if oceans_color == 'water':
        oceans_color = cfeature.COLORS[oceans_color]
    if lands_color == 'land':
        lands_color = cfeature.COLORS[lands_color]
    land = cfeature.NaturalEarthFeature('physical', 
                                        'land', 
                                        scale=scale, 
                                        edgecolor=edgecolor, 
                                        facecolor=lands_color)
    ocean = cfeature.NaturalEarthFeature('physical', 
                                         'ocean',
                                         scale=scale, 
                                         edgecolor=edgecolor, 
                                         facecolor=oceans_color)
    lakes = cfeature.NaturalEarthFeature('physical', 
                                         'lakes',
                                         scale=scale, 
                                         edgecolor=edgecolor, 
                                         facecolor=oceans_color)
    ax.add_feature(land, linewidth=lands_lw)
    ax.add_feature(ocean, linewidth=oceans_lw)
    ax.add_feature(lakes, linewidth=lakes_lw)
    

def make_colorbar(ax, mappable, size='5%', pad='3%', **kwargs):
    """ Prepares and attaches a colorbar to the GeoAxesSubplot
    
    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
    
    mappable : matplotlib.cm.ScalarMappable
    
    size : str
        Width of the colorbar, default is '5%'
    
    pad : str
        Space between the colorbar and ax, default is '3%'
    
    **kwargs
        Additional keyword arguments passed to
        `matplotlib.pyplot.colorbar`
    
    
    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    divider = make_axes_locatable(ax)
    orientation = kwargs.pop('orientation', 'horizontal')
    if orientation == 'vertical':
        loc = 'right'
    elif orientation == 'horizontal':
        loc = 'bottom'  
    cax = divider.append_axes(loc, size, pad=pad, axes_class=mpl.pyplot.Axes)
    cb = ax.get_figure().colorbar(mappable, 
                                  cax=cax, 
                                  orientation=orientation,
                                  **kwargs)
    return cb
    

def colormesh(mesh, c, ax, **kwargs):
    """
    Adaptation of `matplotlib.pyplot.pcolormesh` to the (adaptive) equal-area 
    grid
    
    Parameters
    ----------
    mesh : ndarray (n, 4)
        Equal area grid, consisting of `n` blocks described by lat1, lat2, 
        lon1, lon2
        
    c : list of ndarray (n,)
        Values to plot in each grid cell
        
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Earth image on which `c` will be plotted
        
    **kwargs
        Additional keyword arguments passed to `ax.pcolormesh`
    
    
    Returns
    -------
        img : Instance of `matplotlib.collections.QuadMesh`
    """
    def plot_band(lat1, lat2, lons, v, **kwargs):
        lat_mesh, lon_mesh = np.meshgrid([lat1, lat2], lons)
        img = ax.pcolormesh(lon_mesh, lat_mesh, np.reshape(v, (-1, 1)), 
                            transform=ccrs.PlateCarree(), **kwargs)
        return img
    
    def update_kwargs(c, **kwargs):
        updated = {i: j for i, j in kwargs.items()}
        norm = updated.get('norm')
        if norm is None:
            if updated.get('vmin') is None:
                updated['vmin'] = np.nanmin(c)
            if updated.get('vmax') is None:
                updated['vmax'] = np.nanmax(c)
        else:
            if norm.vmin is None:
                norm.vmin = updated.pop('vmin', np.nanmin(c))
            if norm.vmax is None:
                norm.vmax = updated.pop('vmax', np.nanmax(c))
        return updated

    kwargs = update_kwargs(c, **kwargs)
    
    lat1, lat2 = mesh[0, :2]
    lons = [mesh[0, 2]]
    v = []
    for i, (newlat1, newlat2, lon1, lon2) in enumerate(mesh):
        if newlat1!=lat1 or newlat2!=lat2 or lon1!=lons[-1]:
            img = plot_band(lat1, lat2, lons, v, **kwargs)
            lat1, lat2 = newlat1, newlat2
            lons = [lon1, lon2]
            v = [c[i]]
        else:
            lons.append(lon2)
            v.append(c[i])
    if v:
        img = plot_band(lat1, lat2, lons, v, **kwargs)
    return img


def contourf(mesh, c, ax, smoothing=None, **kwargs):
    """
    Adaptation of matplotlib.pyplot.contourf to the (adaptive) equal area 
    grid.
    
    Parameters
    ----------
    mesh : ndarray (n, 4)
        Equal area grid, consisting of n pixels described by lat1, lat2, 
        lon1, lon2
        
    c : list of ndarray (n,)
        Values to plot in each grid cell
        
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Earth image on which `c` will be plotted
        
    smoothing : float, optional
        Value passed to `scipy.ndimage.filters.gaussian_filter` and used to
        obtain a smooth representation of `c` (default is `None`)
        
    **kwargs
        Additional keyword arguments passed to `ax.contourf`
    
    Returns
    -------
        img : Instance of `matplotlib.contour.QuadContourSet`
    """
    def update_kwargs(c, **kwargs):
        if kwargs.get('norm') is None:
            return kwargs
        
        updated = {i: j for i, j in kwargs.items()}
        norm = updated.get('norm')       
        if norm.vmin is None:
            norm.vmin = updated.pop('vmin', np.nanmin(c))
        if norm.vmax is None:
            norm.vmax = updated.pop('vmax', np.nanmax(c))
            
        if type(norm) == type(LogNorm()):
            levels = updated.get('levels', 50)
            lev_exp = np.linspace(np.log10(norm.vmin), 
                                  np.log10(norm.vmax), 
                                  levels)
            updated['levels'] = np.power(10, lev_exp)
            
        return updated

    kwargs = update_kwargs(c, **kwargs)
    if smoothing is not None:
        c = gaussian_filter(c, sigma=smoothing)
    lats = np.mean(mesh[:, :2], axis=1)
    lons = np.mean(mesh[:, 2:], axis=1)
    img = ax.tricontourf(lons, lats, c, transform=ccrs.PlateCarree(), **kwargs)
    return img
     

def contour(mesh, c, ax, smoothing=None, **kwargs):
    """
    Adaptation of `matplotlib.pyplot.contour` to the (adaptive) equal area 
    grid
    
    Parameters
    ----------
    mesh : ndarray (n, 4)
        Equal area grid, consisting of n pixels described by lat1, lat2, 
        lon1, lon2
        
    c : list of ndarray (n,)
        Values to plot in each grid cell
        
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        Earth image on which `c` will be plotted
        
    smoothing : float, optional
        Value passed to `scipy.ndimage.filters.gaussian_filter` and used to
        obtain a smooth representation of `c` (default is `None`)
        
    **kwargs
        Additional keyword arguments passed to ax.contour
    
    
    Returns
    -------
        img : Instance of `matplotlib.contour.QuadContourSet`
    """
    
    def update_kwargs(c, **kwargs):
        if kwargs.get('norm') is None:
            return kwargs
        
        updated = {i: j for i, j in kwargs.items()}
        norm = updated.get('norm')       
        if norm.vmin is None:
            norm.vmin = updated.pop('vmin', np.nanmin(c))
        if norm.vmax is None:
            norm.vmax = updated.pop('vmax', np.nanmax(c))
            
        if type(norm) == type(LogNorm()):
            levels = updated.get('levels', 50)
            lev_exp = np.linspace(np.log10(norm.vmin), 
                                  np.log10(norm.vmax), 
                                  levels)
            updated['levels'] = np.power(10, lev_exp)
            
        return updated

    kwargs = update_kwargs(c, **kwargs)
    if smoothing is not None:
        c = gaussian_filter(c, sigma=smoothing)
    lats = np.mean(mesh[:, :2], axis=1)
    lons = np.mean(mesh[:, 2:], axis=1)
    img = ax.tricontour(lons, lats, c, transform=ccrs.PlateCarree(), **kwargs)
    return img


def plot_stations(stations, ax=None, show=True, oceans_color='water', 
                  lands_color='land', edgecolor='k', projection='Mercator',
                  resolution='110m', color_by_network=True, legend_dict={}, 
                  **kwargs):
    """ Creates a maps of seismic receivers
    
    Parameters
    ----------
    stations : dict
        Dictionary object containing stations information. This should 
        structured so that each key corresponds to a station code 
        ($network_code.$station_code) and each value is a tuple containing 
        latitude and longitude of the station. For example::
            
            { net1.sta1 : (lat1, lon1),
              net1.sta2 : (lat2, lon2),
              net2.sta3 : (lat3, lon3)
              }
    
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        If not `None`, the receivers are plotted on the `GeoAxesSubplot` instance. 
        Otherwise, a new figure and `GeoAxesSubplot` instance is created
        
    show : bool
        If `True`, the plot is shown. Otherwise, a `GeoAxesSubplot` instance is
        returned. Default is `True`
        
    oceans_color, lands_color : str
        Color of oceans and lands. The arguments are ignored if ax is not
        None. Otherwise, they are passed to `cartopy.feature.NaturalEarthFeature` 
        (to the argument 'facecolor'). Defaults are 'water' and 'land'
        
    edgecolor : str
        Color of the boundaries between, e.g., lakes and land. The argument 
        is ignored if ax is not None. Otherwise, it is passed to 
        cartopy.feature.NaturalEarthFeature (to the argument 'edgecolor'). 
        Default is 'k' (black)
        
    projection : str
        Name of the geographic projection used to create the `GeoAxesSubplot`.
        (Visit the `cartopy website 
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html>`_ 
        for a list of valid projection names.) If ax is not None, `projection` 
        is ignored. Default is 'Mercator'
    
    resolution : {'10m', '50m', '110m'}
        Resolution of the Earth features displayed in the figure. Passed to
        `cartopy.feature.NaturalEarthFeature`. Default is '110m'
    
    color_by_network : bool
        If `True`, each seismic network will have a different color in the
        resulting map, and a legend will be displayed. Otherwise, all
        stations will have the same color. Default is `True`
    
    legend_dict : dict
        Keyword arguments passed to `matplotlib.pyplot.legend`
        
    kwargs : 
        Additional keyword arguments passed to `matplotlib.pyplot.scatter`
        
        
    Returns
    -------
    If `show` is True
        None

    Otherwise 
        `ax`, i.e. the `GeoAxesSubplot`
    """
    def get_coords(stations, color_by_network):
        codes, coords = zip(*[(k, v) for k, v in stations.items() \
                              if '_' not in k])
        if not color_by_network:
            yield np.array(coords), None
        else:
            networks = sorted(set(code.split('.')[0] for code in codes))
            for net in networks:
                idx = [i for i, code in enumerate(codes) if code.startswith(net)]
                yield np.array(coords)[idx], net
    
    def get_map_boundaries(stations):
        coords = np.array([i for i in stations.values()])
        latmin, latmax = np.min(coords[:,0]), np.max(coords[:,0])
        lonmin, lonmax = np.min(coords[:,1]), np.max(coords[:,1])
        dlat = (latmax - latmin) * 0.03
        dlon = (lonmax - lonmin) * 0.03
        lonmin = lonmin-dlon if lonmin-dlon > -180 else lonmin
        lonmax = lonmax+dlon if lonmax+dlon < 180 else lonmax
        latmin = latmin-dlat if latmin-dlat > -90 else latmin
        latmax = latmax+dlat if latmax+dlat < 90 else latmax
        return (lonmin, lonmax, latmin, latmax)
    
    
    transform = ccrs.PlateCarree()
    if ax is None:
        projection = eval('ccrs.%s()'%projection)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        map_boundaries = get_map_boundaries(stations)
        add_earth_features(ax, 
                           scale=resolution,
                           oceans_color=oceans_color, 
                           edgecolor=edgecolor,
                           lands_color=lands_color)
        ax.set_extent(map_boundaries, transform)
    
    marker = kwargs.pop('marker', '^')
    zorder = kwargs.pop('zorder', 100)
    s = kwargs.pop('s', 100)
    for coords, net in get_coords(stations, color_by_network):
        label = kwargs.pop('label', net)
        ax.scatter(*coords.T[::-1], marker=marker, transform=transform, 
                   label=label, zorder=zorder, s=s, **kwargs) 
    
    if color_by_network:
        ax.legend(**legend_dict)
    if show:
        plt.show()
    else:
        return ax


def plot_events(lat, lon, mag=None, ax=None, show=True, oceans_color='water', 
                lands_color='land', edgecolor='k', projection='Mercator',
                resolution='110m', min_size=5, max_size=200, legend_markers=4, 
                legend_dict={}, **kwargs):
    """ Creates a map of epicenters
    
    Parameters
    ----------
    lat, lon : ndarray of shape (n,)
        Latitude and longitude of the epicenters
        
    mag : ndarray of shape(n,), optional
        If not given, the size of each on the map will be constant
    
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        If not `None`, the receivers are plotted on the `GeoAxesSubplot` 
        instance. Otherwise, a new figure and `GeoAxesSubplot` instance is 
        created
        
    show : bool
        If `True`, the plot is shown. Otherwise, a `GeoAxesSubplot` instance is
        returned. Default is `True`
        
    oceans_color, lands_color : str
        Color of oceans and lands. The arguments are ignored if ax is not
        None. Otherwise, they are passed to `cartopy.feature.NaturalEarthFeature` 
        (to the argument 'facecolor'). Defaults are 'water' and 'land'
        
    edgecolor : str
        Color of the boundaries between, e.g., lakes and land. The argument 
        is ignored if ax is not None. Otherwise, it is passed to 
        cartopy.feature.NaturalEarthFeature (to the argument 'edgecolor'). 
        Default is 'k' (black)
        
    projection : str
        Name of the geographic projection used to create the `GeoAxesSubplot`.
        (Visit the `cartopy website 
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html>`_ 
        for a list of valid projection names.) If ax is not None, `projection` 
        is ignored. Default is 'Mercator'
    
    resolution : {'10m', '50m', '110m'}
        Resolution of the Earth features displayed in the figure. Passed to
        `cartopy.feature.NaturalEarthFeature`. Default is '110m'
    
    min_size, max_size : float, optional
        Minimum and maximum size of the epicenters on the map. These are used
        to interpolate all magnitudes associated with each event, so as to
        scale them appropriately on the map. (The final "sizes" are passed to
        the argument `s` of `matplotlib.pyplot.scatter`.) If `mag` is `None`,
        these params are ignored
        
    legend_markers : int
        Number of markers displayed in the legend. Ignored if `s` (size of the
        markers in `matplotlib.pyplot.scatter`) is passed. Only considered if
        `mag` is not `None`
            
    legend_dict : dict
        Keyword arguments passed to `matplotlib.pyplot.legend`
        
    **kwargs 
        Additional keyword arguments passed to `matplotlib.pyplot.scatter`
        
        
    Returns
    -------
    If `show` is True
        None

    Otherwise 
        `ax`, i.e. the `GeoAxesSubplot`
    """
    def get_map_boundaries(lat, lon):
        latmin, latmax = np.min(lat), np.max(lat)
        lonmin, lonmax = np.min(lon), np.max(lon)
        dlat = (latmax - latmin) * 0.03
        dlon = (lonmax - lonmin) * 0.03
        lonmin = lonmin-dlon if lonmin-dlon > -180 else lonmin
        lonmax = lonmax+dlon if lonmax+dlon < 180 else lonmax
        latmin = latmin-dlat if latmin-dlat > -90 else latmin
        latmax = latmax+dlat if latmax+dlat < 90 else latmax
        return (lonmin, lonmax, latmin, latmax)
    
    def get_markers_size(mag, kwargs):
        if mag is None:
            return kwargs.pop('s', None)
        x = np.linspace(min(mag), max(mag))
        y = np.geomspace(min_size, max_size)
        return np.interp(mag, x, y)
    
    def get_rounded_magnitudes(mag):
        magmin, magmax = min(mag), max(mag)
        return np.round(np.geomspace(magmin, magmax, legend_markers), 1)
    
    
    transform = ccrs.PlateCarree()
    if ax is None:
        projection = eval('ccrs.%s()'%projection)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        map_boundaries = get_map_boundaries(lat, lon)
        ax.set_extent(map_boundaries, transform)
        add_earth_features(ax, 
                           scale=resolution,
                           oceans_color=oceans_color, 
                           edgecolor=edgecolor,
                           lands_color=lands_color)
        
    marker = kwargs.pop('marker', '*')
    zorder = kwargs.pop('zorder', 100)
    color = kwargs.pop('color', kwargs.pop('c', 'r'))
    s = kwargs.pop('s', None)
    if s is None:
        s = get_markers_size(mag, kwargs)
    if isinstance(s, Iterable):
        mags_legend = get_rounded_magnitudes(mag)
        idx = [np.argmin(np.abs(mag - mag_i)) for mag_i in mags_legend]
        idx_all = np.setdiff1d(range(len(mag)), idx)
        ax.scatter(lon[idx_all], lat[idx_all], c=color, marker=marker, 
                   transform=transform, s=s[idx_all], zorder=100, **kwargs) 
        for i, mag_legend in zip(idx, mags_legend):
            ax.scatter(lon[i], lat[i], c=color, marker=marker, transform=transform, 
                       s=s[i], label=mag_legend, zorder=100, **kwargs) 
        ax.legend(**legend_dict)
    else:
        ax.scatter(lon, lat, c=color, marker=marker, transform=transform, s=s, 
                   zorder=zorder, **kwargs) 
    if show:
        plt.show()
    else:
        return ax
    

def plot_rays(data_coords, ax=None, show=True, stations_color='r', 
              paths_color='k', oceans_color='water', lands_color='land', 
              edgecolor='k', stations_alpha=None, paths_alpha=0.3, 
              projection='Mercator', resolution='110m', map_boundaries=None, 
              bound_map=True, paths_width=0.2, **kwargs):
    """ 
    Utility function to display the great-circle paths associated with pairs
    of data coordinates
    
    Parameters
    ----------      
    data_coords : ndarray of shape (n, 4)
        Lat1, lon1, lat2, lon2 of the great-circle paths to be plotted
      
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        If not `None`, `c` is plotted on the `GeoAxesSubplot` instance.
        Otherwise, a new figure and `GeoAxesSubplot` instance is created
        
    show : bool
        If `True` (default), the map will be showed once generated. Otherwise
        a `GeoAxesSubplot` instance is returned
        
    stations_color, paths_color : str
        Color of the receivers and of the great-circle paths (see matplotlib 
        documentation for valid color names). Defaults are 'r' (red) and
        'k' (black)
                
    oceans_color, lands_color : str
        Color of oceans and lands. The arguments are ignored if ax is not
        None. Otherwise, they are passed to `cartopy.feature.NaturalEarthFeature` 
        (to the argument 'facecolor'). Defaults are 'water' and 'land'

    edgecolor : str
        Color of the boundaries between, e.g., lakes and land. The argument 
        is ignored if ax is not None. Otherwise, it is passed to 
        cartopy.feature.NaturalEarthFeature (to the argument 'edgecolor'). 
        Default is 'k' (black)

    stations_alpha, paths_alpha : float, optional
        Transparency of the stations and of the great-circle paths. Defaults
        are `None` and 0.3
        
    paths_width : float
        Linewidth of the great-circle paths
                
    projection : str
        Name of the geographic projection used to create the `GeoAxesSubplot`.
        (Visit the `cartopy website 
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html>`_ 
        for a list of valid projection names.) If ax is not None, `projection` 
        is ignored. Default is 'Mercator'
    
    resolution : {'10m', '50m', '110m'}
        Resolution of the Earth features displayed in the figure. Passed to
        `cartopy.feature.NaturalEarthFeature`. Valid arguments are '110m',
        '50m', '10m'. Default is '110m'. Ignored if `ax` is not `None`.
    
    map_boundaries : list or tuple of floats, shape (4,), optional
        Lonmin, lonmax, latmin, latmax (in degrees) defining the extent of
        the map
        
    bound_map : bool
        If `True`, the map boundaries will be automatically determined.
        Ignored if map_boundaries is not `None`
                    

    Returns
    -------
    `None` if `show` is `True`. Otherwise a `GeoAxesSubplot` instance
    """
    def get_map_boundaries(data):
        lats = np.concatenate((data[:,0], data[:, 2]))
        lons = np.concatenate((data[:,1], data[:, 3]))
        latmin, latmax = np.nanmin(lats), np.nanmax(lats)
        lonmin, lonmax = np.nanmin(lons), np.nanmax(lons)
        dlon = (lonmax - lonmin) * 0.03
        dlat = (latmax - latmin) * 0.03
        lonmin = lonmin-dlon if lonmin-dlon > -180 else lonmin
        lonmax = lonmax+dlon if lonmax+dlon < 180 else lonmax
        latmin = latmin-dlat if latmin-dlat > -90 else latmin
        latmax = latmax+dlat if latmax+dlat < 90 else latmax
        return (lonmin, lonmax, latmin, latmax)        
    
    if ax is None:
        projection = lower_threshold_projection(eval('ccrs.%s'%projection))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        add_earth_features(ax, 
                           scale=resolution,
                           oceans_color=oceans_color, 
                           edgecolor=edgecolor,
                           lands_color=lands_color)
        
    transform = ccrs.PlateCarree()
    marker = kwargs.pop('marker', '^')
    
    stations = set()
    for lat1, lon1, lat2, lon2 in data_coords:
        ax.plot([lon1, lon2], 
                [lat1, lat2], 
                color=paths_color, 
                alpha=paths_alpha, 
                lw=paths_width, 
                transform=ccrs.Geodetic())
        if (lat1, lon1) not in stations:
            ax.plot(lon1, 
                    lat1, 
                    marker=marker, 
                    c=stations_color, 
                    transform=transform, 
                    alpha=stations_alpha, 
                    **kwargs) 
            stations.add((lat1, lon1))
        if (lat2, lon2) not in stations:
            ax.plot(lon2, 
                    lat2, 
                    marker=marker, 
                    c=stations_color, 
                    transform=transform, 
                    alpha=stations_alpha, 
                    **kwargs) 
            stations.add((lat2, lon2))
            
    if map_boundaries is None and bound_map:
        map_boundaries = get_map_boundaries(data_coords)
    if map_boundaries is not None:
        ax.set_extent(map_boundaries, transform)  

    if show:
        plt.show()
    else:
        return ax    


def plot_colored_rays(data_coords, c, ax=None, show=True, cmap=scm.roma,
                      vmin=None, vmax=None, stations_color='k', 
                      oceans_color='lightgrey', lands_color='w', edgecolor='k', 
                      stations_alpha=None, paths_alpha=None, resolution='110m',
                      projection='Mercator', map_boundaries=None, bound_map=True, 
                      paths_width=1, colorbar=True, cbar_dict={}, **kwargs):
    """ 
    Utility function to display the great-circle paths associated with pairs
    of data coordinates, colored according to their respective measurements
    
    Parameters
    ----------      
    data_coords : ndarray of shape (n, 4)
        Lat1, lon1, lat2, lon2 of the great-circle paths to be plotted
        
    c : ndarray of shape (n)
        Value associated with each pair of coordinates in `data_coords`
      
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        If not None, `c` is plotted on the GeoAxesSubplot instance.
        Otherwise, a new figure and GeoAxesSubplot instance is created
        
    show : bool
        If True (default), the map will be showed once generated. Otherwise
        a GeoAxesSubplot instance is returned
        
    cmap : str or Colormap
        If str, it should be a valid matplotlib.cm.colormap name (see 
        matplotlib documentation). Otherwise, it should be the name of a
        colormap available in seislib.colormaps (see also the documentation at 
        https://www.fabiocrameri.ch/colourmaps/)
        
    vmin, vmax : float
        Boundaries of the colormap. If None, the minimum and maximum values
        of `c` will be taken
    
    stations_color : str
        Color of the receivers and of the great-circle paths (see matplotlib 
        documentation for valid color names). Defaults are 'r' (red) and
        'k' (black)
                
    oceans_color, lands_color : str
        Color of oceans and lands. The arguments are ignored if ax is not
        None. Otherwise, they are passed to `cartopy.feature.NaturalEarthFeature` 
        (to the argument 'facecolor'). Defaults are 'water' and 'land'

    edgecolor : str
        Color of the boundaries between, e.g., lakes and land. The argument 
        is ignored if ax is not None. Otherwise, it is passed to 
        cartopy.feature.NaturalEarthFeature (to the argument 'edgecolor'). 
        Default is 'k' (black)

    stations_alpha, paths_alpha : float, optional
        Transparency of the stations and of the great-circle paths. Defaults
        are `None` and 0.3
        
    paths_width : float
        Linewidth of the great-circle paths
                
    projection : str
        Name of the geographic projection used to create the `GeoAxesSubplot`.
        (Visit the `cartopy website 
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html>`_ 
        for a list of valid projection names.) If ax is not None, `projection` 
        is ignored. Default is 'Mercator'
        
    resolution : {'10m', '50m', '110m'}
        Resolution of the Earth features displayed in the figure. Passed to
        `cartopy.feature.NaturalEarthFeature`. Valid arguments are '110m',
        '50m', '10m'. Default is '110m'. Ignored if `ax` is not `None`.
    
    map_boundaries : list or tuple of floats, shape (4,), optional
        Lonmin, lonmax, latmin, latmax (in degrees) defining the extent of
        the map
        
    bound_map : bool
        If `True`, the map boundaries will be automatically determined.
        Ignored if `map_boundaries` is not `None`
    
    colorbar : bool
        If `True` (default), a colorbar associated with `c` is displayed on 
        the side of the map        
    
    cbar_dict : dict
        Keyword arguments passed to `matplotlib.colorbar.ColorbarBase`
    
    **kwargs
        Additional keyword arguments passed to ax.plot
    
    Returns
    -------
    If `show` is `True`
        `None` 
    Otherwise 
        `GeoAxesSubplot` instance together with an instance of 
        `matplotlib.colorbar.Colorbar` (if `colorbar` is `True`)
    """
    def get_map_boundaries(data):
        lats = np.concatenate((data[:,0], data[:, 2]))
        lons = np.concatenate((data[:,1], data[:, 3]))
        latmin, latmax = np.nanmin(lats), np.nanmax(lats)
        lonmin, lonmax = np.nanmin(lons), np.nanmax(lons)
        dlon = (lonmax - lonmin) * 0.03
        dlat = (latmax - latmin) * 0.03
        lonmin = lonmin-dlon if lonmin-dlon > -180 else lonmin
        lonmax = lonmax+dlon if lonmax+dlon < 180 else lonmax
        latmin = latmin-dlat if latmin-dlat > -90 else latmin
        latmax = latmax+dlat if latmax+dlat < 90 else latmax
        return (lonmin, lonmax, latmin, latmax)        
    
    def create_colourbar(ax, cmap, norm, **kwargs):
        """Create a colourbar with limits of lwr and upr"""
        divider = make_axes_locatable(ax)
        orientation = kwargs.pop('orientation', 'horizontal')
        if orientation == 'vertical':
            loc = 'right'
        elif orientation == 'horizontal':
            loc = 'bottom'
        cax = divider.append_axes(loc, '5%', pad='3%', axes_class=mpl.pyplot.Axes)
        cb = mpl.colorbar.ColorbarBase(cax, 
                                       cmap=cmap, 
                                       norm=norm, 
                                       orientation=orientation, 
                                       **kwargs)
        return cb

    
    if ax is None:
        projection = lower_threshold_projection(eval('ccrs.%s'%projection))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        add_earth_features(ax, 
                           scale=resolution,
                           oceans_color=oceans_color, 
                           edgecolor=edgecolor,
                           lands_color=lands_color)
        
    transform = ccrs.PlateCarree()
    marker = kwargs.pop('marker', '^')
    vmin = vmin if vmin is not None else np.nanmin(c)
    vmax = vmax if vmax is not None else np.nanmax(c)
    norm = mpl.colors.Normalize(vmin, vmax)
    
    
    stations = set()
    for (lat1, lon1, lat2, lon2), v in zip(data_coords, c):
        ax.plot([lon1, lon2], 
                [lat1, lat2], 
                alpha=paths_alpha, 
                lw=paths_width, 
                color=cmap(norm(v)),
                transform=ccrs.Geodetic())
        if (lat1, lon1) not in stations:
            ax.plot(lon1, 
                    lat1, 
                    marker=marker, 
                    c=stations_color, 
                    transform=transform, 
                    alpha=stations_alpha, 
                    **kwargs) 
            stations.add((lat1, lon1))
        if (lat2, lon2) not in stations:
            ax.plot(lon2, 
                    lat2, 
                    marker=marker, 
                    c=stations_color, 
                    transform=transform, 
                    alpha=stations_alpha, 
                    **kwargs) 
            stations.add((lat2, lon2))
            
    if map_boundaries is None and bound_map:
        map_boundaries = get_map_boundaries(data_coords)
    if map_boundaries is not None:
        ax.set_extent(map_boundaries, transform)  
    if colorbar:
        cb = create_colourbar(ax, cmap, norm, **cbar_dict)    
    if show:
        plt.show()
    else:
        return ax, cb if colorbar else ax       


def plot_map(mesh, c, ax=None, projection='Mercator', map_boundaries=None, 
             bound_map=True, colorbar=True, show=True, style='colormesh', 
             add_features=False, resolution='110m', cbar_dict={}, **kwargs):
    """ 
    Utility function to display the lateral variations of some quantity on 
    a equal-area grid
    
    Parameters
    ----------
    mesh : ndarray, shape (n, 4)
        Grid cells in seislib format, consisting of n pixels described by 
        lat1, lat2, lon1, lon2 (in degrees)
        
    c : ndarray, shape (n,)
        Values associated with the grid cells (mesh)
        
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        If not None, `c` is plotted on the `GeoAxesSubplot` instance. 
        Otherwise, a new figure and `GeoAxesSubplot` instance is created
        
    projection : str
        Name of the geographic projection used to create the `GeoAxesSubplot`.
        (Visit the `cartopy website 
        <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html>`_ 
        for a list of valid projection names.) If ax is not None, `projection` 
        is ignored. Default is 'Mercator'
        
    map_boundaries : list or tuple of floats, shape (4,), optional
        Lonmin, lonmax, latmin, latmax (in degrees) defining the extent of
        the map
        
    bound_map : bool
        If `True`, the map boundaries will be automatically determined.
        Ignored if `map_boundaries` is not None
    
    colorbar : bool
        If `True` (default), a colorbar associated with `c` is displayed on 
        the side of the map
        
    show : bool
        If `True` (default), the map will be showed once generated
        
    style : {'colormesh', 'contourf', 'contour'}
        Possible options are 'colormesh', 'contourf', and 'contour',
        each corresponding to the homonymous method.
        Default is 'colormesh'
        
    add_features : bool
        If `True`, natural Earth features will be added to the `GeoAxesSubplot`.
        Default is `False`. If `ax` is `None`, it is automatically set to `True`
        
    resolution : {'10m', '50m', '110m'}
        Resolution of the Earth features displayed in the figure. Passed to
        `cartopy.feature.NaturalEarthFeature`. Valid arguments are '110m',
        '50m', '10m'. Default is '110m'
        
    cbar_dict : dict
        Keyword arguments passed to `matplotlib.colorbar.ColorbarBase`
        
    **kwargs
        Additional inputs passed to the 'colormesh', 'contourf', and
        'contour' methods of :class:`SeismicTomography`
        
        
    Returns
    -------
    If show is True
        `None`
    Otherwise
        an instance of `matplotlib.collections.QuadMesh`, together with an 
        instance of `matplotlib.colorbar.Colorbar` (if `colorbar` is `True`)
    """
    def get_map_boundaries(mesh):
        latmin, lonmin = np.min(mesh, axis=0)[::2]
        latmax, lonmax = np.max(mesh, axis=0)[1::2]
        dlon = (lonmax - lonmin) * 0.03
        dlat = (latmax - latmin) * 0.03
        
        lonmin = lonmin-dlon if lonmin-dlon > -180 else lonmin
        lonmax = lonmax+dlon if lonmax+dlon < 180 else lonmax
        latmin = latmin-dlat if latmin-dlat > -90 else latmin
        latmax = latmax+dlat if latmax+dlat < 90 else latmax
        
        return (lonmin, lonmax, latmin, latmax)
    
    def set_colorbar_aspect(cb, kwargs):
        norm = kwargs.get('norm', None)
        if norm is not None and type(norm)==type(LogNorm()):
            if style in ['contour', 'contourf']:
                cmin, cmax = cb.vmin, cb.vmax
                ticks = np.power(10, np.arange(np.floor(np.log10(cmin)), 
                                               np.ceil(np.log10(cmax)+1)))
                cb.set_ticks(ticks)
                minorticks = np.hstack([np.arange(2*i, 10*i, 1*i) for i in ticks])
                minorticks = [i for i in minorticks if cmin<i<cmax]
                cb.ax.xaxis.set_ticks(minorticks, minor=True)      
        return cb

    add_features = add_features if ax is not None else True
    if ax is None:
        projection = eval('ccrs.%s()'%projection)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=projection)        
    if add_features:
        add_earth_features(ax, 
                           scale=resolution,
                           oceans_color='none', 
                           edgecolor='k',
                           lands_color='none')
        
    styles = ['colormesh', 'contourf', 'contour']
    if style not in styles:
        raise Exception('`style` not available. Please choose among %s'%styles)
    plotting_func = eval(style)
    
    notnan = np.flatnonzero(~np.isnan(c))
    img = plotting_func(mesh=mesh[notnan], c=c[notnan], ax=ax, **kwargs)
    
    if map_boundaries is None and bound_map:
        map_boundaries = get_map_boundaries(mesh)
    if map_boundaries is not None:
        ax.set_extent(map_boundaries, ccrs.PlateCarree())  
    if colorbar:
        cb = make_colorbar(ax, img, **cbar_dict)
        cb = set_colorbar_aspect(cb, kwargs)
    if show:
        plt.show()
    else:
        return (img, cb) if colorbar else img        


def scientific_label(obj, precision):
    """ Creates a scientific label approximating a real number in LaTex style
    
    Parameters
    ----------
    obj : float
        Number that must be represented
        
    precision : int
        Number of decimal digits in the resulting label
        
        
    Returns
    -------
    scientific_notation : str
        Approximated number, e.g., r'5.000 \\times 10^{-6}', arising from
        the function call scientific_label(5e-06, 3)
        
    
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-2, 2, 150)
    >>> y = x**2 * 1e-5
    >>> plt.plot(x, y)
    >>> ticks = plt.gca().get_yticks()
    >>> labels = [r'$%s$'%scientific_label(i, 1) for i in ticks]
    >>> plt.yticks(ticks=ticks, labels=labels)
    >>> plt.show()
    """
    precision = '%.' + '%se'%precision
    python_notation = precision % obj
    number, power = python_notation.split('e')
    scientific_notation = r'%s \times 10^{%s}'%(number, int(power))
    return scientific_notation














