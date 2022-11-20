#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Parameterization
================

By default, SeisLib discretizes the Earth's surface by means of equal-area grids. 
These prevent from artificially increasing the resolution of the resulting 
tomographic maps at latitudes different than zero (the effect is more prominent nearby
the poles), and should therefore be preferred to Cartesian grids when investigating 
relatively large areas. SeisLib also allows for adaptive parameterizations, with finer 
resolution in the areas characterized by relatively high density of measurements. If 
we consider a given block intersected by more than a certain number of inter-station 
great-circle paths, the finer resolution is achieved by splitting it in four sub-blocks, 
at the midpoint along both latitude an longitude. The operation
can be performed an arbitrary number of times.

"""
from math import radians, degrees
from math import cos, pi, asin, sin
from math import sqrt
from collections import defaultdict, Counter
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Point, Polygon
from seislib.plotting import add_earth_features
SQUARE_DEGREES = 41252.961249419277010
FOUR_PI = 4 * pi





class _Grid():
    
    def __init__(self):    
        pass
    
        
    def __str__(self):
        string = '-------------------------------------\n'
        string += 'GRID PARAMETERS\n'
        string += 'Lonmin - Lonmax : %.3f - %.3f\n'%(self.lonmin, self.lonmax)
        string += 'Latmin - Latmax : %.3f - %.3f\n'%(self.latmin, self.latmax)
        string += 'Number of cells : %s\n'%(self.mesh.shape[0])
        for i in range(self.refined + 1):
            cells = self.cell_size_per_level.get(i, None)
            if cells is not None:
                string += 'Grid cells of %.3f°'%cells
                string += ' : %d\n'%self.ncells_per_level[i]
        string += '-------------------------------------'
        return string
    
    
    def __repr__(self):
        return str(self)
    
    
    def update_grid_params(self, mesh, refined=False):
        """
        Updates the grid boundaries stored in the EqualAreaGrid instance and
        the number of times that the grid has been refined. The update is 
        performed in place.
        
        Parameters
        ----------
        mesh : ndarray (n, 4)
            Grid cells bounded by parallel1, parallel2, meridian1, meridian2
            
        refined : bool
            If `True`, the number of times that the grid has been refined is 
            updated
        """
        self.mesh = mesh
        self.latmin = np.min(mesh[:,0])
        self.lonmin = np.min(mesh[:,2])
        self.latmax = np.max(mesh[:,1])
        self.lonmax = np.max(mesh[:,3])
        if refined:
            self.refined += 1
        if sum(self.ncells_per_level.values()) != mesh.shape[0]:
            for i in range(self.refined):
                if i+1 in self.cell_size_per_level:
                    continue
                self.cell_size_per_level[i+1] = self.cell_size_per_level[i] / 2
            sizes = [self.cell_size_per_level[i] for i in range(self.refined+1)]
            cells_sizes = mesh[:, 1] - mesh[:, 0]
            assert np.all(cells_sizes > 0)
            counts = Counter(np.argmin(np.abs([cells_sizes-size for size in sizes]), 
                                       axis=0))
            self.ncells_per_level = counts
        if self.verbose:
            print('*** GRID UPDATED ***')
            print(self)


    def set_boundaries(self, latmin, latmax, lonmin, lonmax, mesh=None, 
                       inplace=True):
        """ Restricts the mesh to the required boundaries
        
        Parameters
        ----------
        latmin, latmax, lonmin, lonmax : float
            Boundaries of the new mesh. Their units should be consistent with
            those of the mesh
            
        mesh : ndarray (n, 4), optional
            Array containing `n` pixels bounded by parallel1, parallel2, 
            meridian1, meridian2. If `None`, the mesh stored in the 
            :class:`EqualAreaGrid` instance (`self.mesh`) is used. Default is 
            `None`
            
        inplace : bool
             If `True`, the mesh stored in the :class:`EqualAreaGrid` instance 
             (`self.mesh`) is modified permanently (default is `False`)
             
         
        Returns
        -------
        `None`, if inplace is `True`, else the modified `mesh`
        """
        if mesh is None:
            mesh = self.mesh
        if not any([latmin, latmax, lonmin, lonmax]):
            return mesh
        latmin = latmin if latmin is not None else -90
        latmax = latmax if latmax is not None else +90
        lonmin = lonmin if lonmin is not None else -180
        lonmax = lonmax if lonmax is not None else +180
        ilat = np.flatnonzero((mesh[:,0]<=latmax) & (mesh[:,1]>=latmin))
        mesh = mesh[ilat]
        ilon = np.flatnonzero((mesh[:,2]<=lonmax) & (mesh[:,3]>=lonmin))
        mesh = mesh[ilon]
        if inplace:
            return self.update_grid_params(mesh)
        return mesh

    
    def select_cells(self, indexes, inplace=True):
        """ Mesh indexing
        
        Parameters
        ----------
        indexes : list or ndarray
        
        inplace : bool
             If `True`, the mesh stored in the :class:`EqualAreaGrid` 
             instance (`self.mesh`) is modified permanently. Default 
             is `False`.
         
        Returns
        -------
        `None`, if inplace is `True`, else the indexed `mesh`.
        """
        mesh = self.mesh[indexes]
        if inplace:
            return self.update_grid_params(mesh, refined=False)
        return mesh


    def index_lon_lat(self, lon, lat):
        """ Returns the mesh index corresponding with the coordinates (lat, lon)
        
        Parameters
        ----------
        lon, lat : float
            Geographic coordinates. Their units should be consistent with those
            of `mesh`
            
        
        Returns
        -------
        idx : int
            Mesh index corresponding to (lat, lon)
        """
        mesh = self.mesh
        latmax = np.max(mesh[:, 1])
        lonmax = np.max(mesh[:, 3])
        if lat == latmax:
            lat -= 0.000001
        if lon == lonmax:
            lon -= 0.000001
        ilat = np.flatnonzero((mesh[:,0]<=lat) & (lat<mesh[:,1]))
        ilon = np.flatnonzero((mesh[:,2]<=lon) & (lon<mesh[:,3]))
        idx = np.intersect1d(ilat, ilon)
        return int(idx)
    
    
    def midpoints_lon_lat(self):
        """ Returns the midpoints (lon, lat) of each grid's block
        
        
        Returns
        -------
        lon, lat : ndarray of shape (n,)
        """
        lat = np.mean(self.mesh[:, :2], axis=1)
        lon = np.mean(self.mesh[:, 2:], axis=1)
        return lon, lat
    
    
    def indexes_in_region(self, latmin, latmax, lonmin, lonmax):
        """
        Returns the mesh indexes whose midpoints (lon, lat) fall inside the 
        specified rectangular region.
        
        Parameters
        ----------
        latmin, latmax : float
            Latitudinal boundaries defining the rectangular region
            
        lonmin, lonmax : float
            Longitudinal boundaries defining the rectangular region
            
        Returns
        -------
        idx : ndarray
            Mesh indexes falling inside the rectangular region.
        """
        lons, lats = self.midpoints_lon_lat()
        idx = [i for i, (lat, lon) in enumerate(zip(lats, lons)) if \
               lonmin<lon<lonmax and latmin<lat<latmax]
        return np.asarray(idx)
    
    
    def indexes_out_region(self, latmin, latmax, lonmin, lonmax):
        """
        Returns the mesh indexes whose midpoints (lon, lat) fall outside the 
        specified rectangular region.
        
        Parameters
        ----------
        latmin, latmax : float
            Latitudinal boundaries defining the rectangular region
            
        lonmin, lonmax : float
            Longitudinal boundaries defining the rectangular region
            
        Returns
        -------
        idx : ndarray
            Mesh indexes falling outside the rectangular region.
        """
        idx_inside = self.indexes_in_region(latmin, latmax, lonmin, lonmax)
        return np.flatnonzero(~np.in1d(np.arange(self.mesh.shape[0]), idx_inside))
    
    
    def indexes_in_polygon(self, poly):
        """
        Returns the mesh indexes whose midpoints (lon, lat) fall outside the 
        specified rectangular region.
        
        Parameters
        ----------
        poly : ndarray of shape (n, 2), shapely.geometry.Polygon
            Array of points (lon, lat) describing the polygon or instance of 
            `shapely.geometry.Polygon`
            
        Returns
        -------
        idx : ndarray
            Mesh indexes falling inside the polygon.
        """
        lons, lats = self.midpoints_lon_lat()
        if not isinstance(poly, Polygon):
            poly = Polygon(poly)
        idx = [i for i, (lon, lat) in enumerate(zip(lons, lats)) \
               if poly.contains(Point(lon, lat))]
        return np.asarray(idx)

    
    def refine_mesh(self, ipixels, mesh=None, inplace=False):
        """ Halves the size of the specified pixels
        
        Parameters
        ----------
        ipixels : list or ndarray
            mesh indexes to refine
            
        mesh : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2. If None, the mesh stored in the 
            :class:`EqualAreaGrid` instance (`self.mesh`) is used. 
            Default is None
            
        inplace : bool
             If `True`, the mesh stored in the :class:`EqualAreaGrid` 
             instance (`self.mesh`) is modified permanently. Default 
             is False.
             
        Returns
        -------
        `None`, if inplace is `True`, else the modified mesh.
        """
        if mesh is None:
            inplace = True
            mesh = self.mesh
        newmesh_size = mesh.shape[0] + 3*len(ipixels)
        i_newpixel = 0
        #Remove one pixel and add 4 -- the pixels itself divided by four
        newmesh = np.empty((newmesh_size, 4), dtype=np.double)
        to_refine = np.zeros(mesh.shape[0])
        to_refine[ipixels] = 1
        for i_pixel, refine in enumerate(to_refine):
            if not refine:
                newmesh[i_newpixel, 0] = mesh[i_pixel, 0]
                newmesh[i_newpixel, 1] = mesh[i_pixel, 1]
                newmesh[i_newpixel, 2] = mesh[i_pixel, 2]
                newmesh[i_newpixel, 3] = mesh[i_pixel, 3]
                i_newpixel += 1
            else:
                parallel1 = mesh[i_pixel, 0]
                parallel2 = mesh[i_pixel, 1]
                meridian1 = mesh[i_pixel, 2]
                meridian2 = mesh[i_pixel, 3]    
                self.split_pixel(newmesh, i_newpixel, parallel1, parallel2, 
                                 meridian1, meridian2)
                i_newpixel += 4
        if inplace:
            return self.update_grid_params(newmesh, refined=True)
        return newmesh
        
    
    def split_pixel(self, newmesh, i_newpixel, parallel1, parallel2, meridian1, 
                    meridian2):
        """ Called by :meth:`refine_mesh` to create new pixels
        
        Parameters
        ----------
        newmesh : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2
            
        i_newpixel : int
            Mesh index to refine
            
        parallel1, parallel2, meridian1, meridian2 : float
             Geographic coordinates of the boundaries of the pixel to 
             refine. Their units should be consistent with those of the mesh
             
        Returns
        -------
        None, `newmesh` is modified in place
        """
        newmesh[i_newpixel, 0] = (parallel1 + parallel2) / 2
        newmesh[i_newpixel, 1] = parallel2
        newmesh[i_newpixel, 2] = meridian1
        newmesh[i_newpixel, 3] = (meridian1 + meridian2) / 2
        newmesh[i_newpixel+1, 0] = (parallel1 + parallel2) / 2
        newmesh[i_newpixel+1, 1] = parallel2
        newmesh[i_newpixel+1, 2] = (meridian1 + meridian2) / 2
        newmesh[i_newpixel+1, 3] = meridian2
        newmesh[i_newpixel+2, 0] = parallel1
        newmesh[i_newpixel+2, 1] = (parallel1 + parallel2) / 2
        newmesh[i_newpixel+2, 2] = meridian1
        newmesh[i_newpixel+2, 3] = (meridian1 + meridian2) / 2
        newmesh[i_newpixel+3, 0] = parallel1
        newmesh[i_newpixel+3, 1] = (parallel1 + parallel2) / 2
        newmesh[i_newpixel+3, 2] = (meridian1 + meridian2) / 2
        newmesh[i_newpixel+3, 3] = meridian2
    
    
    def plot(self, 
             ax=None, 
             mesh=None, 
             projection='Mercator', 
             meridian_min=-180,
             meridian_max=180,
             show=True, 
             map_boundaries=None, 
             bound_map=True, 
             oceans_color='water', 
             lands_color='land', 
             scale='110m', 
             **kwargs):
        """
        Plots the (adaptive) equal-area grid
        
        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If `None` a new figure and `GeoAxesSubplot` instance is created
            
        mesh : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2. If `None`, the mesh stored in the 
            :class:`EqualAreaGrid` instance (`self.mesh`) is used. 
            Default is `None`.
            
        projection : str
            Name of the geographic projection used to create the `GeoAxesSubplot`.
            (Visit the cartopy website for a list of valid projection names.)
            If ax is not None, `projection` is ignored. Default is 'Mercator'
        
        meridian_min, meridian_max : float
            Minimum and maximum longitude used to bound the parallels. In most
            situations this arguments can be ignored. When plotting the grid
            on particular projections (e.g. LambertConformal), however, these
            values should be tuned: otherwise the parallel lines will not be
            displayed in the final plot. (Due to an unsolved issue in CartoPy.)
        
        show : bool
            If `True` (default), the map will be showed once generated.
          
        map_boundaries : iterable of floats, shape (4,), optional
            Lonmin, lonmax, latmin, latmax (in degrees) defining the extent of
            the map
            
        bound_map : bool
            If `True`, the map boundaries will be automatically determined.
            Ignored if `map_boundaries` is not None
              
        oceans_color, continents_color : str
            Color of oceans and continents in the map. They should be valid 
            matplotlib colors (see `matplotlib.colors` documentation for more 
            details) or be part of `cartopy.cfeature.COLORS`
        
        **kwargs
            Additional keyword arguments passed to
            `matplotlib.pyplot.plot`
        
        Returns
        -------
        `None`, if `show` is True, else a `GeoAxesSubplot` instance
        """
        def get_parallels(mesh):
            parallels = defaultdict(list)
            for parallel1, parallel2, meridian1, meridian2 in mesh:
                for parallel in [parallel1, parallel2]:
                    if not parallels[parallel]:
                        parallels[parallel].append([meridian1, meridian2])
                        continue
                    for merid_list in parallels[parallel]:
                        merid1, merid2 = merid_list
                        if merid1>=meridian1 and merid2<=meridian2:
                            merid_list[0] = meridian1
                            merid_list[1] = meridian2
                            break
                        elif merid1<=meridian1 and merid2>=meridian2:
                            merid_list[0] = merid1
                            merid_list[1] = merid2
                            break
                        elif merid1<=meridian1 and meridian1<=merid2<=meridian2:
                            merid_list[0] = merid1
                            merid_list[1] = meridian2
                            break
                        elif meridian1<=merid1<=meridian2 and merid2>=meridian2:
                            merid_list[0] = meridian1
                            merid_list[1] = merid2
                            break
                    else:
                        parallels[parallel].append([meridian1, meridian2])
            return parallels
                        
        def drawmeridians(mesh, ax, **kwargs_plot):
            drawn_meridians = defaultdict(set)
            transform = ccrs.Geodetic()
            
            # Draw meridians
            for lat1, lat2, lon1, lon2 in mesh:
                ax.plot([lon1, lon1], 
                        [lat1, lat2], 
                        transform=transform,
                        **kwargs_plot)
                drawn_meridians[(lat1, lat2)].add(lon1)
                if lon2 not in drawn_meridians[(lat1, lat2)]:
                    ax.plot([lon2, lon2], 
                            [lat1, lat2], 
                            transform=transform,
                            **kwargs_plot)
                    drawn_meridians[(lat1, lat2)].add(lon2)
            return
        
        def drawparallels(mesh, ax, meridian_min, meridian_max, **kwargs_plot):
            transform = ccrs.PlateCarree()
            parallels = get_parallels(mesh)
            for parallel in parallels:
                for lon1, lon2 in parallels[parallel]:
                    ax.plot([max(lon1, meridian_min), min(lon2, meridian_max)], 
                            [parallel, parallel], 
                            transform=transform,
                            **kwargs_plot)
            return
        
        
        def get_map_boundaries(mesh):
            latmin, lonmin = np.min(mesh, axis=0)[::2]
            latmax, lonmax = np.max(mesh, axis=0)[1::2]
            dlon = (lonmax - lonmin) * 0.01
            dlat = (latmax - latmin) * 0.01
            
            lonmin = lonmin-dlon if lonmin-dlon > -180 else lonmin
            lonmax = lonmax+dlon if lonmax+dlon < 180 else lonmax
            latmin = latmin-dlat if latmin-dlat > -90 else latmin
            latmax = latmax+dlat if latmax+dlat < 90 else latmax
            
            return (lonmin, lonmax, latmin, latmax)
        
    
        if mesh is None:
            mesh = self.mesh
            
        kwargs_default = {'color': kwargs.pop('color', kwargs.pop('c', 'k')),
                          'ls': kwargs.pop('ls', kwargs.pop('linestyle', '-')),
                          'lw': kwargs.pop('lw', kwargs.pop('linewidth', 1)),
                          'zorder': kwargs.pop('zorder', 100)}
        kwargs.update(kwargs_default)
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection=eval('ccrs.%s()'%projection))
            add_earth_features(ax,
                               oceans_color=oceans_color,
                               lands_color=lands_color,
                               scale=scale,
                               edgecolor='none')
        drawmeridians(mesh, ax, **kwargs)
        drawparallels(mesh, ax, meridian_min, meridian_max, **kwargs)
        
        if map_boundaries is None and bound_map:
            map_boundaries = get_map_boundaries(mesh)
        if map_boundaries is not None:
            ax.set_extent(map_boundaries, ccrs.PlateCarree())     
        # Draw parallels
        if show:
            plt.show()
        return ax



class EqualAreaGrid(_Grid):
    """
    Class that allows for creating an equal-area grid covering the whole
    Earth's surface.


    Parameters
    ----------
    cell_size : float
        Size of each side of the equal-area grid
        
    latmin, lonmin, latmax, lonmax : float, optional
        Boundaries (in degrees) of the grid
        
    verbose : bool
        If True, information about the grid will be displayed. Default is 
        True    


    Attributes
    ----------
    verbose : bool
        If True, information about the grid will be displayed.

    refined : int
        Number of times the grid has been "refined"

    lon_span : ndarray of shape (n,)
        Longitudinal span of each block in the `n` latitudinal bands
        defining the grid

    mesh : ndarray of shape (m, 4)
        Blocks bounded by parallel1, parallel2, meridian1, meridian2

    latmin, lonmin, latmax, lonmax : float
        Minimum and maximum latitudes and longitudes of the blocks
        defining the grid

    
    Examples
    --------
    Let's first define an equal-area grid of :math:`10^{\circ} \times 10^{\circ}`.
    By default, this is created on the global scale.
    
    >>> from seislib.tomography import EqualAreaGrid
    >>> grid = EqualAreaGrid(cell_size=10, verbose=True)
    -------------------------------------
    Optimal grid found in 10 iterations
    -------------------------------------
    -------------------------------------
    GRID PARAMETERS
    Lonmin - Lonmax : -180.000 - 180.000
    Latmin - Latmax : -90.000 - 90.000
    Number of cells : 412
    Grid cells of 10.006° : 412
    -------------------------------------

    >>> print(grid.mesh)
    [[ 80.2098   90.     -180.      -60.    ]
    [  80.2098   90.      -60.       60.    ]
    [  80.2098   90.       60.      180.    ]
    ..., 
    [ -90.      -80.2098 -180.      -60.    ]
    [ -90.      -80.2098  -60.       60.    ]
    [ -90.      -80.2098   60.      180.    ]]

    We can now restrict the above parameterization to an arbitrary region, for
    example:

    >>> grid.set_boundaries(latmin=0, 
    ...                     lonmin=0,
    ...                     latmax=10,
    ...                     lonmax=10,
    ...                     inplace=True)
    *** GRID UPDATED ***
    -------------------------------------
    GRID PARAMETERS
    Lonmin - Lonmax : -10.000 - 20.000
    Latmin - Latmax : -10.065 - 10.065
    Number of cells : 6
    Grid cells of 10.006° : 6
    -------------------------------------

    >>> print(grid.mesh)
    [[ 0.      10.0645 -10.       0.    ]
    [  0.      10.0645   0.      10.    ]
    [  0.      10.0645  10.      20.    ]
    [-10.0645   0.     -10.       0.    ]
    [-10.0645   0.       0.      10.    ]
    [-10.0645   0.      10.      20.    ]]


    .. hint::

        The same result can be obtained by passing the boundaries of the
        region of interest directly in the initialization of the class 
        instance, e.g.::

            grid = EqualAreaGrid(cell_size=10, 
                                 latmin=0, 
                                 lonmin=0,
                                 latmax=10,
                                 lonmax=10,
                                 verbose=True)

    We can refine any block, say at the 0th and 1st index, simply by:

    >>> grid.refine_mesh([0, 1], inplace=True)
    *** GRID UPDATED ***
    -------------------------------------
    GRID PARAMETERS
    Lonmin - Lonmax : -10.000 - 20.000
    Latmin - Latmax : -10.065 - 10.065
    Number of cells : 12
    Grid cells of 10.006° : 4
    Grid cells of 5.003° : 8
    -------------------------------------

    >>> print(grid.mesh)
    [[ 5.03225  10.0645  -10.       -5.     ]
    [  5.03225  10.0645   -5.        0.     ]
    [  0.        5.03225 -10.       -5.     ]
    [  0.        5.03225  -5.        0.     ]
    [  5.03225  10.0645    0.        5.     ]
    [  5.03225  10.0645    5.       10.     ]
    [  0.        5.03225   0.        5.     ]
    [  0.        5.03225   5.       10.     ]
    [  0.       10.0645   10.       20.     ]
    [-10.0645    0.      -10.        0.     ]
    [-10.0645    0.        0.       10.     ]
    [-10.0645    0.       10.       20.     ]]

    Note that the size of the two blocks defined at the first two 
    rows of the above have been halved.

    To display the grid, use the :meth:`plot` method
    >>> grid.plot(projection='Mercator')
    """

    def __init__(self, cell_size, latmin=None, lonmin=None, latmax=None, 
                 lonmax=None, verbose=True):
        self.verbose = verbose
        self.refined = 0 #number of times the grid has been "refined", i.e., the some of its pixels has been divided in 4
        ncells, cell_size, lon_span = self.best_grid_parameters(cell_size)
        self.ncells_per_level = Counter({0 : ncells})
        self.cell_size_per_level = {0 : cell_size}
        self.lon_span = lon_span
        self.mesh = self.global_mesh(ncells, lon_span)
        self.set_boundaries(latmin=latmin, latmax=latmax, lonmin=lonmin, 
                            lonmax=lonmax, mesh=self.mesh, inplace=True)
        self.latmin = np.min(self.mesh[:,0])
        self.lonmin = np.min(self.mesh[:,2])
        self.latmax = np.max(self.mesh[:,1])
        self.lonmax = np.max(self.mesh[:,3])
        if self.verbose:
            print(self)

    
    def grid_parameters(self, nrings):
        """
        Computes the grid parameters (number of cells, cells area, cells side, 
        longitude span as a function of latitude) [1]_.
    
        
        Parameters
        ----------
        nrings : int
            Number of latitudinal rings used to subdivide the Earth.
            
            
        Returns
        -------
        ncells : int
            Number of grid cells
            
        cell_side : float
            Latitudinal extent of the blocks in degrees, (corresponding to the 
            sqrt of the area)
            
        lon_span : ndarray
            Longitudinal span the grid cells in each latitudinal band
            
        
        References
        ----------
        .. [1] Malkin 2019, A new equal-area isolatitudinal grid on a spherical 
            surface, The Astronomical Journal
        """
        lon_span = np.zeros(nrings) # longitudinal cell span, degrees
        nrings_half = nrings // 2
        lat_step = 90 / nrings_half # initial lat step
        ncells_half = 0
        for i in range(nrings_half): # North hemisphere
            central_lat = cos(radians( lat_step/2 + lat_step*(nrings_half-1-i) ))
            cells_per_ring = int(round( 360 / (lat_step/central_lat) ))
            lon_span[i] = 360 / cells_per_ring # (in degrees)
            ncells_half = ncells_half + cells_per_ring
        
        # South hemisphere
        lon_span[-nrings_half:] = lon_span[:nrings_half][::-1] 
        ncells = ncells_half * 2
        cell_area = SQUARE_DEGREES / ncells #Cell area, sq. deg
        cell_side = sqrt(cell_area)
        return ncells, cell_side, lon_span
    
    
    def best_grid_parameters(self, cell_side):
        """
        Finds the spatial parameterization that most closely approximates the 
        size of the grid cells required by the user. It exploits a 1-D grid 
        search
        
        Parameters
        ----------
        cell_side : float
            Side's size of the desidered grid cells
        
        
        Returns
        -------
        grid parameters associated with the best parameterization. (See :meth:`grid_parameters`)
        """
        last_sizes = np.array([np.nan, np.nan, np.nan])
        for counter, nrings in enumerate(range(2, 41070, 2), 1):
            ncells, final_side, lon_span = self.grid_parameters(nrings)
            last_sizes[-1] = np.abs(final_side - cell_side)
            if np.nanargmin(last_sizes) == 1:
                if self.verbose:
                    print('-------------------------------------')
                    print('Optimal grid found in %s iterations'%counter)
                    print('-------------------------------------')
                return self.grid_parameters(nrings-2)
            else:
                last_sizes = np.roll(last_sizes, -1)
        else:
            raise Exception('*cell_size* is too small')
            
            
    def global_mesh(self, ncells, lon_span):
        """
        Builds an equal-area global mesh given the number of cells and longitude 
        span as a function of latitude. [1]_
        
        
        Parameters
        ----------
        ncells : int
            Number of grid cells
            
        lon_span : ndarray
            Longitudinal span the grid cells in each latitudinal ring
        
        Returns
        -------
        grid : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2. The grid is rounded to the 4rd decimal digit,
            for improved numerical stability in most applications
            
            
        References
        ----------
        .. [1] Malkin 2019, A new equal-area isolatitudinal grid on a spherical 
            surface, The Astronomical Journal
        """
        grid = np.zeros((ncells, 4)) #4 cols: lat1, lat2, lon1, lon2
        cell_area = self.cell_size_per_level[0]**2
        lat2 = 90
        cell_idx = 0
        for dlon in lon_span:
            arg = sin(radians(lat2)) - (cell_area*FOUR_PI/SQUARE_DEGREES)/(radians(dlon))
            if -1.01 < arg < -1: #avoids numerical errors
                arg = -1
            lat1 = degrees(asin(arg))
            lon1 = -180
            for _ in range(int(round(360 / dlon))):
                lon2 = lon1 + dlon
                grid[cell_idx] = (lat1, lat2, lon1, lon2)
                lon1 = lon2
                cell_idx += 1
            lat2 = lat1        
        return np.round(grid, 4)
            
    
    def parallels_first_pixel(self, mesh=None):
        """ 
        Generator function yielding the indexes at which a change in the parallel 
        coordinates is found.
        
        Parameters
        ----------
        mesh : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2. If `None`, the `mesh` stored in the 
            :class:`EqualAreaGrid` instance (`self.mesh`) is used. Default 
            is `None`.

        Yields
        ------
        i : int
        """
        if mesh is None:
            mesh = self.mesh
        parallel1 = mesh[0][0]
        parallel2 = mesh[0][1]
        yield 0
        for i in range(1, mesh.shape[0]):
            new_parallel1 = mesh[i][0]
            new_parallel2 = mesh[i][1]
            if new_parallel1==parallel1 and new_parallel2==parallel2:
                continue
            parallel1 = new_parallel1
            parallel2 = new_parallel2
            yield i           



class RegularGrid(_Grid):
    r"""
    Class that allows allows for creating a regular grid in the format required by
    :class:`seislib.tomography.tomography.SeismicTomography`. This class is
    particularly suited to tomographic applications at local scale, where the 
    use of equal-area parameterizations does not have clear advantages.


    Parameters
    ----------
    cell_size : float, (2,) tuple
        Size of each side of the regular grid. If a (2,) tuple is passed, this
        will be interpreted as `(dlon, dlat)`, where `dlon` and `dlat` are the
        longitudinal and latitudinal steps characterizing the grid
        
        
    latmin, lonmin, latmax, lonmax : float, optional
        Boundaries (in degrees) of the grid
        
    verbose : bool
        If True, information about the grid will be displayed. Default is 
        True    


    Attributes
    ----------
    verbose : bool
        If True, information about the grid will be displayed.

    refined : int
        Number of times the grid has been "refined"

    mesh : ndarray of shape (m, 4)
        Blocks bounded by parallel1, parallel2, meridian1, meridian2

    latmin, lonmin, latmax, lonmax : float
        Minimum and maximum latitudes and longitudes of the blocks
        defining the grid

    
    Examples
    --------
    Let's first define a regular grid of :math:`0.1^{\circ} \times 0.1^{\circ}`.
    We will restrict the study area to :math:`9 \leq lon \leq 11` and 
    :math:`40 \leq lat \leq 42`.
    
    >>> from seislib.tomography import RegularGrid
    >>> grid = RegularGrid(cell_size=0.1, 
                           latmin=40,
                           latmax=42,
                           lonmin=9,
                           lonmax=11,
                           verbose=True)
    -------------------------------------
    GRID PARAMETERS
    Lonmin - Lonmax : 10.000 - 11.000
    Latmin - Latmax : 40.000 - 41.000
    Number of cells : 100
    Grid cells of 0.100° : 100
    -------------------------------------

    >>> print(grid.mesh)
    [[41.9 42.  10.9 11. ]
     [41.9 42.  10.8 10.9]
     [41.9 42.  10.7 10.8]
     ...
     [40.  40.1  9.2  9.3]
     [40.  40.1  9.1  9.2]
     [40.  40.1  9.   9.1]]


    We can refine any block, say at the 0th and 1st index, simply by:

    >>> grid.refine_mesh([0, 1], inplace=True)
    *** GRID UPDATED ***
    -------------------------------------
    GRID PARAMETERS
    Lonmin - Lonmax : 9.000 - 11.000
    Latmin - Latmax : 40.000 - 42.000
    Number of cells : 406
    Grid cells of 0.100° : 398
    Grid cells of 0.050° : 8
    -------------------------------------

    >>> print(grid.mesh)
    [[41.95 42.   10.9  10.95]
     [41.95 42.   10.95 11.  ]
     [41.9  41.95 10.9  10.95]
     ...
     [40.   40.1   9.2   9.3 ]
     [40.   40.1   9.1   9.2 ]
     [40.   40.1   9.    9.1 ]]


    Note that the size of the two blocks defined at the first two 
    rows of the above have been halved.

    To display the grid, use the :meth:`plot` method
    >>> grid.plot(projection='Mercator')
    """
    def __init__(self, cell_size, latmin=None, lonmin=None, latmax=None, 
                 lonmax=None, verbose=True):
        self.verbose = verbose
        if isinstance(cell_size, Iterable):
            dlon, dlat = cell_size
            cell_size = dlat
        else:
            dlon = dlat = cell_size
        self.mesh = self.create_mesh(dlon,
                                     dlat,
                                     latmin=latmin, 
                                     latmax=latmax, 
                                     lonmin=lonmin,
                                     lonmax=lonmax)
        self.latmin = np.min(self.mesh[:,0])
        self.lonmin = np.min(self.mesh[:,2])
        self.latmax = np.max(self.mesh[:,1])
        self.lonmax = np.max(self.mesh[:,3])

        self.refined = 0
        self.ncells_per_level = Counter({0 : self.mesh.shape[0]})
        self.cell_size_per_level = {0 : cell_size}
        if self.verbose:
            print(self)
            

    def create_mesh(self, dlon, dlat, latmin=None, latmax=None, lonmin=None, 
                    lonmax=None):
        """ Creates the regular grid.
        
        Parameters
        ----------
        dlon, dlat : float
            Longitudinal and latitudinal steps characterizing the grid
            
        latmin, lonmin, latmax, lonmax : float, optional
            Boundaries (in degrees) of the grid
        
        Returns
        -------
        grid : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2. The grid is rounded to the 4rd decimal digit,
            for improved numerical stability in most applications
        """
        latmin = -90 if latmin is None else latmin
        latmax = +90 if latmax is None else latmax
        lonmin = -180 if lonmin is None else lonmin
        lonmax = +180 if lonmax is None else lonmax
        
        mesh = []
        for lat1 in np.arange(latmin, latmax, dlat)[::-1]:
            lat2 = lat1 + dlat
            for lon1 in np.arange(lonmin, lonmax, dlon):
                lon2 = lon1 + dlon
                mesh.append([lat1, lat2, lon1, lon2])
        return np.round(mesh, 4)
















