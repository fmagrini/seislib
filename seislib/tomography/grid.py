#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fabrizio Magrini
@email1: fabrizio.magrini@uniroma3.it
@email2: fabrizio.magrini90@gmail.com
"""

from math import radians, degrees
from math import cos, pi, asin, sin
from math import sqrt
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from seislib.plotting import add_earth_features
SQUARE_DEGREES = 41252.961249419277010
FOUR_PI = 4 * pi


class EqualAreaGrid():

    
    def __init__(self, cell_size, latmin=None, lonmin=None, latmax=None, 
                 lonmax=None, verbose=True):
        """
        Parameters
        ----------
        cell_size : int
            Size of each side of the equal-area grid
            
        latmin, lonmin, latmax, lonmax : float, optional
            Boundaries (in degrees) of the grid
            
        verbose : bool
            If True, information about the grid will be displayed. Default is 
            True
        """
        
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
            If True, the number of times that the grid has been refined is 
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

    
    def grid_parameters(self, nrings):
        """
        Computes the grid parameters (number of cells, cells area, cells side, 
        longitude span as a function of latitude). [For more detail, refer to
        Malkin 2019]
    
        
        Parameters
        ----------
        nrings : int
            number of latitudinal rings used to subdivide the Earth.
            
            
        Returns
        -------
        ncells : int
            Number of grid cells
            
        cell_side : float
            Size of the cells' sides (corresponding to the sqrt of cell_area)
            
        lon_span : ndarray
            Longitudinal span the grid cells in each latitudinal ring
            
        
        References
        ----------
        Malkin 2019, A new equal-area isolatitudinal grid on a spherical 
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
        grid parameters associated with the best parameterization. (See the
        method `grid_parameters`)
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
        span as a function of latitude. [For more detail, refer to Malkin 2019]
        
        
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
        Malkin 2019, A new equal-area isolatitudinal grid on a spherical 
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
    
    
    def set_boundaries(self, latmin, latmax, lonmin, lonmax, mesh=None, 
                       inplace=True):
        """ Restricts the mesh to the required boundaries
        
        Parameters
        ----------
        latmin, latmax, lonmin, lonmax : float
            Boundaries of the new mesh. Their units should be consistent with
            those of the mesh
            
        mesh : ndarray (n, 4), optional
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2. If None, the mesh stored in the EqualAreaGrid 
            instance (self.mesh) is used (default is None)
            
        inplace : bool
             If True, the mesh stored in the EqualAreaGrid instance (self.mesh) 
             is modified permanently (default is False)
             
         
        Returns
        -------
        None, if inplace is True, else returns the modified mesh
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
             If True, the mesh stored in the EqualAreaGrid instance (self.mesh) 
             is modified permanently (default is False).
         
        Returns
        -------
        None, if inplace is True, else returns the indexed mesh.
        """

        mesh = self.mesh[indexes]
        if inplace:
            return self.update_grid_params(mesh, refined=False)
        return mesh
        
    
    def parallels_first_pixel(self, mesh=None):
        """ 
        Generator function yielding the indexes at which a change in the parallel 
        coordinates is found.
        
        Parameters
        ---------
        mesh : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2. If None, the mesh stored in the EqualAreaGrid 
            instance (self.mesh) is used (default is None).
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
            
    
    @classmethod
    def pixel_index(cls, lat, lon, mesh):
        """ Returns the mesh index corresponding with the coordinates (lat, lon)
        
        Parameters
        ----------
        lat, lon : float
            Geographic coordinates. Their units should be consistent with those
            of the mesh
            
        mesh : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2.
        
        Returns
        -------
        idx : int
            Mesh index corresponding with (lat, lon)
        """
        
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
    
    
    def refine_mesh(self, ipixels, mesh=None, inplace=False):
        """ Halves the size of the specified pixels
        
        Parameters
        ----------
        ipixels : list or ndarray
            mesh indexes to refine
            
        mesh : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2. If None, the mesh stored in the EqualAreaGrid 
            instance (self.mesh) is used (default is None)
            
        inplace : bool
             If True, the mesh stored in the EqualAreaGrid instance (self.mesh) 
             is modified permanently (default is False).
             
        Returns
        -------
        None, if inplace is True, else returns the modified mesh.
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
        """ Called by `refine_mesh` to create new pixels
        
        Parameters
        ----------
        newmesh : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2
            
        i_newpixel : int
            mesh index to refine
            
        parallel1, parallel2, meridian1, meridian2 : float
             Geographic coordinates of the boundaries of the pixel to refine. 
             Their units should be consistent with those of the mesh
             
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
    
    
    def plot(self, ax=None, mesh=None, projection='Mercator',  show=True, 
             map_boundaries=None, bound_map=True, oceans_color='aqua', 
             lands_color='coral', scale='110m'):
        """
        Plots the (adaptive) equal-area mesh relying on mpl_toolkits.basemap.Basemap
        
        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If None a new figure and GeoAxesSubplot instance is created
            
        mesh : ndarray (n, 4)
            Array containing n pixels bounded by parallel1, parallel2, 
            meridian1, meridian2. If None, the mesh stored in the EqualAreaGrid 
            instance (self.mesh) is used (default is None)
            
        projection : str
            Name of the geographic projection used to create the GeoAxesSubplot.
            (Visit the cartopy website for a list of valid projection names.)
            If ax is not None, `projection` is ignored. Default is 'Mercator'
                        
        show : bool
            If True (default), the map will be showed once generated. Otherwise
            a cartopy.mpl.geoaxes.GeoAxesSubplot instance is returned
          
        map_boundaries : list or tuple of floats, shape (4,), optional
            Lonmin, lonmax, latmin, latmax (in degrees) defining the extent of
            the map
            
        bound_map : bool
            If True, the map boundaries will be automatically determined.
            Ignored if map_boundaries is not None
              
        oceans_color, continents_color : str
            Color of oceans and continents in the map. They should be valid 
            matplotlib colors (see matplotlib.colors doccumentation for more 
            details) or be part of cartopy.cfeature.COLORS
        
        Returns
        -------
        None, if `show` is True, else a GeoAxesSubplot instance
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
                        
        def drawmeridians(mesh, ax):
            drawn_meridians = defaultdict(set)
            # Draw meridians
            for lat1, lat2, lon1, lon2 in mesh:
                ax.plot([lon1, lon1], [lat1, lat2], color='k', linestyle='-', 
                        lw=1, zorder=100, transform=ccrs.Geodetic())
                drawn_meridians[(lat1, lat2)].add(lon1)
                if lon2 not in drawn_meridians[(lat1, lat2)]:
                    ax.plot([lon2, lon2], [lat1, lat2], color='k', linestyle='-', 
                            lw=1, zorder=100, transform=ccrs.Geodetic())
                    drawn_meridians[(lat1, lat2)].add(lon2)
            return
        
        def drawparallels(mesh, ax):
            parallels = get_parallels(mesh)
            for parallel in parallels:
                for lon1, lon2 in parallels[parallel]:
                    ax.plot([lon1, lon2], [parallel, parallel], color='k', 
                            linestyle='-', lw=1, zorder=100, 
                            transform=ccrs.PlateCarree())
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

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection=eval('ccrs.%s()'%projection))
            add_earth_features(ax,
                               oceans_color=oceans_color,
                               lands_color=lands_color,
                               scale=scale,
                               edgecolor='none')
        drawmeridians(mesh, ax)
        drawparallels(mesh, ax)
        
        if map_boundaries is None and bound_map:
            map_boundaries = get_map_boundaries(mesh)
        if map_boundaries is not None:
            ax.set_extent(map_boundaries, ccrs.PlateCarree())     
        # Draw parallels
        if show:
            plt.show()
        return ax
        
