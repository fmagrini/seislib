#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fabrizio Magrini
@email1: fabrizio.magrini@uniroma3.it
@email2: fabrizio.magrini90@gmail.com
"""

import pickle
import warnings
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
#from scipy.linalg import pinvh, lstsq
from obspy.geodetics import gps2dist_azimuth
from seislib.tomography import EqualAreaGrid
from seislib import colormaps as scm
import seislib.plotting as plotting
from seislib.clib._tomography import _compile_coefficients
from seislib.clib._tomography import _refine_parameterization
from seislib.clib._tomography import _raypaths_per_pixel
from seislib.clib._tomography import _derivatives_lat_lon
from seislib.clib._tomography import _select_parameters
eps = np.finfo(np.float64).eps


class SeismicTomography:
    """ 
    Class to obtain velocity maps using a linearized inversion based on the
    ray theory (infinite frequency approximation)
    
    
    Attributes
    ----------
    verbose : bool
        If True (default), information about the grid and data will be displayed in
        console
        
    grid : seislib.EqualAreaGrid
        Instance of seislib.EqualAreaGrid, corresponding to an equal-area
        parameterization
    
    
    Additional Attributes (available after calling `add_data`)
    ----------------------------------------------------------
    lonmin_data, latmin_data, lonmax_data, latmax_data : float
        Minimum and maximum values of latitude and longitude in the data
    
    data_coords : ndarray of shape (n, 4)
        Lat1, lon1, lat2, lon2 of the pairs of receivers (or epicenter-receiver),
        in degrees (-180<lon<180, -90<lat<90)
        
    velocity : ndarray of shape (n,) (in m/s)
        Velocity measured between each pair of stations or epicenter-station
        
    refvel : float (in m/s)
        Reference velocity, used in the inversion
        
        
    Additional Attributes (available after calling `compile_coefficients`)
    ----------------------------------------------------------------------
    A : ndarray of shape (n, m)
        Where n is the number of data, m is the number of parameters (i.e.,
        number of rows in self.grid.mesh)
            
    
    Methods
    -------
    save(path)
        Saves the SeismicTomography instance to the specified path in the pickle
        compressed format
    
    update_data_info(refvel=None)
        Updates the information on minimum / maximum longitude and latitude of
        the data and the reference velocity
    
    add_data(data=None, refvel=None, src=None, **kwargs)
        Loads the data to be used for tomography
        
    compile_coefficients(refine=False, coeff_matrix=None, keep_empty_cells=True)
        Compiles the matrix of coefficients (`A`) used to perform the inversion. 
    
    reduce_measurements(latmin=-90, latmax=90, lonmin=-180, lonmax=180)
        Any measurement not intersecting the specified region is removed from
        the data
        
    solve(A=None, slowness=None, refvel=None, mesh=None, ndamp=0, rdamp=0)
        Method for solving the (regularized) inverse problem A·x = b
        
    lcurve(A=None, slowness=None, refvel=None, mesh=None, damping='roughness', 
           n=20, damp_min=0.1, damp_max=5, logspace=True, plot=True)
        L-curve analysis. The function calls iteratively the `solve` method
    
    resolution_test(mesh, velocity_model, noise=0, ndamp=0, rdamp=0)
        Resolution test on an arbitrary input model
        
    checkerboard_test(kx, ky, latmin=None, latmax=None, lonmin=None, lonmax=None, 
                      cell_size=5, anom_amp=0.1, refvel=None, noise=0, ndamp=0, 
                      rdamp=0)
        Resolution test, known as "checkerboard test"
        
    spike_test(x0, y0, sigma_x, sigma_y, latmin=None, latmax=None, lonmin=None, 
               lonmax=None, cell_size=5, anom_amp=0.1, refvel=None, noise=0, 
               ndamp=0, rdamp=0)
        Resolution test, known as "spike test"
        
    plot_rays(ax=None, show=True, stations_color='r', gc_paths_color='k',
              oceans_color='water', lands_color='land', edgecolor='k',
              alpha_stations=None, alpha_paths=0.3, projection='Mercator', 
              resolution='110m', map_boundaries=None, bound_map=True, **kwargs)
        Utility function to display the great circle paths associated with pairs
        of data coordinates
        
    plot_colored_rays(ax=None, show=True, cmap=scm.roma, vmin=None, vmax=None, 
                      stations_color='k', oceans_color='lightgrey', 
                      lands_color='w', edgecolor='k', stations_alpha=None, 
                      paths_alpha=None, projection='Mercator', resolution='110m',
                      map_boundaries=None, bound_map=True, paths_width=1, 
                      colorbar=True, cbar_dict={}, **kwargs):
        Utility function to display the great-circle paths associated with pairs
        of data coordinates, colored according to their respective measurements


    Class Methods
    -------------
    load(path):
        Loads the pickle file at the specified path and returns the associated
        instance of SeismicTomography

    delay_to_velocity(lat1, lon1, lat2, lon2, delay, refvel)
        Converts a delay to velocity

    velocity_to_delay(lat1, lon1, lat2, lon2, velocity, refvel)
        Converts a velocity to delay

    gc_distance(lat1, lon1, lat2, lon2)
        Calculates the great circle distance (in m) between coordinate points 
        (in degrees)
        
    azimuth_backazimuth(lat1, lon1, lat2, lon2)
        Calculates the azimuth and backazimuth (in degrees) between coordinate 
        points (in degrees)

    plot_map(mesh, c, ax=None, projection='Mercator', map_boundaries=None, 
             bound_map=True, colorbar=True, show=True, style='colormesh', 
             add_features=False, resolution='110m', cbar_dict={}, **kwargs)
        Utility function to display the lateral variations of some quantity on 
        a equal-area grid
        
    colormesh(mesh, c, ax, **kwargs):
        Adaptation of matplotlib.pyplot.pcolormesh to the (adaptive) equal area 
        grid
        
    contourf(mesh, c, ax, smoothing=None, **kwargs)
        Adaptation of matplotlib.pyplot.contourf to the (adaptive) equal area 
        grid
        
    contour(mesh, c, ax, smoothing=None, **kwargs)
        Adaptation of matplotlib.pyplot.contour to the (adaptive) equal area 
        grid


    Static Methods
    --------------
    checkerboard(ref_value, kx=10, ky=10, latmin=None, latmax=None, lonmin=None, 
                 lonmax=None, anom_amp=0.1)
        Checkerboard-like pattern on Earth

    spike(ref_value, x0, y0, sigma_x=1, sigma_y=1, anom_amp=0.1)
        Spike-like pattern on Earth, created via a 3-D Gaussian
        

    Example
    -------
    The following will calculate a phase-velocity map at a given period, say
    10 s, based on inter-station measurements of seismic ambient noise. (An 
    analogous procedure can be followed based on the dispersion curves obtained
    from teleseismic earthquakes - see documentation on seislib.eq.EQVelocity).
    In practice, our data consist of a ndarray of shape (n, 5), where n is the
    number of inter-station measurements of phase velocity (extracted, for
    example, via seislib.an.AmbientNoiseVelocity), and the five columns consist
    of lat1 (°), lon1 (°), lat2 (°), lon2 (°), velocity (m/s), respectively.
    (-180<lon<180, -90<lat<90). This matrix has been saved to /path/to/data.txt
    
    We will parameterize the resulting map with 2° grid cells, that will be
    iteratively halved in size up to a maximum number of 2 times, to reach a
    minimum size of 0.5° in the areas of the map characterized by a relatively
    large number of measurements. (This refinement of the parameterization can 
    be carried out an arbitrary number of times.)
    
    First, we need to initialize the SeismicTomography instance and load our
    data into memory. We do it as
    
        tomo = SeismicTomography(cell_size=2)
        tomo.add_data(src='/path/to/data')
        
    As you do that, something similar will be printed in console:
        >>>     -------------------------------------
                Optimal grid found in 46 iterations
                -------------------------------------
                -------------------------------------
                GRID PARAMETERS
                Lonmin - Lonmax : -180.000 - 180.000
                Latmin - Latmax : -90.000 - 90.000
                Number of cells : 10312
                Grid cells of 2.000° : 10312
                -------------------------------------
                DATA PARAMETERS
                Lonmin - Lonmax data : -124.178 - -69.292
                Latmin - Latmax data : 23.686 - 48.470
                Number of measurements : 1064
                Source : /path/to/data
                -------------------------------------
    
    ---------------------------------------------------------------------------
    Note1: we could have directly passed the matrix of the data without
        loading it from disk as tomo.add_data(data=your_matrix)
    Note2: you can add how many data sets you wish, simply call `add_data` 
        passing different file names (or matrixes) sequentially. The data
        information will be automatically updated
    Note3: to display general information on the data, type print(tomo). To
        display general information on the parameterization, type 
        print(tomo.grid). (See also the documentation on seislib.EqualAreaGrid)
    ---------------------------------------------------------------------------
    
    Now we can restrict the boundaries of the (global) equal-area 
    parameterization to the minimum and maximum latitude and longitude spanned
    by our data:
        
        tomo.grid.set_boundaries(latmin=tomo.latmin_data, 
                         latmax=tomo.latmax_data,
                         lonmin=tomo.lonmin_data,
                         lonmax=tomo.lonmax_data)
        
        >>>     *** GRID UPDATED ***
                -------------------------------------
                GRID PARAMETERS
                Lonmin - Lonmax : -126.234 - -66.614
                Latmin - Latmax : 22.006 - 50.005
                Number of cells : 324
                Grid cells of 2.000° : 324
                -------------------------------------
    
    Having done so, everything is ready to calculate the coefficients of the
    `A` matrix (i.e., of the data kernel) [for theoretical detail, see, e.g.,
    Boschi & Dziewonski 1999], and to refine the parameterization up to two
    times in the areas characterized by a relatively high density of 
    measurements. We will define such regions (i.e., model parameters) as those
    where there are at least 150 inter-station great-circle paths intersecting
    them. In doing so, we will remove the grid cells of 2° that are not
    intersected by at least one great-circle path (see the argument 
    `keep_empty_cells`).
        
        tomo.compile_coefficients(keep_empty_cells=False)
        tomo.refine_parameterization(hitcounts=150, 
                                     keep_empty_cells=True)    
        tomo.refine_parameterization(hitcounts=150, 
                                     keep_empty_cells=True)
        
        >>>     *** GRID UPDATED ***
                -------------------------------------
                GRID PARAMETERS
                Lonmin - Lonmax : -126.142 - -66.614
                Latmin - Latmax : 22.006 - 50.005
                Number of cells : 202
                Grid cells of 2.000° : 202
                -------------------------------------
                *** GRID UPDATED ***
                -------------------------------------
                GRID PARAMETERS
                Lonmin - Lonmax : -126.142 - -66.614
                Latmin - Latmax : 22.006 - 50.005
                Number of cells : 808
                Grid cells of 2.000° : 0
                Grid cells of 1.000° : 808
                -------------------------------------
                *** GRID UPDATED ***
                -------------------------------------
                GRID PARAMETERS
                Lonmin - Lonmax : -126.142 - -66.614
                Latmin - Latmax : 22.006 - 50.005
                Number of cells : 2284
                Grid cells of 2.000° : 0
                Grid cells of 1.000° : 316
                Grid cells of 0.500° : 1968
                -------------------------------------
                
    To obtain the velocity map, we now need to carry out the inversion. We will
    apply a roughness damping regularization, using a roughness coefficient
    equal to 3e-3 (to select a proper roughness damping, check the method
    `lcurve`). We then plot the retrieved velocity.
        
        c = 1 / tomo.solve(rdamp=3e-3)
        tomo.plot_map(tomo.grid.mesh, c)
        
    (Note that the `solve` method returns slowness, hence we took the inverse
    of the solution.) A checkerboard test, to verify the ability of our data to 
    resolve the lateral differences in the study area, can simply be performed
    and visualized by
    
        restest = tomo.checkerboard_test(kx=6,
                                         ky=6,
                                         latmin=tomo.grid.latmin,
                                         latmax=tomo.grid.latmax,
                                         lonmin=tomo.grid.lonmin,
                                         lonmax=tomo.grid.lonmax,
                                         cell_size=0.5,
                                         anom_amp=0.1,
                                         noise=0,
                                         rdamp=3e-3)
        
        input_model = 1 / restest['synth_model']
        input_mesh = restest['mesh']
        retrieved_model = 1 / restest['retrieved_model']
        
        tomo.plot_map(tomo.grid.mesh, retrieved_model)
        tomo.plot_map(input_mesh, input_model)

    ---------------------------------------------------------------------------
    Note: the resolution test (check also the methods `resolution_test` and
        `spike_test`) should be performed after (and only after) the coefficients
        in the data-kernel (tomo.A matrix) have been complied, i.e., after
        having called the method `compile_coefficients` (and eventually
        `refine_parameterization`, which updates the A matrix)
    ---------------------------------------------------------------------------
    
    Finally, we show how we can obtain a heat map of the raypaths intersecting
    each model parameter:
        
        raypaths = tomo.raypaths_per_pixel()
        img, cb = tomo.plot_map(mesh=tomo.grid.mesh, 
                                c=raypaths, 
                                cmap='cividis', 
                                style='contourf', 
                                levels=20,
                                show=False)
        cb.set_label('Raycounts')
        
    ---------------------------------------------------------------------------
    Note1: when the argument `show` is False, the image and colorbar will be
        returned. That allow us to change the label on the colorbar.
    Note2: for more control on the plots, consider creating your own instance 
        of GeoAxesSubplot (see cartopy documentation) before calling the
        plot_map function (or, alternatively, the methods `colormesh`, `contour`,
        or `contourf` of SeismicTomography). For example:
            
        from seislib.plotting import make_colorbar
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
        ax.coastlines(resolution='50m', color='k', lw=1, zorder=100)
        img = tomo.colormesh(mesh=tomo.grid.mesh, 
                             c=raypaths, 
                             ax=ax, 
                             cmap='cividis', 
                             shading='flat', 
                             edgecolors='face')
        map_boundaries = (tomo.grid.lonmin, tomo.grid.lonmax, 
                          tomo.grid.latmin, tomo.grid.latmax)
        ax.set_extent(map_boundaries, ccrs.PlateCarree())  
        cb = make_colorbar(ax, img, orientation='horizontal')
        cb.set_label(label='Raycounts', labelpad=10, fontsize=22)
    ---------------------------------------------------------------------------
    
    
    
    References
    ----------
    Boschi & Dziewonski 1999, High- and low-resolution images of the Earth's 
        mantle: Implications of different approaches to tomographic modeling,
        JGR
    
    """ 
    
    def __init__(self, cell_size=4, latmin=None, lonmin=None, latmax=None, 
                 lonmax=None, verbose=True):
        """
        Parameters
        ----------
        cell_size : int
            Size of each side of the equal-area grid
            
        latmin, lonmin, latmax, lonmax : float, optional
            Boundaries (in degrees) of the grid
            
        verbose : bool
            If True, information about the grid and data will be displayed in
            console. (Default is True)
        """
          
        self.verbose = verbose
        self.grid = EqualAreaGrid(cell_size=cell_size,
                                  latmin=latmin,
                                  lonmin=lonmin, 
                                  latmax=latmax,
                                  lonmax=lonmax,
                                  verbose=verbose)

               
    def __repr__(self):
        return str(self)
              
    
    def __str__(self):
        string = 'DATA PARAMETERS\n'
        if not 'data_coords' in self.__dict__:
            return string + 'No data available'
        string += 'Lonmin - Lonmax data : %.3f - %.3f\n'%(self.lonmin_data,
                                                          self.lonmax_data)
        string += 'Latmin - Latmax data : %.3f - %.3f\n'%(self.latmin_data,
                                                          self.latmax_data)
        string += 'Number of measurements : %s\n'%self.velocity.size
        string += 'Source : %s\n'%self.src
        string += '-------------------------------------\n'
        return string
    
    
    def save(self, path):
        """ 
        Saves the SeismicTomography instance to the specified path in the pickle
        compressed format.
        
        Parameters
        ----------
        path : str
            Absolute path of the resulting file
        """
        
        if not path.endswith('.pickle'):
            path += '.pickle'    
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
            
    @classmethod
    def load(cls, path):
        """
        Loads the pickle file at the specified path and returns the associated
        instance of SeismicTomography.
        
        Parameters
        ----------
        path : str
            Absolute path of the file
        """
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    
    def update_data_info(self, refvel=None):
        """ 
        Updates the information on minimum / maximum longitude and latitude of
        the data and the reference velocity
        
        Parameters
        ----------
        refvel : float, optional
            If not None, the reference velocity is updated along with minimum
            and maximum longitudes and latitudes
        """
        
        self.refvel = np.mean(self.velocity) if refvel is None else refvel
        self.latmin_data = min(np.min(self.data_coords[:, (0, 2)], axis=0))
        self.lonmin_data = min(np.min(self.data_coords[:, (1, 3)], axis=0))
        self.latmax_data = max(np.max(self.data_coords[:, (0, 2)], axis=0))
        self.lonmax_data = max(np.max(self.data_coords[:, (1, 3)], axis=0))
        if self.verbose:
            print(self)
    
    
    def add_data(self, data=None, refvel=None, src=None, **kwargs):
        """Loads the data to be used for tomography.

        If the argument `data` isn't passed in, the data are loaded from `src`.
        If the instance of SeismicTomography already contains data, the new data
        will be added (concatenated) to the previously loaded ones.
        
        NOTE: In the current implementation, the 6th column of the data (i.e., 
        the standard deviation), even if present, is not employed in the 
        inversion (see SeismicTomography.solve). This behaviour will be changed
        in a future release.
        

        Parameters
        ----------
        data : ndarray, optional
            The array must be of shape (n, 5) or (n, 6). Cols 1 to 4 must contain
            the coordinates of the measurements (lat1, lon1, lat2, lon2), col 5
            the velocity, and col 6 the standard deviation (optional). 
            (-180<lon<180, -90<lat<90, velocity in m/s)
            
        refvel : float, optional (in m/s)
            Reference velocity. If None, the average velocity is used as reference
            
        src : str, optional
            If `data` is None, `src` is used to load the data. The file extension
            must be .txt
            
        **kwargs : optional
            Arguments to be passed to numpy.loadtxt
        
        
        Returns
        -------
        None
            The data are stored in self.__dict__ and can be accessed through 
            self.data_coords, self.velocity, self.refvel, self.std (if present), 
            self.latmin_data, self.lonmin_data, self.latmax_data, 
            self.lonmax_data
        """

        if data is None:
            data = np.loadtxt(src, **kwargs)
        else:
            data = data.astype(np.float64)
        if 'velocity' not in self.__dict__:
            self.src = 'Unknown' if src is None else src
            self.data_coords = data[:, :4]
            self.velocity = data[:, 4]
            if data.shape[1] == 6:
                self.std = data[:, 5] 
        else:
            self.src += ' + Unknown' if src is None else ' + %s'%src
            if data.shape[1] == 6:
                if 'std' in self.__dict__:
                    self.std = np.concatenate((self.std, data[:, 5]))
                else:
                    self.std = np.concatenate((np.zeros(self.velocity.shape),
                                               data[:, 5]))
            else:
                if 'std' in self.__dict__:
                    self.std = np.concatenate((self.std, 
                                               np.zeros(self.velocity.shape)))
            self.data_coords = np.row_stack((self.data_coords, data[:, :4]))
            self.velocity = np.concatenate((self.velocity, data[:, 4]))
        
        self.update_data_info(refvel=refvel)
        

    @classmethod
    def delay_to_velocity(cls, lat1, lon1, lat2, lon2, delay, refvel):
        """ Converts a delay to velocity
        
        A negative delay dt corresponds to faster velocity with respect to the 
        reference velocity v0:
            dt = x/v - x/v0
            v = x / (dt + x/v0),
        where x denotes distance and v the observed velocity.
        
        Parameters
        ----------
        lat1, lon1, lat2, lon2 : float or array-like of shape (n,)
            Coordinates (in radians) associated with the dalay
            
        delay : float or array-like of shape (n,)
            Measured delay (in seconds)
            
        refvel : float
            Reference velocity used to calculate the delay (in m/s)
            
            
        Returns
        -------
        float or array-like of shape (n,)
        """
        
        dist = np.asarray(cls.gc_distance(lat1, lon1, lat2, lon2))
        return dist / (delay + dist/refvel)
        
    
    @classmethod
    def velocity_to_delay(cls, lat1, lon1, lat2, lon2, velocity, refvel):
        """ Converts a velocity to delay
        
        Velocities (v) faster than the reference (v0) correspond to negative 
        delays (dt):
            dt = x/v - x/v0
        where x denotes distance.
        
        Parameters
        ----------
        lat1, lon1, lat2, lon2 : float or array-like of shape (n,)
            Coordinates (in radians) associated with the velocity
            
        velocity : float or array-like of shape (n,)
            Measured velocity (in m/s)
            
        refvel : float
            Reference velocity to be used to calculate the delay (in m/s)
            
            
        Returns
        -------
        float or array-like of shape (n,)
        """   
        
        dist = np.asarray(cls.gc_distance(lat1, lon1, lat2, lon2))
        return dist/velocity - dist/refvel
    
    
    @classmethod
    def gc_distance(cls, lat1, lon1, lat2, lon2):
        """ 
        Calculates the great circle distance (in m) between coordinate points 
        (in degrees). This function calls directly the obspy gps2dist_azimuth,
        it only extends its functionality through the numpy.vectorize decorator.
        
        Parameters
        ----------
        lat1, lon1, lat2, lon2 : float or array-like of shape (n,) (in degrees)
        
        
        Returns
        -------
        Great-circle distance (in m). If the input is an array (or list) of
        coordinates, an array of distances is returned
        """
        
        func = np.vectorize(gps2dist_azimuth)
        return func(lat1, lon1, lat2, lon2)[0]
    
    
    @classmethod
    def azimuth_backazimuth(cls, lat1, lon1, lat2, lon2):
        """ 
        Calculates the azimuth and backazimuth (in degrees) between coordinate 
        points (in degrees). This function calls directly the obspy 
        gps2dist_azimuth, it only extends its functionality through the 
        numpy.vectorize decorator
        
        Parameters
        ----------
        lat1, lon1, lat2, lon2 : float or array-like of shape (n,) (in degrees)
        
        
        Returns
        -------
        tuple of shape (2,) or ndarray of shape (n, 2) : [azimuth, backazimuth]
        """
        
        func = np.vectorize(gps2dist_azimuth)
        return func(lat1, lon1, lat2, lon2)[1:]
    
    
    def compile_coefficients(self, refine=False, coeff_matrix=None, 
                             keep_empty_cells=True):
        """ 
        Compiles the matrix of coefficients (`A`) used to perform the inversion. 
        It calls the _compile_coefficients function, written in cython.
        
        
        Parameters
        ----------
        refine : bool
            should be True if the user has refined the parameterization of the
            mesh (by calling `refine_parameterization`). This allows to reduce
            computation time. (Default is False)
            
        coeff_matrix : if None, a new `A` will be created and stored in the
            SeismicTomography instance (self.A). `A` should be passed
            in after the parameterization refinement. (Default is None)
            
        keep_empty_cells : bool
            If False, cells intersected by no raypaths are removed
        
        
        Returns
        -------
        None 
            The SeismicTomography instance will be updated, so that `A` can be
            accessed typing self.A
        """
        
        mesh_latmax = np.radians(self.grid.latmax)
        mesh_lonmax = np.radians(self.grid.lonmax)
        mesh = np.radians(self.grid.mesh)
        data_coords = np.radians(self.data_coords)
        self.A = _compile_coefficients(data_coords=data_coords,
                                       mesh=mesh, 
                                       mesh_latmax=mesh_latmax, 
                                       mesh_lonmax=mesh_lonmax,
                                       refine=refine,
                                       coeff_matrix=coeff_matrix)
        if not keep_empty_cells:
            raycounts = self.raypaths_per_pixel()
            keep_pixels = np.flatnonzero(raycounts > 0).astype(np.int32)
            self.A = _select_parameters(self.A, keep_pixels)
            self.grid.select_cells(indexes=keep_pixels)
        
    
    def refine_parameterization(self, hitcounts=100, keep_empty_cells=True,
                                latmin=None, latmax=None, lonmin=None, lonmax=None):
        """ 
        Halves the dimension of the pixels intersected by a number of 
        raypaths >= hitcounts.        
        
        Parameters
        ----------
        hitcounts : int (default is 100)        
            Each parameter (grid cell) intersected by a number of raypaths
            equal or greater than this threshold is "refined", i.e., splitted
            in four equal-area sub-parameters
            
        keep_empty_cells : bool
            If False, cells intersected by no raypaths are removed
            
        lat1, lon1, lat2, lon2 : float (in degrees)
            If specified, only the region inside this boundary is considered
            for refining the parameterization. (-180<lon<180, -90<lat<90)


        Returns
        -------
        None
            The coordinates mesh stored in the SeismicTomography instance
            (self.grid.mesh) and the coefficients matrix (self.A) are updated
        """
        
        mesh = np.radians(self.grid.mesh)
        if any([latmin, latmax, lonmin, lonmax]):
            latmin = latmin if latmin is not None else -90
            latmax = latmax if latmax is not None else +90
            lonmin = lonmin if lonmin is not None else -180
            lonmax = lonmax if lonmax is not None else +180
            region_to_refine = np.radians([latmin, latmax, lonmin, lonmax])
        else:
            region_to_refine = None

        newmesh, A = _refine_parameterization(mesh, 
                                              self.A, 
                                              hitcounts=hitcounts,
                                              region_to_refine=region_to_refine)
        self.grid.update_grid_params(np.degrees(newmesh), refined=True)
        self.compile_coefficients(refine=True, coeff_matrix=A, 
                                  keep_empty_cells=keep_empty_cells)

    
    def raypaths_per_pixel(self):
        """ 
        Calculates the number of raypaths in each pixel of the mesh stored in
        the SeismicTomography instance (self.grid.mesh)
        
        Returns
        -------
        ndarray of dimension self.grid.mesh.shape[0]            
        """
        
        if 'A' not in self.__dict__:
            self.compile_coefficients()
        return np.asarray(_raypaths_per_pixel(self.A))
                    
    
    def reduce_measurements(self, latmin=-90, latmax=90, lonmin=-180, lonmax=180):
        """
        Any measurement not intersecting the specified region is removed from
        the data. Consider using this function in presence of large datasets,
        for filtering out those rays that do not constrain the parameters (i.e.,
        the grid cells) of interest in the inversion.
        
        Parameters
        ----------
        lat1, lon1, lat2, lon2 : float (in degrees)
            Boundaries of the region of interest. (-180<lon<180, -90<lat<90)
        
        
        Returns
        -------
        None
            The instance of SeismicTomography is updated with the new data

        """
        
        latmin = latmin if latmin is not None else -90
        latmax = latmax if latmax is not None else +90
        lonmin = lonmin if lonmin is not None else -180
        lonmax = lonmax if lonmax is not None else +180
        mesh = self.grid.mesh
        ilat = np.flatnonzero((mesh[:,0]<=latmax) & (mesh[:,1]>=latmin))
        ilon = np.flatnonzero((mesh[:,2]<=lonmax) & (mesh[:,3]>=lonmin))
        ipixels = np.intersect1d(ilat, ilon)
        if 'A' not in self.__dict__:
            self.compile_coefficients(keep_empty_cells=True)
        idata = [i for i in range(self.A.shape[0]) if np.any(self.A[i, ipixels])]
        self.data_coords = self.data_coords[idata]
        self.velocity = self.velocity[idata]
        del self.A
        self.update_data_info()
    
    
    def solve(self, A=None, slowness=None, refvel=None, mesh=None, ndamp=0, 
              rdamp=0):
        """
        Method for solving the (regularized) inverse problem A·x = b, where A is
        the matrix of coefficients, x is the sought model of slowness, b is the
        array of measurements expressed in slowness. The method first computes
        the array of residuals r = b - A·x0, where x0 denotes the reference 
        slowness model passed to the function as `refvel` (and converted to
        slowness prior to the inversion). The residuals are then used to retrieve
        the least square solution x* = (A'·A + µ^2·I + ρ^2·I·G)^-1 · A'·r,
        where A' is the transpose of A, I denotes the identity matrix, and
        µ and ρ are norm and roughness damping, respectively. The square 
        matrix G has the same dimensions of the model x. x* is then used to
        compute the final model x = x0 + x*.
        
        Parameters
        ----------
        A : ndarray, shape (m, n), optional
            Matrix of the coefficients. If None (default), the `A` matrix
            stored in the SeismicTomography instance is used
            
        slowness : ndarray, shape (m,), optional
            Inverse of the velocity measurements. Corresponds to b in the 
            equation Ax = b. If None (default), the velocity stored in the 
            SeismicTomography instance is used to compute slowness
            
        refvel : float, optional
            Reference velocity (in m/s) used for the initial guess of the velocity
            model. If provided, should stabilize the inversion and make it 
            converge faster. If None (default), the reference velocity stored
            in the SeismicTomography instance is used
            
        mesh : ndarray, shape (n, 4), optional
            Grid cells in seislib format, consisting of n pixels described by 
            lat1, lat2, lon1, lon2 (in degrees). If None (default), the mesh 
            stored in the SeismicTomography instance is used
            
        ndamp : float
            Extent of norm damping (default is zero)
            
        rdamp : float
            Extent of roughness damping (default is zero)
            
        
        Returns
        -------
        ndarray of shape (n,)
            Least-square solution (slowness, in s/m)
        """
        
        def norm_damping(mesh, damp):
            I = np.identity(mesh.shape[0])
            I *= np.diff(mesh[:, :2])**2
            return scipy.sparse.csr_matrix(I * damp**2)
        
        def roughness_damping(mesh, damp):
            G_lat, G_lon = _derivatives_lat_lon(mesh)
            G_lat = scipy.sparse.csr_matrix(G_lat)
            G_lon = scipy.sparse.csr_matrix(G_lon)
            return damp**2 * (G_lat.T.dot(G_lat) + G_lon.T.dot(G_lon))
                        
        A = A if A is not None else self.A
        if not isinstance(A, scipy.sparse.csr_matrix):
            A = scipy.sparse.csr_matrix(A)
        slowness = slowness if slowness is not None else 1/self.velocity
        refvel = refvel if refvel is not None else self.refvel
        mesh = np.radians(mesh) if mesh is not None else np.radians(self.grid.mesh)
        x0 = np.ones(A.shape[1]) * 1/refvel
        residuals = slowness - A.dot(x0)
        lhs = A.T @ A
        if ndamp > 0:
            lhs += norm_damping(mesh=mesh, damp=ndamp)
        if rdamp > 0:
            lhs += roughness_damping(mesh=mesh, damp=rdamp)
        rhs = A.T.dot(residuals)      
        return x0 + scipy.linalg.solve(lhs.todense(), rhs)

    
    def lcurve(self, A=None, slowness=None, refvel=None, mesh=None, damping='roughness', 
               n=20, damp_min=1e-5, damp_max=1e-1, logspace=True, show=True):
        """ L-curve analysis. The function calls iteratively the `solve` method
        
        Parameters
        ----------
        A : ndarray, shape (m, n), optional
            Matrix of the coefficients. If None (default), the `A` matrix
            stored in the SeismicTomography instance is used
            
        slowness : ndarray, shape (m,), optional
            Inverse of the velocity measurements. Corresponds to b in the 
            equation Ax = b. If None (default), the velocity stored in the 
            SeismicTomography instance is used to compute slowness
            
        refvel : float, optional
            Reference velocity (in m/s) used for the initial guess of the velocity
            model. If provided, should stabilize the inversion and make it 
            converge faster. If None (default), the reference velocity stored
            in the SeismicTomography instance is used
            
        mesh : ndarray, shape (n, 4), optional
            Grid cells in seislib format, consisting of n pixels described by 
            lat1, lat2, lon1, lon2. If None (default), the mesh stored in the 
            SeismicTomography instance is used
            
        damping : str
            Damping criterion used in the inversion. If "norm", a norm damping
            is applied, otherwise the inversion is regularized using roughness
            damping (default)
            
        n : int
            Number of inversions performed in the analysis
            
        damp_min : float
            Minimum damping term (default is 1e-5)
            
        damp_max : float
            Maximum damping term (default is 1e-1)
            
        logspace : bool
            If True (default), the damping range is defined to be linearly 
            spaced in logarithmic scale between `damp_min` and `damp_max`, i.e.,
            numpy.logspace(np.log10(damp_min), np.log10(damp_ax), n). False 
            corresponds to numpy.linspace(damp_min, damp_max, n)
            
        show : bool
            If True (default), the result of the analysis is displayed
        
        Returns
        -------
        damp_range : ndarray (n,)
            Damping range used in the analysis
            
        result : tuple of shape (2,)
            (i) Residual norm |A·x - b| and (ii) norm of the damped solution 
            |G·x|, where G = I (identity matrix) if the L-curve was performed 
            using a norm damping (`ndamp`>0), or the roughness operator if the 
            roughness damping was used instead (`rdamp`>0)
        """
        
        def plot(dampings, models, ylabel, scale):
            plt.figure(figsize=(10, 8))
            for damping, (residual_norm, yaxis) in zip(dampings, models):
                plt.plot(residual_norm, yaxis, 'ro')
                plt.text(residual_norm - residual_norm*2e-3, 
                         yaxis - yaxis*2e-3, 
                         s=r'$%s$'%plotting.scientific_label(damping, 1), 
                         va='top', 
                         ha='right', 
                         fontsize=11,
                         color='r')
            plt.xlabel(r'$||$ A$\cdot$x - b $||$')
            plt.ylabel(ylabel, fontsize=20, labelpad=10)
            plt.yscale(scale)
            plt.xscale(scale)
            plt.show()
        
        A = A if A is not None else self.A
        if not isinstance(A, scipy.sparse.csr_matrix):
            A = scipy.sparse.csr_matrix(A)
        slowness = slowness if slowness is not None else 1/self.velocity
        refvel = refvel if refvel is not None else self.refvel
        mesh = mesh if mesh is not None else self.grid.mesh
        if damping!='roughness' and damping!='norm':
            string = 'Unrecognized damping type: only "roughness" and "norm"'
            string += ' are allowed. Roughness damping will be used instead.'
            warnings.warn(string)
        if logspace:
            damp_range = np.logspace(np.log10(damp_min), np.log10(damp_max), n)
        else:
            damp_range = np.linspace(damp_min, damp_max, n)
        result = []
        if damping == 'roughness':
            G_lat, G_lon = _derivatives_lat_lon(mesh)
            G_lat = scipy.sparse.csr_matrix(G_lat)
            G_lon = scipy.sparse.csr_matrix(G_lon)
            G = (G_lat.T.dot(G_lat) + G_lon.T.dot(G_lon))
        for damp in damp_range:
            if damping == 'roughness':
                ndamp = 0
                rdamp = damp
            else:
                ndamp = damp
                rdamp = 0     
            x = self.solve(A=A, slowness=slowness, mesh=mesh, ndamp=ndamp, 
                           rdamp=rdamp, refvel=refvel)
            residual_norm = np.linalg.norm(slowness - A.dot(x))
            yaxis = np.linalg.norm(x) if ndamp else np.linalg.norm(G.dot(x))
            result.append((residual_norm, yaxis))
        if show:
            ylabel = r'$||$ G $\cdot$ x $||$' if damping=='roughness' else r'$||$ x $||$'
            scale = 'log' if logspace else 'linear'
            plot(damp_range, result, ylabel, scale)
            
        return damp_range, result
    
        
    @staticmethod
    def checkerboard(ref_value, kx=10, ky=10, latmin=None, latmax=None, lonmin=None, 
                     lonmax=None, anom_amp=0.1):
        """ Checkerboard-like pattern on Earth
        
        Parameters
        ----------
        ref_value : float
            Reference value with respect to which the anomalies are generated
            
        kx, ky : int | float
            Frequency of anomalies along longitude (kx) and latitude (ky) within
            the boundaries defined by `lonmin`, `lonmax`, `latmin`, `latmax`
            
        latmin, latmax, lonmin, lonmax : float, optional (in degrees)
            Boundaries of the checkerboard. 
            
        anom_amp : float
            Amplitude of the anomalies. Default is 0.1 (i.e. 10% of the reference
            value)
        
        
        Returns
        -------
        Generating function, which should be called passing the values of longitude
        and latitude where the values of the checkerboard want to be retrieved.
        
        
        Example
        -------
        tomo = SeismicTomography(5)
        lat = np.mean(tomo.grid.mesh[:, :2], axis=1)
        lon = np.mean(tomo.grid.mesh[:, 2:], axis=1)
        pattern = tomo.checkerboard(ref_value=3, kx=6, ky=6)
        c = pattern(lon, lat)
        img, cb = tomo.plot_map(tomo.grid.mesh, c, projection='Robinson', 
                                show=False)
        cb.set_label('Checkerboard Pattern')
        """
        
        def generating_function(lon, lat):
            smooth_poles = np.cos(np.radians(lat))
            lon_pattern = np.sin(x_spatial_freq * lon)
            lat_pattern = np.sin(y_spatial_freq * lat)
            values = (lon_pattern + lat_pattern) * smooth_poles
            values *= anom_amp / np.max(np.abs(values))
            if ref_value is not None:
                return ref_value + values*ref_value
            return values
            

        latmin = latmin if latmin is not None else -90
        lonmin = lonmin if lonmin is not None else -180
        latmax = latmax if latmax is not None else 90
        lonmax = lonmax if lonmax is not None else 180       
        x_spatial_freq = kx * 2 * np.pi / (lonmax - lonmin)
        y_spatial_freq = ky * 2 * np.pi / (latmax - latmin)
        
        return generating_function
            
  
    @staticmethod
    def spike(ref_value, x0, y0, sigma_x=1, sigma_y=1, anom_amp=0.1):
        """ Spike-like pattern on Earth, created via a 3-D Gaussian
        
        Parameters
        ----------
        ref_value : float
            Reference value with respect to which the anomalies are generated
            
        x0, y0 : float
            Central longitude and latitude of the anomaly
            
        sigma_x, sigma_y : float (default is 1)
            Standard deviations in the longitude and latitude directions of the
            3-D Gaussian
            
        anom_amp : float
            Amplitude of the anomalies. Default is 0.1 (i.e. 10% of the reference
            value)
            
        Returns
        -------
        Generating function, which should be called passing the values of longitude
        and latitude where the values of the spike pattern want to be retrieved
        

        Example
        -------
        tomo = SeismicTomography(1, latmin=-20, latmax=20, lonmin=-20, lonmax=20)
        lat = np.mean(tomo.grid.mesh[:, :2], axis=1)
        lon = np.mean(tomo.grid.mesh[:, 2:], axis=1)
        pattern = tomo.spike(ref_value=3, 
                             x0=0, 
                             y0=0, 
                             sigma_x=5, 
                             sigma_y=5,
                             anom_amp=0.2)
        c = pattern(lon, lat)
        img, cb = tomo.plot_map(tomo.grid.mesh, c, projection='Robinson', 
                                show=False)
        cb.set_label('Spike Anomaly')
        """
        
        def gaussian_3d(x, y):
            x_term = (x - x0)**2 / (2 * sigma_x**2)
            y_term = (y - y0)**2 / (2 * sigma_y**2)
            return ref_value + anom_amp * ref_value * np.exp(-(x_term + y_term))
            
        return gaussian_3d
    
    
    def resolution_test(self, mesh, velocity_model, noise=0, ndamp=0, rdamp=0):
        """ Resolution test on an arbitrary input model
        
        This function is called under the hood to perform both the checkerboard
        and the spike test.
        
        Parameters
        ----------
        mesh : ndarray, shape (n, 4)
            Grid cells in seislib format, consisting of n pixels described by 
            lat1, lat2, lon1, lon2 (in degrees). (-180<lon<180, -90<lat<90)
            
        velocity_model : ndarray, shape (n,)
            Synthetic velocity model associated with `mesh`
            
        noise : float, optional
            If > 0, random noise expressed as percentage of the synthetic data
            
        ndamp : float
            Extent of norm damping (default is zero)
            
        rdamp : float
            Extent of roughness damping (default is zero)
            
            
        Returns
        -------
        Dictionary object structured as follows:
            
            'synth_data': ndarray of shape (m,), where m corresponds to the
                number of rows in the A matrix (i.e., in the data kernel)
                
            'synth_model': inverse of `velocity_model`, used to create the
                synthetic data
                
            'retrieved_model': model retrieved from the inversion of the
                synthetic data
                
            'mesh': same as the input `mesh`
        """

        mesh_latmax = np.radians(np.max(mesh[:, 1]))
        mesh_lonmax = np.radians(np.max(mesh[:, 3]))
        data_coords = np.radians(self.data_coords)
        A_test = _compile_coefficients(data_coords=data_coords,
                                       mesh=np.radians(mesh), 
                                       mesh_latmax=mesh_latmax,
                                       mesh_lonmax=mesh_lonmax)
        dist = np.asarray(self.gc_distance(*self.data_coords.T))
        slowness = 1 / velocity_model
        synth_vel = dist / np.dot(A_test*dist.reshape(-1, 1), slowness)
        if noise > 0:
            coeffs = synth_vel * noise
            synth_vel += np.random.uniform(-coeffs, coeffs)         
        retrieved_model = self.solve(A=self.A, 
                                     slowness=1/synth_vel,
                                     mesh=self.grid.mesh,
                                     ndamp=ndamp, 
                                     rdamp=rdamp, 
                                     refvel=np.mean(velocity_model))
        results = {'synth_data': synth_vel, 
                   'synth_model': slowness, 
                   'retrieved_model': retrieved_model, 
                   'mesh': mesh}
        return results
        

    def checkerboard_test(self, kx, ky, latmin=None, latmax=None, lonmin=None, 
                          lonmax=None, cell_size=5, anom_amp=0.1, refvel=None, 
                          noise=0, ndamp=0, rdamp=0):
        """ 
        Resolution test, known as "checkerboard test". The method first builds
        synthetic data (velocities) using a checkerboard pattern as an input 
        (velocity) model. The checkerboard is built using 
        SeismicTomography.checkerboard.
        
        Parameters
        ----------
        kx, ky : int , float
            Frequency of anomalies along longitude (kx) and latitude (ky) within
            the boundaries defined by `lonmin`, `lonmax`, `latmin`, `latmax`
            
        latmin, latmax, lonmin, lonmax : float, optional
            Boundaries of the checkerboard, in degrees. (-180<lon<180, 
            -90<lat<90)
            
        cell_size : int | float
            Size of the checkerboard grid-cell sides (in degrees). Larger values
            are preferrable to avoid the resolution of an overly large inverse 
            problem
            
        anom_amp : float
            Intensity of the anomalies with respect to `ref_value`. A value of 
            0.1 means 10% of `ref_value`
            
        refvel : float, optional
            Original values of the anomalies (black and white squares of a 
            checkerboard). By default, their value is +-1. Therefore, if 
            `anom_extent` is 0.1 (say) their final values will be +-0.1
            
        noise : float, optional
            If > 0, random noise expressed as percentage of the synthetic data
            
        ndamp : float
            Extent of norm damping (default is zero)
            
        rdamp : float
            Extent of roughness damping (default is zero)
            
        
        Returns
        -------
        Dictionary object structured as follows:
            'synth_data': ndarray of shape (m,), where m corresponds to the
                number of rows in the A matrix (i.e., in the data kernel)
                
            'synth_model': ndarray of shape (n,), synthetic model (slowness) 
                used to create the synthetic data
                
            'retrieved_model': model retrieved from the inversion of the
                synthetic data
                
            'mesh': ndarray of shape (n, 4), parameterization associated with 
                the synthetic model
        """
        
        if refvel is None:
            refvel = self.refvel
            
        pattern = SeismicTomography.checkerboard(ref_value=refvel,
                                                 kx=kx,
                                                 ky=ky,
                                                 latmin=latmin,
                                                 latmax=latmax,
                                                 lonmin=lonmin,
                                                 lonmax=lonmax,
                                                 anom_amp=anom_amp
                                                 )
        mesh = EqualAreaGrid(cell_size, 
                             latmin=latmin, 
                             latmax=latmax, 
                             lonmin=lonmin,
                             lonmax=lonmax,
                             verbose=False).mesh
        lons = (mesh[:,2] + mesh[:,3]) / 2
        lats = (mesh[:,0] + mesh[:,1]) / 2                                        
        velocity = pattern(lons, lats)
        
        return self.resolution_test(mesh=mesh, 
                                    velocity_model=velocity, 
                                    noise=noise, 
                                    ndamp=ndamp, 
                                    rdamp=rdamp)
        
    
    def spike_test(self, x0, y0, sigma_x, sigma_y, latmin=None, latmax=None, 
                   lonmin=None, lonmax=None, cell_size=5, anom_amp=0.1, 
                   refvel=None, noise=0, ndamp=0, rdamp=0):
        """ 
        Resolution test, known as "spike test". The method first builds
        synthetic data (velocities) using a spike pattern as an input (velocity) 
        model. The spike is built using SeismicTomography.spike.
        
        Parameters
        ----------
        x0, y0 : float
            Central longitude and latitude of the anomaly. (-180<lon<180, 
            -90<lat<90)
            
        sigma_x, sigma_y : float (default is 1)
            Standard deviations in the longitude and latitude directions of the
            3-D Gaussian
            
        latmin, latmax, lonmin, lonmax : float, optional
            Boundaries of the checkerboard, in degrees
            
        cell_size : int | float
            Size of the checkerboard grid-cell sides (in degrees). Larger values
            are preferrable to avoid the resolution of an overly large inverse 
            problem
            
        anom_amp : float
            Intensity of the anomalies with respect to `ref_value`. A value of 
            0.1 means 10% of `ref_value`
            
        refvel : float, optional
            Original values of the anomalies (black and white squares of a 
            checkerboard). By default, their value is +-1. Therefore, if 
            `anom_extent` is 0.1 (say) their final values will be +-0.1
            
        noise : float, optional
            If > 0, random noise expressed as percentage of the synthetic data
            
        ndamp : float
            Extent of norm damping (default is zero)
            
        rdamp : float
            Extent of roughness damping (default is zero)
            
        
        Returns
        -------
        Dictionary object structured as follows:
            'synth_data': ndarray of shape (m,), where m corresponds to the
                number of rows in the A matrix (i.e., in the data kernel)
                
            'synth_model': ndarray of shape (n,), synthetic model (slowness) 
                used to create the synthetic data
                
            'retrieved_model': model retrieved from the inversion of the
                synthetic data
                
            'mesh': ndarray of shape (n, 4), parameterization associated with 
                the synthetic model
        """
        
        if refvel is None:
            refvel = self.refvel
            
        pattern = SeismicTomography.spike(ref_value=refvel, 
                                          x0=x0, 
                                          y0=y0, 
                                          sigma_x=sigma_x, 
                                          sigma_y=sigma_y, 
                                          anom_amp=anom_amp)
        mesh = EqualAreaGrid(cell_size, 
                             latmin=latmin, 
                             latmax=latmax, 
                             lonmin=lonmin,
                             lonmax=lonmax,
                             verbose=False).mesh
        lons = (mesh[:,2] + mesh[:,3]) / 2
        lats = (mesh[:,0] + mesh[:,1]) / 2                                        
        velocity = pattern(lons, lats)
        
        return self.resolution_test(mesh=mesh, 
                                    velocity_model=velocity, 
                                    noise=noise, 
                                    ndamp=ndamp, 
                                    rdamp=rdamp)


    @classmethod
    def plot_map(cls, mesh, c, ax=None, projection='Mercator', map_boundaries=None, 
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
            If not None, `c` is plotted on the GeoAxesSubplot instance. 
            Otherwise, a new figure and GeoAxesSubplot instance is created
            
        projection : str
            Name of the geographic projection used to create the GeoAxesSubplot.
            (Visit the cartopy website for a list of valid projection names.)
            If ax is not None, `projection` is ignored. Default is 'Mercator'
            
        map_boundaries : list or tuple of floats, shape (4,), optional
            Lonmin, lonmax, latmin, latmax (in degrees) defining the extent of
            the map
            
        bound_map : bool
            If True, the map boundaries will be automatically determined.
            Ignored if map_boundaries is not None
        
        colorbar : bool
            If True (default), a colorbar associated with `c` is displayed on 
            the side of the map
            
        show : bool
            If True (default), the map will be showed once generated
            
        style : str
            Possible options are 'colormesh', 'contourf', and 'contour',
            each corresponding to the homonymous method of SeismicTomography.
            Default is 'colormesh'
            
        add_features : bool
            If True, natural Earth features will be added to the GeoAxesSubplot.
            Default is False. If `ax` is None, it is automatically set to True
            
        resolution : str
            Resolution of the Earth features displayed in the figure. Passed to
            cartopy.feature.NaturalEarthFeature. Valid arguments are '110m',
            '50m', '10m'. Default is '110m'
            
        cbar_dict : dict
            Keyword arguments passed to matplotlib.colorbar.ColorbarBase
            
        kwargs : Additional inputs passed to the 'colormesh', 'contourf', and
            'contour' methods of SeismicTomography
            
            
        Returns
        -------
        If show is False: None
        Otherwise, an instance of matplotlib.collections.QuadMesh is returned,
        together with an instance of matplotlib.colorbar.Colorbar (if `colorbar`
        is True)
        """
        
        return plotting.plot_map(mesh=mesh, 
                                 c=c, 
                                 ax=ax, 
                                 projection=projection, 
                                 map_boundaries=map_boundaries, 
                                 bound_map=bound_map, 
                                 colorbar=colorbar, 
                                 show=show, 
                                 style=style, 
                                 add_features=add_features, 
                                 resolution=resolution,
                                 cbar_dict=cbar_dict,
                                 **kwargs)


    @classmethod
    def colormesh(cls, mesh, c, ax, **kwargs):
        """
        Adaptation of matplotlib.pyplot.pcolormesh to the (adaptive) equal area 
        grid.
        
        Parameters
        ---------
        mesh : ndarray (n, 4)
            Equal area grid, consisting of n pixels described by lat1, lat2, 
            lon1, lon2
            
        c : list of ndarray (n,)
            Values to plot in each grid cell
            
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not None, the receivers are plotted on the GeoAxesSubplot instance. 
            Otherwise, a new figure and GeoAxesSubplot instance is created
            
        kwargs : Additional inputs passed to seislib.plotting.colormesh
        
        
        Returns
        -------
            img : Instance of matplotlib.collections.QuadMesh
        """
        
        return plotting.colormesh(mesh, c, ax=ax, **kwargs)
        
    
    @classmethod
    def contourf(cls, mesh, c, ax, smoothing=None, **kwargs):
        """
        Adaptation of matplotlib.pyplot.contourf to the (adaptive) equal area 
        grid
        
        Parameters
        ---------
        mesh : ndarray (n, 4)
            Equal area grid, consisting of n pixels described by lat1, lat2, 
            lon1, lon2
            
        c : list of ndarray (n,)
            Values to plot in each grid cell
            
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not None, the receivers are plotted on the GeoAxesSubplot instance. 
            Otherwise, a new figure and GeoAxesSubplot instance is created
             
        smoothing : float, optional
            Value passed to scipy.ndimage.filters.gaussian_filter and used to
            obtain a smooth representation of `c` (default is None)
            
        kwargs : Additional inputs passed to seislib.plotting.contourf
        
        Returns
        -------
            img : Instance of matplotlib.contour.QuadContourSet
        """
        
        return plotting.contourf(mesh, c, ax=ax, smoothing=smoothing, **kwargs)


    @classmethod     
    def contour(cls, mesh, c, ax, smoothing=None, **kwargs):
        """
        Adaptation of matplotlib.pyplot.contour to the (adaptive) equal area 
        grid
        
        Parameters
        ---------
        mesh : ndarray (n, 4)
            Equal area grid, consisting of n pixels described by lat1, lat2, 
            lon1, lon2
            
        c : list of ndarray (n,)
            Values to plot in each grid cell
            
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not None, the receivers are plotted on the GeoAxesSubplot instance. 
            Otherwise, a new figure and GeoAxesSubplot instance is created
            
        smoothing : float, optional
            Value passed to scipy.ndimage.filters.gaussian_filter and used to
            obtain a smooth representation of `c` (default is None)
            
        kwargs : Additional inputs passed to seislib.plotting.contourf

        
        Returns
        -------
            img : Instance of matplotlib.contour.QuadContourSet
        """
        
        return plotting.contour(mesh, c, ax, smoothing=smoothing, **kwargs)
    
    
    def plot_rays(self, ax=None, show=True, stations_color='r', 
                  paths_color='k', oceans_color='water', lands_color='land', 
                  edgecolor='k', stations_alpha=None, paths_alpha=0.3, 
                  projection='Mercator', resolution='110m', map_boundaries=None, 
                  bound_map=True, paths_width=0.2, **kwargs):
        """ 
        Utility function to display the great-circle paths associated with pairs
        of data coordinates
        
        Parameters
        ----------            
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not None, `c` is plotted on the GeoAxesSubplot instance.
            Otherwise, a new figure and GeoAxesSubplot instance is created
            
        show : bool
            If True (default), the map will be showed once generated. Otherwise
            a GeoAxesSubplot instance is returned
            
        stations_color, paths_color : str
            Color of the receivers and of the great-circle paths (see matplotlib 
            documentation for valid color names). Defaults are 'r' (red) and
            'k' (black)
                    
        oceans_color, lands_color : str
            Color of oceans and lands. The arguments are ignored if ax is not
            None. Otherwise, they are passed to cartopy.feature.GSHHSFeature 
            (to the argument 'facecolor'). Defaults are 'water' and 'land'
    
        edgecolor : str
            Color of the boundaries between, e.g., lakes and land. The argument 
            is ignored if ax is not None. Otherwise, it is passed to 
            cartopy.feature.GSHHSFeature (to the argument 'edgecolor'). Default
            is 'k' (black)
    
        stations_alpha, paths_alpha : float, optional
            Transparency of the stations and of the great-circle paths. Defaults
            are None and 0.3
            
        paths_width : float
            Linewidth of the great-circle paths
                    
        projection : str
            Name of the geographic projection used to create the GeoAxesSubplot.
            (Visit the cartopy website for a list of valid projection names.)
            If ax is not None, `projection` is ignored. Default is 'Mercator'
     
        resolution : str
            Resolution of the Earth features displayed in the figure. Passed to
            cartopy.feature.NaturalEarthFeature. Valid arguments are '110m',
            '50m', '10m'. Default is '110m'
        
        map_boundaries : list or tuple of floats, shape (4,), optional
            Lonmin, lonmax, latmin, latmax (in degrees) defining the extent of
            the map
            
        bound_map : bool
            If True, the map boundaries will be automatically determined.
            Ignored if map_boundaries is not None
                        
    
        Returns
        -------
        None if show is False. Otherwise a GeoAxesSubplot instance
        """
      
        return plotting.plot_rays(self.data_coords, 
                                  ax=ax, 
                                  show=show, 
                                  stations_color=stations_color, 
                                  paths_color=paths_color, 
                                  oceans_color=oceans_color, 
                                  lands_color=lands_color, 
                                  edgecolor=edgecolor, 
                                  stations_alpha=stations_alpha, 
                                  paths_alpha=paths_alpha, 
                                  projection=projection, 
                                  resolution=resolution,
                                  map_boundaries=map_boundaries, 
                                  bound_map=bound_map, 
                                  paths_width=paths_width, 
                                  **kwargs)


    def plot_colored_rays(self, ax=None, show=True, cmap=scm.roma, vmin=None, 
                          vmax=None, stations_color='k', oceans_color='lightgrey', 
                          lands_color='w', edgecolor='k', stations_alpha=None, 
                          paths_alpha=None, projection='Mercator', resolution='110m',
                          map_boundaries=None, bound_map=True, paths_width=1, 
                          colorbar=True, cbar_dict={}, **kwargs):
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
            None. Otherwise, they are passed to cartopy.feature.GSHHSFeature 
            (to the argument 'facecolor'). Defaults are 'water' and 'land'
    
        edgecolor : str
            Color of the boundaries between, e.g., lakes and land. The argument 
            is ignored if ax is not None. Otherwise, it is passed to 
            cartopy.feature.GSHHSFeature (to the argument 'edgecolor'). Default
            is 'k' (black)
    
        stations_alpha, paths_alpha : float, optional
            Transparency of the stations and of the great-circle paths. Defaults
            are None and 0.3
            
        paths_width : float
            Linewidth of the great-circle paths
                    
        projection : str
            Name of the geographic projection used to create the GeoAxesSubplot.
            (Visit the cartopy website for a list of valid projection names.)
            If ax is not None, `projection` is ignored. Default is 'Mercator'
            
        resolution : str
            Resolution of the Earth features displayed in the figure. Passed to
            cartopy.feature.NaturalEarthFeature. Valid arguments are '110m',
            '50m', '10m'. Default is '110m'
            
        map_boundaries : list or tuple of floats, shape (4,), optional
            Lonmin, lonmax, latmin, latmax (in degrees) defining the extent of
            the map
            
        bound_map : bool
            If True, the map boundaries will be automatically determined.
            Ignored if map_boundaries is not None
        
        colorbar : bool
            If True (default), a colorbar associated with `c` is displayed on 
            the side of the map        
        
        cbar_dict : dict
            Keyword arguments passed to matplotlib.colorbar.ColorbarBase
        
        kwargs:
            Additional keyword arguments passed to ax.plot
        
        Returns
        -------
        None if show is False. Otherwise a GeoAxesSubplot instance together with an 
        instance of matplotlib.colorbar.Colorbar (if `colorbar` is True)
        """
              
        return plotting.plot_colored_rays(data_coords=self.data_coords, 
                                          c=self.velocity,
                                          ax=ax, 
                                          show=show, 
                                          cmap=cmap, 
                                          vmin=vmin, 
                                          vmax=vmax, 
                                          stations_color=stations_color, 
                                          oceans_color=oceans_color, 
                                          lands_color=lands_color, 
                                          edgecolor=edgecolor, 
                                          stations_alpha=stations_alpha, 
                                          paths_alpha=paths_alpha, 
                                          projection=projection, 
                                          resolution=resolution,
                                          map_boundaries=map_boundaries, 
                                          bound_map=bound_map, 
                                          paths_width=paths_width, 
                                          colorbar=colorbar, 
                                          cbar_dict=cbar_dict, 
                                          **kwargs)


        
        
