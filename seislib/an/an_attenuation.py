#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Rayleigh-Wave Attenuation
========================= 

The below class provides support for calculating Rayleigh-Wave 
attenuation based on continuous ambient-noise recordings of a 
dense seismic array.

As opposed to the problem of calculating phase velocities, which 
are only related to the phase of the empirical Green's function, 
estimates of attenuation should rely on its `amplitude`. Previous 
studies showed how the amplitude of the empirical Green's function 
should be treated with caution, because of its sensitivity to 
parameters such as the distribution of noise sources [1]_. Accordingly, 
[2]_ designed a procedure that should contribute to "regularizing" 
the subsequent inversion for the Rayleigh-wave attenuation coefficient 
:math:`\alpha`, as verified by a suite of numerical tests. Such procedure 
builds on the previous work of [3]_, and allows for measuring the
frequency-dependency of :math:`\alpha`.

If we consider a dense seismic array, whose receivers are located at 
:math:`{\bf x}`, SeisLib first retrieves the normalized cross-spectra
associated with each combination of station pairs `ij` 

.. math::

    \rho = 
    \Re{ \Bigg\lbrace \frac{u ({\bf x}_i,\omega) u^{\ast} ({\bf x}_j,\omega)}
    {\left\langle \vert u({\bf x},\omega) 
    \vert^2 \right\rangle_{\bf x}}} \Bigg\rbrace, 

where `u` denotes the ambient-noise recording in the frequency domain, 
:math:`^{\ast}` complex conjugation, and :math:`\Re{\big\lbrace \dotso \big\rbrace}` 
maps a complex number into its real part. The normalization term 
:math:`\left\langle \vert u({\bf x},\omega) \vert^2 \right\rangle_{\bf x}` 
corresponds to the average power spectral density (PSD) recorded by the seismic 
array. 

Then, :math:`\alpha` can be constrained by minimizing the cost function [2]_

.. math::

    C(\alpha, \omega)= 
    \sum_{i,j}
    \Delta_{ij}^2
    \Biggr\rvert
    \mbox{env}\left[\rho \right]
    - 
    \mbox{env}\left[
    J_0\left(\frac{\omega \Delta_{ij}}{c_{ij}(\omega)}\right) 
    \mbox{e}^{-\alpha(\omega) \Delta_{ij}}
    \right]
    \Biggr\rvert^2,

where :math:`\Delta`, :math:`\omega`, and `c` denote inter-station 
distance, angular frequency, and phase velocity, respectively, 
and :math:`J_0` a zeroth order Bessel function of the 
first kind. The weight :math:`\Delta_{ij}^2` is used to compensate for the 
decrease in the amplitude of :math:`J_0` with inter-station distance, due to 
geometrical spreading, and the envelope function :math:`\mbox{env}` has beneficial 
effects on the stability of the inversion.

In the above equation, phase velocity `c` is assumed to be known, i.e., it 
should be calculated in a preliminary step (see :mod:`seislib.an.an_velocity`).
The minimum of :math:`C(\alpha, \omega)` 
can then be found via "grid-search" over :math:`\alpha`, for a discrete set of 
values of :math:`\omega`. The alert reader might notice at this point that, 
since the sum is carried out over all possible combinations of station pairs 
belonging to the considered array, only one attenuation curve can be extracted 
from such minimization. This strategy, albeit in a sense restrictive, has been 
shown to yield robust estimates of the frequency dependency of :math:`\alpha` 
even in presence of a heterogeneous distribution of noise sources [2]_. If 
the array has good azimuthal coverage, using all station pairs as an ensemble 
in the minimization of :math:`C(\alpha, \omega)` allows for sampling most 
azimuths of wave propagation. In turn, this should "regularize" the inversion 
by decreasing unwanted effects due to inhomogeneities in the distribution of 
the noise sources, or compensating for a non-perfectly diffuse ambient seismic 
field.

Since the above equation only allows to obtain one attenuation curve for
seismic array, to retrieve the spatial variations in attenuation SeisLib 
implements the strategy applied by [4]_ to USArray data. In practice,
:meth:`parameterize` allows for subdividing a given array in many (possibly
overlapping) sub-arrays. These are identified by equal-area blocks of
arbitrary size, and used to obtain one attenuation curve for each of them.


References
----------
.. [1] Tsai, (2011). Understanding the amplitudes of noise correlation 
    measurements. JGR

.. [2] Magrini & Boschi, (2021). Surface‐Wave Attenuation From Seismic 
    Ambient Noise: Numerical Validation and Application. JGR

.. [3] Boschi et al., (2019). On seismic ambient noise cross-correlation 
    and surface-wave attenuation. GJI

.. [4] Magrini et al., (2021). Rayleigh-wave attenuation across the
    conterminous United States in the microseism frequency band. Scientific
    Reports
"""
import os
import itertools as it
from collections import defaultdict
from types import GeneratorType
import gc
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import detrend
from scipy.interpolate import interp1d, CubicSpline
from scipy.special import j0, y0
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
from obspy import read
from obspy import UTCDateTime as UTC
from obspy.signal.invsim import cosine_taper
from obspy.geodetics.base import gps2dist_azimuth
from obspy.io.sac.util import SacIOError
from seislib.tomography import EqualAreaGrid, RegularGrid
from seislib.utils import load_pickle, save_pickle, remove_file, resample
from seislib.utils import azimuth_backazimuth
from seislib.plotting import add_earth_features
from seislib.plotting import colormesh, contour, contourf
from seislib.plotting import plot_stations, plot_map, make_colorbar
from seislib.an import velocity_filter


def _read(*args, verbose=True, **kwargs):
    try:
        return read(*args, **kwargs)
    except SacIOError:
        if verbose:
            print('SacIOError', *args)


class AmbientNoiseAttenuation:
    
    r"""
    Class to obtain Rayleigh-wave attenuation out of continuous seismograms,
    via cross-correlation of seismic ambient noise. 
        
    Parameters
    ----------
    src : str
        Absolute path to the directory containing the files associated with 
        the continuous seismograms
        
    savedir : str
        Absolute path to the directory where the results are saved

    verbose : bool
        Whether or not information on progress is printed in the console


    Attributes
    ----------
    src : str
        Absolute path to the directory containing the files associated with the
        continuous seismograms
        
    savedir : str
        Absolute path to the directory where the results are saved
        
    files : list
        List of files to be processed to calculate the attenuation curves
        
    verbose : bool
        Whether or not information on progress is printed in the console
        
    stations : dict
        Coordinates of the station pairs that can be employed to retrieve
        dispersion curves (can be accessed after calling the method
        `prepare_data`)
    
    parameterization : dict
        Information on the parameterization of the study area, consisting
        of a dictionary object of the form::

            {
            'grid': np.array([[lat1, lat2, lon1, lon2],
                              [lat3, lat4, lon3, lon4]
                              [etc.]]),
            'no_stations': np.array([int1, int2, etc.]),
            'station_codes': [[sta1, sta2, sta3, etc.],
                              [sta4, sta5, sta6, etc.]]
            }
        
        where lat1, lat2, lon1, lon2 identify the first block of the
        parameterization, int1 is the number of stations in the first
        block, and sta1, sta2, sta3, etc. are the associated station 
        codes.
               
        
    Examples
    --------
    In the following, we will calculate an attenuation map for a region
    characterized by a dense distribution of receivers. When applied to 
    the transportable component of the USArray, the code will produce
    maps analogous to those in [1]_. To do so, we carry out the following
    steps squentially: (i) define an overlapping parameterization of the study 
    area, (ii) compute the Fourier transforms of our continuous seismograms, 
    (iii) use the thus retrieved Fourier transforms to compute the average 
    power-spectral density in each grid cell of our parameterization; these 
    are used in the calculation of the cross-spectra as normalization term 
    [2]_. (iv) Prepare the inversion data-set, for each sub-array associated with
    each grid cell of our parameterization. This data-set consists of the 
    cross-spectra and of the previously computed inter-station phase velocities.
    (v) Invert the data-set associated with each grid cell, so as to retrieve 
    one attenuation curve per grid cell, and (vi) plot the results.
    
    First, we need to define an :class:`AmbientNoiseAttenuation` instance, and 
    retrieve the geographic coordinates for each receivers. Our waveform data 
    are in .sac format, with the header compiled with the needed information. 
    (To download seismic data in the proper format, consider using 
    :class:`seislib.an.an_downloader.ANDownloader`) These are stored in 
    /path/to/data. We will save the results in /savedir::
    
        src = '/path/to/data'
        save = '/savedir'
        an = AmbientNoiseAttenuation(src=src, savedir=save, verbose=True)
        an.prepare_data()
        
    :meth:`prepare_data` will automatically save a station.pickle file in $save
    
    .. note::
        If the files containing the continuous seismograms do not have a sac 
        header, or if they have a different extension than .sac, the user can 
        create the station.pickle file on his own, before calling 
        :meth:`prepare_data`. (However, each file name should be in the format 
        net.sta.loc.cha', where net, sta, and loc (optional) are network, 
        station, and location code, respectively, and cha is channel (e.g., 
        BHZ, BHN, or BHE). For example, TA.N38A..LHZ.sac.
        
        The stations.pickle file should contain a dictionary object where each key
        corresponds to a station code ($net.$sta) and each value should be a tuple
        containing latitude and longitude of the station. For example::
            
            { net1.sta1 : (lat1, lon1),
            net1.sta2 : (lat2, lon2),
            net2.sta3 : (lat3, lon3)
            }
    
    We now parameterize the study area, by means of an equal-area grid of
    cell dimensions of 2.5°, with a spatial overlap of 50% on both latitude and
    longitude. Those grid cells containing less than 6 receivers are discarded::
    
        an.parameterize(cell_size=2.5, 
                        overlap=0.5, 
                        min_no_stations=6)
        
    The above will create the file paramerization.pickle in $save, and set
    the corresponding dictionary object to the attribute `paramterization` of
    `an`. This can be accessed typing `an.parameterization`, and will be
    used under the hood in the following processing. We now compute the Fourier 
    transforms and the cross-spectra. To compute the Fourier transforms, we use 
    time windows of 6h, and in the calculation of the cross spectra we will 
    discard every grid cell for which at least 6 station pairs having at least 
    30 days of simultaneous recordings are not available. (`fs` indicates the 
    sampling rate.)::
        
        an.compute_ffts(fs=1, window_length=3600*6)
        an.compute_corr_spectra(min_no_pairs=6, 
                                min_no_days=30)
    
    The above operation might require a long time and occupy a relatively large
    space on disk (even more than the size of the original data, depending on
    the target sampling rate `fs`). Three directories (fft, psd, cross-spectra)
    will be created in $save, where the result of the processing is stored.
    
    Before preparing the final data-set for the inversion, we need to calculate
    inter-station phase velocities. We did so by using 
    :class:`seislib.an.an_velocity.AmbientNoiseVelocity`. The retrieved 
    dispersion curves are stored in /path/to/dispcurves::
        
        an.prepare_inversion(src_velocity='/path/to/dispcurves',
                             freqmin=1/15, 
                             freqmax=0.4, 
                             nfreq=300)
    
    The above will prepare the data-set needed by the inversion, using a 
    frequency range of 1/15 - 0.4 Hz, sampled by 300 points so that each
    frequency is equally spaced from each other on logarithmic scale. The 
    data-set, built for each grid cell individually, is automatically saved to 
    $save/inversion_dataset. To retrieve one attenuation curve for grid cell,
    we can now run::
        
        an.inversion(alphamin=5e-8, alphamax=1e-4, nalpha=350, min_no_pairs=6)
        
    that will search for the optimal frequency-dependent attenuation coefficient
    in the range :math:`5 \times 10^{-8}` - :math:`1 \times 10^{-4}` 
    1/m via a 2-D grid search carried out over
    :math:`\alpha` and frequency. The :math:`\alpha` range is sampled through 
    350 points equally spaced on a logarithmic scale. All grid cells for which 
    data are not available from at least 6 station pairs are discarded.
    
    The above will save the results in $save/results. These can be visualized
    in the form of maps at different periods. In the following, we calculate
    an attenuation map at the period of 4 s, parameterizing it as an equal-area
    grid of pixel-size = 1°. Only the grid cells of this equal-area grid for 
    which at least 2 attenuation curves are available (remember that we used a 
    50% overlapping parameterization for the calculation of the attenuation 
    curves) will be attributed an attenuation value. (This behaviour is 
    controlled by the parameter `min_overlapping_pixels`). Once the attenuation 
    map is created, we plot it::
        
        grid, attenuation = an.get_attenuation_map(period=4, 
                                                   cell_size=1,
                                                   min_overlapping_pixels=2)
        an.plot_map(grid, attenuation)
    
    
    .. note::

        Once you have prepared the inversion data set by 
        :meth:`prepare_inversion` (and you are satisfied by the parameters 
        passed therein), you can free the previously occupied space 
        on disk by deleting the directories $save/ffts, $save/psd, since they
        will not be used in the subsequent inversion.

    References
    ----------
    .. [1] Magrini et al., (2021). Rayleigh-wave attenuation across the
        conterminous United States in the microseism frequency band. Scientific
        Reports

    .. [2] Magrini & Boschi, (2021). Surface‐Wave Attenuation From Seismic 
        Ambient Noise: Numerical Validation and Application. JGR
    """
    
    def __init__(self, src, savedir=None, verbose=True):
        self.verbose = verbose
        self.src = src
        savedir = os.path.dirname(src) if savedir is None else savedir
        self.savedir = os.path.join(savedir, 'an_attenuation')
        os.makedirs(self.savedir, exist_ok=True)
        self.files = self.get_files()
        
    
    def __str__(self):
        string = '\nRAYLEIGH-WAVE ATTENUATION FROM SEISMIC AMBIENT NOISE'
        separators = len(string)
        string += '\n%s'%('='*separators)
        stations = set(['.'.join(i.split('.')[:2]) for i in self.files])
        string += '\nRECEIVERS: %s'%(len(stations))
        if self.__dict__.get('parameterization') is not None:
            string += '\nSUB-ARRAYS: %s'%(self.parameterization['grid'].shape[0])
        string += '\n%s'%('='*separators)
        string += '\nSOURCE DIR: %s'%self.src
        string += '\nSAVE DIR: %s'%self.savedir
        return string
    
    
    def __repr__(self):
        return str(self)
        
    
    def get_files(self):
        """ 
        Retrieves the files to be processed for extracting Rayleigh-wave
        attenuation on the vertical component.
        
        Returns
        -------
        files : list of str
            e.g., ['net1.sta1.00.BHZ.sac', 'net1.sta2.00.BHZ.sac']
        """

        files = []
        for file in sorted(os.listdir(self.src)):
            channel = file.split('.')[-2]
            if 'HZ' in channel:
                files.append(file)
        return files
    
    
    def get_stations_coords(self, files):
        """ 
        Retrieves the geographic coordinates associated with each receiver
        
        Parameters
        ----------
        files: list of str
            Names of the files corresponding with the continuous seismograms,
            located in the `src` directory
            
        Returns
        -------
        coords : dict
            each key corresponds to a station code ($network_code.$station_code) 
            and each value is a tuple containing latitude and longitude of the 
            station. For example::
                
                { net1.sta1 : (lat1, lon1),
                  net1.sta2 : (lat2, lon2),
                  net2.sta3 : (lat3, lon3)
                  }
        """
        coords = {}
        for file in files:
            station_code = '.'.join(file.split('.')[:2])
            tr = _read(os.path.join(self.src, file), 
                       headonly=True,
                       verbose=self.verbose)[0]
            lat, lon = tr.stats.sac.stla, tr.stats.sac.stlo
            coords[station_code] = (lat, lon)
            if self.verbose:
                print(station_code, '%.3f'%lat, '%.3f'%lon)   
        return coords
    
                
    def prepare_data(self, recompute=False):
        """ 
        Saves to disk the geographic coordinates associated with each receiver. 
        These are saved to self.savedir/stations.pickle
        
        The stations.pickle file contains a dictionary object where each key
        corresponds to a station code ($network_code.$station_code) and each 
        value is a tuple containing latitude and longitude of the station. 
        For example::
            
            { net1.sta1 : (lat1, lon1),
              net1.sta2 : (lat2, lon2),
              net2.sta3 : (lat3, lon3)
              }
            
        If self.savedir/parameterization.pickle exists, the corresponding
        dictionary object is set as an attribute of the 
        :class:`AmbientNoiseAttenuation` instance under the name 
        `parameterization`. This includes information on the parameterization 
        of the study area, and has the form::

            {
            'grid': np.array([[lat1, lat2, lon1, lon2],
                              [lat3, lat4, lon3, lon4]
                              [etc.]]),
            'no_stations': np.array([int1, int2, etc.]),
            'station_codes': [[sta1, sta2, sta3, etc.],
                              [sta4, sta5, sta6, etc.]]
            }
        
        where lat1, lat2, lon1, lon2 identify the first block of the
        parameterization, int1 is the number of stations in the first
        block, and sta1, sta2, sta3, etc. are the associated station 
        codes.
        
        Parameters
        ----------
        recompute : bool
            If `True`, the station coordinates and times will be removed from
            disk and recalculated. Otherwise (default), if they are present,
            they will be loaded into memory, avoiding any computation. This
            parameter should be set to `True` whenever one has added new files 
            to the source directory
        """
        savecoords = os.path.join(self.savedir, 'stations.pickle')
        parameterization = os.path.join(self.savedir, 'parameterization.pickle')
        if recompute:
            remove_file(savecoords)
        if not os.path.exists(savecoords):
            coords = self.get_stations_coords(self.files)
            save_pickle(savecoords, coords)
        else:
            coords = load_pickle(savecoords)
        self.stations = coords
        if os.path.exists(parameterization):
            self.parameterization = load_pickle(parameterization)
            
    
    def parameterize(self, cell_size, overlap=0.5, min_no_stations=6,
                     regular_grid=False, plotting=True, plot_resolution='110m'):
        """
        Creates the equal area (possibly overlapping) parameterization used in 
        the subsequent analysis. The equal-area grid is created through the 
        :class:seislib.tomography.grid.EqualAreaGrid class of seislib.
        
        The parameterization is saved at $savedir/parameterization.pickle and
        set as an attribute of the :class:`AmbientNoiseAttenuation` instance 
        under the name `parameterization`. The corresponding dictionary object
        includes information on the parameterization of the study area, and 
        has the form::

            {
            'grid': np.array([[lat1, lat2, lon1, lon2],
                              [lat3, lat4, lon3, lon4]
                              [etc.]]),
            'no_stations': np.array([int1, int2, etc.]),
            'station_codes': [[sta1, sta2, sta3, etc.],
                              [sta4, sta5, sta6, etc.]]
            }
        
        where lat1, lat2, lon1, lon2 identify the first block of the
        parameterization, int1 is the number of stations in the first
        block, and sta1, sta2, sta3, etc. are the associated station 
        codes.
        
        Parameters
        ----------
        cell_size : float
            Size of each grid cell of the resulting parameterization (in 
            degrees)
        
        overlap : float
            If > 0, the parameterization will be overlapping in space by the
            specified extent [1]_. Default is 0.5
        
        min_no_stations : int
            Minimum number of stations falling within each grid-cell. If the
            value is not reached, the grid-cell in question is removed from the
            parameterization
        
        regular_grid : bool
            If False (default), the study area is discretized using an equal-area
            parameterization. Otherwise, a regular grid is employed.
        
        plotting : bool
            If `True`, a figure on the resulting parameterization is displayed
            
        plot_resolution : {'10m', '50m', '110m'}
            Resolution of the Earth features displayed in the figure. Passed to
            `cartopy.feature.NaturalEarthFeature`. Valid arguments are '110m',
            '50m', '10m'. Default is '110m'
        
        References
        ----------
        .. [1] Magrini et al., (2021). Rayleigh-wave attenuation across the
            conterminous United States in the microseism frequency band. Scientific
            Reports
        """
        def add_overlap(grid, overlap_east=0.5, overlap_north=0.5):        
            mesh = grid.mesh
            dlat = (1-overlap_north) * (mesh[:,1]-mesh[:,0])
            old_mesh = mesh.copy()
            for i in range(1, round(1 / (1-overlap_north))):
                new_lats = old_mesh[:, :2] + dlat.reshape(-1, 1)*i
                mesh = np.row_stack((
                        mesh, np.column_stack((new_lats, old_mesh[:, 2:]))
                        ))
            
            dlon = (1-overlap_east) * (mesh[:,3]-mesh[:,2])       
            old_mesh = mesh.copy()
            for i in range(1, round(1 / (1-overlap_east))):
                new_lons = old_mesh[:, 2:] + dlon.reshape(-1, 1)*i
                mesh = np.row_stack((
                        mesh, np.column_stack((old_mesh[:, :2], new_lons))
                        ))
            return sort_mesh(mesh)
        
        def sort_mesh(mesh):
            mesh = mesh[np.argsort(mesh[:,0])][::-1]
            final_indexes = []
            indexes = [0]
            lb, ub = mesh[0, :2]
            for i, (lat1, lat2, lon1, lon2) in enumerate(mesh[1:], 1):
                if lat1==lb and lat2==ub:
                    indexes.append(i)
                else:
                    final_indexes.extend(list(
                            np.array(indexes)[np.argsort(mesh[indexes, 2])]
                            ))
                    indexes = [i]
                    lb, ub = lat1, lat2
            if indexes:
                final_indexes.extend(list(
                        np.array(indexes)[np.argsort(mesh[indexes, 2])]
                        ))
            return mesh[final_indexes]

        def stations_per_pixel(grid, station_codes, station_coords):
            sta_per_pixel = []
            sta_in_pixel = []
            for lat1, lat2, lon1, lon2 in grid.mesh:
                idx_stations_in_pixel = np.flatnonzero(
                        (station_coords[:,0]>=lat1) \
                        & (station_coords[:,0]<=lat2) \
                        & (station_coords[:,1]>=lon1) \
                        & (station_coords[:,1]<=lon2))
                n_stations = idx_stations_in_pixel.size
                sta_per_pixel.append(n_stations)
                sta_in_pixel.append(list(station_codes[idx_stations_in_pixel]))
            return np.array(sta_per_pixel), sta_in_pixel
        
        def plot_stations_and_grid(station_coords, grid, map_boundaries):
            
            def plot_one_pixel(coords, ax):
                lat1, lat2, lon1, lon2 = coords
                paths = [
                        ((lon1, lon1), (lat1, lat2)),
                        ((lon1, lon2), (lat2, lat2)),
                        ((lon2, lon2), (lat2, lat1)),
                        ((lon2, lon1), (lat1, lat1)),
                        ]
                for x, y in paths:
                    ax.plot(x, y, 'r', transform=transform)
                    
            transform = ccrs.PlateCarree()
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
            ax.set_extent(map_boundaries, ccrs.PlateCarree())
            add_earth_features(ax, 
                               scale=plot_resolution, 
                               oceans_color='water', 
                               lands_color='land')
            ax = grid.plot(ax=ax, show=False)
            ax.plot(*station_coords.T[::-1], '^b', transform=transform) 
            plot_one_pixel(grid.mesh[0], ax)
            plt.show()
            
        def plot_stations_per_pixel(sta_per_pixel):
            plt.figure()
            plt.hist(sta_per_pixel, bins=10, ec='w')
            plt.xlabel('Stations per sub-array', labelpad=10)
            plt.show()

        def get_map_boundaries(grid):
            dlon = (grid.lonmax - grid.lonmin) * 0.05
            dlat = (grid.latmax - grid.latmin) * 0.05
            lonmin = grid.lonmin-dlon if grid.lonmin-dlon > -180 else grid.lonmin
            lonmax = grid.lonmax+dlon if grid.lonmax+dlon < 180 else grid.lonmax
            latmin = grid.latmin-dlat if grid.latmin-dlat > -90 else grid.latmin
            latmax = grid.latmax+dlat if grid.latmax+dlat < 90 else grid.latmax
            return (lonmin, lonmax, latmin, latmax)            
        
        
        station_codes, station_coords = zip(*list(self.stations.items()))
        station_codes = np.array(station_codes)
        station_coords = np.array(station_coords)  
        latmin, latmax = station_coords[:,0].min(), station_coords[:,0].max()
        lonmin, lonmax = station_coords[:,1].min(), station_coords[:,1].max()
        grid_type = EqualAreaGrid if not regular_grid else RegularGrid
        grid = grid_type(cell_size, 
                         latmin=latmin,
                         latmax=latmax,
                         lonmin=lonmin,
                         lonmax=lonmax,
                         )
        if overlap > 0:
            mesh = add_overlap(grid, overlap_east=overlap, overlap_north=overlap)
        grid.update_grid_params(mesh)
        no_stations = stations_per_pixel(grid, station_codes, station_coords)[0]
        grid.select_cells(np.flatnonzero(no_stations >= min_no_stations))
        no_stations, codes = stations_per_pixel(grid, 
                                                station_codes, 
                                                station_coords)
        parameterization = {
                'grid': grid.mesh,
                'no_stations': no_stations,
                'station_codes': [sorted(c) for c in codes]
        }
        save_pickle(os.path.join(self.savedir, 'parameterization.pickle'),
                    parameterization)
        self.parameterization = parameterization
        
        if plotting:
            map_boundaries = get_map_boundaries(grid)
            plot_stations_and_grid(station_coords, grid, map_boundaries)
            plot_stations_per_pixel(no_stations)
            
            
    def compute_ffts(self, fs=1, window_length=3600):
        """ Computes the Fourier transforms for all the continuous data.
        
        A directory $self.savedir/fft will be created, and all Fourier 
        transforms will be saved within it. Within the same directory, two other
        files will be created: (i) the frequency vector associated with each 
        Fourier transform, named frequencies.npy, and (ii) a dictionary object
        where the starting and ending times of all continuous seismograms
        are stored, named times.pickle
        
        Parameters
        ----------
        fs : int, float
            Target sampling rate of the continuous seismograms. If some
            seismogram is characterized by a different sampling rate, it will
            be resampled
        
        window_lenght : int
            Length of the time windows (in s) used to perform the 
            cross-correlations
        
        .. warning::

            The operation might take a relatively large space on disk, depending on
            the amount of continuous seismograms available and on the sampling rate
            chosen. (Even more than the size of the original data.) This,
            however, is necessary to speed up (consistently) all the subsequent 
            operations, involving cross-correlations and calculation of the average
            power spectral density of each sub-array.
        """
        def fourier_transform(x, window_samples):      
            taper = cosine_taper(window_samples, p=0.05)
            x = detrend(x, type='constant')
            x *= taper           
            return rfft(x, window_samples)    

        def start_end_indexes(tr, times):
            starttime = tr.stats.starttime.timestamp
            endtime = tr.stats.endtime.timestamp
            start_idx = np.argmin( np.abs(np.array(times) - starttime) )
            end_idx = np.argmin( np.abs(np.array(times) - endtime) )    
            return start_idx, end_idx
        
        def round_time(t, window_length, kind='starttime'):
            dt = window_length if kind=='starttime' else 0
            tstamp = t.timestamp
            return UTC( (tstamp//window_length)*window_length + dt )

                    
        save_ffts = os.path.join(self.savedir, 'fft')
        os.makedirs(save_ffts, exist_ok=True)
        
        dt = 1 / fs
        window_length = int(window_length)
        window_samples = int(window_length * fs)
        freq = rfftfreq(window_samples, dt)
        np.save(os.path.join(save_ffts, 'frequencies.npy'), freq)
        
        times_per_station = defaultdict(list)
        stations = set([code for codes in self.parameterization['station_codes'] \
                    for code in codes])
        for file in self.files:
            if self.verbose:
                print(file)
            station_code = '.'.join(file.split('.')[:2])
            if station_code not in stations:
                continue
            if os.path.exists(os.path.join(save_ffts, '%s.npy'%station_code)):
                continue
            tr = _read(os.path.join(self.src, file), verbose=self.verbose)[0]
            if tr is None:
                continue
            if tr.stats.sampling_rate != fs:
                tr = resample(tr, fs)
            tstart = round_time(tr.stats.starttime, window_length, kind='starttime')
            tend = round_time(tr.stats.endtime, window_length, kind='endtime')
            tr = tr.slice(tstart, tend)
            data = tr.data           
            times = np.arange(tstart.timestamp, tend.timestamp, window_length)
            store = []
            for i, time in enumerate(times):
                data_tmp = data[i*window_samples : (i+1)*window_samples]
                if np.all(data_tmp == 0):
                    continue
                fft = fourier_transform(data_tmp, window_samples)
                store.append(fft)
                times_per_station[station_code].append(time)
            store = np.array(store)
            if store.size:
                np.save(os.path.join(save_ffts, '%s.npy'%station_code), store)
                del store
                gc.collect()
        times_per_station = {k: np.array(v) for k, v in times_per_station.items()}
        save_pickle(os.path.join(save_ffts, 'times.pickle'), times_per_station)
        
        
    def compute_corr_spectra(self, min_no_pairs=6, min_no_days=30, 
                             ram_available=9000, ram_split=4):
        """ 
        Computes the corr-spectra for each pair of receivers in each grid cell
        of the parameterization. In practice, all the Fourier transform 
        associated with the continuous seismograms found in a given cell are 
        first loaded into memory. Then, they are used to calculate the
        average power-spectral density of the grid-cell, which is then used
        as a normalization term of the cross-correlations (see [1]_, [2]_,
        and [3]_).
        
        Two directories are created: $self.savedir/corr_spectra and 
        $self.savedir/psd, where the cross-spectra and the average psd will
        be saved.
        
        Parameters
        ----------
        min_no_pairs : int
            Minimum number of pairs of receivers in a given grid cell available
            for computing the cross-spectrum. Grid cells not reaching this
            number are not processed. Default is 6. Larger values yield more
            robust estimates of Rayleigh-wave attenuation [3]_. Smaller values 
            are suggested against
            
        min_no_days : int, float
            Minimum number of simultaneous recordings available for a given
            station pair to be considered for the calculation of the cross-
            spectrum. Station pairs not reaching this number do not count as
            available station pairs (see `min_no_pairs`)
            
        ram_available : int, float
            RAM available (in Mb) for the analysis. The parameter allows for 
            estabilishing, in a given pixel, if all the Fourier transforms 
            related to the pixel can be loaded into memory or a generator is
            otherwise required. Lower values are suggested to avoid MemoryErrors,
            although they would result in a longer computational time. Default
            is 9000 (i.e., 9 Gb)
            
        ram_split : int
            When the (estimated) RAM necessary to load the Fourier transforms
            of a given pixel into memory exceeds `ram_available`, the task is
            splitted in `ram_split` operations. Larger values help avoiding 
            MemoryErrors, but result in a longer computational time. Default 
            is 4
        
        References
        ----------
        .. [1] Boschi et al., (2019). On seismic ambient noise cross-correlation and 
            surface-wave attenuation. GJI
        
        .. [2] Boschi et al., (2020). Erratum: On seismic ambient noise cross-correlation 
            and surface-wave attenuation. GJI
        
        .. [3] Magrini & Boschi (2021). Surface‐Wave Attenuation From Seismic Ambient 
            Noise: Numerical Validation and Application. JGR   
        """
        def load_done(path):
            if os.path.exists(path):
                return [int(i.strip()) for i in open(path)]
            return []
        
        def update_done(ipixel, path):
            with open(path, 'a') as f:
                f.write('%s\n'%ipixel)
                
        def overlapping_times(recording_stations, times_per_station, 
                              window_length):
            common_days = {}
            for sta1, sta2 in it.combinations(recording_stations, 2):
                times1 = times_per_station[sta1]
                times2 = times_per_station[sta2]
                common_times = np.intersect1d(times1, times2)
                common_days[(sta1, sta2)] = len(common_times) * window_length / 86400
            return common_days       
        
        def ram_required(stations):
            required = 0
            for station in stations:
                size = os.path.getsize(
                        os.path.join(src_ffts, '%s.npy'%station)
                        ) / 1000000
                required += size
            return required
        
        def load_ffts(ffts, stations, times):
            if ffts is None or isinstance(ffts, GeneratorType):
                ffts = {}
            else:
                ffts = {sta: ffts[sta] for sta in stations if sta in ffts}
            
            for sta in stations:
                if sta not in ffts:
                    ffts[sta] = np.load(os.path.join(src_ffts, '%s.npy'%sta))
                
            return ffts
            
        def load_ffts_generator(stations, times, ram_split, verbose):
            
            def starttime_and_endtime(stations, times):
                starttime = min([times[sta].min() for sta in stations])
                endtime = max([times[sta].max() for sta in stations])
                return starttime, endtime
                
            starttime, endtime = starttime_and_endtime(stations, times)
            timespan = (endtime - starttime) / ram_split
            for i in range(ram_split):
                if verbose:
                    print('--- loading part', i+1)
                start, end = starttime + i*timespan, starttime + (i+1)*timespan
                ffts = {}
                times_partial = {}
                for sta in stations:
                    itimes = np.flatnonzero((times[sta]>=start) \
                        & (times[sta]<end))
                    if itimes.size:
                        times_partial[sta] = times[sta][itimes]
                        ft = np.load(os.path.join(src_ffts, '%s.npy'%sta))
                        ffts[sta] = ft[itimes]               
                if ffts:
                    yield ffts, times_partial
                    del ffts, times_partial
                    gc.collect()
            
        def get_psd(psd_file, ffts, stations, times, window_length, 
                    min_no_pairs, min_no_days, ram_split, verbose):
            if os.path.exists(psd_file):
                return load_pickle(psd_file)
                
            psd_dict = compute_psd(ffts=ffts, 
                                   stations=stations, 
                                   times=times,  
                                   min_no_pairs=min_no_pairs)
            
            days_available = len(psd_dict) * window_length / 86400
            if days_available < min_no_days:
                return None
            save_pickle(psd_file, psd_dict)
            return psd_dict
        
        def compute_psd(ffts, stations, times, min_no_pairs):         
            if isinstance(ffts, GeneratorType):
                return compute_psd_generator(ffts_gen=ffts,
                                             min_no_pairs=min_no_pairs)            
            psd_dict = {}
            times_recordings = np.sort(
                    np.unique(np.concatenate([times[sta] for sta in stations]))
                    )            
            for time in times_recordings:
                avg_psd = compute_avg_psd(ffts=ffts,
                                          stations=stations, 
                                          times=times, 
                                          time=time, 
                                          min_no_pairs=min_no_pairs) 
                if avg_psd is not None:
                    psd_dict[time] = avg_psd    

            return psd_dict
            
        def compute_psd_generator(ffts_gen, min_no_pairs):
            psd_dict = {}
            for ffts, times_partial in ffts_gen:
                stations = sorted(times_partial.keys()) 
                psd_dict_tmp = compute_psd(ffts=ffts,
                                           stations=stations,
                                           times=times_partial,
                                           min_no_pairs=min_no_pairs)
                psd_dict.update(psd_dict_tmp)
            
            return psd_dict


        def compute_avg_psd(ffts, stations, times, time, min_no_pairs):
            psd = 0
            counter = 0
            for sta in stations:
                idx = np.flatnonzero(times[sta] == time)
                if idx.size:
                    ft = ffts[sta][np.asscalar(idx)]
                    psd += np.real(ft)**2 + np.imag(ft)**2
                    counter += 1
    
            if counter*(counter-1)/2 >= min_no_pairs:
                return psd / counter
            return None
                
        def compute_egfs(frequency, ffts, psd_dict, times, station_pairs, 
                         window_length, min_no_days, src_ffts, pixel_dir,
                         save_memory):

            for sta1, sta2 in station_pairs:
                save_file = os.path.join(pixel_dir, '%s__%s.npy'%(sta1, sta2))
                if os.path.exists(save_file):
                    continue
                if save_memory:
                    ffts = load_ffts(ffts=ffts, 
                                     stations=(sta1, sta2), 
                                     times=times)
                    
                egf = noisecorr_psd(sta1=sta1, 
                                    sta2=sta2, 
                                    ffts=ffts, 
                                    times=times, 
                                    psd_dict=psd_dict, 
                                    frequency=frequency,
                                    window_length=window_length, 
                                    min_no_days=min_no_days)
                if egf is not None:
                    os.makedirs(pixel_dir, exist_ok=True)
                    np.save(save_file, egf)

            
        def noisecorr_psd(sta1, sta2, ffts, times, psd_dict, frequency,
                          window_length, min_no_days):            
            times1, times2 = times[sta1], times[sta2]
            common_times = np.intersect1d(
                    np.intersect1d(times1, times2), list(psd_dict.keys())
                    )

            if common_times.size*window_length/86400 < min_no_days:
                return None

            idx1 = np.in1d(times1, common_times).nonzero()[0]
            idx2 = np.in1d(times2, common_times).nonzero()[0]
            common_times = times1[idx1]    
            
            ft1 = ffts[sta1][idx1]
            ft2 = ffts[sta2][idx2]
            psd = np.array([psd_dict[t] for t in common_times])
            
            corr_normalized = np.conj(ft1)*ft2 / psd
            egf = np.sum(corr_normalized, axis=0) / psd.shape[0]
            #In rare cases one of the PSD positions is zero ---> nans
            if np.any(np.isnan(egf)):
                mask = np.isfinite(egf)
                egf = np.interp(frequency, frequency[mask], egf[mask])
                
            return egf
        
        # MAIN FUNCTION COMPUTE_CORR_SPECTRA
        
        src_ffts = os.path.join(self.savedir, 'fft')
        save_corr = os.path.join(self.savedir, 'corr_spectra')
        save_psd = os.path.join(self.savedir, 'psd')
        os.makedirs(save_corr, exist_ok=True)
        os.makedirs(save_psd, exist_ok=True)
        
        freq = np.load(os.path.join(src_ffts, 'frequencies.npy'))
        np.save(os.path.join(save_corr, 'frequencies.npy'), freq)
        window_length = (freq.size-1) * 2 # seconds
        
        times = load_pickle(os.path.join(src_ffts, 'times.pickle'))
        save_done = os.path.join(self.savedir, 'corr_spectra', 'DONE.txt')
        
        done = load_done(save_done)
        ffts = None
        for ipixel in range(self.parameterization['grid'].shape[0]):
            if ipixel in done:
                continue
            if self.verbose:
                print('PIXEL %s'%ipixel)
            stations = self.parameterization['station_codes'][ipixel]
            common_days = overlapping_times(recording_stations=stations, 
                                            times_per_station=times,
                                            window_length=window_length)
            station_pairs = [k for k, v in common_days.items() if v>min_no_days]
            if len(station_pairs) < min_no_pairs:
                update_done(ipixel, save_done)
                if self.verbose:
                    print('- not enough simultaneous recordings')
                continue
            
            ram_estimate = ram_required(stations)
            save_memory = True if ram_estimate>ram_available else False
            if self.verbose:
                print('- loading FFTs (RAM required ~%.1f Mb)'%ram_estimate)
                if save_memory:
                    string = '- splitting the operation in %s to '%ram_split
                    string += 'avoid exceeding the RAM available '
                    string += "(see params ``ram_available'' and ``ram_split'')"
                    print(string)
                    
            if save_memory:
                del ffts
                gc.collect()
                ffts = load_ffts_generator(stations=stations, 
                                           times=times, 
                                           ram_split=ram_split, 
                                           verbose=self.verbose)
            else:
                ffts = load_ffts(ffts=ffts, stations=stations, times=times)
                gc.collect()
            
            psd_file = os.path.join(save_psd, 'pixel%s.pickle'%ipixel)
            psd_dict = get_psd(psd_file=psd_file, 
                               ffts=ffts, 
                               stations=stations, 
                               times=times,
                               window_length=window_length,
                               min_no_pairs=min_no_pairs,
                               min_no_days=min_no_days,
                               ram_split=ram_split,
                               verbose=self.verbose)
            if psd_dict is None:
                update_done(ipixel, save_done)
                if self.verbose:
                    print('- not enough simultaneous recordings')
                continue
            
            if self.verbose:
                print('- Computing EGFs for %s station pairs'%len(station_pairs))
            
            compute_egfs(frequency=freq, 
                         ffts=ffts, 
                         psd_dict=psd_dict, 
                         times=times, 
                         station_pairs=station_pairs, 
                         window_length=window_length, 
                         min_no_days=min_no_days, 
                         src_ffts=src_ffts, 
                         pixel_dir=os.path.join(save_corr, 'pixel%s'%ipixel),
                         save_memory=save_memory
                         )
            update_done(ipixel, save_done)
            gc.collect()
        
    
    def prepare_inversion(self, src_velocity, freqmin=0.05, freqmax=0.4, 
                          nfreq=300, smooth=False, smoothing_window=25, 
                          smoothing_poly=2):
        r""" Prepares the data set to be inverted for Rayleigh-wave attenuation.
        
        The data set used in the inversion, for each station pair, consists of 
        (i) envelope of the cross-spectrum, (ii) envelope of the associated 
        Bessel function :math:`J_0`, (iii) phase velocity, (iv) inter-station 
        distance. These objects are collected, for each grid cell associated 
        with the parameterization, for all station pairs available, and stored 
        into a dictionary object that is written on disk at 
        $self.savedir/inversion_dataset. The individual dictionaries are saved
        in the pickle format and named after the index of the grid cell in the
        belonging parameterization.
        
        Parameters
        ----------
        src_velocity : str
            Absolute path to the directory containing the phase-velocity
            dispersion curves associated with the station pairs for which
            cross-spectra are available
        
        freqmin, freqmax : float
            Frequency range to analyse. Default is 0.05 and 0.4 Hz
            
        nfreq : float
            The frequency range in question will be subdivided in `nfreq` points
            geometrically spaced (see `numpy.geomspace`)
            
        smooth : bool
            If `True`, the cross-spectra envelopes are smoothing with a 
            Savitzky-Golay filter. Default is `False`
            
        smoothing_window : odd int
            Passed to `scipy.signal.savgol_filter`. Default is 25
            
        smoothing_poly : int
            Passed to `scipy.signal.savgol_filter`. Default is 2
        """
        def load_tmp_data(corr_dir, src_velocity, velocity_files_dict,
                          stations, freqmin, freqmax, verbose):
            freq_tmp = np.geomspace(freqmin/5, freqmax*2, 2000)
            omega_tmp = 2 * np.pi * freq_tmp
            tmp_data = {}
            for corr_file in os.listdir(corr_dir):
                sta1, sta2 = corr_file.strip('.npy').split('__')
                if verbose:
                    print('-', sta1, sta2)
                dispcurve = load_dispcurve(sta1=sta1, 
                                           sta2=sta2, 
                                           src_velocity=src_velocity, 
                                           velocity_files_dict=velocity_files_dict)
                if dispcurve is not None:
                    if dispcurve[0, 1] / 10 < 1:
                        dispcurve[:, 1] *= 1000 # convert in m/s
                    c_w = interp1d(dispcurve[:,0], 
                                   dispcurve[:,1],
                                   bounds_error=False,
                                   fill_value=np.nan)(freq_tmp)
                    egf = np.load(os.path.join(corr_dir, corr_file))
                    dist = get_distance(sta1, sta2, stations)
                    tmp_data['%s__%s'%(sta1, sta2)] = {
                            'r': dist,
                            'c_w': c_w,
                            'egf': egf,
                            'j0': j0(omega_tmp * dist / c_w),
                            'y0': y0(omega_tmp * dist / c_w),
                            'freq': freq_tmp
                            }
            return tmp_data

        def load_dispcurve(sta1, sta2, src_velocity, velocity_files_dict):
            velocity_file = velocity_files_dict.get((sta1, sta2), None)
            if velocity_file is not None:
                return np.load(os.path.join(src_velocity, velocity_file))   
            return None                 
        
        def get_distance(sta1, sta2, stations):
            lat1, lon1 = stations[sta1]
            lat2, lon2 = stations[sta2]
            return gps2dist_azimuth(lat1, lon1, lat2, lon2)[0]

        def get_velocity_files_dict(src_velocity):
            velocity_files_dict = {}
            for file in os.listdir(src_velocity):
                if file.endswith('npy'):
                    tr1, tr2 = file.split('__')
                    sta1 = '.'.join(tr1.split('.')[:2])
                    sta2 = '.'.join(tr2.split('.')[:2])
                    velocity_files_dict[(sta1, sta2)] = file
            return velocity_files_dict
        
        def get_envelope(x, y, freq, smoothed=False, smoothing_poly=None, 
                         smoothing_window=None):
            """
            Return envelope at FREQUENCIES
            """
            y = np.abs(y)
            peaks = find_peaks(y)[0]
            spline = CubicSpline(x[peaks], 
                                 y[peaks], 
                                 bc_type='natural', 
                                 extrapolate=False)   
            envelope = spline(freq)
            if smoothed:
                notnan = np.flatnonzero(~np.isnan(envelope))
                envelope[notnan] = savgol_filter(envelope[notnan], 
                                                 smoothing_window, 
                                                 smoothing_poly)    
            if np.any(envelope < 0):
                envelope = np.where(envelope>0, envelope, np.nan)
            return envelope
        
        def get_data_dict(station_pair, tmp_dict, freq, freq_corr, **kwargs):
            c_w = interp1d(tmp_dict['freq'], 
                           tmp_dict['c_w'],
                           bounds_error=False,
                           fill_value=np.nan)(freq)
            try:
                smoothed_egf = velocity_filter(freq_corr, 
                                               tmp_dict['egf'], 
                                               tmp_dict['r']/1000, 
                                               p=0.1, 
                                               cmin=1.5, 
                                               cmax=5)
                envelope_egf = get_envelope(x=freq_corr,
                                            y=smoothed_egf.real,
                                            freq=freq,
                                            **kwargs)
                envelope_j0 = get_envelope(x=tmp_dict['freq'], 
                                           y=tmp_dict['j0'],
                                           freq=freq,
                                           **kwargs)
            except ValueError:
                return
            
            smoothed_egf = np.interp(freq, freq_corr, smoothed_egf, 
                                     left=np.nan, right=np.nan)
            bessel = np.interp(freq, tmp_dict['freq'], tmp_dict['j0'], 
                               left=np.nan, right=np.nan)
            return {
                    'egf': smoothed_egf, 
                    'j0': bessel, 
                    'egf_env': envelope_egf, 
                    'j0_env': envelope_j0,  
                    'c_w': c_w, 
                    'r': tmp_dict['r']
                    }    
   
        src_corr = os.path.join(self.savedir, 'corr_spectra')
        freq_corr = np.load(os.path.join(src_corr, 'frequencies.npy'))
        freq = np.geomspace(freqmin, freqmax, nfreq)
        save_dataset = os.path.join(self.savedir, 'inversion_dataset')
        os.makedirs(save_dataset, exist_ok=True)
        np.save(os.path.join(save_dataset, 'frequencies.npy'), freq)
        velocity_files_dict = get_velocity_files_dict(src_velocity) 
        
        envelope_kwargs = dict(smoothed=smooth, 
                               smoothing_poly=smoothing_poly,
                               smoothing_window=smoothing_window)
        
        pixels = sorted([i for i in os.listdir(src_corr) if 'pixel' in i],
                         key=lambda i: int(i.strip('pixel')))
        for pixel in pixels:
            if os.path.exists(os.path.join(save_dataset, '%s.pickle'%pixel)):
                continue        
            if self.verbose:
                print(pixel.upper())
            tmp_data = load_tmp_data(corr_dir=os.path.join(src_corr, pixel),
                                     src_velocity=src_velocity,
                                     velocity_files_dict=velocity_files_dict,
                                     stations=self.stations,
                                     freqmin=freqmin,
                                     freqmax=freqmax,
                                     verbose=self.verbose)
            data = {}
            for station_pair, tmp_dict in tmp_data.items():      
                data_dict = get_data_dict(station_pair=station_pair,
                                          tmp_dict=tmp_dict,
                                          freq=freq,
                                          freq_corr=freq_corr,
                                          **envelope_kwargs)
                if data_dict is not None:
                    data[station_pair] = data_dict
            if data:
                save_pickle(os.path.join(save_dataset, '%s.pickle'%pixel), data)
                
    
    def inversion(self, alphamin=5e-7, alphamax=5e-4, nalpha=350, min_no_pairs=6):
        r""" Carries out the inversion for Rayleigh-wave attenuation
        
        For each pickle file available in $self.savedir/inversion_dataset,
        an inversion is carried out by minimizing the misfit between the corr-
        spectra and the corresponding Bessel functions, so as to retrieve
        one attenuation curve for each pixel [1]_. 
        
        The results (one attenuation curve per pickle file), are saved
        at $self.savedir/results, and consist of (i) best_alphas.pickle: 
        a dictionary object containing, for each grid cell, the retrieved 
        attenuation curve and the number of station pairs used to constrain it;
        (ii) frequencies.npy: frequency vector associated with the attenuation
        curves; (iii) pixel$d.pickle (where $d indicates the index of the grid 
        cell in $self.savedir/parameterization.pickle): dictionary object
        containing, along with the information stored in (i), the minimized
        misfit matrix to retrieve the attenuation curve.
        
        
        Parameters
        ----------
        alphamin, alphamax : float
            Attenuation range (in 1/m) within which the best-fitting 
            attenuation is sought as a function of frequency. Defaults are 
            5e-7 and 5e-4
            
        nalpha : int
            Number of values of :math:`\alpha` used to subdivide the attenuation 
            range geometrically (see `numpy.geomspace`). Default is 350
        
        min_no_pairs : int
            Minimum number of stations pairs required to carry out the 
            inversion. If, in a given sub-array, the number of available data
            is smaller than this number, the pixel is not inverted for. Default
            is 6. Smaller values are suggested against.
        
        References
        ----------
        .. [1] Magrini & Boschi (2021). Surface‐Wave Attenuation From Seismic 
            Ambient Noise: Numerical Validation and Application. JGR  
        """
        def nans_where_allzeros(matrix):
            matrix = matrix.T
            for row in matrix:
                if not np.any(row):
                    row[:] = np.nan                
            return matrix.T
        
        def get_best_alpha(total_cost, alphas):
            best_alphas = []
            for row in total_cost.T:
                if np.all(np.isnan(row)):
                    best_alphas.append(np.nan)
                else:
                    best_alphas.append(alphas[np.nanargmin(row)])
        #    idx = [np.nanargmin(i) if not np.all(np.isnan(i)) else np.nan for i in total_cost.T]
            return np.array(best_alphas)
        
        def cost_function(j0_env, egf_env, alpha, r):
            model = j0_env * np.exp(-alpha*r)
            cost = np.abs(egf_env - model)**2
            return model, cost * r**2
        
        def get_no_measurements(cost_matrix):
            no_measurements = np.nansum(cost_matrix, axis=0)
            return np.where(no_measurements>0, 1, 0)
        
        def invert_data(data, freq, alphas):
            results = {}
            total_cost = 0
            no_measurements = 0
            for pair in data:
                cost_matrix = np.zeros((alphas.size, freq.size))
                egf_env = data[pair]['egf_env']
                j0_env = data[pair]['j0_env']
                r = data[pair]['r'] 
                for ialpha, alpha in enumerate(alphas):
                    model, cost = cost_function(j0_env=j0_env,
                                                egf_env=egf_env,
                                                alpha=alpha,
                                                r=r)
                    cost_matrix[ialpha,:] = cost
                    
                no_measurements += get_no_measurements(cost_matrix)
#                results[pair] = cost_matrix
                total_cost += np.where(np.isnan(cost_matrix), 0, cost_matrix) 
                
            results['cost'] = nans_where_allzeros(total_cost)   
            results['best_alphas'] = get_best_alpha(total_cost, alphas)
            results['no_measurements'] = no_measurements
            return results
        
        src_dataset = os.path.join(self.savedir, 'inversion_dataset')
        save = os.path.join(self.savedir, 'results')
        os.makedirs(save, exist_ok=True)
        freq = np.load(os.path.join(src_dataset, 'frequencies.npy'))
        np.save(os.path.join(save, 'frequencies.npy'), freq)
        alphas = np.geomspace(alphamin, alphamax, nalpha)
        np.save(os.path.join(save, 'alphas.npy'), alphas)
        
        pixels = sorted([i for i in os.listdir(src_dataset) if 'pixel' in i],
                         key=lambda i: int(i.strip('pixel.pickle')))
        best_alpha_dict = {}
        for pixel in pixels:
            if self.verbose:
                print('PIXEL%s'%pixel.strip('pixel.pickle'))
            data = load_pickle(os.path.join(src_dataset, pixel))
            results = invert_data(data=data, freq=freq, alphas=alphas)
            if np.all(results['no_measurements'] < min_no_pairs):
                continue
            ipixel = int(pixel.strip('.ceiklpx'))
            best_alpha_dict[ipixel] = (
                    results['best_alphas'], results['no_measurements']
                    )                            
            save_pickle(os.path.join(save, pixel), results)
            
        save_pickle(os.path.join(save, 'best_alphas.pickle'), best_alpha_dict)
            
    
    def get_attenuation_map(self, period, cell_size, min_overlapping_pixels=2):
        r""" 
        Calculates an attenuation maps from the inversion results.
        
        The map will be parameterized using an equal-area grid, built thorugh
        :class:`seislib.tomography.grid.EqualAreaGrid`
        
        Parameters
        ----------
        period : float, int
        
        cell_size : float, int
            Size of each grid cell (in degrees) of the resulting 
            parameterization
            
        min_overlapping_pixels: int
            At a given period, each value of :math:`\alpha` in the resulting 
            map will be given by a weighted average of the attenuation curves 
            retrieved in the inversion (the weights are determined by the 
            number of station pairs used in the inversion of each sub-array). 
            This parameters determines the minimum number of attenuation 
            curves used to constrain :math:`\alpha` in a given pixel of the 
            resulting map. Default is 2
            
        Returns
        -------
        new_mesh : ndarray of shape (n, 4)
            Parameterization of the final map, where each row corresponds to
            a grid cell and the columns define its boundaries (lat1, lat2, 
            lon1, lon2)
            
        attenuation : ndarray of shape (n,)
            Value of attenuation associated with each grid cell of 
            `new_mesh`
        """
        def interpolate_results(freq, results, period):
            alphas = np.zeros(mesh.shape[0])
            no_measurements = np.zeros(mesh.shape[0])
            for i, (pixel, (a, n)) in enumerate(sorted(results.items())):
                alphas[i] = np.interp(1/period, freq, a)
                no_measurements[i] = int(np.interp(1/period, freq, n))
            return alphas, no_measurements
        
        def get_related_pixels(mesh, lon1, lon2, lat1, lat2):
            lats1, lats2, lons1, lons2 = mesh.T
            idx = np.flatnonzero(
                   (((lon1>lons1) & (lon1<lons2) & \
                     (lat1>lats1) & (lat1<lats2))) | \
                   (((lon2>lons1) & (lon2<lons2) & \
                     (lat1>lats1) & (lat1<lats2))) | \
                   (((lon1>lons1) & (lon1<lons2) & \
                     (lat2>lats1) & (lat2<lats2))) | \
                   (((lon2>lons1) & (lon2<lons2) & \
                     (lat2>lats1) & (lat2<lats2))) 
                   )       
            return idx

        src_results = os.path.join(self.savedir, 'results')        
        freq = np.load(os.path.join(src_results, 'frequencies.npy'))
        mesh = self.parameterization['grid']
        results = load_pickle(os.path.join(src_results, 'best_alphas.pickle'))
        alphas, no_measurements = interpolate_results(freq=freq,
                                                      results=results, 
                                                      period=period)
        
        mesh = mesh[sorted(results)]
        latmin, lonmin = np.min(mesh, axis=0)[::2]
        latmax, lonmax = np.max(mesh, axis=0)[1::2]
        new_mesh = EqualAreaGrid(cell_size=cell_size, 
                                 lonmin=lonmin,
                                 lonmax=lonmax,
                                 latmin=latmin,
                                 latmax=latmax).mesh
        ipixels = []              
        attenuation = []
        for i, (lat1, lat2, lon1, lon2) in enumerate(new_mesh):
            related_pixels = get_related_pixels(mesh, 
                                                lon1=lon1, 
                                                lon2=lon2, 
                                                lat1=lat1, 
                                                lat2=lat2)
            if related_pixels.size < min_overlapping_pixels:
                continue
            measures = alphas[related_pixels]
            weights = no_measurements[related_pixels]
            attenuation.append(np.sum(measures * weights) / weights.sum())
            ipixels.append(i)
        new_mesh = new_mesh[ipixels]
        
        return new_mesh, np.array(attenuation)
        
        
    @classmethod
    def plot_map(cls, mesh, c, ax=None, projection='Mercator', map_boundaries=None, 
                 bound_map=True, colorbar=True, show=True, style='colormesh', 
                 add_features=False, resolution='110m', cbar_dict={}, **kwargs):
        """ 
        Displays an attenuation map.
        
        Parameters
        ----------
        mesh : ndarray of shape (n, 4)
            Grid cells in SeisLib format, consisting of n pixels described by 
            lat1, lat2, lon1, lon2 (in degrees).
            
        c : ndarray of shape (n,)
            Values associated with the grid cells (mesh)
            
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not `None`, `c` is plotted on the `GeoAxesSubplot` instance. 
            Otherwise, a new figure and `GeoAxesSubplot` instance is created
            
        projection : str
            Name of the geographic projection used to create the `GeoAxesSubplot`.
            (Visit the `cartopy website 
            <https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html>`_ 
            for a list of valid projection names.) If ax is not None, `projection` 
            is ignored. Default is 'Mercator'
            
        map_boundaries : iterable of floats, shape (4,), optional
            Lonmin, lonmax, latmin, latmax (in degrees) defining the extent of
            the map
            
        bound_map : bool
            If `True`, the map boundaries will be automatically determined.
            Default is `False`
        
        colorbar : bool
            If `True` (default), a colorbar associated with `c` is displayed on 
            the side of the map
            
        show : bool
            If `True` (default), the map will be showed once generated
            
        style : {'colormesh', 'contourf', 'contour'}
            Possible options are 'colormesh', 'contourf', and 'contour'.
            Default is 'colormesh', corresponding to :meth:`colormesh`.

        add_features : bool
            If `True`, natural Earth features will be added to the 
            `GeoAxesSubplot` through 
            :meth:`seislib.plotting.plotting.add_earth_features`.
            Default is `False`. If `ax` is `None`, it is automatically set 
            to `True`
            
        resolution : {'10m', '50m', '110m'}
            Resolution of the Earth features displayed in the figure. Passed to
            `cartopy.feature.NaturalEarthFeature`. Valid arguments are '110m',
            '50m', '10m'. Default is '110m'. Ignored if `ax` is not `None`.
            
        cbar_dict : dict, optional
            Keyword arguments passed to `matplotlib.colorbar.ColorbarBase`
         
        **kwargs : dict, optional
             Additional inputs passed to the method defined by `style`.
            
        Returns
        -------
        `None` if show is `False` else an instance of 
        `matplotlib.collections.QuadMesh` (together with an instance
        of `matplotlib.colorbar.Colorbar`, if colorbar is True)
        """
        norm = kwargs.pop('norm', LogNorm())
        cmap = kwargs.pop('cmap', 'inferno')
        
        return plot_map(mesh=mesh, 
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
                        norm=norm,
                        cmap=cmap,
                        cbar_dict=cbar_dict,
                        **kwargs)
         
        
    @classmethod
    def colormesh(cls, mesh, c, ax, **kwargs):
        """
        Adaptation of `matplotlib.pyplot.pcolormesh` to an (adaptive) 
        equal-area grid.
        
        Parameters
        ----------
        mesh : ndarray of shape (n, 4)
            Equal area grid, consisting of n pixels described by lat1, lat2, 
            lon1, lon2
            
        c : array-like of shape (n,)
            Values to plot in each grid cell
            
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not `None`, the receivers are plotted on the `GeoAxesSubplot` 
            instance. Otherwise, a new figure and `GeoAxesSubplot` instance 
            is created
            
        **kwargs : dict, optional 
            Additional inputs passed to `seislib.plotting.plotting.colormesh`
        
        Returns
        -------
        img : Instance of matplotlib.collections.QuadMesh
        """
        return colormesh(mesh, c, ax=ax, **kwargs)
        
    
    @classmethod
    def contourf(cls, mesh, c, ax, smoothing=None, **kwargs):
        """
        Adaptation of `matplotlib.pyplot.contourf` to an (adaptive) 
        equal-area grid
        
        Parameters
        ----------
        mesh : ndarray of shape (n, 4)
            Equal area grid, consisting of n pixels described by lat1, lat2, 
            lon1, lon2
            
        c : array-like of shape (n,)
            Values to plot in each grid cell
            
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not `None`, the receivers are plotted on the `GeoAxesSubplot` 
            instance. Otherwise, a new figure and `GeoAxesSubplot` instance 
            is created
             
        smoothing : float, optional
            Value passed to `scipy.ndimage.filters.gaussian_filter` and used to
            obtain a smooth representation of `c` (default is `None`)
            
        **kwargs : dict, optional 
            Additional inputs passed to `seislib.plotting.plotting.colormesh`
        
        Returns
        -------
        img : Instance of matplotlib.contour.QuadContourSet
        """
        return contourf(mesh, c, ax=ax, smoothing=smoothing, **kwargs)


    @classmethod     
    def contour(cls, mesh, c, ax, smoothing=None, **kwargs):
        """
        Adaptation of matplotlib.pyplot.contour to the (adaptive) equal area 
        grid
        
        Parameters
        ----------
        mesh : ndarray of shape (n, 4)
            Equal area grid, consisting of n pixels described by lat1, lat2, 
            lon1, lon2
            
        c : array-like of shape (n,)
            Values to plot in each grid cell
            
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not `None`, the receivers are plotted on the `GeoAxesSubplot` 
            instance. Otherwise, a new figure and `GeoAxesSubplot` instance 
            is created
             
        smoothing : float, optional
            Value passed to `scipy.ndimage.filters.gaussian_filter` and used to
            obtain a smooth representation of `c` (default is `None`)
            
        **kwargs : dict, optional 
            Additional inputs passed to `seislib.plotting.plotting.colormesh`
        
        Returns
        -------
        img : Instance of matplotlib.contour.QuadContourSet
        """
        return contour(mesh, c, ax, smoothing=smoothing, **kwargs)
    
    
    def plot_stations(self, ax=None, show=True, oceans_color='water', 
                      lands_color='land', edgecolor='k', projection='Mercator',
                      resolution='110m', color_by_network=True, legend_dict={}, 
                      **kwargs):
        """ Maps the seismic receivers for which data are available
        
        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not `None`, the receivers are plotted on the `GeoAxesSubplot` instance. 
            Otherwise, a new figure and `GeoAxesSubplot` instance is created
            
        show : bool
            If `True`, the plot is shown. Otherwise, a `GeoAxesSubplot` instance is
            returned. Default is `True`
            
        oceans_color, lands_color : str
            Color of oceans and lands. The arguments are ignored if ax is not
            `None`. Otherwise, they are passed to `cartopy.feature.NaturalEarthFeature` 
            (to the argument 'facecolor'). Defaults are 'water' and 'land'
            
        edgecolor : str
            Color of the boundaries between, e.g., lakes and land. The argument 
            is ignored if ax is not `None`. Otherwise, it is passed to 
            `cartopy.feature.NaturalEarthFeature` (to the argument 'edgecolor'). 
            Default is 'k' (black)
            
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
            
        color_by_network : bool
            If `True`, each seismic network will have a different color in the
            resulting map, and a legend will be displayed. Otherwise, all
            stations will have the same color. Default is `True`
        
        legend_dict : dict, optional
            Dictionary of keyword arguments passed to `matplotlib.pyplot.legend`
        
        **kwargs : dict, optional
            Additional keyword arguments passed to `matplotlib.pyplot.scatter` 
            
        Returns
        -------
        `None` if `show` is `True`, else `ax`, i.e. the GeoAxesSubplot
        """                
        return plot_stations(stations=self.stations,
                             ax=ax, 
                             show=show, 
                             oceans_color=oceans_color, 
                             lands_color=lands_color, 
                             edgecolor=edgecolor, 
                             projection=projection,
                             resolution=resolution,
                             color_by_network=color_by_network, 
                             legend_dict=legend_dict,
                             **kwargs)
        
    
    def plot_azimuthal_coverage(self, ipixel, ax=None, bins=None, show=True,
                                **kwargs):
        """ Plots a polar histogram of the azimuthal coverage for one pixel

        Parameters
        ----------
        ipixel : int
            Index of the pixel used in the inversion for alpha
            
        ax : matplotlib instance of PolarAxesSubplot, optional
            If `None`, a new figure and `PolarAxesSubplot` is created
            
        bins : array-like, optional
            Azimuthal bins (in degrees). If `None`, bins of 30 degrees 
            between 0 and 360 are used
            
        show : bool
            If `False`, `ax` is returned. Else, the figure is displayed.
            Default is `True`
            
        **kwargs : dict, optional 
            Additional arguments passed to `matplotlib.pyplot.bar`     
            
            
        Returns
        -------
        `None` if `show` is `True`, else `ax`
        """
        data = load_pickle(os.path.join(self.savedir, 
                                        'inversion_dataset',
                                        'pixel%s.pickle'%ipixel))
        pairs = [i.split('__') for i in data]
        coords = np.array([[*self.stations[pair[0]], *self.stations[pair[1]]] \
                           for pair in pairs])
        azimuths = np.array([*azimuth_backazimuth(*coords.T)]).ravel()
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

        bins = bins if bins is not None else np.arange(0, 360+30, 30)
        bin_size = np.radians(abs(bins[1] - bins[0]))
        hist, bins = np.histogram(azimuths, bins=bins)
        centers = np.radians(np.ediff1d(bins)//2 + bins[:-1])
                
        color = kwargs.pop('color', '.8')
        bottom = kwargs.pop('bottom', 0)
        edgecolor = kwargs.pop('edgecolor', 'k')
        ax.bar(centers, 
               hist, 
               width=bin_size, 
               bottom=bottom, 
               color=color, 
               edgecolor=edgecolor,
               **kwargs)
        ax.tick_params(axis='both', which='major')
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2.0)
        if not show:
            return ax
        plt.show()
        
    
    def plot_cost(self, ipixel, ax=None, alphamin=None, alphamax=None, 
                  freqmin=None, freqmax=None, show=True, curve_dict={},
                  **kwargs):
        """ Plots the values of cost obtained in the inversion of a given pixel
        
        Parameters
        ----------
        ipixel : int
            Index of the pixel used in the inversion for alpha
            
        ax : matplotlib instance of AxesSubplot, optional
            If `None`, a new figure and `ax` is created
            
        show : bool
            If `False`, `ax` is returned. Else, the figure is displayed.

        alphamin, alphamax : float, optional
            Alpha range used to bound the yaxis in the figure. If `None`, 
            the whole alpha range used in the inversion is displayed
            
        freqmin, freqmax : float, optional
            Frequency range used to bound the xaxis in the figure. If `None`, 
            the whole Frequency range used in the inversion is displayed
            
        show : bool
            If `False`, `ax` is returned. Else, the figure is displayed.
            Default is `True`
            
        curve_dict : dict, optional
            Arguments passed to `matplotlib.pyplot.plot`, to control the 
            aspect of the attenuation curve
        
        **kwargs : dict, optional
            Additional arguments passed to `matplotlib.pyplot.pcolormesh`


        Returns
        -------
        If `show` is `True`, `None`, else a `(2,) tuple` containing

        - `matplotlib.collections.QuadMesh` (associated with pcolormesh); 
        - `matplotlib.pyplot.colorbar`
        """
        def normalize_cost(matrix):
            matrix = matrix.T.copy()
            for row in matrix:
                if not np.all(np.isnan(row)):
                    assert not np.any(np.isnan(row))
                    max_row, min_row = np.max(row), np.min(row)
                    row[:] = (row-min_row) / (max_row-min_row)
            return matrix.T
        
        src = os.path.join(self.savedir, 'results')
        result = load_pickle(os.path.join(src, 'pixel%s.pickle'%ipixel))
        alphas_dict = load_pickle(os.path.join(src, 'best_alphas.pickle'))
        frequency = np.load(os.path.join(src, 'frequencies.npy'))
        alpha = np.load(os.path.join(src, 'alphas.npy'))
        best_alphas, no_pairs = alphas_dict[ipixel]
        cost = result['cost']
        cost_normalized = normalize_cost(cost)
        
        no_stations = self.parameterization['no_stations'][ipixel]
        no_measurements = max(result['no_measurements'])
        text = 'Stations: %s\n'%no_stations
        text += 'Cross-spectra: %s'%no_measurements
        
        alphamin = alphamin if alphamin is not None else alpha.min()
        alphamax = alphamax if alphamax is not None else alpha.max()
        freqmin = freqmin if freqmin is not None else frequency.min()
        freqmax = freqmax if freqmax is not None else frequency.max()
        cmap = kwargs.pop('cmap', 'Greys_r')
        if 'norm' not in kwargs:
            min_lev = np.nanmin(cost_normalized)
            max_lev = np.nanmax(cost_normalized)
            norm = plt.Normalize(vmin=min_lev, vmax=max_lev/30)
            
        bbox_props = dict(boxstyle='round', 
                          fc='w', 
                          ec='none', 
                          lw=1.5, 
                          pad=0.1, 
                          alpha=0.85)
        
        if ax is None:
            fig, ax = plt.subplots()
        img = ax.pcolormesh(frequency, 
                       alpha, 
                       cost_normalized, 
                       norm=norm, 
                       cmap=cmap,
                       **kwargs)
        ax.plot(frequency, best_alphas, **curve_dict)
        cb = make_colorbar(ax, img, orientation='vertical')
        # cb = fig.colorbar(img, ax=ax, pad=0.02)
        cb.set_label(label='Normalized cost')
        cb.ax.ticklabel_format(style='sci', scilimits=(0,0))
        ax.set_yscale('log')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'$\alpha\ [m^{-1}]$')
        ax.tick_params(axis='both', which='both', direction='in')
        ax.set_ylim(alphamin, alphamax)
        ax.set_xlim(frequency.min(), frequency.max())
        ax.text(0.02, 0.97, s=text, va='top', ha='left', transform=ax.transAxes, 
                bbox=bbox_props)
        if not show:
            return img, cb
        plt.show()
        
    
    
    
    

