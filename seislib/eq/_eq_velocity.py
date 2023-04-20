#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Inter-Station Dispersion Curves
=============================== 

The below classes provide automated routines to calculate
Rayleigh and Love inter-station dispersion curves based on teleseismic
earthquakes, through the two-station method.

"""

import os
import shutil
import itertools as it
from collections import defaultdict
import numpy as np
from numpy import pi
from numpy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
from obspy import read, read_events
from obspy.geodetics.base import gps2dist_azimuth
from obspy.io.sac.util import SacIOError
from seislib.utils import adapt_timespan, zeropad, bandpass_gaussian
from seislib.utils import load_pickle, save_pickle, remove_file
from seislib.utils import gaussian, skewed_normal, next_power_of_2
from seislib.plotting import plot_stations, plot_events
from seislib.exceptions import DispersionCurveException
from seislib.tomography import SeismicTomography



def _read(*args, verbose=True, **kwargs):
    try:
        return read(*args, **kwargs)
    except SacIOError:
        if verbose:
            print('SacIOError', *args)
            

class EQVelocity:
    """
    Class to obtain surface-wave phase velocities from teleseismic earthquakes,
    using the two-station method.
    
    Parameters
    ----------
    src : str
        Absolute path to the directory containing the files associated with 
        the earthquake recordings. The directory should contain,
        for each event, a sub-directory named after the origin time of the
        earthquake in timestamp format (see `obspy.UTCDateTime.timestamp`).
        Inside each sub-directory, the seismograms associated with the
        respective event recorded at all the available receivers should be
        stored into sacfiles. These should be named after the trace id, 
        i.e., net.sta.loc.cha.sac (see `obspy.Trace.id`). Furthermore, inside 
        each event directory there should be either an xml containing the 
        event coordinates (latitude, longitude), or these coordinates should 
        be stored in the sac header of the related seismograms. 
        
        .. note::
            To download the data in the proper format, it is suggested to use
            `seislib.eq.EQDownloader`.

    savedir : str, optional
        Absolute path to the directory where the results are saved. If not
        provided, its value will be set to the parent directory of `src` (i.e.,
        $src/..). All the results will be saved into the directory
        $savedir/eq_velocity/$component (see the `component` parameter)

    component : {'Z', 'R', 'T'}
        Either 'Z', 'R', (corresponding to vertically and radially  polarized 
        Rayleigh waves, respectively), or 'T' (for Love waves)

    verbose : bool
        Whether or not information on progress is printed in the console

    Attributes
    ----------
    src : str
        Absolute path to the directory containing the files associated with the
        earthquake-based recordings

    savedir : str
        Absolute path to the directory where the results are saved
    
    component : {'Z', 'R', or 'T'}
    
    events : list
        List of available events to calculate the dispersion curves
    
    verbose : bool
        Whether or not information on progress is printed in the console
    
    stations : dict
        Coordinates of the station pairs that can be employed to retrieve
        dispersion curves (can be accessed after calling the method
        `prepare_data`)
    
    triplets : dict
        Triplets of epicenters-receivers that can be employed to retrieve
        dispersion curves (can be accessed after calling the method
        `prepare_data`)        
        
    Examples
    --------
    The following will calculate the dispersion curves from teleseismic
    earthquakes recorded at a set of stations. These are stored in the directory 
    src='/path/to/data'. The data have been downloaded using 
    :class:`seislib.eq.eq_downloader.EQDownloader`. The dispersion curves are extracted 
    for vertically-polarized Rayleigh waves (on the vertical - Z - component). We first 
    define the `EQVelocity` instance and prepare the data for the subsequent analysis:
    
    >>> from seislib.eq import EQVelocity
    >>> eq = EQVelocity(src, component='Z')
    >>> eq.prepare_data(azimuth_tolerance=7, 
                        distmin=None, 
                        distmax=2500, 
                        min_no_events=8)
    >>> print(eq)
    RAYLEIGH-WAVE VELOCITY FROM TELESEISMIC EARTHQUAKES (TWO-STATION METHOD)
            =========================================================================
            EVENTS: X
            RECEIVERS: N.A.
            TRIPLETS (EPICENTER-RECEIVERS) AVAILABLE: N.A.
            DISPERSION CURVES RETRIEVED: Y
            =========================================================================
            SOURCE DIR: /path/to/data
            SAVE DIR: /path/to/eq_velocity/Z
        
    The above will find, among our seismograms, station pairs approximately lying (azimuthal tolerance of 
    :math:`\pm7^{\circ}`) on the same great-circle path as the epicenters. We set a maximum inter-station 
    distance of 2500 km, and all pairs of receivers for which less than 8 events are available for 
    measuring the dispersion curves are discarded. Printing `eq` will give information on the data available.
    (In the above output, X and Y are integers depending on your data.) 
        
    .. note::
        :meth:`prepare_data` will extract the geographic coordinates of each seismic 
        receivers from the header of the corresponding sac files, those of the seismic events, 
        and store the information on the triplets of epicenters-receivers that will be used to 
        obtain the dispersion curves.

        The geographic coordinates of the seismic receivers are saved at 
        $eq.savedir/stations.pickle, and are stored in a dictionary object 
        where each key corresponds to a station code ($network_code.$station_code) 
        and each value is a tuple containing latitude and longitude of the station.
        For example::
            { 
            net1.sta1 : (lat1, lon1),
            net1.sta2 : (lat2, lon2),
            net2.sta3 : (lat3, lon3)
                }
    
    The geographic coordinates of the events are saved at 
    $eq.savedir/events.pickle, and have a similar structure to the above:
    each key corresponds to an event origin time (in the
    obspy.UTCDateTime.timestamp format), and each value is (lat, lon, mag) of
    the earthquake, where mag is magnitude.

    The triplets of epicenter-receivers are saved at $eq.savedir/triplets.pickle, 
    and are stored in a dictionary object where each key is a tuple corresponding 
    to a station pair and each value is a list of events that are aligned on the 
    same great circle path as the receivers (within the limits defined by the 
    input params of :meth:`prepare_data`). For example::
        
        { 
        (net1.sta1, net2.sta2) : ['1222701570.84',
                                  '1237378863.3', 
                                  '1237486660.74',
                                  '1238981562.62',
                                  '1239825695.33',
                                  etc.],
            
        (net3.sta3, net4.sta4) : ['1222701570.84',
                                  '1237378863.3',
                                  '1237486660.74',
                                  '1238981562.62',
                                  '1239825695.33',
                                  etc.]
            }
        
    .. note:: 
        Each event origin time corresponds to a sub-directory of
        the source data (`src`). Since the `min_no_events` passed to 
        :meth:`prepare_data` is 8, the length of the lists of events 
        associated with each receiver pair will be at least 8.

    To visualize the location of the epicenters that will be used
    to calculate inter-station dispersion curves, run, for example,::

        eq.plot_events(color='r', 
                       min_size=5, 
                       max_size=250, 
                       marker='*', 
                       legend_dict=dict(fontsize=12))
    
    We can now extract the dispersion curves in a automated fashion. 
    The following will allow us to extract dispersion measurements at 
    75 different surface-wave periods, linearly spaced in the range 15-150 s. 
    All  values of phase velocity outside the velocity range 2.5-5 km/s
    will be discarded, and will only periods for which a ratio between
    inter-station distance and wavelength > 1 will be considered. (The 
    wavelength is inferred from the reference curve.)::

        eq.extract_dispcurves(refcurve, 
                              periodmin=15, 
                              periodmax=150, 
                              no_periods=75, 
                              cmin=2.5, 
                              cmax=5, 
                              min_no_wavelengths=1, 
                              plotting=True)
    """
    
    def __init__(self, src, savedir=None, component='Z', verbose=True):   
        self.src = src
        savedir = os.path.dirname(src) if savedir is None else savedir
        self.verbose = verbose
        if component not in ['Z', 'R', 'T']:
            msg = '`component` should be either `Z` (Rayleigh, vertical), `T`'
            msg += ' (Love, transverse), or `R` (Rayleigh, radial), but'
            msg += ' `%s` was passed.'%component
            raise ValueError(msg)
        self.component = component
        self.savedir = os.path.join(savedir, 'eq_velocity', component)
        os.makedirs(self.savedir, exist_ok=True)
        self.events = sorted([i for i in os.listdir(self.src) \
                              if os.path.isdir(os.path.join(self.src, i))])
        
        
    def __str__(self):
        phase = 'love'if self.component=='T' else 'rayleigh'
        if phase=='rayleigh' and self.component=='R':
            phase = phase + ' (radial)'
        string = '\n%s-WAVE VELOCITY FROM TELESEISMIC EARTHQUAKES'%(phase.upper())
        string += ' (TWO-STATION METHOD)'
        separators = len(string)
        string += '\n%s'%('='*separators)
        no_events = len(self.events)
        string += '\nEVENTS: %s'%(no_events)
        no_stations = len(self.stations) if 'stations' in self.__dict__ else 'N.A.'
        string += '\nRECEIVERS: %s\n'%(no_stations)
        try:
            done = len(os.listdir(os.path.join(self.savedir, 'dispcurves')))
        except FileNotFoundError:
            done = 0
        triplets = len(self.triplets) if 'triplets' in self.__dict__ else 'N.A.'
        string += 'TRIPLETS (EPICENTER-RECEIVERS) AVAILABLE: %s\n'%(triplets)
        string += 'DISPERSION CURVES RETRIEVED: %s'%(done)
        string += '\n%s'%('='*separators)
        string += '\nSOURCE DIR: %s'%self.src
        string += '\nSAVE DIR: %s'%self.savedir

        return string

    
    def __repr__(self):
        return str(self)
    
    
    def get_coords_and_triplets(self, events, azimuth_tolerance=5, distmin=None,
                                distmax=None):
        """ 
        Retrieves stations, events information, and the triplets of 
        epicenter-receivers to be used to calculate the phase velocities
        
        Parameters
        ----------
        events : list
            List of events (i.e., sub-directories in the data folder `src`)
        
        azimuth_tolerance : float
            Maximum allowed deviation from the great circle path in degrees
        
        distmin, distmax : float, optional
            Minimum and maximum allowed inter-station distance (in km). Default 
            is None
        
        Returns
        -------
        stations : dict
            Each key corresponds to a station code ($network_code.$station_code) 
            and each value is a tuple containing latitude and longitude of the 
            station. For example::
                
                { net1.sta1 : (lat1, lon1),
                  net1.sta2 : (lat2, lon2),
                  net2.sta3 : (lat3, lon3)
                  }
        
        events_info : dict
            Each key corresponds to an event origin time and each value is a 
            tuple containing latitude, longitude, and magnitude of the event::
                    
                { '1222701570.84' : (lat1, lon1, mag1),
                    '1237486660.74' : (lat2, lon2, mag2),
                    '1239825695.33' : (lat3, lon3, mag3)
                    }                 
        
        triplets : dict
            Each key is a tuple corresponding to a station pair and each value 
            is a list of events that are aligned on the same great circle path
            as the receivers (within the limits defined by the input params). 
            For example::
                
                { (net1.sta1, net2.sta2) : ['1222701570.84',
                                            '1237378863.3', 
                                            '1237486660.74',
                                            '1238981562.62',
                                            '1239825695.33'],
                    
                  (net3.sta3, net4.sta4) : ['1222701570.84',
                                            '1237378863.3', 
                                            '1237486660.74',
                                            '1238981562.62',
                                            '1239825695.33']
                  }
            Note that each event in the list corresponds to a sub-directory of
            the source data `src`.
        """
        def get_event_coords_from_xml(eventfile):
            event = read_events(eventfile)[0]
            evla = event.origins[0].latitude
            evlo = event.origins[0].longitude
            mag = event.magnitudes[0].mag
            return evla, evlo, mag
        
        def get_event_coords_from_sac(sacfile):
            tr = _read(sacfile, 
                       headonly=True,
                       verbose=self.verbose)[0]
            evla = tr.stats.sac.evla
            evlo = tr.stats.sac.evlo
            mag = tr.stats.sac.mag
            return evla, evlo, mag
        
        def get_station_coords(evdir, sacfile):
            nonlocal stations
            code = '.'.join(sacfile.split('.')[:2])
            if code in stations:
                return code, stations[code]
            st = _read(os.path.join(evdir, sacfile), 
                       headonly=True, 
                       verbose=self.verbose)
            coords = st[0].stats.sac.stla, st[0].stats.sac.stlo
            stations[code] = coords
            return code, coords
        
        
        stations = {}
        events_info = {}
        triplets = defaultdict(list)
        no_events = len(events)
        iprint = no_events//10 if no_events>10 else 1
        for ievent, event in enumerate(events, 1):
            if self.verbose:
                if not ievent % iprint:
                    print('EVENTS PROCESSED: %d/%d'%(ievent, no_events))
                
            evdir = os.path.join(self.src, event)
            sacfiles = sorted([i for i in os.listdir(evdir) \
                               if i.endswith('sac') and i[-5]==self.component])
            try:
                evla, evlo, mag = get_event_coords_from_xml(
                        os.path.join(evdir, '%s.xml'%event)
                        )
            except FileNotFoundError:
                evla, evlo, mag = get_event_coords_from_sac(
                        os.path.join(evdir, sacfiles[0])
                        )
            events_info[event] = (evla, evlo, mag)
            for sac1, sac2 in it.combinations(sacfiles, 2):
                sta1, (stla1, stlo1) = get_station_coords(evdir, sac1)
                sta2, (stla2, stlo2) = get_station_coords(evdir, sac2)
                if self.lie_on_same_gc(stla1=stla1, 
                                       stlo1=stlo1, 
                                       stla2=stla2, 
                                       stlo2=stlo2, 
                                       evla=evla, 
                                       evlo=evlo,
                                       azimuth_tolerance=azimuth_tolerance, 
                                       distmin=distmin,
                                       distmax=distmax):
                    triplets[(sta1, sta2)].append(event)
                    
        return stations, events_info, triplets
                
               
    @classmethod
    def lie_on_same_gc(cls, stla1, stlo1, stla2, stlo2, evla, evlo, 
                       azimuth_tolerance=5, distmin=None, distmax=None):
        """
        Boolean function. If the station pair and the epicenter lie on the same 
        great circle path, it returns `True`.
        
        Parameters
        ----------
        stla1, stlo1 : float
            Latitude and longitude of station 1
        
        stla2, stlo2 : float
            Latitude and longitude of station 2
        
        evla, evlo : float
            Latitude and longitude of the epicenter
        
        azimuth_tolerance : float 
            Maximum deviation from the great circle path in degrees
        
        distmin, distmax : float, optional
            Minimum and maximum allowed inter-station distance (in km). Default 
            is None
        
        
        Returns
        -------
        bool
        """
        dist1, az1, _ = gps2dist_azimuth(stla1, stlo1, evla, evlo)
        dist2, az2, _ = gps2dist_azimuth(stla2, stlo2, evla, evlo)  
        if dist1 > dist2:
            (dist1, az1), (dist2, az2) = (dist2, az2), (dist1, az1)
        dist3, az3, _ = gps2dist_azimuth(stla2, stlo2, stla1, stlo1)
        dist3 /= 1000
        if distmax is not None and dist3>distmax:
            return False    
        if distmin is not None and dist3<distmin:
            return False    
        if az3-azimuth_tolerance <= az2 <= az3+azimuth_tolerance:
            return True   
        return False        
        
        
    def prepare_data(self, 
                     azimuth_tolerance=5, 
                     distmin=None, 
                     distmax=None, 
                     min_no_events=5, 
                     recompute=False, 
                     delete_unused_files=False):
        """
        Saves to disk the geographic coordinates of the seismic receivers and of
        the seismic events, along with the triplets of epicenters-receivers to 
        be used for retrieving the dispersion curves.
        
        Parameters
        ----------
        azimuth_tolerance : float
            Maximum allowed deviation from the great circle path in degrees.
            All triplets of epicenter-receivers for which the receivers are not
            aligned within the tolerance indicated are rejected. Larger values
            will identify more triplets to be used in the following analysis.
            But if this value is too large, the assumptions behind the two-
            station method [1]_ may not be met. Suggested values are between 
            3 and 8. Default is 5.
            
        distmin, distmax: float, optional
            Minimum and maximum allowed inter-station distance (in km). Default 
            is `None`
            
        min_no_events : int
            Minimum number of events available for a given station pair to be
            considered in the calculation of the phase velocities.
            
        recompute : bool
            If `True`, the station coordinates and triplets will be removed from
            disk and recalculated. Otherwise (default), if they are present,
            they will be loaded into memory, avoiding any computation. This
            parameter should be set to `True` whenever one wants to change the
            other parameters of this function, which control the selection of
            the epicenter-receivers triplets
            
        delete_unused_files : bool
            If `True`, every waveform-file that is not contained in the triplets
            object (i.e., those that are not used to extract dispersion curves
            in the subsequent analysis) will be permanently deleted from the
            system.
            
        Notes
        -----
        The geographic coordinates of the seismic receivers are saved at 
        $self.savedir/stations.pickle, and are stored in a dictionary object 
        where each key corresponds to a station code ($network_code.$station_code) 
        and each value is a tuple containing latitude and longitude of the station. 
        For example::
            
            { net1.sta1 : (lat1, lon1),
              net1.sta2 : (lat2, lon2),
              net2.sta3 : (lat3, lon3)
              }
            
        The geographic coordinates of the events are saved at 
        $self.savedir/events.pickle, and have a similar structure to the above:
        each key corresponds to an event origin time (in 
        `obspy.UTCDateTime.timestamp` format), and each value is (lat, lon, mag) 
        of the epicenter, where mag is the magnitude of the event.

        The triplets of epicenter-receivers are saved at 
        $self.savedir/triplets.pickle, and are stored in a dictionary object 
        where each key is a tuple corresponding to a station pair and each value 
        is a list of events that are aligned on the same great circle path
        as the receivers (within the limits defined by the input params). 
        For example::
            
            { (net1.sta1, net2.sta2) : ['1222701570.84',
                                        '1237378863.3', 
                                        '1237486660.74',
                                        '1238981562.62',
                                        '1239825695.33'],
                
              (net3.sta3, net4.sta4) : ['1222701570.84',
                                        '1237378863.3', 
                                        '1237486660.74',
                                        '1238981562.62',
                                        '1239825695.33']
              }
            
        Note that each event in the list corresponds to a sub-directory of
        the source data `src`.
        
        
        References
        ----------
        .. [1] Magrini et al. 2020, Arrival-angle effects on two-receiver measurements 
            of phase velocity, GJI
        """       
        save_stations = os.path.join(self.savedir, 'stations.pickle')
        save_events = os.path.join(self.savedir, 'events.pickle')
        save_triplets = os.path.join(self.savedir, 'triplets.pickle')
        if recompute:
            remove_file(save_stations)
            remove_file(save_events)
            remove_file(save_triplets)
        exist_stations = os.path.exists(save_stations)
        exist_events = os.path.exists(save_events)
        exist_triplets = os.path.exists(save_triplets)
        if not exist_stations or not exist_events or not exist_triplets:
            stations, events_info, triplets = self.get_coords_and_triplets(
                    self.events, 
                    azimuth_tolerance=azimuth_tolerance,
                    distmin=distmin,
                    distmax=distmax
                    )
            triplets = {k: v for k, v in triplets.items() if len(v)>=min_no_events}
            save_pickle(save_stations, stations)
            save_pickle(save_events, events_info)
            save_pickle(save_triplets, triplets)
        else:
            stations = load_pickle(save_stations)
            events_info = load_pickle(save_events)
            triplets = load_pickle(save_triplets)
        self.min_no_events = min_no_events
        self.stations = stations
        self.events_info = events_info
        self.triplets = triplets
        if delete_unused_files:
            self.delete_unused_files()
            
            
    def get_events_used(self):
        """ 
        Retrieves the events id for which triplets of epicenter-receivers are
        available to extract dispersion measurements
        
        Returns
        -------
        events_used : dict
            Dictionary object where each key corresponds to an event (origin
            time in `obspy.UTCDateTime.timestamp` format, i.e., the name of the
            respective directory in `self.src`), and the associated values 
            include all the station codes that exploit that event to extract
            a dispersion measurement
        """
        events_used = defaultdict(set)
        for stapair, evlist in self.triplets.items():
            for event in evlist:
                events_used[event] = events_used[event].union(stapair)
        return events_used
    
    
    def delete_unused_files(self):
        """ 
        Deletes every file in the data directory which is not useful for 
        extracting dispersion curves (i.e., those waveform-files that are not 
        included in triplets dict).
        
        .. warning:: 
            Use it with caution. It will not be possible to restore the 
            deleted files.
        """
        def get_unused(sacfiles, used_codes):
            for sac in sacfiles:
                code = '.'.join(sac.split('.')[:2])
                if code not in used_codes:
                    yield sac
        
        def remove_folder(path):
            shutil.rmtree(folder)
            return 1
        
        def remove_files(folder, files):
            for i, file in enumerate(files, 1):
                os.remove(os.path.join(folder, file))
            try:
                return i
            except UnboundLocalError:
                return 0
        
        deleted_files = 0
        deleted_folders = 0
        events_used = self.get_events_used()
        for event in sorted(self.events):
            folder = os.path.join(self.src, event)
            sacfiles = set([i for i in os.listdir(folder) if i.endswith('.sac')])
            if not sacfiles:
                deleted_folders += remove_folder(folder)
                continue
            
            to_delete = list(get_unused(sacfiles, used_codes=events_used[event]))
            if len(to_delete) == len(sacfiles):
                deleted_folders += remove_folder(folder)
                deleted_files += len(to_delete)
            else:
                deleted_files += remove_files(folder, to_delete)
        if self.verbose:
            print('DIRECTORIES REMOVED:', deleted_folders)
            print('FILES REMOVED:', deleted_files)    
    
    def extract_dispcurves(self, 
                           refcurve, 
                           periodmin=15, 
                           periodmax=150, 
                           no_periods=75, 
                           cmin=2.5, 
                           cmax=5, 
                           min_no_wavelengths=1.5,
                           approach='freq', 
                           prob_min=0.25, 
                           prior_sigma_10s=0.7,
                           prior_sigma_200s=0.3, 
                           manual_picking=False,
                           plotting=False,
                           show=True):
        """
        Automatic extraction of the dispersion curves for all available pairs
        of receivers.
        
        The results are saved to $self.savedir/dispcurves in the npy format,
        and consist of ndarrays of shape (n, 2), where the 1st column is period 
        and the 2nd phase velocity (in m/s).
        
        The routine iterates over all the available pair of receivers for which
        there are epicenters aligned on the same great circle path as the
        receivers (see the EQVelocity.prepare_data method); for each such pair
        of stations, (i) it first extracts dispersion measurements from all the
        event available, and then (ii) merges the dispersion measurements to
        obtain a "probability" density distribution of the thus retrieved
        dispersion measurements, which is function of period and phase velocity.
        (iii) Finally, the dispersion curve is extracted from the regions of
        greater "probability". All this is done under the hood calling the 
        :meth:`measure_dispersion` and :meth:`extract_dispcurve` of 
        :class:`TwoStationMethod`.
        
        
        Parameters
        ----------
        refcurve : ndarray of shape (n, 2)
            Reference curve used to extract the dispersion curves. The 1st
            column should be period, the 2nd velocity (in either
            km/s or m/s). The reference curve is automatically converted to
            km/s, the physical unit employed in the subsequent analysis.
            
        periodmin, periodmax : float
            Minimum and maximum period analysed by the algorithm (default
            are 15 and 150 s). The resulting dispersion curves will be limited
            to this period range
            
        no_periods : int
            Number of periods between periodmin and periodmax (included) used 
            in the subsequent analysis. The resulting periods will be equally 
            spaced (linearly) from each other. Default is 75
            
        cmin, cmax : float
            Estimated velocity range spanned by the dispersion curves (default
            values are 2.5 and 5 km/s). The resulting dispersion curves will be 
            limited to this velocity range
            
        min_no_wavelengths : float
            Ratio between the estimated wavelength :math:`\lambda = T c_{ref}`
            of the surface-wave at a given period *T* and interstation distance
            :math:`\Delta`. Whenever this ratio is > `min_no_wavelength`, 
            the period in question is not used to retrieve a dispersion measurement. 
            Values < 1 are suggested against. Default is 1.5
            
        approach : {'time', 'freq'}
            Passed to :meth:`TwoStationMethod.measure_dispersion`. It tells 
            if the dispersion measurements are extracted in the frequency domain 
            ('freq') or in the time domain ('time'). Default is 'freq'
            
        prob_min : float
            Minimum acceptable "probability" in the density of dispersion 
            measurements, at a given period, below which the dispersion curve 
            is not picked. Larger values are more restrictive. Good values are 
            between ~0.2 and ~0.35. Default is 0.25. See 
            :meth:`TwoStationMethod.extract_dispcurve`. 
            
        prior_sigma_10s, prior_sigma_200s : float
            Standard deviations of the Gaussian functions built around the 
            reference model (`refcurve`) at the periods of 10 and 200 s to 
            calculate the prior probability of the dispersion measurements.
            At each analysed period, the standard deviation is interpolated (and
            eventually linearly extrapolated) based on these two values. Smaller
            values give more "weight" to the reference curve in the picking of
            the phase velocities. Defaults are 0.7 and 0.3. See
            :meth:`TwoStationMethod.extract_dispcurve`. 
            
        plotting : bool
            If `True`, a figure is created for each retrieved dispersion curve.
            This is automatically displayed and saved in $self.savedir/figures
            
        manual_picking : bool
            If True, the user is required to pick the dispersion curve manually.
            The picking is carried out through an interactive plot.
        """
        def percentage_done(no_pairs, no_done):
            return '\nPERCENTAGE DONE: %.2f\n'%(no_done/no_pairs * 100)
        
        def load_done(file):
            if os.path.exists(file):
                return set([i.strip() for i in open(file)])
            else:
                return set()
    
        def update_done(sta1, sta2):
            with open(save_done, 'a') as f:
                f.write('%s__%s\n'%(sta1, sta2))
            # done.add('%s_%s'%(sta1, sta2))
            
        def dist_az_backaz(stations, sta1, sta2):
            stla1, stlo1 = stations[sta1]
            stla2, stlo2 = stations[sta2]
            dist, az, baz = gps2dist_azimuth(stla1, stlo1, stla2, stlo2)
            dist /= 1000.
            return dist, az, baz  
        

        save_dispersion = os.path.join(self.savedir, 'dispersion')
        save_pv = os.path.join(self.savedir, 'dispcurves')
        save_tmp = os.path.join(self.savedir, 'tmp')
        save_done = os.path.join(save_tmp, 'DONE.txt')
        save_fig = os.path.join(self.savedir, 'figures')
        os.makedirs(save_pv, exist_ok=True)
        os.makedirs(save_dispersion, exist_ok=True)
        os.makedirs(save_tmp, exist_ok=True)
        if plotting is not None:
            os.makedirs(save_fig, exist_ok=True)

        if refcurve[0, 1] / 10 > 1:
            refcurve = np.column_stack((refcurve[:, 0], refcurve[:, 1]/1000))
        tsm = TwoStationMethod(refcurve=refcurve,
                               periodmin=periodmin,
                               periodmax=periodmax,
                               no_periods=no_periods,
                               cmin=cmin, 
                               cmax=cmax, 
                               ttol=0.3,
                               min_no_wavelengths=min_no_wavelengths, 
                               approach=approach)
        done = load_done(save_done)
        for ndone, ((sta1, sta2), events) in enumerate(self.triplets.items()):
            if '%s__%s'%(sta1, sta2) in done:
                continue
            if not ndone % 100:
                done = load_done(save_done)
                if self.verbose:
                    print(percentage_done(len(self.triplets), len(done)))
            if self.verbose:
                print(sta1, '-', sta2, ':', len(events), 'EVENTS')
            savedir = os.path.join(save_dispersion, '%s__%s'%(sta1, sta2))
            os.makedirs(savedir, exist_ok=True)
            for otime in events:
                st1 = _read(os.path.join(self.src, otime, '%s*%s.sac'%(sta1, self.component)),
                            verbose=self.verbose)
                st2 = _read(os.path.join(self.src, otime, '%s*%s.sac'%(sta2, self.component)),
                            verbose=self.verbose)
                try:
                    tsm.preprocess(st1, st2, float(otime), fs=2)
                    dispersion = tsm.measure_dispersion()
                except:
                    continue
                if dispersion.size:
                    file = os.path.join(savedir, '%s__%s__%s.npy'%(sta1, sta2, otime))
                    np.save(file, dispersion)
            if len(os.listdir(savedir)) >= self.min_no_events:
                dist = dist_az_backaz(self.stations, sta1, sta2)[0]
                outimg = os.path.join(save_fig, '%s__%s.png'%(sta1, sta2))
                try:
                    dispcurve = tsm.extract_dispcurve(refcurve,
                                                      src=savedir,
                                                      dist_km=dist,
                                                      prob_min=prob_min,
                                                      smoothing=0.3,
                                                      plotting=plotting,
                                                      savefig=outimg,
                                                      sta1=sta1,
                                                      sta2=sta2,
                                                      prior_sigma_10s=prior_sigma_10s,
                                                      prior_sigma_200s=prior_sigma_200s,
                                                      manual_picking=manual_picking,
                                                      show=show)
                except:
                    update_done(sta1, sta2)
                    continue
                    
                dispcurve[:,1] *= 1000
                np.save(os.path.join(save_pv, '%s__%s.npy'%(sta1, sta2)),
                        dispcurve)
                
            update_done(sta1, sta2)
                
    
    def prepare_input_tomography(self, savedir, period, min_no_wavelengths=1.5,
                                 outfile='input_%.2fs.txt'):
        """ 
        Prepares a txt file for each specified period, to be used for 
        calculating phase-velocity maps using :class:`seislib.tomography.SeismicTomography`.
        
        Parameters
        ----------
        savedir : str
            Absolute path to the directory where the file(s) is (are) saved.
            If savedir does not exist, it will be created
            
        period : float, array-like
            Period (or periods) at which the dispersion curves will be 
            interpolated using :meth:`interpolate_dispcurves`

        min_no_wavelengths : float
            Ratio between the estimated wavelength :math:`\lambda = T c_{ref}`
            of the surface-wave at a given period *T* and interstation distance
            :math:`\Delta`. Whenever this ratio is > `min_no_wavelength`, 
            the period in question is not used to retrieve a dispersion measurement. 
            Values < 1 are suggested against. Default is 1.5

        outfile : str
            Format for the file names. It must include either %s or %.Xf (where
            X is integer), since this will be replaced by each period analysed
            (one for file)

        Examples
        --------
        Assume you calculated dispersion curves from your data, which are 
        located in /path/to/data, and you had initialized your 
        class:`EQVelocity` instance, to calculate inter-station 
        dispersion curves, as::
            
            from seislib.eq import EQVelocity
            eq = EQVelocity(src=SRC, component='Z')
            eq.prepare_data(azimuth_tolerance=7,
                            distmin=None, 
                            distmax=3000, 
                            min_no_events=8)

        You can, even if the computation of the dispersion curves is not 
        finished, use the above instance to extract the information needed
        to produce phase-velocity maps as follows::

            savedir = /arbitrary/path/to/savedir
            periods = [20, 30, 40, 50, 75, 100]
            eq.prepare_input_tomography(savedir=savedir,
                                        period=periods)

        The above will save one txt file for each specified period.
        """
        os.makedirs(savedir, exist_ok=True)
        period = np.array([period]) if np.isscalar(period) else np.array(period)
        codes, coords, velocity = self.interpolate_dispcurves(period)
        dist = SeismicTomography.gc_distance(*coords.T)
        wavelength = velocity * period
        ratios = dist.reshape(-1, 1) / wavelength
        velocity_final = np.where(ratios>min_no_wavelengths, velocity, np.nan)
        for iperiod, p in enumerate(period):
            save = os.path.join(savedir, outfile%p)
            vel = velocity_final[:, iperiod]
            notnan = np.flatnonzero(~np.isnan(vel))
            if self.verbose:
                print('Measurements at %.2fs:'%p, notnan.size)
            np.savetxt(save, np.column_stack((coords[notnan], vel[notnan])))
            
    
    def interpolate_dispcurves(self, period):
        """ 
        Interpolates the dispersion curves found at $self.savedir/dispcurves
        at the specified period(s). (No extrapolation is made.)
        
        Parameters
        ----------
        period : float, array-like
            Period (or periods) at which the dispersion curves will be 
            interpolated
            
        Returns
        -------
        codes : ndarray of shape (n, 2)
            Codes associated with the station pairs for which a dispersion curve
            has been calculated
            
        coords : ndarray of shape (n, 4)
            Coordinates (lat1, lon1, lat2, lon2) of the station pairs corresponding
            to the station codes
            
        measurements : ndarray of shape (n, p)
            Phase velocity calculated for station pair contained in coords at
            the wanted period(s). `p` is the number of periods
            
            .. note:: 
                *measurements* could contain nans

        Examples
        --------
        Assume you calculated dispersion curves from your data, which are 
        located in /path/to/data, and you had initialized your 
        class:`EQVelocity` instance, to calculate inter-station 
        dispersion curves, as::
            
            from seislib.eq import EQVelocity
            
            eq = EQVelocity(src=SRC, component='Z')
            eq.prepare_data(azimuth_tolerance=7,
                            distmin=None, 
                            distmax=3000, 
                            min_no_events=8)

        Even if the computation of the dispersion curves is not yet 
        finished, you can use the above instance to interpolate all the
        available dispersion curves as follows::

            period = [20, 30, 40, 50]
            codes, coords, measurements = eq.interpolate_dispcurves(period)  
        """   
        def display_progress(no_files, done, verbose=False):
            if verbose and not done % int(0.05*no_files + 1):
                print('FILES PROCESSED: %d/%d'%(done, no_files))

        src = os.path.join(self.savedir, 'dispcurves')
        files = [i for i in sorted(os.listdir(src)) if i.endswith('npy')]
        files_size = len(files)
        period_size = 1 if np.isscalar(period) else len(period)
        measurements = np.zeros((files_size, period_size))
        coords = np.zeros((files_size, 4))
        codes = []
        for i, file in enumerate(files):
            display_progress(no_files=files_size, done=i, verbose=self.verbose)
            periods, vel = np.load(os.path.join(src, file)).T
            interp_vel = interp1d(periods, vel, bounds_error=False)(period)
            measurements[i] = interp_vel
            code1, code2 = file.split('.npy')[0].split('__')
            sta1 = '.'.join(code1.split('.')[:2])
            sta2 = '.'.join(code2.split('.')[:2])
            coords[i] = (*self.stations[sta1], *self.stations[sta2])
            codes.append((sta1, sta2))
        return np.array(codes), coords, measurements
    
    
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
        If `show` is True, None, else `ax`, i.e. the GeoAxesSubplot
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
                
        
    def plot_events(self, ax=None, show=True, oceans_color='water', lands_color='land', 
                    edgecolor='k', projection='Mercator', resolution='110m', 
                    min_size=5, max_size=150, legend_markers=4, legend_dict={}, 
                    **kwargs):
        """ Creates a map of epicenters
        
        Parameters
        ----------
        lat, lon : ndarray of shape (n,)
            Latitude and longitude of the epicenters
            
        mag : ndarray of shape(n,), optional
            If not given, the size of each on the map will be constant
        
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not `None`, the receivers are plotted on the `GeoAxesSubplot` instance. 
            Otherwise, a new figure and `GeoAxesSubplot` instance is created
            
        show : bool
            If `True`, the plot is shown. Otherwise, a `GeoAxesSubplot` instance is
            returned. Default is `True`
            
        oceans_color, lands_color : str
            Color of oceans and lands. The arguments are ignored if `ax` is not
            `None`. Otherwise, they are passed to `cartopy.feature.NaturalEarthFeature` 
            (to the argument 'facecolor'). Defaults are 'water' and 'land'
            
        edgecolor : str
            Color of the boundaries between, e.g., lakes and land. The argument 
            is ignored if `ax` is not `None`. Otherwise, it is passed to 
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
        
        min_size, max_size : float
            Minimum and maximum size of the epicenters on the map. These are used
            to interpolate all magnitudes associated with each event, so as to
            scale them appropriately on the map. (The final "sizes" are passed to
            the argument `s` of `matplotlib.pyplot.scatter`)
        
        legend_markers : int
            Number of markers displayed in the legend. Ignored if `s` (size of the
            markers in `matplotlib.pyplot.scatter`) is passed
        
        legend_dict : dict, optional
            Keyword arguments passed to `matplotlib.pyplot.legend`
            
        **kwargs : dict, optional
            Additional keyword arguments passed to `matplotlib.pyplot.scatter`  
            
        Returns
        -------
        If `show` is True, None, else `ax`, i.e. the GeoAxesSubplot
        """
        events_used = self.get_events_used().keys()
        events_info = np.array([j for i, j in self.events_info.items() \
                                if i in events_used])
        lat, lon, mag = events_info.T
        return plot_events(lat=lat,
                           lon=lon,
                           mag=mag,
                           ax=ax, 
                           show=show, 
                           oceans_color=oceans_color, 
                           lands_color=lands_color, 
                           edgecolor=edgecolor, 
                           projection=projection,
                           resolution=resolution, 
                           min_size=min_size,
                           max_size=max_size,
                           legend_markers=legend_markers,
                           legend_dict=legend_dict, 
                           **kwargs)
        
###############################################################################
        
        
        
        

class TwoStationMethod:
    """ 
    Class providing methods to process and extract dispersion curves from
    earthquake recordings generated by strong teleseismic earthquakes.
    
    Parameters
    ----------
    refcurve : ndarray of shape (n, 2)
        Reference curve used to extract the dispersion curves. The 1st
        column should be period, the 2nd velocity (in either km/s or m/s). 
        The reference curve is automatically converted to
        km/s, the physical units employed in the subsequent analysis.
        
    periodmin, periodmax : float
        Minimum and maximum period analysed by the algorithm (default
        are 15 and 150 s). The resulting dispersion curves will be limited
        to this period range
        
    no_periods : int
        Number of periods between `periodmin` and `periodmax` (included) 
        used in the subsequent analysis. The resulting periods will be 
        equally spaced (linearly) from each other. Default is 75
        
    cmin, cmax : float
        Estimated velocity range spanned by the dispersion curves (default
        values are 2.5 and 5 km/s). The resulting dispersion curves will 
        be limited to this velocity range
        
    ttol : float
        Tolerance, with respect to the reference curve, used to taper the 
        seismograms around the expected arrival time of the surface wave 
        (at a given period). In practice, at a given period, everything 
        outside of the time range given by tmin and tmax (see below) is set 
        to zero through a taper. tmin and tmax are defined as::
        
            tmin = dist / (ref_vel + ref_vel*ttol)
            tmax = dist / (ref_vel - ref_vel*ttol)
            
        where dist is inter-station distance. Default is 0.3, i.e., 30% of
        the reference velocity
    
    min_no_wavelengths : float
        Ratio between the estimated wavelength :math:`\lambda = T c_{ref}`
        of the surface-wave at a given period *T* and interstation distance
        :math:`\Delta`. Whenever this ratio is > `min_no_wavelength`, 
        the period in question is not used to retrieve a dispersion measurement. 
        Values < 1 are suggested against. Default is 1.5
        
    approach : {'freq', 'time'}
        Passed to :meth:`TwoStationMethod.measure_dispersion`. It tells 
        if the dispersion measurements are extracted in the frequency domain 
        ('freq') or in the time domain ('time'). Default is 'freq'
        
    gamma_f : float
        Controls the width of the bandpass filters, at a given period, used 
        to isolate the fundamental mode in the seismogram [1]_.
        
    gamma_w, distances : ndarray of shape (m,), optional
        Control the width of tapers used to taper, at a given period, the
        cross correlations in the frequency domain (these two parameters 
        are ignored if `approach` is 'time'). `distances` should be in km. 
        If not given, they will be automatically set to::
        
            gamma_w = np.linspace(5, 50, 100)
            distances = np.linspace(100, 3000, 100)
            
        These two arrays are used as interpolators to calculate `gamma`
        based on the inter-station distance. `gamma` is the parameter that
        actually controls the width of the tapers [1]_, and is defined as::
        
            gamma = np.interp(dist, distances, gamma_w)       


    Attributes
    ----------
    periods : ndarray of shape (n,)
        Array of periods to extract the dispersion measurements from
        
    ref_vel : ndarray of shape (n,)
        Array of reference velocities corresponding to `periods`
        
    ref_model : scipy.interpolate.interpolate.interp1d
        Interpolating function. Input: period. Returns: phase velocity
        
    min_no_wavelegnths : float
        Minimum number of wavelengths between two receivers, at a given period,
        below which the dispersion measurements are not extracted
        
    cmin, cmax : float
        Estimated velocity range (in km/s) spanned by the dispersion curves
        
    ttol : float
        Tolerance (percentage), with respect to the reference curve, used to 
        taper the seismograms around the expected arrival time of the surface 
        wave (at a given period).
    
    gamma_f : float, int
        Controls the width of the bandpass filters, at a given period, used 
        to isolate the fundamental mode in the seismogram.
        
    gamma_w, distances : ndarrays of shape (m,), optional
        Control the width of tapers used to taper, at a given period, the
        cross correlations in the frequency domain
        
    approach : str
        It indicates if the dispersion measurements are extracted in the 
        frequency domain ('freq') or in the time domain ('time')
        
    otime : float
        Origin time of the earthquake. Available after calling
        :meth:`TwoStationMethod.preprocess`
        
    dist1, dist2 : float
        Epicentral distances at the two receivers. Available after calling
        :meth:`TwoStationMethod.preprocess`
        
    dist : float
        Inter-station distance, calculated as dist2-dist1. Available after 
        calling :meth:`TwoStationMethod.preprocess`
        
    dt : float
        Time sampling of the seismograms. Available after calling
        :meth:`TwoStationMethod.preprocess`
        
    fs : float
        Sampling rate of the seismograms. Available after calling
        :meth:`TwoStationMethod.preprocess`
        
    data1, data2 : ndarray of shape (n,)
        Preprocessed seismograms. Available after calling
        :meth:`TwoStationMethod.preprocess`
        
    times : ndarray of shape (n,)
        Array of times corresponding to the two seismograms, starting at the
        origin time. Available after calling 
        :meth:`TwoStationMethod.preprocess`
        
    periods_masked : ndarray of shape (n,)
        Subset of `periods`, where the wavelength is such that there is a
        number of wavelenghts between the two receivers > `min_no_wavelength`. 
        Available after calling :meth:`TwoStationMethod.preprocess`        
    
    Examples
    --------
    
    In the following we will calculate a dispersion curve based on several
    earthquakes recorded at two receivers (sta1, sta2) on the vertical 
    components (Z). The seismograms (two per earthquake) are stored in as many directories 
    as the number of earthquakes available in the analysis. These directories are named 
    after the event origin time (in the obspy.UTCDateTime.timestamp format) and located
    at /path/to/data. The seismic data must be in the .sac format, with the information on 
    the epicentral distance (dist) compiled in the header. It is assumed that receivers 
    and epicenter lie on the same great-circle path.

    We will retrieve dispersion measurements for 75 periods linearly spaced
    between 15 and 150s. We assume that in this period range surface waves travel
    in the velocity range 2.5 - 5 km/s. We need to pass a reference curve
    consisting of 2 columns (1st: period, 2nd: phase velocity in either km/s or
    m/s), that will be used in the subsequent analysis. We will only employ 
    periods for which the ratio between expected wavelength (based on the
    reference curve) and inter-station distance is not smaller than 1.5.

    First, we initialize the TwoStationMethod class as::
        
        from seislib.eq import TwoStationMethod

        tsm = TwoStationMethod(refcurve=refcurve,
                               periodmin=15,
                               periodmax=150,
                               no_periods=75,
                               cmin=2.5, 
                               cmax=5, 
                               min_no_wavelengths=1.5)
        
    Then, we use this class obtain dispersion measurements for all events
    available in the data directory. (We are assuming that we only have folders
    corresponding to events that are aligned on the same great circle path as
    the epicenter. For a more automatic processing of station pairs and events,
    see :meth:`EQVelocity.prepare_data`.) We will store all
    the dispersion measurements in the folder /path/to/savedir/dispersion, in
    the npy format::
    
        import os
        import numpy as np
        from obspy import read

        src = /path/to/data
        dispersion_dir = /path/to/savedir/dispersion
        
        events = os.listdir(src)
        for origin_time in events:
            st1 = read(os.path.join(src, origin_time, '%s*'%sta1))
            st2 = read(os.path.join(src, origin_time, '%s*'%sta2))    
            tsm.preprocess(st1, st2, float(origin_time))
            dispersion = tsm.measure_dispersion()
            outfile = os.path.join(dispersion_dir, 
                                   '%s_%s_%s.npy'%(sta1, sta2, origin_time))
            np.save(outfile, dispersion)
                 
            
    Now that all the dispersion measurements have been extracted, we can
    calculate a dispersion curve based on them. (dist_km is inter-station 
    distance in km). The results will displayed in the console::
    
        dispcurve = tsm.extract_dispcurve(refcurve=refcurve,
                                          src=dispersion_dir,
                                          dist_km=dist_km,
                                          plotting=True,
                                          sta1=sta1,
                                          sta2=sta2)

    References
    ----------
    .. [1] Soomro et al. (2016), Phase velocities of Rayleigh and Love waves in 
        central and northern Europe from automated, broad-band, interstation 
        measurements, GJI
    """
    
    def __init__(self, refcurve, periodmin=15, periodmax=150, no_periods=75, 
                 cmin=2.5, cmax=5, ttol=0.3, min_no_wavelengths=1.5, 
                 approach='freq', gamma_f=14, gamma_w=None, distances=None):

        self.periods = np.linspace(periodmin, periodmax, no_periods, dtype='float32')
        refcurve = self.convert_to_kms(refcurve)
        self.ref_model = interp1d(refcurve[:,0], 
                                  refcurve[:,1],
                                  bounds_error=False, 
                                  fill_value='extrapolate')
        self.ref_vel = self.ref_model(self.periods)
        self.min_no_wavelengths = min_no_wavelengths
        self.cmin = cmin
        self.cmax = cmax
        self.ttol = ttol
        self.gamma_f = gamma_f
        if gamma_w is None or distances is None:
            self.gamma_w = np.linspace(5, 50, 100)
            self.distances = np.linspace(100, 3000, 100)
        else:
            self.gamma_w = gamma_w
            self.distances = distances

        self.approach = approach.lower().replace(' ', '')
        if self.approach not in ['time', 'freq']:
            string = '`%s` not supported: `approach` should be either `freq` or \
            `time`'%approach
            raise NotImplementedError(string)
                
    
    def frequency_w_and_alphaf(self, period):
        r""" 
        Retrieves frequency, angular frequency, and `alpha_f` at a given period
        
        Parameters
        ----------
        period : float
        
        Returns
        -------
        frequency : float
            Inverse of period
            
        w : float
            Angular frequency, i.e. :math:`2\pi \times` `frequency`
            
        alpha_f : float
            Controls the width of the bandpass filters, at a given period, used 
            to isolate the fundamental mode in the seismogram [1]_. It is a function
            of `gamma_f` (See :class:`TwoStationMethod`).
            
            
        References
        ----------
        .. [1] Soomro et al. (2016), Phase velocities of Rayleigh and Love waves in 
            central and northern Europe from automated, broad-band, interstation 
            measurements, GJI
        """
        frequency = 1 / period
        w = 2 * pi * frequency
        alpha_f = w * self.dt * self.gamma_f**2       
        return frequency, w, alpha_f
    
    
    def build_taper(self, center_idx, taper_size, data_size, 
                    taper_type=signal.tukey, alpha=0.1):
        """ Defines the taper used for tapering the seismograms
        
        Parameters
        ----------
        center_idx : int 
            Index of corresponding to the center of the taper
            
        taper_size : int
            Lenght of the taper. The taper will be centred onto `taper_idx` and
            extend around the center by `taper_size`/2
            
        data_size :
            Length of the data, which corresponds to the final size of the 
            taper
            
        taper_type : func
            Taper function. Default is `scipy.signal.tukey 
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
            
        alpha : float 
            Shape parameter of the taper window.
        
        Returns
        -------
        taper : numpy.ndarray
        """
        
        l_idx = center_idx - taper_size//2
        u_idx = center_idx + taper_size//2 + 1
        idx = list(range(l_idx, u_idx))
        taper = np.zeros(data_size)
        taper[idx] = taper_type(len(idx), alpha=alpha)
        return taper


    def times_for_taper(self, dist, ref_vel, tolerance):
        """ 
        Identifies the starttime and endtime of the tapers to be applied to the 
        seismograms
        
        Parameters
        ----------
        dist : float
            Inter-station distance (in km)
            
        ref_vel : float
            Reference velocity (at a given period in km/s)
            
        tolerance : float
            Tolerance, with respect to ref_vel, used to identify the starting
            and ending time of the taper
            
        Returns
        -------
        tmin, tmax : float
            Starting and ending time of the taper
        """ 
        
        tmin = int(dist / (ref_vel + ref_vel*tolerance))
        tmax = int(dist / (ref_vel - ref_vel*tolerance))
        return tmin, tmax
    
    
    def taper_from_times(self, tmin, tmax, taper_p=0.2):
        """ Retrieves the tapers to be applied to the seismograms
        
        Parameters
        ----------
        tmin, tmax : float
            Starting and ending time of the taper
            
        taper_p : float
            Shape parameter of the taper. Default is 0.2
        
        Returns
        -------
        taper : ndarray of shape (n,)
        """
        
        idx = np.where((self.times>tmin) & (self.times<tmax))[0]
        taper = self.build_taper(center_idx=(np.max(idx) + np.min(idx)) // 2, 
                                 taper_size=len(idx), 
                                 data_size=self.times.size, 
                                 alpha=taper_p)       
        return taper
        
    
    def adapt_startime_to_t(self, data, dt, starttime, origintime, 
                            power_2_sized=True):
        """ 
        Zero-pads data in order to make the start time coincide with the event 
        origin time
        
        Parameters
        ----------
        data : ndarray
        
        dt : float 
            Time sampling interval
            
        starttime, origintime : `obspy.UTCDateTime.timestamp`
            Starttime and origintime of the waveform
            
        power_2_sized : bool
            If True, the returned data size will be a power of 2. This allows
            to take the Fourier transform more efficiently
        
        Returns
        -------
        numpy.ndarray
            Padded data
        """
        
        time_difference = origintime - starttime
        if time_difference != 0:
            quotient, remainder = divmod(time_difference, dt)
            if remainder > dt/2:
                quotient += 1
            if quotient < 0:
                data = np.concatenate((np.zeros(int(-quotient)), data))
            else:
                data = data[int(quotient):]
                
        if power_2_sized:
            n = next_power_of_2(data.size)
            data = np.concatenate((data, np.zeros(n-data.size)))
            
        return data


    def adapt_times(self, tr1, tr2):
        """ 
        Zero-pads and slices the two traces so that their startting and ending 
        times coincide
        
        Parameters
        ----------
        tr1, tr2 : obspy.Trace
        
        Returns
        -------
        tr1, tr2 : obspy.Trace
            Modified copy of the original input
        """
        tstarts = [tr1.stats.starttime, tr2.stats.starttime]
        tends = [tr1.stats.endtime, tr2.stats.endtime]
        if len(set([t.timestamp for t in tstarts])) == 1 \
        and len(set([t.timestamp for t in tends])) == 1:
            return tr1, tr2
        
        tstart = min(tstarts)
        tend = max(tends)
        tr1 = zeropad(tr1, tstart, tend)
        assert np.all(tr1.data == tr1.slice(tr1.stats.starttime, tr1.stats.endtime).data)
        tr2 = zeropad(tr2, tstart, tend)
        assert np.all(tr2.data == tr2.slice(tr2.stats.starttime, tr2.stats.endtime).data)
        
        tr1, tr2 = adapt_timespan(tr1, tr2)
        assert tr1.stats.starttime.timestamp == tr2.stats.starttime.timestamp
        return tr1, tr2
            

    def preprocess(self, st1, st2, otime, fs=2):
        """
        Read sac infos and prepare to pv extraction. The sac headers must be 
        compiled with traces and event information
        
        Parameters
        ----------
        st1, st2 : obspy.core.stream.Stream
        
        otime : obspy.UTCDateTime.timestamp
            Event origin time
            
        fs : int
            Sampling rate
                            
        taper_p : float
            Percentage taper to apply to the streams.
        """
        self.otime = otime
        tr1, tr2 = st1[0], st2[0]
        if tr1.stats.sac.dist > tr2.stats.sac.dist:
            tr1, tr2 = tr2, tr1
            
        self.dist1 = tr1.stats.sac.dist
        self.dist2 = tr2.stats.sac.dist
        self.dist = self.dist2 - self.dist1    
        
        if tr1.stats.sampling_rate != fs:
            tr1 = tr1.interpolate(sampling_rate=fs, method="weighted_average_slopes")
        if tr2.stats.sampling_rate != fs:
            tr2 = tr2.interpolate(sampling_rate=fs, method="weighted_average_slopes")
        self.dt = tr1.stats.delta
        self.fs = tr1.stats.sampling_rate
        
        tr1.taper(max_percentage=0.05)
        tr2.taper(max_percentage=0.05)     
        tr1, tr2 = self.adapt_times(tr1, tr2)
        starttime = tr1.stats.starttime.timestamp
        
        self.data1 = self.adapt_startime_to_t(data=tr1.data, 
                                              dt=self.dt, 
                                              starttime=starttime,
                                              origintime=self.otime, 
                                              power_2_sized=True)
        self.data2 = self.adapt_startime_to_t(data=tr2.data, 
                                              dt=self.dt, 
                                              starttime=starttime,
                                              origintime=self.otime, 
                                              power_2_sized=True)
        self.times = np.arange(0, self.data1.size*self.dt, self.dt)           
        lambda_min = self.ref_vel * self.periods * self.min_no_wavelengths
        mask = lambda_min < self.dist
        self.periods_masked = self.periods[mask]
    

    def freq_domain_dispersion(self, taper_p=0.2):
        """
        Extract dispersion measurements using a frequency-domain approach.
        
        Parameters
        ----------
        taper_p : float
            Taper percentage
                
        Returns
        -------
        solutions : ndarray of shape (n, 2)
            The 1st column is period, the 2nd is phase velocity. Note that, in
            general, at a given period more than one phase-velocity measurement
            will be present in solutions, because of the :math:`2\pi` phase 
            ambiguity [1]_. All phase velocities outside the velocity range 
            cmin-cmax are discarded (see :class:`TwoStationMethod`).
            
        References
        ----------
        .. [1] Magrini et al. 2020, Arrival-angle effects on two-receiver measurements 
            of phase velocity, GJI
        """        
        dist = self.dist
        gamma = np.interp(dist, self.distances, self.gamma_w) 
        freq = rfftfreq(len(self.times), self.dt)
        velocities = []
        frequencies = []
        
        periods = self.periods_masked
        indexes = reversed(range(len(periods)))
        for idx, period in zip(indexes, sorted(periods)[::-1]):               
            frequency, w, alpha_f = self.frequency_w_and_alphaf(period)
            alpha_w = w * self.dt * gamma**2 
            ref_vel = self.ref_model(period)
            
            tmin1, tmax1 = self.times_for_taper(self.dist1, ref_vel, self.ttol)
            tmin2, tmax2 = self.times_for_taper(self.dist2, ref_vel, self.ttol)         
            taper_u1 = self.taper_from_times(tmin1, tmax1, taper_p=taper_p)
            taper_u2 = self.taper_from_times(tmin2, tmax2, taper_p=taper_p)
            u1 = bandpass_gaussian(self.data1, self.dt, period, alpha_f) * taper_u1
            u2 = bandpass_gaussian(self.data2, self.dt, period, alpha_f) * taper_u2           
            
            xcorr = signal.correlate(u2, u1, 'full')[u1.size - 1:]
            t_max = self.times[np.argmax(xcorr)]
            if t_max == 0:
                continue
                  
            taper_xcorr = np.exp(-w**2 * (self.times - t_max)**2 / (4*alpha_w))
            xcorr *= taper_xcorr
            
            xcorr_w = np.conj(rfft(xcorr))
            phi = np.unwrap(np.angle(xcorr_w))
            phi_interp = np.interp(frequency, freq, phi)       
    ##################BECAUSE#################################################
    #irfft(U1 * np.exp(-1j*(phi))) == irfft(U1 * np.exp(-1j*(phi+2*pi))) == u2
    ##########################################################################   
            for i in range(-20, 21, 2):
                c_w = w*dist / (phi_interp + i*pi)
                if self.cmin <= c_w <= self.cmax:
                    velocities.append(c_w)
                    frequencies.append(1 / period)

        solutions = np.column_stack((1/np.asarray(frequencies), velocities))
        return solutions


    def time_domain_dispersion(self, taper_p=0.2):
        """
        Extract dispersion measurements using a time-domain approach.
        
        Parameters
        ----------
        taper_p : float
            Taper percentage
                
        Returns
        -------
        solutions : ndarray of shape (n, 2)
            The 1st column is period, the 2nd is phase velocity. Note that, in
            general, at a given period more than one phase-velocity measurement
            will be present in solutions, because of the :math:`2\pi` phase 
            ambiguity [1]_. All phase velocities outside the velocity range 
            cmin-cmax are discarded (see :class:`TwoStationMethod`).
            
        References
        ----------
        .. [1] Magrini et al. 2020, Arrival-angle effects on two-receiver measurements 
            of phase velocity, GJI
        """   
        
        dist = self.dist
        velocities = []
        frequencies = []
        
        periods = self.periods_masked
        indexes = reversed(range(len(periods)))
        for idx, period in zip(indexes, sorted(periods)[::-1]):                   
            frequency, w, alpha_f = self.frequency_w_and_alphaf(period)
            
            ref_vel = self.ref_model(period)
            tmin1, tmax1 = self.times_for_taper(self.dist1, ref_vel, self.ttol)
            tmin2, tmax2 = self.times_for_taper(self.dist2, ref_vel, self.ttol)            
            taper_u1 = self.taper_from_times(tmin1, tmax1, taper_p=taper_p)
            taper_u2 = self.taper_from_times(tmin2, tmax2, taper_p=taper_p)
            u1 = bandpass_gaussian(self.data1, self.dt, period, alpha_f) * taper_u1
            u2 = bandpass_gaussian(self.data2, self.dt, period, alpha_f) * taper_u2
            
            xcorr = signal.correlate(u2, u1, 'full')[u1.size - 1:]
            xcorr /= np.nanmax(xcorr)
            args_max = signal.find_peaks(xcorr, height=1e-3)[0]
            t_maxima = self.times[args_max]
            for t_max in t_maxima:
                c_w = dist / t_max
                if c_w<self.cmin or c_w>self.cmax:
                    continue
                velocities.append(c_w)
                frequencies.append(1 / period)
            
        solutions = np.column_stack((1/np.asarray(frequencies), velocities)) 
        return solutions
    
    
    def measure_dispersion(self, taper_p=0.2):
        """ 
        Extract dispersion measurements using either a time-domain or a 
        frequency-domain approach, depending on the parameter `approach` passed
        to :class:`TwoStationMethod`. The function will call either 
        :meth:`time_domain_dispersion` or :meth:`freq_domain_dispersion`, 
        respectively.
        
        Parameters
        ----------
        taper_p : float
            Taper percentage

        solutions : ndarray of shape (n, 2)
            The 1st column is period, the 2nd is phase velocity. Note that, in
            general, at a given period more than one phase-velocity measurement
            will be present in solutions, because of the :math:`2\pi` phase 
            ambiguity [1]_. All phase velocities outside the velocity range 
            cmin-cmax are discarded (see :class:`TwoStationMethod`).
            
        References
        ----------
        .. [1] Magrini et al. 2020, Arrival-angle effects on two-receiver measurements 
            of phase velocity, GJI
        """
        def approach(string):   
            nonlocal self
            if string == 'time': 
                return self.time_domain_dispersion
            if string == 'freq': 
                return self.freq_domain_dispersion
                       
        method = approach(self.approach)
        return method(taper_p=taper_p)
    
    
    @classmethod
    def convert_to_kms(cls, dispcurve):
        """ Converts from m/s to km/s the dispersion curve (if necessary).
        
        Parameters
        ----------
        dispcurve : ndarray of shape (n, 2)
            The 1st column (typically frequency or period) is ignored. The
            2nd column should be velocity. If the first value of velocity
            divided by 10 is greater than 1, the 2nd column is divided by
            1000. Otherwise, the dispersion curve is left unchanged.
            
        Returns
        -------
        dispcurve : ndarray of shape (n, 2)
            Dispersion curve eventually converted to km/s
        """
        if dispcurve[0, 1] / 10 > 1:
            dispcurve = np.column_stack((dispcurve[:, 0], dispcurve[:, 1]/1000))
        return dispcurve
    
    
    @classmethod
    def extract_dispcurve(cls, refcurve, src, dist_km, prob_min=0.25, smoothing=0.3,
                          min_derivative=-0.01, prior_sigma_10s=0.7, prior_sigma_200s=0.3, 
                          plotting=False, savefig=None, sta1=None, sta2=None,
                          manual_picking=False, show=True):
        """ 
        Extraction of the dispersion curve from a given set of dispersion
        measurements. These (one per earthquake) should be stored in a directory
        as npy files.
        
        The algorithm will (i) first gather all the dispersion measurments in a
        unique `ndarray of shape (n, 2)`, where the 1st column is period and
        the 2nd phase velocity. From this ensemble, (ii) it will then create 
        a 2-D image representing the density of measurements, via a method 
        similar to a kernel-density estimate (KDE). This image (of shape (k, l)
        and henceforth referred to as `P_obs`), is normalized at each period by 
        the maximum at the same period. (iii) A second image, call it `P_ref`, 
        is obtained from the reference curve: the weights given to each pixel at 
        a given period are defined by a Gaussian function centred onto the
        reference velocity at the same period, so that they decrease with 
        distance from the reference velocity. As in `P_obs`, the image has a shape
        of (k, l) and is normalized at each period by the maximum. (iv) Element-
        wise multiplication of `P_ref` and `P_obs` yields a third image, call it 
        `P_cond`, which can be interpreted as the probability to measure a
        dispersion curve at given period and velocity given the dispersion
        observations and the a-priori knowledge of the reference curve. (v) The
        dispersion curve is picked starting from the longest periods, where
        the phase ambiguity is less pronounced, and is driven by two quality
        parameters: `min_derivative` (so as to avoid strong velocity decreases
        with period) and `prob_min` (areas in `P_cond` characterized by values 
        smaller than `prob_min` are not considered).
        
        Parameters
        ----------
        refcurve : ndarray of shape (n, 2)
            Reference curve used to extract the dispersion curves. The 1st
            column should be period, the 2nd velocity (in either
            km/s or m/s). The reference curve is automatically converted to
            km/s, the physical unit employed in the subsequent analysis.
            
        src : str
            Absolute path to the directory containing the dispersion measurements
        
        dist_km : float
            Inter-station distance (in km)
            
        prob_min : float
            "probability" in `P_cond`, at a given period, below which the 
            dispersion curve is not picked. Larger values are more restrictive. 
            Good values are between ~0.2 and ~0.35. Default is 0.25
            
        prior_sigma_10s, prior_sigma_200s : float
            Standard deviations of the Gaussians built around the reference 
            model (refcurve) at the periods of 10 and 200 s to calculate `P_ref`.
            At each analysed period, the standard deviation is interpolated (and
            eventually linearly extrapolated) based on these two values. Smaller
            values give more "weight" to the reference curve in the picking of
            the phase velocities. Defaults are 0.7 and 0.3
            
        smoothing : float
            Final smoothing applied to the dispersion curve. The smoothing is
            performed using `scipy.signal.savgol_filter`. This parameters allows
            for defining the window_length passed to the SciPy function, via::
            
                window_length = int(len(dispcurve) * smoothing)
                
            Default is 0.3
            
        min_derivative : float
            Minimum derivative of phase-velocity as a function of period. If
            exceeded, a quality selection will truncate the dispersion curve.
            To avoid the presence of gaps in the final dispersion curves, if 
            they are present due to this truncation, the branch of dispersion 
            curve associated with the largest probability (calculated from 
            `P_cond`) is returned, the other(s) are discarded. 
            
            Default is -0.01. Smaller (negative) values are less restrictive.
        
        plotting : bool
            If `True`, a figure is created for each retrieved dispersion curve
        
        savefig : str, optional
            Absolute path to the output image. Ignored if `plotting` is False
        
        sta1, sta2 : str, optional
            Station codes of the two receivers used to calculate the dispersion
            curve. If `plotting` is True, they are displayed in title of the
            title of the resulting figure. Otherwise they are ignored
        
        manual_picking : bool
            If `True`, the user is required to pick each dispersion curve manually.
            The picking is carried out through an interactive plot
            
        show : bool
            If `False`, the figure is closed before being shown. This is only
            relevant when the user has the interactive plot enabled: in that
            case, the figure will not be displayed. Default is `True`
            
        Returns
        -------
        dispcurve : ndarray of shape (m, 2)
            Dispersion curve, where he 1st column is period and the 2nd
            is phase velocity in km/s
            

        Raises
        ------
        DispersionCurveException
            If the calculation of the dispersion curve fails.
        """
    
        def load_dispersion_measurements(src):
            data = []
            for file in os.listdir(src):
                data.append(np.load(os.path.join(src, file)))
            return np.vstack(data)    
        
        def dv_cycle_jump(frequency, velocity, dist):
            return np.abs( velocity - 1 / (1/(dist*frequency) + 1/velocity) )
        
        def get_weight_field(points, X, Y, dist, rel_ellipse_width=3, 
                             rel_ellipse_height=0.5):
            
            weight_field = np.zeros_like(X)
            # assuming that the x axis is equally spaced
            ell_width = rel_ellipse_width*np.diff(X[0])[0]  
            for period, velocity in points:
                # The ellipse
                ell_center = (period, velocity)
                ell_height = (rel_ellipse_height * dv_cycle_jump(1/period,
                                                                 velocity,
                                                                 dist))
                            
                xct = X - ell_center[0]
                yct = Y - ell_center[1]             
                rad_cc = (xct**2 / (ell_width/2)**2) + (yct**2 / (ell_height/2)**2)
                weight_field[rad_cc<=1] += np.cos(np.pi/2 * rad_cc[rad_cc<=1])
                
            return weight_field
        
        def posterior_prob(data, dist):
            periods = np.unique(data[:,0])
            velocities = data[:,1]
            period_min = np.min(periods)
            cmin, cmax = np.min(velocities), np.max(velocities)
            min_dv_cycle = dv_cycle_jump(1./period_min, cmin, dist)
            c = np.arange(cmin, cmax, min_dv_cycle/10.)
            X, Y = np.meshgrid(periods, c)
            weight_field = get_weight_field(points=data, 
                                            X=X, 
                                            Y=Y, 
                                            dist=dist,
                                            rel_ellipse_width=5,
                                            rel_ellipse_height=0.5)
            p_post = weight_field / np.abs(weight_field).max()
            return periods, c, (X, Y), p_post
        
        def prior_prob(period, c, ref_model, p_post, prior_sigma_10s=0.7, 
                       prior_sigma_200s=0.3):
            
            def sigma(period):
                x = np.linspace(10, 200)
                y = np.geomspace(prior_sigma_10s, prior_sigma_200s)
                interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
                return interp(period)
            
            refvel = ref_model(period)
            p_prior = np.zeros(p_post.shape)
            
            for i, mu in enumerate(refvel):
                prior = gaussian(c, mu, sigma(period[i]))
                p_prior[:, i] = prior / prior.max()
            return p_prior
        
        def get_probabilistic_velocity(prob, vel, min_rel_height=1.5, 
                                       prominence=0.05):
            peaks = signal.find_peaks(prob, prominence=prominence)[0]
            if peaks.size == 1:
                return vel[peaks[0]], prob[peaks[0]]
            if peaks.size > 1:
                amp = prob[peaks]
                argmax = np.argmax(amp)
                notmax = [i for i in range(amp.size) if i!=argmax]
                if np.all(amp[argmax] > min_rel_height * amp[notmax]):
                    return vel[peaks[argmax]], amp[argmax]
            return None, None

        def get_dispcurve(period, c, p_cond, smoothing=0.3, min_derivative=-0.01):
            
            def get_first_picks(period, c, p_cond):
                probabilities = []
                periods = []
                dispcurve = []
                for iperiod in range(period.size-1, -1, -1):
                    prob = p_cond[:, iperiod]
                    if np.max(prob) < 0.6:
                        if len(dispcurve) >= 3:
                            return iperiod, periods, dispcurve, probabilities
                        else:
                            continue
                    velocity, probability = get_probabilistic_velocity(prob=prob, 
                                                                       vel=c, 
                                                                       min_rel_height=2)
                    if velocity is not None:
                        dispcurve.insert(0, velocity)
                        periods.insert(0, period[iperiod])
                        probabilities.insert(0, probability)
                    else:
                        if len(dispcurve)>=3:
                            return iperiod, periods, dispcurve, probabilities
                return iperiod, periods, dispcurve, probabilities
            
            def get_remaining_picks(ref_model, period, c, p_cond, iperiod, periods, 
                                    dispcurve, probabilities):

                if iperiod == 0:
                    return periods, dispcurve, probabilities
                smoothing_window = c.size // 20
                if not smoothing_window % 2:
                        smoothing_window += 1
                
                for iperiod in range(iperiod, -1, -1):
                    prob = p_cond[:, iperiod]
                    dy = ref_model(periods[1]) - ref_model(periods[0])
                    dx = periods[1] - periods[0]
                    derivative_c =  dy / dx
                    
                    dperiod = abs(period[iperiod] - periods[0])
                    c_pred = dispcurve[0] - derivative_c * dperiod
#                    prob_prior = gaussian(c, c_pred, 0.15)
                    prob_prior = skewed_normal(c, c_pred, 0.2, -5)
                    iloc = np.argmin(np.abs(c - c_pred))
                    prob_prior = np.roll(prob_prior, iloc - np.argmax(prob_prior))
                    prob_cond = prob_prior/prob_prior.max() * prob
                    prob_cond = signal.savgol_filter(prob_cond, smoothing_window, 2)
                    velocity, probability = get_probabilistic_velocity(prob=prob_cond, 
                                                                       vel=c, 
                                                                       min_rel_height=1.3,
                                                                       prominence=0.025)
                    if velocity is not None:
                        dispcurve.insert(0, velocity)
                        periods.insert(0, period[iperiod])
                        probabilities.insert(0, probability)
                    else:
                        break
                return periods, dispcurve, probabilities
                
            def quality_selection(periods, vel, probability, min_derivative=-0.01):
                
                def select_segment(periods, vel, probability, inans):
                    segments = []
                    measurements = []
                    for iperiod in range(len(periods)):
                        if iperiod not in inans:
                            measurements.append((periods[iperiod], 
                                                 vel[iperiod], 
                                                 probability[iperiod]))
                        else:
                            if measurements:
                                segments.append(np.array(measurements))
                                measurements = []
                    if measurements:
                        segments.append(np.array(measurements))
                    
                    segments = [seg for seg in segments if seg.shape[0]>=3]
                    best_prob = 0.6
                    best_segment = None
                    for isegment, segment in enumerate(segments):
                        prob = np.mean(segment[:,2])
                        if prob > best_prob:
                            best_segment = isegment
                            best_prob = prob
                    if best_segment is None:
                        raise DispersionCurveException
                    return segments[best_segment]  
                
                
                derivative = np.gradient(vel) / np.gradient(periods)
                inans = np.flatnonzero(derivative < min_derivative)
                p, v, prob = select_segment(periods, vel, probability, inans).T
                dp = min(np.diff(p))
                inans = np.flatnonzero(np.diff(p) > dp*10)
                if inans:
                    p, v, _ = select_segment(p, v, prob, inans).T
                return np.column_stack((p, v))
            
            
            def quality_check_first_picks(ref_model, periods, vel):
                if len(vel) < 3:
                    raise DispersionCurveException
                dy = vel[-1] - vel[0]
                dx = periods[-1] - periods[0]
                derivative_obs = dy / dx
                dy_ref = ref_model(periods[-1]) - ref_model(periods[0])
                derivative_ref = dy_ref / dx
                if derivative_obs > derivative_ref * 5:
                    raise DispersionCurveException
                
            iperiod, periods, dispcurve, prob = get_first_picks(period, c, p_cond)
            quality_check_first_picks(ref_model, periods, dispcurve)
            periods, dispcurve, prob = get_remaining_picks(ref_model=ref_model,
                                                           period=period, 
                                                           c=c, 
                                                           p_cond=p_cond, 
                                                           iperiod=iperiod,
                                                           periods=periods,
                                                           dispcurve=dispcurve,
                                                           probabilities=prob)
            smoothing_window = int(len(dispcurve) * smoothing)
            if not smoothing_window % 2:
                smoothing_window += 1
            dispcurve = signal.savgol_filter(dispcurve, 
                                             window_length=max(smoothing_window, 3), 
                                             polyorder=2)
            return quality_selection(periods, dispcurve, prob, min_derivative=min_derivative)
        

        def plot(data, mesh, p_prior, p_post, p_cond, p_cond_filtered, dispcurve, 
                 dist, no_earthquakes, sta1=None, sta2=None, savefig=None, show=True):
            suptitle = 'Dist: %.2f km No. Earthquakes: %d'%(dist, no_earthquakes)
            if sta1 is not None and sta2 is not None:
                suptitle = '%s - %s '%(sta1, sta2) + suptitle
            xlim = np.min(mesh[0][0]), np.max(mesh[0][0])
            ylim = np.min(mesh[1][:, 0]), np.max(mesh[1][:, 0])
            
            plt.figure(figsize=plt.figaspect(0.8))
            plt.subplot(2, 2, 1)
            plt.pcolormesh(*mesh, p_prior)
            plt.plot(refcurve[:,0], refcurve[:,1], 'b', label='Reference')
            plt.ylim(*ylim)
            plt.xlim(*xlim)
            plt.ylabel('Velocity [km/s]')
            plt.legend(loc='upper right')
            plt.title('Prior')
            
            plt.subplot(2, 2, 2)
            plt.pcolormesh(*mesh, p_post)
            plt.plot(data[:,0], data[:,1], 'ro', ms=0.5, label='Obs. Dispersion')
            plt.plot(dispcurve[:,0], dispcurve[:,1], 'r', label='Retrieved')
            plt.ylim(*ylim)
            plt.xlim(*xlim)
            plt.legend(loc='upper right')
            plt.title('Density of observations')
            
            plt.subplot(2, 2, 3)
            plt.pcolormesh(*mesh, p_cond)
            plt.plot(dispcurve[:,0], dispcurve[:,1], 'r', label='Retrieved')
            plt.ylim(*ylim)
            plt.xlim(*xlim)
            plt.ylabel('Velocity [km/s]')
            plt.xlabel('Period [s]')
            plt.legend(loc='upper right')
            plt.title('Weighted density of obs.')
            
            plt.subplot(2, 2, 4)
            plt.pcolormesh(*mesh, p_cond_filtered)
            plt.plot(refcurve[:,0], refcurve[:,1], 'b', label='Reference')
            plt.plot(dispcurve[:,0], dispcurve[:,1], 'r', label='Retrieved')
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.xlabel('Period [s]')
            plt.legend(loc='upper right')
            plt.title(r'Weighted and filtered ($prob\_min$=%.2f)'%prob_min)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.91)
            plt.suptitle(suptitle, y=0.98)
            if savefig is not None:
                plt.savefig(savefig, dpi=(200))
            if not show:
                plt.close()
            else:
                plt.show()

        def pick_curve_manually(data, mesh, p_post, refcurve, dist, no_earthquakes,
                                plotting=False, savefig=None, sta1=None, sta2=None):

            def plot_mesh_and_data():
                plt.pcolormesh(*mesh, p_post)
                plt.plot(data[:,0], data[:,1], 'ro', ms=0.5, label='Obs. Dispersion')
                plt.plot(*refcurve.T, label='Reference', color='k')
                plt.ylim(*ylim)
                plt.xlim(*xlim)
            
            from IPython import get_ipython
            get_ipython().magic('matplotlib auto')
            title = r'$\bf{Pick}$: Left click, $\bf{Delete}$: Right click, $\bf{When\ finished}$: Enter'
            question = r'$\bf{Are\ you\ satisfied?\ Answer\ in\ the\ console\ (y/n)}$'
            xlim = np.min(mesh[0][0]), np.max(mesh[0][0])
            ylim = np.min(mesh[1][:, 0]), np.max(mesh[1][:, 0])
            while True:
                fig = plt.figure(figsize=(13, 9))
                plot_mesh_and_data()
                plt.suptitle(title, fontsize=23)
                plt.legend()
                dispcurve = np.array(plt.ginput(-1, 0))
                if not dispcurve.size:
                    raise DispersionCurveException
    #            dispcurve[:,1] = savgol_filter(dispcurve[:,1], 5, 2)
                plt.plot(*dispcurve.T, label='Picked', color='b', lw=2)
                plt.legend()
                plt.suptitle(question, fontsize=23)
                print('Are you satisfied (y/n)? Answer below')
                fig.canvas.draw()
                fig.canvas.flush_events()
                answer = input()
                plt.show()
                plt.close()
                if answer.lower() == 'y': 
                    if plotting:
                        suptitle = 'Dist: %.2f km No. Earthquakes: %d'%(dist, no_earthquakes)
                        if sta1 is not None and sta2 is not None:
                            suptitle = '%s - %s '%(sta1, sta2) + suptitle
                        fig = plt.figure(figsize=(13, 9))
                        plot_mesh_and_data()
                        plt.plot(*dispcurve.T, label='Picked', color='b', lw=2)
                        plt.legend(loc='upper right')  
                        plt.suptitle(suptitle, y=0.98)
                        if savefig is not None:
                            plt.savefig(savefig, dpi=(200))
                        plt.show()
                        plt.close()
                    return dispcurve
            
        
        refcurve = cls.convert_to_kms(refcurve)
        ref_model = interp1d(refcurve[:,0], 
                             refcurve[:,1],
                             bounds_error=False, 
                             fill_value='extrapolate')
        data = load_dispersion_measurements(src)
        period, c, mesh, p_post = posterior_prob(data, dist_km)
        if manual_picking:
            dispcurve = pick_curve_manually(data=data, 
                                            mesh=mesh, 
                                            p_post=p_post, 
                                            refcurve=refcurve, 
                                            plotting=plotting,
                                            savefig=savefig,
                                            dist=dist_km,
                                            no_earthquakes=len(os.listdir(src)),
                                            sta1=sta1,
                                            sta2=sta2)
        else:
            p_prior = prior_prob(period=period, 
                                 c=c, 
                                 ref_model=ref_model, 
                                 p_post=p_post, 
                                 prior_sigma_10s=prior_sigma_10s, 
                                 prior_sigma_200s=prior_sigma_200s)
            p_cond = p_prior * p_post
            p_cond_filtered = np.where(p_cond<prob_min, 0, p_cond)
            dispcurve = get_dispcurve(period, 
                                      c, 
                                      p_cond_filtered, 
                                      smoothing,
                                      min_derivative=min_derivative)
            if plotting:
                plot(data=data, 
                     mesh=mesh, 
                     p_prior=p_prior, 
                     p_post=p_post, 
                     p_cond=p_cond, 
                     p_cond_filtered=p_cond_filtered, 
                     dispcurve=dispcurve, 
                     dist=dist_km,
                     no_earthquakes=len(os.listdir(src)), 
                     sta1=sta1,
                     sta2=sta2, 
                     savefig=savefig,
                     show=show)
        return dispcurve


