#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download of Teleseismic-Earthquake Recordings
============================================= 

The below class allows for automated downloads of seismograms
recording teleseismic surface waves. The data will be saved on disk
in the format required by :class:`seislib.eq.eq_velocity.EQVelocity` and
:class:`seislib.eq.eq_velocity.TwoStationMethod` to calculate inter-station
dispersion curves based on a two-station method.

"""

import os
import warnings
import time
import socket
from obspy import read_inventory, Catalog
from obspy import UTCDateTime as UTC
from obspy.core import AttribDict
from obspy.geodetics import gps2dist_azimuth, degrees2kilometers
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException
from obspy.io.mseed import InternalMSEEDError
from http.client import IncompleteRead
from collections import defaultdict
from seislib.utils import resample, rotate_stream, remove_file
from seislib.plotting import plot_stations
warnings.simplefilter(action='ignore')
ONE_YEAR = 365 * 86400



class EQDownloader:
    """ Downloads surface-wave data to be used via the two-station method

    Parameters
    ----------
    savedir : str
        Absolute path to the directory where the data will be saved. Inside 
        this directory, a folder named 'data' will be created. Inside 'data'
        the seismograms will be organized in subdirectories (one for each 
        seismic event) in the format net.sta.loc.cha.sac.
        
    inventory_name : str
        Name of the inventory (`obspy.core.inventory.inventory.Inventory`) 
        associated with the downloads. This will be saved in xml format
        in the `savedir`
        
    inv_provider : str
        Provider of station metadata, passed to `obspy.clients.fdsn.Client`. 
        Default is `iris`
        
    user, password : str, optional
        User name and password for accessing restricted data. Passed to 
        `obspy.clients.fdsn.Client`
        
    ev_provider : str
        Provider of earthquake metadata, passed to `obspy.clients.fdsn.Client`. 
        Default is `iris`
        
    vmin, vmax : float
        Minimum and maximum velocity of the surface waves. These values are
        used to establish the starting and ending time of the seismograms
        to be downloaded with respect to the origin time of given earthquake.
        Relatively loose limits are to be preferred. Default values are 1.5 
        and 5.5 (km/s), respectively
        
    sampling_rate : int, float
        Final sampling rate of the waveforms, in Hz. Default is 1 Hz
        
    units : {'DISP', 'VEL', 'ACC'}
        Physical units of the final waveforms. Can be either 'DISP', 'VEL',
        or 'ACC' (case insensitive). Default is 'DISP'.
        
    prefilter : (4, ) tuple of floats
        Bandapass filter applied to the waveforms before removing the
        intrument response. Default is (0.001, 0.005, 0.05, 0.4)
        
    attach_response : bool
        If `True`, the details of the instrument response are attached to the 
        header of the waveforms during their downloads. It can slow down the 
        downloads, but it will make errors due to response information less 
        likely. Default is `False`
        
    stations_config : dict
        Python dictionary containing optional keys and values passed to
        `obspy.clients.fdsn.Client.get_stations`.
        
    events_config : dict
        Python dictionary containing optional keys and values passed to
        `obspy.clients.fdsn.Client.get_events`. Default values are::
            distmin = 2223.9 (km, corresponding to 20°)
            distmax = 15567.3 (km, corresponding to 140°)
            depthmax = 50 (km)
            magmin = 6
            magmax = 8.5
            starttime = UTC(2000, 1, 1)
            endtime = UTC(2021, 1, 1)

        .. warning:: 
            The obspy key words 'minmag', 'maxmag', and 'maxdepth' have been 
            renamed as 'magmin', 'magmax', and 'depthmax', for better clarity 
            in the code. Passing to `events_config` one of the (obspy) key 
            words among 'minmag', 'maxmag', and 'maxdepth' will result in an 
            error.
        
        .. note::
            The above listed default values include 'distmin' and 'distmax' (in km). These refer 
            to the minimum and maximum distance of the epicenter from a given receiver. Note 
            that these two key words correspond to the obspy's 
            'minradius' and 'maxradius', with the only difference that the latters 
            are expressed in degrees. 'distmin' and 'distmax' have been introduced only for 
            achieving a higher consistency in the use of physical units throughout the code. 
            The user should be aware that, although the use of obspy's 'minradius' and 
            'maxradius' will not result in any error, their use is suggested against. 
            In fact, if 'distmin' and 'distmax' are not specified, their default value 
            will be used in the downloads. And if this default is more "restrictive" than 
            'minradius' and 'maxradius', 'minradius' and 'maxradius' will simpy be ignored 
            in the downloads.
        
    sleep_time_after_exception : float
        Time to wait (in s) after an `obspy.clients.fdsn.header.FDSNException`.
        The excpetion could be due to temporal issues of the client, so
        waiting a little bit before the next download could be useful to
        get things back to normality.
        
    verbose : bool
        If `True`, information on the downloads will be printed in console


    Attributes
    ----------
    savedir : str
        Absolute path of the directory that will be created
        
    staclient : obspy.clients.fdsn.Client
        Provider of station metadata and waveforms
        
    evclient : obspy.clients.fdsn.Client
        Provider of events metadata
        
    channels : list
        List of candidate channels for the downloads
        
    components : {3, 1}
        Either 3 (ZNE) or 1 (Z) 
        
    vmin, vmax : float
        Minimum and maximum expected velocity of the surface waves
        
    sampling_rate : float
        Sampling rate of the final seismograms
        
    prefilter : tuple
        Filter applied to the waveform before removal of the instrument response
        
    units : str
        Physical units of the final seismogram
        
    attach_response : bool     
        Whether or not the response information is attached to the waveforms
        during the download
        
    verbose : bool
        Whether or not progress information are displayed in the console
        
    sleep_time_after_exception : float
        The downloads are stopped for the specified time (in s) after a FDSN 
        exception
        
    distmin, distmax : float
        Minimum and maximum distance of a station from the epicenter (km)
        
    depthmax : float
        Maximum depth of the earthquakes (km)
        
    magmin, magmax : float
        Minimum and maximum magnitude
        
    starttime, endtime : obspy.core.utcdatetime.UTCDateTime
        Starttime and entime of the events catalogue
        
    stations_config, events_config : dict
        Dictionary object with information relevant to the download of the
        stations and events metadata
        
    inventory : obspy.core.inventory.inventory.Inventory
        Stations metadata
        
    _exceptions : collections.defaultdict(int)
        Dictionary object with information on the occurred exceptions
        
    _downloaded : int
        Number of waveforms downloaded
        
    _no_stations : int
        Number of stations to download
        
    _events_done : int
        Number of events for which the downloads have been finished


    Examples
    --------
    The following will download seismic data from all the seismic networks 
    and stations managed by iris within the region specified by minlatitude, 
    maxlatitude, minlongitude, maxlongitude. All such parameters should be passed to
    `stations_config`::

        stations_config=dict(
                channel='LH*,BH*,HH*',
                includerestricted=False,
                maxlatitude=12,
                minlatitude=-18,
                minlongitude=90,
                maxlongitude=140)    
            
    The above also specifies the preference of downloading 3-component seismograms (ZNE), 
    as implied by the asterisk sign. For a given receiver recording an earthquake, either 
    of the LH, BH, or HH channels will be downloaded, with priority given from left to right, 
    i.e., LH>BH>HH: if LH is not available, BH will be downloaded; if BH is not available, HH will 
    be downloaded.
        
    In this example, we use all earthquakes characterized by a 
    magnitude between 6 and 8.5, by a maximum depth of 50km, and generated 
    between the 1.1.2000 and 1.1.2021. The distance between each earthquake
    and a given receiver should be more than 20° (expressed in km) and less
    than 140° (in km)::

        from obspy import UTCDateTime as UTC
        from obspy.geodetics import degrees2kilometers

        events_config=dict(
                starttime=UTC(2000, 1, 1),
                endtime=UTC(2021, 1, 1),
                depthmax=50,
                magmin=6, 
                magmax=8.5,
                distmin=degrees2kilometers(20),
                distmax=degrees2kilometers(140),
                    )      
        
    We initialize the EQDownloader instance, and then start it::
        
        from seislib.eq import EQDownloader
        
        downloader = EQDownloader(savedir='/path/to/directory',
                                  inv_provider='iris',
                                  ev_provider='iris',
                                  inventory_name='iris.xml',
                                  sampling_rate=1,
                                  prefilter=(0.001, 0.005, 0.1, 0.4),
                                  vmin=1.5,
                                  vmax=5.5,
                                  units='disp',
                                  attach_response=False,
                                  stations_config=stations_config,
                                  events_config=events_config,
                                  verbose=True)        
        downloader.start()
    """


    def __init__(self, savedir, inventory_name, inv_provider='iris', user=None, 
                 password=None, ev_provider='iris', vmin=1.5, vmax=5.5, 
                 sampling_rate=1, units='disp', prefilter=(0.001, 0.005, 0.05, 0.4), 
                 attach_response=False, stations_config={}, events_config={},
                 sleep_time_after_exception=30, verbose=True):

        self.savedir = savedir
        os.makedirs(savedir, exist_ok=True)
        self.staclient = Client(inv_provider, user=user, password=password)
        self.evclient = Client(ev_provider)

        self.channels = stations_config.get('channel', None).split(',')
        self.components = 3 if self.channels[0][2]=='*' else 1
        self.sampling_rate = sampling_rate
        #self.npts = self.sampling_rate * self.duration
        self.vmin = vmin
        self.vmax = vmax
        self.prefilter = prefilter
        self.units = units
        self.attach_response = attach_response
            
        self.verbose = verbose
        self.sleep_time_after_exception = sleep_time_after_exception
        
        self.distmin = events_config.get('distmin', degrees2kilometers(20))
        self.distmax = events_config.get('distmax', degrees2kilometers(140))
        self.depthmax = events_config.get('depthmax', 50)
        self.magmin = events_config.get('magmin', 6)
        self.magmax = events_config.get('magmax', 8.5)
        self.starttime = events_config.get('starttime', UTC(2000, 1, 1))
        self.endtime = events_config.get('endtime', UTC(2021, 1, 1))
        keys = ['distmin', 'distmax', 'depthmax', 'magmin', 'magmax',
                'starttime', 'endtime']
        self.events_config = {k: v for k, v in events_config.items() \
                              if k not in keys}

        self.stations_config = stations_config
        inventory_path = os.path.join(savedir, inventory_name)
        if not os.path.exists(inventory_path):
            self.inventory = self.build_inventory(**stations_config)
            self.inventory.write(inventory_path, format="STATIONXML")
        else:
            self.inventory = read_inventory(inventory_path)
        
        self._exceptions = defaultdict(int)
        self._downloaded = 0
        self._events_done = 0
    
    
    def __str__(self):
        string = '\nWAVEFORMS DOWNLOADED: %d\n'%self._downloaded
        if len(self._exceptions) > 0:
            string += 'EXCEPTIONS:\n'
            for key, value in sorted(self._exceptions.items()):
                string += ''.join([' '*5, '%s: %d\n'%(key.upper(), value)])
        string += ''.join(['-'*40, '\n', ' '*5, '* '*3])
        string += 'EVENTS DONE: %d'%(self._events_done)
        string += ''.join([' *'*3, '\n'])
        string += ''.join(['-'*40, '\n'])
        return string

    def build_inventory(self, **kwargs):
        """ Builds an obspy inventory containing stations information
        
        The strategy is to first download inventory information at station level.
        Then, for each station, the instrument response is downloaded for each
        channel. This may increase the downloading time, but prevents possible
        "timed out errors".
        
        Parameters
        ----------
        **kwargs : dict, optional
            Additional key word arguments passed to the get_stations method of 
            `obspy.clients.fdsn.client`
            
        Returns
        -------
        inv : obspy.core.inventory.inventory.Inventory
        """
        def attach_channels(inv, **kwargs):
            
            tmp_kwargs = {i:j for i, j in kwargs.items() \
                          if i not in ['network', 'station']}
            for network in inv:
                for station in network:
                    if station.__dict__.get('creation_date') is None:
                        station.creation_date = station.start_date
                    while 1:
                        try:
                            tmp = self.staclient.get_stations(network=network.code, 
                                                              station=station.code, 
                                                              level='response',
                                                              **tmp_kwargs)
                            station.channels.extend(tmp[0][0].channels)
                            break
                        except FDSNNoDataException:
                            break
                        except FDSNException:
                            time.sleep(0.5)
                            continue
                        except TypeError:
                            continue
                        except ConnectionResetError:
                            time.sleep(0.5)
                            continue
                        except socket.timeout:
                            time.sleep(0.5)
                            continue
            return inv

        if self.verbose > 0:
            print('\n***COLLECTING STATIONS INFO***\n')
        inv = self.staclient.get_stations(**kwargs)
        if self.verbose:
            print(inv)
        return attach_channels(inv, **kwargs)

    @classmethod
    def inventory_iterator(cls, inventory, reverse=False):
        """ Generator function to iterate over an obspy inventory
        
        Parameters
        ----------
        inventory : obspy.core.inventory.inventory.Inventory  
        
        reverse : bool
            If `True`, the inventory will be iterated over from bottom to top
            
            
        Yields
        ------
        (2,) tuple 
            Yields network and station information at each iteration
        """
        def func(iterable, reverse=False):
            if reverse:
                return reversed(iterable)
            return iterable
        
        for network in func(inventory, reverse=reverse):
            for station in func(network, reverse=reverse):
                yield network, station
                
                
    @classmethod
    def event_coordinates_and_time(cls, event):
        """ Fetch event coordinates (lat, lon, depth) and origin time.
        
        Parameters
        ----------
        event : obspy.core.event.event.Event
        
        Returns
        -------
        (2,) Tuple
            (latitude, longitude, depth) and origin time in 
            `obspy.UTCDateTime.timestamp` format
    
        """
        otime = event.origins[0].time
        evla = event.origins[0].latitude
        evlo = event.origins[0].longitude
        evdp = event.origins[0].depth
        return (evla, evlo, evdp), otime
    
    
    @classmethod
    def station_coordinates(cls, station_info):
        """ Fetch station coordinates (lat, lon, elevation).
        
        Parameters
        ----------
        station_info : obspy.core.inventory.station.Station
        
        
        Returns
        -------
        (3,) Tuple of floats
            latitude, longitude, elevation
        """
        stla = station_info.latitude
        stlo = station_info.longitude
        stel = station_info.elevation
        return (stla, stlo, stel)


    @classmethod
    def get_event_info(cls, event):
        """ Fetch event information
        
        Parameters
        ----------
        event : obspy.core.event.event.Event
        
        Returns
        -------
        (2, ) Tuple
            (origin time, magnitude), (latitude, longitude, depth)
        """
        (evla, evlo, evdp), otime = cls.event_coordinates_and_time(event)
        mag = event.magnitudes[0].mag
        return (otime, mag), (evla, evlo, evdp)    
    
    
    def fetch_catalog(self, t1, t2, **kwargs):
        """ Fetch catalog of seismic events
        
        Parameters
        ----------
        t1, t2 : obspy.core.utcdatetime.UTCDateTime
            Starttime and endtime of the catalog
            
        **kwargs : dict, optional
            Additional key-word arguments passed to 
            `obspy.clients.fdsn.client.Client.get_events`
        
        Returns
        -------
        obspy.core.event.catalog.Catalog
            Catalog of seismic events
        """
        catalog = self.evclient.get_events(
                starttime=t1, 
                endtime=t2, 
                maxdepth=self.depthmax,
                minmagnitude=self.magmin,
                maxmagnitude=self.magmax,
                **kwargs
                )
        return Catalog(sorted(catalog, key=lambda ev: ev.origins[0].time))
      
        
    @classmethod     
    def station_was_active(cls, station, time):
        """ Wheater or not the seismic station was active at the given time
        
        Parameters
        ----------
        station : obspy.core.inventory.station.Station
        
        time : obspy.core.utcdatetime.UTCDateTime
        
        
        Returns
        -------
        bool
            Whether or not the receiver was active at the specified time
        """
        if station.start_date and (time<station.start_date):
            return False
        if station.end_date and (time>station.end_date):
            return False
        return True
    
    
    def active_channels(self, station):
        """ Channels available for the given station among those to download
        
        Parameters
        ----------
        station : obspy.core.inventory.station.Station
        
        Returns
        -------
        List
        """
        channels = set([channel.code[:2] for channel in station.channels])
        return [cha for cha in self.channels if cha[:2] in channels]
    
    
    def collect_waveforms(self, network, station, channels, starttime, endtime):
        """ Downloads obspy stream
        
        Parameters
        ----------
        network, station : str 
            network and station codes
            
        channels : iterable of str
            iterable containing the channels codes suited to the download. Each
            channel will be tried to be used in the downloads. The first successfull
            attempt determines the returned waveforms. (Higher priority is given to
            decreasing indexes)
            
        starttime, endtime : obspy.core.utcdatetime.UTCDateTime
            start and end time of the stream.
        
        Returns
        -------
        obspy.core.stream.Stream if the download is successful, else None
        """
        
        def gaps_are_present(st):
            return False if not st.get_gaps() else True
        
        for cha in channels:
            try:
                st = self.staclient.get_waveforms(network=network, 
                                                  station=station, 
                                                  location='*', 
                                                  channel=cha, 
                                                  starttime=starttime, 
                                                  endtime=endtime,
                                                  attach_response=self.attach_response)
                if gaps_are_present(st):
                    self._exceptions['Gaps in stream'] += 1
                    return None
#                st.merge(method=1, interpolation_samples=2, fill_value=0)
                return st               
                
            except FDSNNoDataException:
                self._exceptions['FDSNNoDataException'] += 1
            except InternalMSEEDError:
                self._exceptions['InternalMSEEDError'] += 1
            except FDSNException:
                time.sleep(self.sleep_time_after_exception)
            except IncompleteRead:
                self._exceptions['IncompleteRead'] += 1
                
        return None
    
    
    def handle_multiple_locations(self, st, station_info):
        """ Automatic selection of location.
        
        Parameters
        ----------
        st : obspy.core.stream.Stream
        
        station_info : obspy.core.inventory.station.Station
        
        
        Returns
        -------
        st : obspy.core.stream.Stream
        """
        locations = set([tr.stats.location for tr in st])
        if len(locations) > 1:
            channel_type = st[0].stats.channel[0]
            for channel in station_info:
                if channel.location_code not in locations:
                    continue
                if channel.code[0] != channel_type:
                    continue
                if 'borehole' in str(channel.sensor.description).lower():
                    location = channel.location_code
                    break
            else:
                location = sorted(set([tr.stats.location for tr in st]))[0]    
            return st.select(location=location)
        
        return st


    def prepare_data(self, st):
        """ Demean, detrend, tapering, removing response and resampling
        
        Parameters
        ----------
        st : obspy.core.stream.Stream
        
        
        Returns
        -------
        obspy.core.stream.Stream if the processing is successful, else None
        """
        try:
            st.detrend('demean')
            st.detrend('linear')
        except:
            self._exceptions['detrend'] += 1
            return
        st.taper(type='hann', max_percentage=0.05)        
        inventory = None if self.attach_response else self.inventory
        try:
            st.remove_response(output=self.units, 
                               inventory=inventory,
                               pre_filt=self.prefilter)
        except ValueError:
            self._exceptions['Response error'] += 1
            return
            
        sampling_rates = set([tr.stats.sampling_rate for tr in st])
        if len(sampling_rates) > 1:
            self._exceptions['Multiple sampling rates'] += 1
            return
        if sampling_rates.pop() != self.sampling_rate:
            st = resample(st, self.sampling_rate)
        return st


    def adjust_channels(self, st):
        """
        If the stream contains the Z12 channels, these are rotated towards ZNE.
        
        Parameters
        ----------
        st : obspy.core.stream.Stream
        
        Returns
        -------
        obspy.core.stream.Stream if the rotation is successful, else None
        """
        channels = set([tr.stats.channel[-1] for tr in st])
        if 'Z' in channels and 'N' in channels and 'E' in channels:
            return st
        sorting_priorities = {'Z':0, '1':1, '2':2, '3':3}
        sorting_method = lambda i: sorting_priorities[i]
        try:
            channels = ''.join(sorted(channels, key=sorting_method))
        except KeyError:
            self._exceptions['rotation (unknown orientation)'] += 1
        try:
            st = rotate_stream(st, method='->ZNE', inventory=self.inventory, 
                               components=[channels])
        except Exception:
            self._exceptions['rotation'] += 1
            return
        channels = [tr.stats.channel[-1] for tr in st]
        assert 'Z' in channels and 'N' in channels and 'E' in channels
        return st


    def select_components(self, st, baz):
        """ Handles the absence of some components in the final stream
        
        Parameters
        ----------
        st : obspy.Stream
        
        baz : int, float (in degrees)
            Back-azimuth used for the rotation NE->RT
        
        Returns
        -------
        st : obspy.core.stream.Stream
            If the expected three-component stream lacks of one or both the
            horizontal components, the vertical component is returned. If the
            stream lacks the vertical component but has the two horizontal ones,
            it returns the horizontal components rotated towards the back
            azimuth. If only one horizontal component is available, it returns 
            None        
        """
        channels = [tr.stats.channel[-1] for tr in st]
        if 'Z' in channels:
            return st.select(channel='*Z')
        if 'E' in channels and 'N' in channels:
            assert len(st) == 2
            try:
                st = rotate_stream(st, method='NE->RT', back_azimuth=baz)
            except ValueError:
                self._exceptions['rotation'] += 1
                return None
            return st


    def preprocessing(self, st, station, baz):
        """ Preprocessing of the obspy stream
        
        The function calls sequentially the methods :meth:`handle_multiple_locations`,
        :meth:`prepare_data`, and :meth:`adjust_channels`.
        
        Parameters
        ----------
        st : obspy.core.stream.Stream
        
        station : obspy.core.inventory.station.Station
        
        baz : float
            Back azimuth (degrees) of the epicenter with respect to the receiver
        
        Returns
        -------
        obspy.core.stream.Stream if the preprocessing is successful, else None
        """
        if st is None or len(st)==0:# or len(st)<self.components:
            return
        st = self.handle_multiple_locations(st, station)
        st = self.prepare_data(st)
        if st is None or len(st)>self.components:# or len(st)!=self.components:
            return
        
        if len(st) < self.components:
            return self.select_components(st, baz)
        
        elif len(st) == self.components == 3:
            st = self.adjust_channels(st)
            if st is None:
                return
            try:
                return rotate_stream(st, method='NE->RT', back_azimuth=baz)
            except ValueError:
                self._exceptions['rotation'] += 1
                return
            
        return st
    
    
    def compile_header_and_save(self, st, savedir, stla, stlo, stel, evla, evlo, evdp, 
                                otime, mag, dist, az, baz):
        """ Compiles the header of the obspy stream (sac format) and writes to disk
        
        Parameters
        ----------
        stream : obspy.core.stream.Stream
        
        stla, stlo, stel : float
            Latitude, longitude, elevation of the seismic station
            
        evla, evlo, evdp : float
            Latitude, longitude, depth of the earthquake
            
        otime : obspy.core.utcdatetime.UTCDateTime
            Origin time of the earthquake
            
        mag, dist : float
            Event magnitude and distance (km) of the event from the receiver
            
        az, baz : float
            Azimuth and back azimuth of the epicenter with respect to the receiver
        """
        for tr in st:
            tr.stats.sac = AttribDict()
            tr.stats.sac.stla = stla
            tr.stats.sac.stlo = stlo
            tr.stats.sac.stel = stel
            tr.stats.sac.nzhour = tr.stats.starttime.hour
            tr.stats.sac.nzjday = tr.stats.starttime.julday
            tr.stats.sac.nzmin = tr.stats.starttime.minute
            tr.stats.sac.nzmsec = tr.stats.starttime.microsecond
            tr.stats.sac.nzsec = tr.stats.starttime.second
            tr.stats.sac.nzyear = tr.stats.starttime.year
            tr.stats.sac.e = tr.stats.endtime - tr.stats.starttime
            tr.stats.sac.evla = evla
            tr.stats.sac.evlo = evlo
            tr.stats.sac.evdp = evdp
            tr.stats.sac.mag = mag
            tr.stats.sac.o = otime.timestamp - tr.stats.starttime.timestamp
            tr.stats.sac.dist = dist
            tr.stats.sac.az = az
            tr.stats.sac.baz = baz
            outfile = os.path.join(savedir, str(otime.timestamp), '%s.sac'%tr.id)
            tr.write(outfile, format='sac')
        
    
    def plot_stations(self, ax=None, show=True, oceans_color='water', 
                      lands_color='land', edgecolor='k', projection='Mercator',
                      resolution='110m', color_by_network=True, legend_dict={}, 
                      **kwargs):
        """ Creates a maps of seismic receivers available for download
        
        Parameters
        ----------
        stations : dict
            Dictionary object containing stations information. This should 
            structured so that each key corresponds to a station code 
            ($network_code.$station_code) and each value is a tuple containing 
            latitude and longitude of the station. 
            
            For example::
                
                { net1.sta1 : (lat1, lon1),
                  net1.sta2 : (lat2, lon2),
                  net2.sta3 : (lat3, lon3)
                  }
        
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            If not None, the receivers are plotted on the `GeoAxesSubplot` instance. 
            Otherwise, a new figure and `GeoAxesSubplot` instance is created
            
        show : bool
            If True, the plot is shown. Otherwise, a `GeoAxesSubplot` instance is
            returned. Default is `True`
            
        oceans_color, lands_color : str
            Color of oceans and lands. The arguments are ignored if ax is not
            `None`. Otherwise, they are passed to `cartopy.feature.NaturalEarthFeature` 
            (to the argument 'facecolor'). Defaults are 'water' and 'land'
            
        edgecolor : str
            Color of the boundaries between, e.g., lakes and land. The argument 
            is ignored if ax is not None. Otherwise, it is passed to 
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
            '50m', '10m'. Default is '110m'
        
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
        stations = {}
        for net, sta in self.inventory_iterator(self.inventory):
            station_code = '%s.%s'%(net.code, sta.code)
            stla, stlo, stel = self.station_coordinates(sta)
            stations[station_code] = (stla, stlo)
        
        return plot_stations(stations=stations,
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
            
    
    
    def start(self):
        """ Starts the downloads.
        
        The catalog of seismic events is iterated over and, for each event, all
        the waveforms from the receivers associated with `stations_config`
        will be downloaded, if available. The waveforms are detrended, demeaned, 
        removed of the instrument response, and resampled. Finally, they are
        saved in savedir/data/event, where 'event' is the origin time (in the
        obspy timestamp format) of the considered earthquake.
        
        The algorithm keeps track of the progress made. This allows one to stop
        the downloads and start from where it was left off at any time.
        """
        
        def get_done_stations(savedir, event):
            done_file = os.path.join(savedir, event, 'DONE.txt')
            if not os.path.exists(done_file):
                return set()
            return set([i.strip() for i in open(done_file)])
        
        def get_starttime(savedir, inv):
            remove_file(os.path.join(savedir, '.DS_STORE')) # mac tmp file
            events = sorted(os.listdir(savedir), key=lambda i: float(i))
            if not events:
                return None
            station_codes = set([
                    '%s.%s'%(net.code, sta.code) \
                    for net, sta in EQDownloader.inventory_iterator(inv)
            ])
            for event in events:
                done_stations = get_done_stations(savedir, event)
                if station_codes.difference(done_stations):
                    return UTC(float(event)) - 1
            return UTC(float(event)) + 1
        
        def update_done_file(file, station_code):
            with open(file, 'a') as f:
                f.write('%s\n'%station_code)
                
                
        savedir = os.path.join(self.savedir, 'data')
        os.makedirs(savedir, exist_ok=True)
        t1 = get_starttime(savedir, self.inventory)
        
        if t1 is None:
            t1 = self.starttime
        endtime = self.endtime
        while endtime > t1:
            t2 = t1+ONE_YEAR if t1+ONE_YEAR<=endtime else endtime
            
            catalog = self.fetch_catalog(t1, t2, **self.events_config)
            for event in catalog:
                (otime, mag), (evla, evlo, evdp) = self.get_event_info(event)
                outdir = os.path.join(savedir, str(otime.timestamp))
                os.makedirs(outdir, exist_ok=True)
                event.write(os.path.join(outdir, '%s.xml'%str(otime.timestamp)),
                            format='quakeml')
                done_file = os.path.join(savedir, str(otime.timestamp), 'DONE.txt')
                print(os.path.join(savedir, str(otime.timestamp)))
                done_stations = get_done_stations(savedir, str(otime.timestamp))
                if self.verbose:
                    print('EVENT:', otime.isoformat().split('T')[0])
                    print('- Mag: %s\n- Lat: %.2f\n- Lon: %.2f'%(mag, evla, evlo))
                for net, sta in self.inventory_iterator(self.inventory):
                    station_code = '%s.%s'%(net.code, sta.code)
                    if station_code in done_stations:
                        continue
                    if not self.station_was_active(sta, otime):
                        update_done_file(done_file, station_code)
                        continue
                    stla, stlo, stel = self.station_coordinates(sta)
                    dist, az, baz = gps2dist_azimuth(evla, evlo, stla, stlo)
                    dist /= 1000
                    if dist>self.distmax or dist<self.distmin:
                        update_done_file(done_file, station_code)
                        continue
                    channels = self.active_channels(sta)
                    st = self.collect_waveforms(network=net.code, 
                                                station=sta.code,
                                                channels=channels,
                                                starttime=otime + dist/self.vmax,
                                                endtime=otime + dist/self.vmin)
                    st = self.preprocessing(st, sta, baz)
                    if st is None:
                        update_done_file(done_file, station_code)
                        continue
                    self.compile_header_and_save(st=st, 
                                                 savedir=savedir, 
                                                 stla=stla, 
                                                 stlo=stlo, 
                                                 stel=stel, 
                                                 evla=evla, 
                                                 evlo=evlo, 
                                                 evdp=evdp, 
                                                 otime=otime, 
                                                 mag=mag, 
                                                 dist=dist, 
                                                 az=az, 
                                                 baz=baz)
                    update_done_file(done_file, station_code)
                    self._downloaded += self.components
                    if self.verbose:
                        print(station_code)
                
                t1 = t2
                self._events_done += 1
                if self.verbose:
                    print(self)            
                
            
        
        


















