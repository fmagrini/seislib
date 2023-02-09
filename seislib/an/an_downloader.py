#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download of Continuous Seismograms
================================== 

The below class allows for automated downloads of continuous seismograms
to be used in seismic-ambient-noise tomography. The data will be saved on 
disk in the format required by :class:`seislib.an.AmbientNoiseVelocity` 
and :class:`seislib.an.AmbientNoiseAttenuation` to calculate 
surface-wave dispersion curves and Rayleigh-wave attenuation, respectively.

"""
import os
import warnings
import time
import shutil
import socket
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from obspy import read
from obspy import read_inventory
from obspy import Stream
from obspy import UTCDateTime as UTC
from obspy.core import AttribDict
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException
from obspy.io.mseed import InternalMSEEDError
from http.client import IncompleteRead
from seislib.utils import rotate_stream
from seislib.plotting import plot_stations
warnings.simplefilter(action='ignore')
        


class ANDownloader:
    """ Downloads seismic data to be used in ambient noise interferometry
    
    Parameters
    ----------
    savedir : str
        Absolute path to the directory where the data will be saved. Inside 
        this, a sub-directory named 'data' will be created, inside which
        the continuous data will be stored in the format net.sta.loc.cha.sac.
        (net=network code, sta=station, loc=location, cha=channel)

    inventory_name : str
        Name of the inventory (`obspy.core.inventory.inventory.Inventory`) 
        associated with the downloads. This will be saved in xml format
        in the `savedir`

    provider : str
        Provider of seismic data, passed to `obspy.clients.fdsn.Client`. 
        Default is `iris`

    user, password : str, optional
        User name and password for accessing restricted data. Passed to 
        `obspy.clients.fdsn.Client`

    duration : float, optional
        Duration (in seconds) of the individual seismic waveforms to be 
        downloaded. They will be merged together once all individual 
        waveforms from a given station are downloaded. Default is 43200s, 
        i.e., 12h.

    sampling_rate : float
        Final sampling rate of the waveforms, in Hz. Default is 1 Hz

    prefilter : tuple of floats
        Bandapass filter applied to the waveforms before removing the
        intrument response. Default is (0.005, 0.01, 0.25, 0.5).

    units : {'DISP', 'VEL', 'ACC'}
        Physical units of the final waveforms. Can be either 'DISP', 'VEL',
        or 'ACC' (case insensitive). Default is 'DISP'.

    attach_response : bool
        If `True`, the details of the instrument response are attached to the 
        header of the waveforms during their downloads. It can slow down the 
        downloads, but it will make errors due to response information less 
        likely. Default is `False`

    stations_config : dict
        python dictionary containing optional keys and values passed to
        `obspy.clients.fdsn.Client.get_stations`.

    sleep_time_after_exception : float
        Time to wait (in seconds) after an `obspy.clients.fdsn.header.FDSNException`.
        The exception could be due to temporal issues of the client, so
        waiting a little bit before the next download could be useful to
        get things back to normality.

    verbose : bool
        If `True`, information on the downloads will be printed in console

    reversed_iterations : bool
        If `True`, the station inventory will be iterated over in a reversed
        order.

    max_final_stream_duration : float
        Maximum lenght (in seconds) of the continuous waveforms. Default is 
        126144000s, i.e. 4 years. By default, when more than 4 years of data 
        are available, the first 4 years are stored in the 'data' directory 
        with the conventional name net.sta.loc.cha.sac; the following batches 
        are saved as net.sta.loc.cha_1.sac (for the 4th-8th years of data), 
        net.sta.loc.cha_2.sac (for the 8th-12th years),
        net.sta.loc.cha_3.sac (for the 12th-16th years), etc.           


    Attributes
    ----------
    savedir : str
        Absolute path of the directory that will be created
        
    staclient : obspy.clients.fdsn.Client
        Provider of station metadata and waveforms
        
    channels : list
        List of candidate channels for the downloads
        
    components : int
        Either 3 (ZNE) or 1 (Z) 
        
    duration : int, float
        Duration of the waveforms downloaded (in s)
        
    max_final_stream_duration : float
        Maximum duration of the final continuous seismograms
        
    sampling_rate : float
        Sampling rate of the final seismograms
        
    prefilter : tuple of floats
        Filter applied to the waveform before removal of the instrument response
        
    units : str
        Physical units of the final seismogram
        
    attach_response : bool     
        Whether or not the response information is attached to the waveforms
        during the download
        
    verbose : bool
        Whether or not progress information are displayed in the console
        
    reversed_iterations : bool
        Whether or not the inventory of stations is reversely iterated over
        
    sleep_time_after_exception : float
        The downloads are stopped for the specified time (in s) after a FDSN 
        exception
        
    stations_config : dict
        Dictionary object with information relevant to the download of the
        station metadata
        
    inventory : obspy.core.inventory.inventory.Inventory
        Stations metadata
        
    _exceptions : collections.defaultdict(int)
        Dictionary object with information on the occurred exceptions
        
    _downloaded : int
        Number of waveforms downloaded
        
    _no_stations : int
        Number of stations to download
        
    _stations_done : int
        Number of stations for which the downloads have been finished
        
        
    Examples
    --------
    The following will download all the continuous waveforms available from
    the 1.1.2020 to the 1.1.2021 for the network II and for the channel
    BH (3 components: BHZ, BHE, BHN), within the region specified by
    `minlatitude`, `maxlatitude`, `minlongitude`, maxlongitude. Since we do 
    not have access to restricted data, in this case, `includerestricted` is 
    set to `False`. See obspy documentation on `Client.get_stations` for more
    info on the possible arguments passed to `stations_config`::
        
        from obspy import UTCDateTime as UTC
        
        stations_config = dict(network='II',
                               channel='BH*',
                               starttime=UTC(2020, 1, 1),
                               endtime=UTC(2021, 1, 1),
                               includerestricted=False,
                               maxlatitude=12,
                               minlatitude=-18,
                               minlongitude=90,
                               maxlongitude=140)
    
    .. note:: 
        If `channel` is not specified in `stations_config`, the default
        will be 'HH*' (i.e., HHZ, HHE, HHN). Multiple channels can be passed
        using a comma as, e.g., 'HH*,BH*', or 'HHZ,BHZ'. In the downloads,
        priority is given to the first specified channel (in the above,
        the BH* is downloaded only if HH* is not available).
        
    
    We initialize the ANDownloader instance, and then start it::
        
        downloader = ANDownloader(savedir=/path/to/directory, 
                                  inventory_name='II.xml',
                                  provider='iris',
                                  sampling_rate=1,
                                  prefilter=(0.005, 0.01, 0.5, 1),
                                  units='disp',
                                  attach_response=False,
                                  stations_config=stations_config,
                                  verbose=True)
        downloader.start()

    .. note::
        You can kill the above process without any fear: the next time you 
        will run the code, it will pick up where it left off.

    .. hint::
        If instead you want to modify your `station_configs`, make sure to
        either delete the .xml file created at the previous run of the 
        algorithm or use a different `inventory_name`!

    """
    
    def __init__(self, savedir, inventory_name, provider='iris', user=None,
                 password=None, duration=43200, sampling_rate=1, units='disp',
                 prefilter=(0.005, 0.01, 0.25, 0.5), attach_response=False, 
                 stations_config={}, sleep_time_after_exception=10, verbose=True, 
                 reversed_iterations=False, max_final_stream_duration=126144000):
        self.savedir = savedir
        os.makedirs(savedir, exist_ok=True)
        self.staclient = Client(provider, user=user, password=password)
        if user is not None and password is not None:
            includerestricted = stations_config.pop('includerestricted', None)
            if includerestricted is None:
                includerestricted = True
            stations_config['includerestricted'] = includerestricted
        
        self.channels = stations_config.get('channel', 'HH*').split(',')
        self.components = 3 if self.channels[0][2]=='*' else 1
        self.duration = duration
        self.max_final_stream_duration = max_final_stream_duration
        self.sampling_rate = sampling_rate
        #self.npts = self.sampling_rate * self.duration
        self.prefilter = prefilter
        self.units = units
        self.attach_response = attach_response
            
        self.verbose = verbose
        self.reversed_iterations = reversed_iterations
        self.sleep_time_after_exception = 30
        inventory_path = os.path.join(savedir, inventory_name)
        self.stations_config = stations_config
        if not os.path.exists(inventory_path):
            self.inventory = self.build_inventory(**self.stations_config)
            self.inventory.write(inventory_path, format="STATIONXML")
        else:
            self.inventory = read_inventory(inventory_path)

        self._exceptions = defaultdict(int)
        self._downloaded = 0
        self._no_stations = sum([len(net) for net in self.inventory])
        self._stations_done = 0
    
        
    def __str__(self):
        string = '\nWAVEFORMS DOWNLOADED: %d\n'%self._downloaded
        if len(self._exceptions) > 0:
            string += 'EXCEPTIONS:\n'
            for key, value in sorted(self._exceptions.items()):
                string += ''.join([' '*5, '%s: %d\n'%(key.upper(), value)])
        string += ''.join(['-'*40, '\n', ' '*5, '* '*3])
        string += 'STATIONS DONE: %d/%d'%(self._stations_done, self._no_stations)
        string += ''.join([' *'*3, '\n'])
        string += ''.join(['-'*40, '\n'])
        return string
            
    
    def update_stats(self, stations=False, downloaded=False):
        """ 
        Updates information on number of downloaded waveforms and done stations

        Parameters
        ----------
        stations, downloaded : bool
            If `True`, the attributes `_stations_done` and `_downloaded` are
            updated.
        """
        if stations:
            self._stations_done += 1
        if downloaded:
            self._downloaded += self.components
                   
        
    @classmethod
    def station_coordinates(cls, station_info):
        """ Fetch station coordinates (lat, lon, elevation).
        
        Parameters
        ----------
        station_info : obspy.core.inventory.station.Station
        
        Returns
        -------
        latitude, longitude, elevation : float
        """
        stla = station_info.latitude
        stlo = station_info.longitude
        stel = station_info.elevation
        return (stla, stlo, stel)
    
    
    def handle_multiple_locations(self, st, station_info):
        """ Automatic selection of location.
        
        Parameters
        ----------
        st : obspy.core.stream.Stream

        station_info : obspy.core.inventory.station.Station
        
        Returns
        -------
        Obspy stream
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


    def collect_waveforms(self, network, station, channels, starttime, endtime):
        """ Downloads obspy stream
        
        Parameters
        ----------
        network, station, location : str 
            network, station and location codes

        channels : iterable
            iterable containing the channel codes suited to the download. Each
            channel will be tried to be used in the downloads. The first successfull
            attempt determines the returned waveforms. (Higher priority is given to
            decreasing indexes)

        starttime, endtime : obspy.core.utcdatetime.UTCDateTime
            start and end time of the stream.
        
        Returns
        -------
        obspy.core.stream.Stream if the download is successful, else None
        """
        for cha in channels:
            try:
                st = self.staclient.get_waveforms(network=network, 
                                                  station=station, 
                                                  location='*', 
                                                  channel=cha, 
                                                  starttime=starttime, 
                                                  endtime=endtime,
                                                  attach_response=self.attach_response)
                break
            except FDSNNoDataException:
                self._exceptions['FDSNNoDataException'] += 1
            except InternalMSEEDError:
                self._exceptions['InternalMSEEDError'] += 1
            except FDSNException:
                time.sleep(self.sleep_time_after_exception)
            except IncompleteRead:
                self._exceptions['IncompleteRead'] += 1
        else:
#            if self.verbose:
#                print('NO DATA AVAILABLE')
            return
        if not st:
            return
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
        st.filter('lowpass', freq=self.sampling_rate/2-0.05, corners=4, zerophase=True)

        sampling_rates = set([tr.stats.sampling_rate for tr in st])
        if len(sampling_rates) > 1:
            self._exceptions['sampling rate'] += 1
            return
        if sampling_rates.pop() != self.sampling_rate:
            try:
                st.interpolate(sampling_rate=self.sampling_rate, 
                               method="weighted_average_slopes")
            except:
                return
        
        inventory = None if self.attach_response else self.inventory
        try:
            st.remove_response(output=self.units, 
                               inventory=inventory,
                               pre_filt=self.prefilter)
        except ValueError:
            self._exceptions['response'] += 1
            return
            
        st.merge(method=1, interpolation_samples=2, fill_value=0)  
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
        channels = ''.join(sorted(channels, key=sorting_method))
        try:
            st = rotate_stream(st, method='->ZNE', inventory=self.inventory, 
                               components=[channels])
        except Exception:
            self._exceptions['rotation'] += 1
            return
        channels = [tr.stats.channel[-1] for tr in st]
        if not ('Z' in channels and 'N' in channels and 'E' in channels):
            return
        return st
    
    
    def compile_header(self, stream, stla, stlo, stel):
        """ Returns the obspy stream with the header compiled in sac format.
        
        Parameters
        ----------
        stream : obspy.core.stream.Stream

        stla, stlo, stel : float
            latitude, longitude, elevation of the seismic station
        
        Returns
        -------
        obspy.core.stream.Stream
        """
        for tr in stream:
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
        
        return stream
           

    def save_tmp_files(self, stream, outdir, starttime):
        """ 
        Saves temporary files (of duration equal to `self.duration`) that will be
        merged and saved in the 'data' directory
        
        Parameters
        ----------
        stream : obspy.core.stream.Stream

        outdir : str
            Absolute path to saving directory

        starttime : str, float (timestamp format)
            Starttime of the stream, only used to name the temporary file
        """
        for tr in stream:
            outfile = os.path.join(outdir, '%s_%s.sac'%(tr.id, starttime))
            tr.write(outfile, format='sac')


    def merge_days_and_save(self, station_code):
        """
        Merges all temporary files and saves them as a unique file in the 'data' 
        directory. If these files exceed `self.max_final_stream_duration`, they
        are splitted and saved in several batches.
        
        Parameters
        ----------
        station_code : str
            Station code in the format net.sta
        """
        def get_code_to_write(station_code, savedir):
            station_codes = set(['.'.join(i.split('.')[:2]) \
                                 for i in os.listdir(savedir) if station_code in i])
            if station_code not in station_codes:
                return station_code
            station_code += '_1'
            while station_code in station_codes:
                suffix = station_code.split('_')[-1]
                station_code = station_code.replace('_' + suffix, 
                                                    '_' + str(int(suffix)+1))
            return station_code
        
        def adjust_stats(st, stat):
            stats = Counter([tr.stats[stat][:2] for tr in st])
            if len(stats) > 1:
                newstat = stats.most_common(1)[0][0]
                for tr in st:
                    if stat == 'location':
                        tr.stats[stat] = newstat
                    else:
                        oldstat = tr.stats[stat]
                        tr.stats[stat] = oldstat.replace(oldstat[:2], newstat)
            return st

        
        def batch_files(files, max_final_stream_duration):
            while files:
                files = sorted(files, key=lambda i: i.split('.sac')[0].split('_')[-1])
                times_str = sorted(set([i.split('.sac')[0].split('_')[-1] for i in files]),
                                   key=lambda i: float(i))
                times_float = np.array(times_str, dtype=float)
                t1 = np.min(times_float)
                t2 = t1 + max_final_stream_duration
                idx_t2 = np.argmin(np.abs(t2 - times_float))
                tmp = times_str[:idx_t2 + 1]
                to_yield = [file for file in files \
                            if file.split('.sac')[0].split('_')[-1] in tmp]
                yield to_yield
                files = set(files).difference(set(to_yield))
                

        def read_files_and_merge(files, directory, components):
            st = Stream()
            for file in files:
                try:
                    st.extend(read(os.path.join(directory, file)))
                except Exception as e:
                    print('Problem with file %s:'%file)
                    print(e)
                    continue
            st.merge(method=1, interpolation_samples=2, fill_value=0)
            if len(st) > self.components:
                st = adjust_stats(st, 'channel')
                st = adjust_stats(st, 'location')
                st.merge(method=1, interpolation_samples=2, fill_value=0)       
                if len(st) > components:
                    print(st)
                    raise Exception('Something went :-/, check the stream')  
            return st
        

        tmp_directory = os.path.join(self.savedir, 'tmp', station_code) 
        tmp_files = [i for i in os.listdir(tmp_directory) if i.endswith('.sac')]
        if not tmp_files:
            return 
        if self.verbose:
            print('\nLoading all the traces into memory for the final merging',
                  'of %s'%station_code)
        
        savedir = os.path.join(self.savedir, 'data')
        os.makedirs(savedir, exist_ok=True)
        for files in batch_files(files=tmp_files,
                                 max_final_stream_duration=self.max_final_stream_duration):
            st = read_files_and_merge(files=files, 
                                      directory=tmp_directory,
                                      components=self.components)
            if self.verbose:
                print('Saving')
            code_to_write = get_code_to_write(station_code, savedir=savedir) 
            for tr in st:
                net, sta = code_to_write.split('.')
                tr.stats.network = net
                tr.stats.station = sta
                if self.verbose:
                    print(tr.id)
                tr.write(os.path.join(savedir, '%s.sac'%tr.id), format='sac')


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
        List of channels
        """
        channels = set([channel.code[:2] for channel in station.channels])
        return [cha for cha in self.channels if cha[:2] in channels]


    @classmethod
    def operating_times_formatter(station):
        """ Utility function to print an obspy.UTCDateTime in a nice way
        
        Parameters
        ----------
        station : obspy.core.inventory.station.Station
        
        Returns
        -------
        start, end : str
            Starting and ending operating time of the seismic station
        """
        start = station.start_date
        end = station.end_date
        if start is not None:
            start = start.isoformat().split('T')[0]
        if end is not None:
            end = end.isoformat().split('T')[0]
        return start, end
    
    
    def select_components(self, st):
        """ Handles the absence of some components in the final stream
        
        Parameters
        ----------
        st : obspy.Stream
        
        Returns
        -------
        st : obspy.Stream
            If the expected three-component stream lacks of one or both the
            horizontal components, the vertical component is returned. If the
            stream lacks the vertical component but has the two horizontal ones,
            it returns the horizontal components. If only one horizontal
            component is available, it returns None.    
        """
        channels = [tr.stats.channel[-1] for tr in st]
        if 'Z' in channels:
            return st.select(channel='*Z')
        if 'E' in channels and 'N' in channels:
            return st
        return None
    
    
    def preprocessing(self, st, station):
        """ Preprocessing of the obspy stream
        
        The function calls sequentially the methods :meth:`handle_multiple_locations`,
        :meth:`prepare_data`, and :meth:`adjust_channels`.
        
        Parameters
        ----------
        st : obspy.core.stream.Stream

        station : obspy.core.inventory.station.Station
        
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
            return self.select_components(st)
        elif self.components == 3:
            return self.adjust_channels(st)
        
        return st

    
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
        (2,) tuple containing network and station information at each iteration
        """
        def func(iterable, reverse=False):
            if reverse:
                return reversed(iterable)
            return iterable
        
        for network in func(inventory, reverse=reverse):
            for station in func(network, reverse=reverse):
                yield network, station
        
    
    def build_inventory(self, **kwargs):
        """ Builds an obspy inventory containing stations information
        
        The strategy is to first download inventory information at station level.
        Then, for each station, the instrument response is downloaded for each
        channel. This may increase the downloading time, but prevents possible
        "timed out errors".
        
        Parameters
        ----------
        kwargs :
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
                    if station.__dict__.get('creation_date') is None: #FIXES A BUG IN WRITING xml
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
        
        if self.verbose:
            print('\n***COLLECTING STATIONS INFO***\n')
        inv = self.staclient.get_stations(**kwargs)
        return attach_channels(inv, **kwargs)      
    
    
    def update_done_stations(self, station_code):
        """ Updates the 'done_stations.txt' file in the /tmp directory
        
        Parameters
        ----------
        station_code : str
            Station code in the format net.sta
        """
        f = open(os.path.join(self.savedir, 'tmp', 'done_stations.txt'), 'a')
        f.write('%s\n'%station_code)
        f.close()


    def last_time_downloaded(self, tmp_folder):
        """ Retrieves the last time downloaded at the previous run of the algorithm 
        
        Parameters
        ----------
        tmp_folder : str
            Absolute path to the temporary directory where the individual waveforms
            associated with a given station are saved
            
        Returns
        -------
        `obspy.core.utcdatetime.UTCDateTime.timestamp` if files are present in the
            temporary directory, else `None`
        """
        tmp_files = os.listdir(tmp_folder)
        if tmp_files:
            return max([float(i.split('_')[1].split('.sac')[0]) for i in tmp_files])
        else:
            return None


    def starttime_for_download(self, tmp_folder, station):
        """ 
        Identifies the start time from which the downloads will begin for a 
        given station
        
        Parameters
        ----------
        tmp_folder : str
            Absolute path to the temporary folder where the individual waveforms
            associated with a given station are saved

        station : obspy.core.inventory.station.Station
            
        Returns
        -------
        obspy.core.utcdatetime.UTCDateTime
        """
        last_time = self.last_time_downloaded(tmp_folder)
        if last_time is None:
            if 'starttime' in self.stations_config:
                return self.stations_config['starttime'] - self.duration
            return station.start_date
        return UTC(last_time)


    def endtime_for_download(self, station):
        """ 
        Gives the end time of the last time window to be downloaded for a given 
        station
        
        Parameters
        ----------
        station : obspy.core.inventory.station.Station
            
        Returns
        -------
        obspy.core.utcdatetime.UTCDateTime
        """
        if 'endtime' in self.stations_config:
            return self.stations_config['endtime']
        today = UTC.utcnow()
        return station.end_date if station.end_date<today else today
    

    def plot_stations(self, ax=None, show=True, oceans_color='water', 
                      lands_color='land', edgecolor='k', projection='Mercator',
                      resolution='110m', color_by_network=True, legend_dict={}, 
                      **kwargs):
        """ Maps the seismic receivers available for download.
        
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
        """ Starts the downloads
        
        The inventory of stations will be iterated over and, for each
        station code, all the seismic waveforms available for download will be 
        stored (after after detrending, demeaning, removing the instrument 
        response, and resampling) in a temporary directory ($savedir/tmp). Once, for 
        the same station, the downloads are completed, the individual waveforms are 
        merged to a unique continuous seismogram (if that does not exceed the maximum 
        duration of the stream indicated by the user, otherwise it will be splitted 
        into different files. (See param `max_final_stream_duration` in 
        :class:`ANDownloader`)
        
        The algorithm keeps track of the progress made. This allows one to stop
        the downloads and start from where it was left off at any time.
        """
        
        def stations_processed(savedir):
            os.makedirs(os.path.join(savedir, 'tmp'), exist_ok=True)
            donefile = os.path.join(savedir, 'tmp', 'done_stations.txt')
            if not os.path.exists(donefile):
                with open(donefile, 'w'):
                    return []
            return [i.strip() for i in open(donefile)]
        
        inventory = self.inventory_iterator(self.inventory, reverse=self.reversed_iterations)
        for network, station in inventory:
            net, sta = network.code, station.code
            station_code = '%s.%s'%(net, sta)
            done_stations = stations_processed(self.savedir)
            if station_code in done_stations:
                self.update_stats(stations=True)
                continue
            if self.verbose:
                print(self, end='\n\n')
                print('* * * DOWNLOADING %s * * *'%station_code, end='\n\n')
            channels = self.active_channels(station)

            stla, stlo, stel = self.station_coordinates(station)
            tmp_folder = os.path.join(self.savedir, 'tmp', station_code) 
            os.makedirs(tmp_folder, exist_ok=True)

            t1 = self.starttime_for_download(tmp_folder, station)
            endtime = self.endtime_for_download(station)
            windows_no = (endtime - t1) / self.duration
            windows_done = 0
            while t1 <= endtime:
                windows_done += 1
                if self.verbose and not windows_done % int(0.05*windows_no + 1):
                    print()
                    print('%s'%station_code, '---> %d/%d'%(windows_done, windows_no), end=' ')
                    print(datetime.now(), end='\n\n')
                t1 += self.duration
                if not self.station_was_active(station, t1):
                    continue
                st = self.collect_waveforms(network=net, 
                                            station=sta,
                                            channels=channels, 
                                            starttime=t1,
                                            endtime=t1 + self.duration)
                st = self.preprocessing(st, station)
                if st is None:
                    continue

                st = self.compile_header(st, stla, stlo, stel)
                self.save_tmp_files(st, tmp_folder, t1.timestamp)
                self.update_stats(downloaded=True)
               
        
            self.merge_days_and_save(station_code)
            self.update_done_stations(station_code)
            shutil.rmtree(tmp_folder)
            self.update_stats(stations=True)


        
    





