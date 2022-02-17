Download of Seismograms Recording Teleseismic Earthquakes
---------------------------------------------------------

.. py:class:: EQDownloader(savedir, inventory_name, inv_provider='iris', user=None, password=None, ev_provider='iris', vmin=1.5, vmax=5.5, sampling_rate=1, units='disp', prefilter=(0.001, 0.005, 0.05, 0.4), attach_response=False, stations_config={}, events_config={}, sleep_time_after_exception=30, verbose=True)
   :module: seislib.eq

   Downloads surface-wave data to be used via the two-station method


   **Attributes**

        - savedir : str
            Absolute path of the directory that will be created

        - staclient : obspy.clients.fdsn.Client
            Provider of station metadata and waveforms

        - evclient : obspy.clients.fdsn.Client
            Provider of events metadata

        - channels : list
            List of candidate channels for the downloads

        - components : int
            Either 3 (ZNE) or 1 (Z) 

        - vmin, vmax : float, int
            Minimum and maximum expected velocity of the surface waves

        - sampling_rate : int, float
            Sampling rate of the final seismograms

        - prefilter : tuple
            Filter applied to the waveform before removal of the instrument response

        - units : str
            Physical units of the final seismogram

        - attach_response : bool     
            Whether or not the response information is attached to the waveforms
            during the download

        - verbose : bool
            Whether or not progress information are displayed in the console

        - sleep_time_after_exception : int, float
            The downloads are stopped for the specified time (in s) after a FDSN 
            exception

        - distmin, distmax : float
            Minimum and maximum distance of a station from the epicenter (km)

        - depthmax : float
            Maximum depth of the earthquakes (km)

        - magmin, magmax : float
            Minimum and maximum magnitude

        - starttime, endtime : obspy.core.utcdatetime.UTCDateTime
            Starttime and entime of the events catalogue

        - stations_config, events_config : dict
            Dictionary object with information relevant to the download of the
            stations and events metadata

        - inventory : obspy.core.inventory.inventory.Inventory
            Stations metadata

        - _exceptions : collections.defaultdict(int)
            Dictionary object with information on the occurred exceptions

        - _downloaded : int
            Number of waveforms downloaded

        - _no_stations : int
            Number of stations to download

        - _events_done : int
            Number of events for which the downloads have been finished


   **Methods**
   
        - handle_multiple_locations(st, station_info)
            Location selection in case multiple ones are available

        - collect_waveforms(network, station, channels, starttime, endtime)
            Downloads obspy stream

        - prepare_data(st)
            Demean, detrend, tapering, removing response and resampling

        - adjust_channels(st)
            If the stream contains the Z12 channels, these are rotated towards ZNE

        - compile_header_and_save(st, savedir, stla, stlo, stel, evla, evlo, evdp, 
                                otime, mag, dist, az, baz)
            Compiles the header of the obspy stream (sac format) and writes to disk

        - select_components(st, baz)
            Handles the absence of some components in the final stream

        - preprocessing(st, station)
            Preprocessing of the obspy stream

        - build_inventory(**kwargs)
            Builds an obspy inventory containing stations information    

        - active_channels(station)
            Channels available for the given station among those to download

        - fetch_catalog(t1, t2, **kwargs)
            Fetches a catalog of seismic events

        - start()
            Starts the downloads


   **Class Methods**
   
        - station_coordinates(station_info)
            Fetches station coordinates (lat, lon, elevation)

        - station_was_active(station, time)
            Wheater or not the seismic station was active at the given time

        - inventory_iterator(inventory, reverse=False):
            Generator function to iterate over an obspy inventory

        - event_coordinates_and_time(event)
            Fetch event coordinates (lat, lon, depth) and origin time.

        - get_event_info(event)
            Fetch event information


   .. py:method:: EQDownloader.active_channels(station)
      :module: seislib.eq.MassDownloader_EQ

      Channels available for the given station among those to download

      **Parameters**

        - station : obspy.core.inventory.station.Station

      **Returns**
      
        - List of channels
