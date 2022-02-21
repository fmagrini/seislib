Low-Level Calculation of Earthquake-Based Phase Velocities: ``Two-Station Method``
==================================================================================
Class used under the hood by :doc:`seislib.eq.EQVelocity <seislib.eq_velocity>` to calculate inter-station 
phase velocities based on pairs of receivers aligned on the same great-circle path as the epicenter of 
teleseismic earthquakes.


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

    tsm = TwoStationMethod(refcurve=refcurve,
                           periodmin=15,
                           periodmax=150,
                           no_periods=75,
                           cmin=2.5, 
                           cmax=5, 
                           min_no_wavelengths=1.5)
..

Then, we use this class obtain dispersion measurements for all events
available in the data directory. (We are assuming that we only have folders
corresponding to events that are aligned on the same great circle path as
the epicenter. For a more automatic processing of station pairs and events,
see the method `prepare_data` of :doc:`seislib.eq.EQVelocity <seislib.eq_velocity>`.) 
For each event, we will store the associated dispersion measurements in /path/to/savedir/dispersion, 
using the npy format::

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
..
                
        
Now that all the dispersion measurements have been extracted, we can
calculate one dispersion curve based on them. (dist_km is inter-station 
distance in km). The result will displayed in the console (plotting=True)::
        
    dispcurve = tsm.extract_dispcurve(refcurve=refcurve,
                                      src=dispersion_dir,
                                      dist_km=dist_km,
                                      plotting=True,
                                      sta1=sta1,
                                      sta2=sta2)
..


Parameters
----------


	**refcurve** (``(n, 2) ndarray``): Reference curve used to extract the dispersion curves. The first column should be period, the second column velocity (in either km/s or m/s). The reference curve is automatically converted to km/s, the physical unit employed in the subsequent analysis.

	**periodmin**, **periodmax** (``float``): Minimum and maximum period analysed by the algorithm (default are 15 and 150 s). The resulting dispersion curves will be limited to this period range

	**no_periods** (``int``): Number of periods between periodmin and periodmax (included) used in the subsequent analysis. The resulting periods will be equally spaced (linearly) from each other. Default is 75

	**cmin**, **cmax** (``float``): Estimated velocity range (in km/s) spanned by the dispersion curves (default values are 2.5 and 5). The resulting dispersion curves will be limited to this velocity range

	**ttol** (``float``): Tolerance, with respect to the reference curve, used to taper the seismograms around the expected arrival time of the surface wave (at a given period). In practice, at a given period, 
        everything outside of the time range given by tmin and tmax (see below) is set to zero through a taper. 
        tmin and tmax are defined as::
            tmin = dist / (ref_vel + ref_vel*ttol) 
            tmax = dist / (ref_vel - ref_vel*ttol)
        ..
        where dist is inter-station distance. Default is 0.3, i.e., 30% of the reference velocity

	**min_no_wavelengths** (``float``): Ratio between the estimated wavelength of the surface-wave at a given period (lambda = period * c_ref) and interstation distance. If lambda/dist > min_no_wavelength, the period in question is not used to retrieve a dispersion measurement. Values < 1 are suggested against. Default is 1.5

	**approach** (``str``): Passed to TwoStationMethod.measure_dispersion. It indicates if the dispersion measurements are extracted in the frequency domain ('freq') or in the time domain ('time'). Default is 'freq'

	**gamma_f** (``float`` | ``int``): Controls the width of the bandpass filters, at a given period, used to isolate the fundamental 
        mode in the seismogram. For technical details, refer to :cite:t:`soomro16`.

	**gamma_w**, **distances** (``(m,) ndarray``, optional): Control the width of tapers used to taper, at a given period, the cross correlations in the frequency domain (these two parameters are ignored if `approach` is 'time'). `distances` should be in km. If not given, they will be automatically set to gamma_w = np.linspace(5, 50, 100) distances = np.linspace(100, 3000, 100) These two arrays are used as interpolators to calculate `gamma` based on the inter-station distance. `gamma` is the parameter that actually controls the width of the tapers, and is defined as gamma = np.interp(dist, distances, gamma_w)



Attributes
----------

Methods
-------

Class Methods
-------------



