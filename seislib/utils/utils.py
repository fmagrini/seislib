#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import os
import pickle
import numpy as np
from numpy.fft import rfft, rfftfreq, irfft
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.integrate import quad
from scipy.stats import pearsonr
from obspy import Trace, Stream
from obspy.geodetics import gps2dist_azimuth
from obspy import UTCDateTime as UTC
from seislib.exceptions import TimeSpanException


def gc_distance(lat1, lon1, lat2, lon2):
    """ 
    Calculates the great circle distance (in m) between coordinate points 
    (in degrees). This function calls directly the obspy `gps2dist_azimuth`,
    it only extends its functionality through the `numpy.vectorize` decorator.
    
    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float or array-like of shape (n,)
        Coordinates of the points on the Earth's surface, in degrees.        
    
    Returns
    -------
    Great-circle distance (in m) 
        If the input is an array (or list) of coordinates, an array of 
        distances is returned
    """
    func = np.vectorize(gps2dist_azimuth)
    return func(lat1, lon1, lat2, lon2)[0]


def azimuth_backazimuth(lat1, lon1, lat2, lon2):
    """ 
    Calculates the azimuth and backazimuth (in degrees) between coordinate 
    points (in degrees). This function calls directly the obspy 
    `gps2dist_azimuth`, it only extends its functionality through the 
    numpy.vectorize decorator
    
    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float or array-like of shape (n,)
        Coordinates of the points on the Earth's surface, in degrees.               
    
    Returns
    -------
    tuple of shape (2,) or ndarray of shape (n, 2)
        The columns consist of azimuth azimuth and backazimuth    
    """
    func = np.vectorize(gps2dist_azimuth)
    return func(lat1, lon1, lat2, lon2)[1:]


def adapt_timespan(st1, st2):
    """
    Slices all traces from the two input streams to the overlapping timerange.
    Then returns a copy of the sliced streams.
        
    Parameters
    ----------
    st1, st2 : obspy.Stream, obspy.Trace

    Returns
    -------
    st1, st2 : obspy.Stream, obspy.Trace
        Obspy stream or trace depending on the input. The original input is 
        not permanently modified (a copy is returned)

    Raises
    ------
    TimeSpanException
        If no overlap is available

    Notes
    -----
    The maximum precision achieved by this function is governed by the
    samling rate. If sub-sample precision is required, consider using 
    :meth:`adapt_timespan_interpolate`
    """
    is_trace = False
    if isinstance(st1, Trace) or isinstance(st2, Trace):
        is_trace = True
    st1 = Stream(st1) if isinstance(st1, Trace) else st1
    st2 = Stream(st2) if isinstance(st2, Trace) else st2
    
    # this has to be done twice, because otherwise there sometimes occurs a 1s timeshift
    for adapt in range(2):
        starttime = max([tr.stats.starttime for tr in st1]
                        + [tr.stats.starttime for tr in st2])   
        endtime = min([tr.stats.endtime for tr in st1]
                      + [tr.stats.endtime for tr in st2])
        if starttime >= endtime:
            raise TimeSpanException(st1, st2)
            
        st1 = st1.slice(starttime, endtime)
        st2 = st2.slice(starttime, endtime)
        for tr in st1:
            tr.stats.starttime = starttime
        for tr in st2:
            tr.stats.starttime = starttime
            
    return (st1, st2) if not is_trace else (st1[0], st2[0])


def adapt_timespan_interpolate(st1, st2, min_overlap=0):
    """
    Slices all traces from the two input streams to the overlapping timerange.
    Then returns a copy of the sliced streams. If the starttime of the sliced 
    traces do not fit exactly (because of sub-sample time shifts), the traces 
    are interpolated to remove the time shift.       
    
    Parameters
    ----------
    st1, st2 : obspy.Stream, obspy.Trace

    Returns
    -------
    st1_out, st2_out : obspy.Stream, obspy.Trace
        Obspy stream or trace depending on the input. The original input
        is not permanently modified (a copy is returned)

    Raises
    ------
    TimeSpanException
        If no overlap is available

    Notes
    -----
    Interpolation can require a relatively long time depending on the
    size of the sliced stream. If speed is preferred to (sub-sample) 
    precision, consider using :meth:`adapt_timespan`
    """
    def slice_streams(st1, st2, starttime, endtime):
        st1_out = st1.slice(starttime, endtime)
        st2_out = st2.slice(starttime, endtime)
        return st1_out, st2_out
    
    is_trace = False
    if isinstance(st1, Trace) or isinstance(st2, Trace):
        is_trace = True
    st1 = Stream(st1) if isinstance(st1, Trace) else st1
    st2 = Stream(st2) if isinstance(st2, Trace) else st2
    starttime = max([tr.stats.starttime for tr in st1]
                    + [tr.stats.starttime for tr in st2])
    endtime = min([tr.stats.endtime for tr in st1]
                  + [tr.stats.endtime for tr in st2])
    
    if starttime > endtime:
        raise TimeSpanException(st1, st2)
    
    sr1 = st1[0].stats.sampling_rate
    sr2 = st2[0].stats.sampling_rate
    st1_out, st2_out = slice_streams(st1, st2, starttime, endtime)
    
    starttimes = set([tr.stats.starttime.timestamp for tr in st1_out]
                     + [tr.stats.starttime.timestamp for tr in st2_out])
    if len(starttimes) > 1:
        starttime = UTC(max(starttimes))
        st1_out.interpolate(sr1, starttime=starttime)
        st2_out.interpolate(sr2, starttime=starttime)
        endtime = min([tr.stats.endtime for tr in st1_out]
                      + [tr.stats.endtime for tr in st2_out])
        st1_out, st2_out = slice_streams(st1_out, st2_out, starttime, endtime)

    return (st1_out, st2_out) if not is_trace else (st1_out[0], st2_out[0])
    

def adapt_sampling_rate(st1, st2):
    """ 
    If the input streams (or traces) have different sampling rates, the one
    characterized by the largest sampling rate is downsampled to the sampling
    rate of the other stream (or trace).
    
    The downsampling is carried out via the :meth:`resample`, which modifies 
    the input streams in place.
    
    Parameters
    ----------
    st1, st2 : obspy.Stream, obspy.Trace
    
    Returns
    -------
    st1, st2 : obspy.Stream, obspy.Trace
        Obspy stream or trace depending on the input. The input is permanently 
        modified
    """
    is_trace = False
    if isinstance(st1, Trace) or isinstance(st2, Trace):
        is_trace = True
    st1 = Stream(st1) if isinstance(st1, Trace) else st1
    st2 = Stream(st2) if isinstance(st2, Trace) else st2
    fs1, fs2 = st1[0].stats.sampling_rate, st2[0].stats.sampling_rate  
    if fs1 < fs2:
        st2 = resample(st2, fs1)
    elif fs2 < fs1:
        st1 = resample(st1, fs2)
    return (st1, st2) if not is_trace else (st1[0], st2[0])


def resample(x, fs):
    """ 
    If the input streams (or traces) have different sampling rates, the one
    characterized by the largest sampling rate is downsampled to the sampling
    rate of the other stream (or trace).
    
    Parameters
    ----------
    st1, st2 : obspy.Stream, obspy.Trace
    

    Returns
    -------
    st1, st2 : obspy.Stream, obspy.Trace
        Obspy stream or trace depending on the input. The input is permanently 
        modified
    """
    nyquist_f = fs/2 - (fs/2)*0.01
    try:
        x.filter('lowpass', freq=nyquist_f, corners=4, zerophase=True)
    except ValueError: 
        pass # when fs > sampling_rate(x), filtering is not needed
    x.interpolate(sampling_rate=fs, method="weighted_average_slopes")
    return x



def bandpass_gaussian(data, dt, period, alpha):
    """ Gaussian filter of real-valued data carried out in the frequency domain
    
    The bandpass filter is carried out with a Gaussian filter centered at 
    `period`, whose width is controlled by `alpha`::

        exp(-alpha * ((f-f0)/f0)**2)

    where f is frequency and f0 = 1 / period. 
    
    Parameters
    ----------
    data : ndarray of shape (n,)
        Real-valued data to be filtered

    dt : float
        Time sampling interval of the data

    period : float
        Central period, around which the (tight) bandapass filter is carried out

    alpha : float
        Parameter that controls the width of the Gaussian filter
        
    Returns
    -------
    numpy.ndarray of shape (n,) 
        Filtered data
    """
    ft = rfft(data)
    freq = rfftfreq(len(data), d=dt)
    f0 = 1.0 / period
    ft *= np.exp( -alpha * ((freq-f0) / f0)**2 )
    return irfft(ft, n=len(data))


def zeropad(tr, starttime, endtime):
    """ 
    Zeropads an `obspy.Trace` so as to cover the time window 
    specified by `starttime`'and `endtime`
    
    Parameters
    ----------
    tr : obspy.Trace

    starttime, endtime : obspy.UTCDateTime
    
    Returns
    -------
    trace : obspy.Trace
        Zeropadded copy of the input trace.
    """
    trace = Trace()
    for key, value in tr.stats.items():
        if key not in ['endtime', 'npts']:
            trace.stats[key] = value
    fs = tr.stats.sampling_rate
    samples_before = int((tr.stats.starttime - starttime) * fs)
    samples_after = int((endtime - tr.stats.endtime) * fs)
    data = tr.data
    if samples_before > 0:
        trace.stats.starttime = tr.stats.starttime - ((samples_before+1) / fs)
        data = np.concatenate((np.zeros(samples_before+1), data))
    if samples_after > 0:
        data = np.concatenate((data, np.zeros(samples_after+1)))
    
    trace.data = data
    return trace
   

def rotate(r, t, dphi):
    """
    Rotation of radial and transverse component of the seismogram by a specified
    angle, following the obspy signs convention.
    
    Parameters
    ----------
    r, t : numpy.ndarray
        Radial (r) and transverse (t) components
    dphi : float
        Angle in degrees
    
    Returns
    -------
    rnew, tnew : numpy.ndarray
        Rotated components
    """
    rnew = -t*np.sin(np.radians((dphi+180)%360)) - r*np.cos(np.radians((dphi+180)%360))
    tnew = -t*np.cos(np.radians((dphi+180)%360)) + r*np.sin(np.radians((dphi+180)%360))
    return rnew, tnew
                

def rotate_stream(st, **kwargs):
    """        
    The method calls the `obspy.Stream.rotate` method, which sometimes runs into
    errors if differences are present among the starttimes and/or endtimes of 
    the traces constituting the stream. These are prevented by slicing the 
    stream to a common time window and (if necessary) interpolating it so as to 
    avoid sub-sample differences.
    
    Parameters
    ----------
    st : obspy.Stream

    **kwargs
        Optional arguments passed to `obspy.Stream.rotate`
    
    Returns
    -------
    st : obspy.Stream
        Rotated copy of the input Stream
    """
    def starttime_and_endtime(st):
        starttime = max([tr.stats.starttime for tr in st])
        endtime = min([tr.stats.endtime for tr in st])
        return starttime, endtime
    
    try:
        starttime, endtime = starttime_and_endtime(st)
        st = st.slice(starttime, endtime)
        st = st.rotate(**kwargs)
    except ValueError:
        starttime, endtime = starttime_and_endtime(st)
        st.interpolate(sampling_rate=st[0].stats.sampling_rate, 
                       starttime=starttime)
        starttime, endtime = starttime_and_endtime(st)
        st = st.slice(starttime, endtime)
        st = st.rotate(**kwargs)
    return st


def running_mean(x, N):
    """ Moving average
    
    Parameters
    ----------
    x : ndarray of shape (m,)
        Data vector

    N : int
        Controls the extent of the smoothing (larger values correspond to larger
        smoothing)
    
    Returns
    -------
    runmean : ndarray of shape (m,)
        Smoothed input
    
    Notes
    -----
    This is a simple implementation of a moving average. More sofisticated 
    functions can be found, e.g., in `scipy.signal.savgol_filter` or 
    `scipy.ndimage.filters.uniform_filter1d`
    """
    if N%2 == 0:
        N+=1
    idx0 = int((N-1)/2)
    runmean = np.zeros(len(x))
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    runmean[idx0:-idx0] = (cumsum[N:] - cumsum[:-N]) / N
    for i in range(idx0):
        runmean[i] = np.mean(x[:2*i+1])
        runmean[-i-1] = np.mean(x[-2*i-1:])
    return runmean


def scatter_to_mesh(lats, lons, c, mesh, method='linear'):
    """ Translates scattered data into a seislib mesh
    
    Parameters
    ----------
    lats, lons : ndarray (n,)
        Coordinates of the scattered data

    c : ndarray (n,)
        Values of the scattered data

    mesh : ndarray (m, 4)
        mesh in SeisLib's format, where the four columns correspond 
        to the boundaries of each pixel, i.e., lat1, lat2, lon1, lon2

    method : {'linear', 'nearest', 'cubic'}
        Interpolation method. Supported: 'linear' (default), 'nearest', 
        and 'cubic'. The three methods call `LinearNDInterpolator`, 
        `NearestNDInterpolator`, and `CloughTocher2DInterpolator` of the 
        `scipy.interpolate` module, respectively.
    
    Returns
    -------
    1-D ndarray containing the `c` values interpolated on `mesh`

    Raises
    ------
    NotImplementedError
        If `method` is not supported
    """
    if method == 'linear':
        interpolator = LinearNDInterpolator(np.column_stack((lons, lats)), c)
    elif method == 'nearest':
        interpolator = NearestNDInterpolator(np.column_stack((lons, lats)), c)
    elif method == 'cubic':
        interpolator = CloughTocher2DInterpolator(np.column_stack((lons, lats)), c)
    else:
        msg = '`%s` interpolation not supported. Supported '%method
        msg += 'methods are `linear`, `nearest`, and `cubic`'
        raise NotImplementedError(msg)
    mesh_central_coords = np.column_stack(((mesh[:,2]+mesh[:,3]) / 2, 
                                           (mesh[:,0]+mesh[:,1]) / 2))
    return interpolator(mesh_central_coords)
    
    
def pearson_corrcoef(v1, v2):
    """ Pearson coerrelation coefficient between two vectors
    
    Parameters
    ----------
    v1, v2 : lists or ndarrays (n,)
        
    
    Returns
    -------
    corrcoeff : float
        Pearson correlation coefficient

    pvalue : float
    

    Raises
    ------
    ValueError
        If v1 and v2 have different shapes

    Notes
    -----
    This function calls `scipy.stats.pearsonr`, bu handles the presence
    of `nan` values.
    """
    if v1.shape != v2.shape:
        raise ValueError('Shapes %s and %s are inconsistent'%(v1.shape, v2.shape))
    notnan = np.intersect1d(np.flatnonzero(~np.isnan(v1)), 
                            np.flatnonzero(~np.isnan(v2)))
    return pearsonr(v1[notnan], v2[notnan])


@np.vectorize
def gaussian(x, mu, sigma):
    """ Gaussian function
    
    Parameters
    ----------
    x : float or ndarray
        Indipendent variable
        
    mu : float
        Mean of the Gaussian
        
    sigma : float
        Standard deviation of the Gaussian
        
    
    Returns
    -------
    float or ndarray
        Gaussian evaluated at x
    """
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((x-mu)/sigma)**2)
   

@np.vectorize
def skewed_normal(x, mu, sigma, skewness):   
    """ Skewed Normal distribution
    
    Parameters
    ----------
    x : float or ndarray
        Indipendent variable
        
    mu : float
        Mean of the resulting Skewed Normal
        
    sigma : float
        Standard deviation of the Skewed Normal
        
    skewness : float
        Parameter regulating the skewness of the function. The function is
        right-skewed if `skewness`>0, and left-skewed if `skewness`<0

    Returns
    -------
    float or ndarray
        Skewed Normal distribution evaluated at x
    """    
    loc = mu - (np.sqrt(2 / np.pi)) * (sigma*skewness / np.sqrt(1+skewness**2))
    integrand = lambda t: np.exp(-t**2 / 2)
    const = 1 / (sigma*np.pi) * np.exp(-(x-loc)**2 / (2*sigma**2))
    suplim = skewness*((x - loc)/sigma)
    return const * quad(integrand, -np.inf, suplim)[0]


def next_power_of_2(x):  
    """ Closest power of two larger or equal to x
    
    Parameters
    ----------
    x : int
    
    
    Returns
    -------
    int
    """
    return 1 if x==0 else 2**(x - 1).bit_length()


def load_pickle(path):
    """ Loads a .pickle file
    
    Parameters
    ----------
    path : str
        Absolute path to the file
    
    Returns
    -------
    Object contained in the .pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f) 


def save_pickle(file, obj):
    """ Saves an object to a .pickle file
    
    Parameters
    ----------
    file : str
        Absolute path to the resulting file
    
    obj : python object 
        Object to be saved (see documentation on the pickle module to know
        more on which Python objects can be stored into .pickle files)
    """
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def remove_file(file):
    """ Removes a file from disk, handling eventual exceptions
    
    Parameters
    ----------
    file : str
        Absolute path to the file to be removed
    """
    try:
        os.remove(file)
    except FileNotFoundError:
        pass



