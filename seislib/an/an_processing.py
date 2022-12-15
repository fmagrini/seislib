#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lower-Level Support
===================

This module provides useful functions for the processing of continuous 
recordings of seismic ambient noise. All these functions are called, 
under the hood, by :class:`seislib.an.AmbientNoiseVelocity` and
:class:`seislib.an.AmbientNoiseAttenuation`.

"""
import numpy as np
from scipy.signal import detrend, find_peaks, savgol_filter
from scipy.stats import linregress
from scipy.interpolate import interp1d, griddata
from scipy.special import j0, jn_zeros, jv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from obspy import Stream
from obspy.signal.invsim import cosine_taper
from seislib.utils import adapt_timespan, adapt_sampling_rate, running_mean
from seislib.exceptions import TimeSpanException, DispersionCurveException
from seislib.exceptions import NonFiniteDataException

                
def noisecorr(tr1, tr2, window_length=3600, overlap=0.5, whiten=True, psd=False,
              waterlevel=1e-10):
    """ Cross correlation of continuous seismograms.
    
    The two seismograms are first sliced to a common time window and, if they
    are characterized by two different sampling rate, the one with the larger
    sampling rate is downsampled. The two seismograms are then subdivided into
    (possibly overlapping) time windows, which are cross correlated in the
    frequency domain. The cross correlations are then stacked and ensemble
    averaged to obtain the final cross spectrum in the frequency domain.
    
    Parameters
    ----------
    tr1, tr2 : obspy.Trace
        Continuous seismograms to be cross-correlated
        
    window_lenght : float
        Length of the time windows (in seconds) used to perform the cross 
        correlations. Default is 3600
        
    overlap : float
            Should be >=0 and <1 (strictly smaller than 1). Rules the extent
            of overlap between one time window and the following [2]_. Default 
            is 0.5
        
    whiten : bool
        Whether or not the individual cross correlations are normalized by
        whitening [1]_. Default is `True`
        
    psd : bool
        Whether or not the individual cross correlations are normalized by
        the average psd. Default is `False`. If `whiten` is `True`, `psd` 
        is ignored
        
    waterlevel : float
        Only applied if `whiten` is `True`. It prevents `ZeroDivisionError`. 
        Default is 1e-10
        
        
    Returns
    -------
    freq : ndarray of shape (n,)
        Frequency associated with the cross spectrum. It is calculated as
        `np.fft.rfftfreq(win_samples, dt)`, where `dt` is the time delta in 
        the input seismograms, and `win_samples` is `int(window_length / dt)`
        
    corr_spectrum : ndarray of shape (n,)
        Ensemble average of the individual cross correlations
    
    
    Raises
    ------
    NonFiniteDataException
        If one of the two traces contains non-finite elements
    
    TimeSpanException
        If the input traces do not span common times
    
    
    References
    ----------
    .. [1] Bensen et al. 2007, Processing seismic ambient noise data to obtain reliable 
        broad-band surface wave dispersion measurements, GJI
        
    .. [2] Seats et al. 2012, Improved ambient noise correlation functions using 
        Welch's method, GJI
    """
    
    if ( np.isnan(tr1.data).any() or np.isnan(tr2.data).any() or
         np.isinf(tr1.data).any() or np.isinf(tr2.data).any() ):
        raise NonFiniteDataException(tr1, tr2)
        
    st1, st2 = adapt_timespan(Stream(tr1), Stream(tr2))    
    if (st1[0].stats.endtime - st1[0].stats.starttime) < window_length:
        msg = 'Common time range shorter than %.1f s.'%window_length
        raise TimeSpanException(st1[0], st2[0], message=msg)
    
    st1, st2 = adapt_sampling_rate(st1, st2)
    dt = 1 / st1[0].stats.sampling_rate
    data1 = st1[0].data
    data2 = st2[0].data
        
    win_samples = int(window_length / dt)
    no_windows = int( (data1.size-win_samples) / ((1-overlap)*win_samples) ) + 1
    freq = np.fft.rfftfreq(win_samples, dt)
    taper = cosine_taper(win_samples, p=0.05)     

    counter = 0
    result = 0
    for i in range(no_windows):
        istart = int( i * (1-overlap) * win_samples )
        iend = istart + win_samples
        d1 = data1[istart : iend]
        d2 = data2[istart : iend]
        d1 = detrend(d1, type='constant')
        d2 = detrend(d2, type='constant')
        d1 *= taper
        d2 *= taper
        D1 = np.fft.rfft(d1)
        D2 = np.fft.rfft(d2)
        xcorr = np.conj(D1) * D2
        if np.all(xcorr == 0):
            continue
        if whiten:
            norm = (np.abs(D1) + waterlevel) * (np.abs(D2) + waterlevel)
        elif psd:
            psd1 = np.real(D1)**2 + np.imag(D1)**2
            psd2 = np.real(D2)**2 + np.imag(D2)**2
            norm = (psd1 + psd2) / 2
        else:
            norm = 1

        result += xcorr/norm
        counter += 1
    
    return freq, result/counter
 

def velocity_filter(freq, corr_spectrum, interstation_distance, cmin=1.0, 
                    cmax=5.0, p=0.05):
    """ 
    Filters a frequency-domain cross-spectrum so as to remove all signal
    corresponding to a specified velocity range.
    
    In practice, the procedure (i) inverse-Fourier transforms the cross spectrum 
    to the time domain; (ii) it zero-pads the resulting time-domain signal at 
    times corresponding to velocities outside the specified velocity range by 
    applying a cosine taper (the same cosine taper is applied at the two ends of 
    the interval); (iii) a forward-Fourier transform brings back the padded 
    cross correlation to the frequency domain [1]_.
    
    Parameters
    ----------
    freq : ndarray of shape (n,)
        Frequency vector
        
    cross_spectrum : ndarray of shape (n,)
        Complex-valued frequency-domain cross_spectrum
        
    interstation_distance : float (in km)
    
    cmin, cmax : float (in km/s)
        Velocity range. Default values are 1 and 5
        
    p : float
        Decimal percentage of cosine taper. Default is 0.05 (5%)
    
    
    Returns
    -------
    corr : ndarray of shape (n,)
        Filtered cross-spectrum
        
    References
    ----------
    .. [1] Sadeghisorkhani et al. 2018, GSpecDisp: a matlab GUI package for 
        phase-velocity dispersion measurements from ambient-noise correlations,
        Computers & Geosciences 
    """
    
    dt = 1 / (2 * freq[-1])
    idx_tmin = int((interstation_distance/cmax)/dt * (1-p/2)) # 5percent extra for taper
    idx_tmax = int((interstation_distance/cmin)/dt * (1+p/2)) # 5% extra for taper
    vel_filt_window = cosine_taper(idx_tmax-idx_tmin, p=p)
    tcorr = np.fft.irfft(corr_spectrum)
    vel_filt = np.zeros(len(tcorr))
    vel_filt[idx_tmin : idx_tmax] = vel_filt_window
    vel_filt[-idx_tmax+1 : -idx_tmin+1] = vel_filt_window #+1 is just for symmetry reasons
    tcorr *= vel_filt
    corr = np.fft.rfft(tcorr)
    return corr


def get_zero_crossings(freq, xcorr, dist, freqmin=0, freqmax=1, cmin=1, cmax=5, 
                       horizontal_polarization=False, return_indexes=False):
    r"""
    Returns the zero crossings from the smoothed complex cross-correlation
    spectrum.
        
    Parameters
    ----------
    freq : ndarray of shape (n,)
        Frequency vector
        
    xcorr : ndarray of shape (n,)
        Real or complex-valued array containing the cross-correlation. (Only 
        the real part is used.)
        
    dist : float
        Interstation distance in km
        
    freqmin, freqmax : float
        Restricts the zero crossings to the corresponding frequency range. 
        Values in frequency vector outside this range are ignored. Default
        values are 0 and 1
        
    cmin, cmax : float (in km/s)
        Min and max (estimated) surface-wave velocities. Only zero crossings 
        corresponding to velocities within this range are considered. Default
        values are 1 and 5
        
    horizontal_polarization : bool
        If `True`, the zero crossings from the cross-spectrum are compared to 
        the difference function :math:`J_0 - J_2` (Bessel functions of the first 
        kind of order 0 and 2 respectively) [1]_. Should be `True` for Love and 
        radially-polarized Rayleigh waves. If `False` (default) only 
        :math:`J_0` is used
    
    return_indexes : bool
        If `True`, the different branches in the zero-crossings are enumerated
        and returned. Default is `False`
        
    Returns
    -------
    If `return_indexes` is `False` (default):
        ndarray of shape (m, 2)
            2-D array containing frequencies (1st column) and corresponding
            velocities (2nd column, in km/s) at the zero crossings
    Otherwise:
        ndarray of shape (m, 3)
            Where the third column is the index corresponding to each branch
        
    
    References
    ----------
    .. [1] Kästle et al. 2016, Two-receiver measurements of phase velocity: 
        cross- validation of ambient-noise and earthquake-based observations, 
        GJI
    """  
   
    def get_crossings(x, y, return_zeros=False):
        icross = np.flatnonzero(y[:-1]*y[1:] < 0)
        x_cross = x[icross]
        y_icross = y[icross]
        dx = x[icross+1] - x_cross
        dy = y[icross+1] - y_icross
        crossings = -y_icross * dx / dy + x_cross
        if return_zeros:
            return crossings
        pos_crossings = crossings[y_icross < 0]
        neg_crossings = crossings[y_icross > 0]
        return pos_crossings, neg_crossings
        
    
    ifreq = np.flatnonzero((freq>=freqmin) & (freq<=freqmax))
    freq = freq[ifreq]
    xcorr = xcorr.real[ifreq]
    omega = 2 * np.pi * freq[-1]

    pos_crossings, neg_crossings = get_crossings(freq, xcorr)

    # maximum number of zero crossings for a Bessel function from 0 to freqmax,
    # dist and cmin
    no_bessel_zeros = int(omega * dist / cmin / np.pi)
    
    # when dealing with horizontally polarized waves (Love and Rayleigh radial) 
    # the zeros are calculated as J0(x)-J2(x)
    if horizontal_polarization: 
        j02_arg = np.linspace(0, omega * dist / cmin, no_bessel_zeros*5)
        j02 = j0(j02_arg) - jv(2, j02_arg)
        j_zeros = get_crossings(j02_arg, j02, return_zeros=True)
    else:
        j_zeros = jn_zeros(0, no_bessel_zeros)
        
    pos_bessel_zeros = j_zeros[1::2]
    neg_bessel_zeros = j_zeros[0::2]
    
    # All found zero crossings in the smoothed spectrum are compared to all
    # possible zero crossings in the Bessel function. Resulting velocities
    # that are within the cmin - cmax range are stored and returned.
    crossings1 = []
    crossings2 = []
    for j, pcross in enumerate(pos_crossings):
        velocities = pcross * 2 * np.pi * dist / pos_bessel_zeros
        branch_indices = np.arange(len(velocities)) - j
        idx_valid = (velocities>cmin) * (velocities<cmax)
        velocities=velocities[idx_valid]
        branch_indices = branch_indices[idx_valid]
        crossings1.append(np.column_stack((np.ones(len(velocities)) * pcross,
                                           velocities,
                                           branch_indices)))
    for j, ncross in enumerate(neg_crossings):
        velocities = ncross * 2 * np.pi * dist / neg_bessel_zeros
        branch_indices = np.arange(len(velocities)) - j
        idx_valid = (velocities>cmin) * (velocities<cmax)
        velocities = velocities[idx_valid]
        branch_indices = branch_indices[idx_valid]
        crossings2.append(np.column_stack((np.ones(len(velocities)) * ncross,
                                           velocities,
                                           branch_indices))) 
    # check for branch mixups
    crossings1 = np.vstack(crossings1)
    crossings2 = np.vstack(crossings2)
    teststd = []
    for j in range(-1 ,2):
        testcross = np.vstack((crossings1[crossings1[:,2] == 0],
                               crossings2[crossings2[:,2]+j == 0]))
        testcross = testcross[testcross[:,0].argsort(), 1]
        teststd.append(np.std(np.diff(testcross)))
    shift_index = np.argmin(teststd) - 1
    crossings2[:, 2] += shift_index
           
    crossings = np.vstack((crossings1, crossings2))
    idxmin = np.abs(np.min(crossings[:, 2]))
    crossings[:, 2] += idxmin
    crossings = crossings[crossings[:, 0].argsort()]
    
    return crossings if return_indexes else crossings[:, :2]


def extract_dispcurve(frequencies, 
                      corr_spectrum, 
                      interstation_distance, 
                      ref_curve, 
                      freqmin=0, 
                      freqmax=1, 
                      cmin=1, 
                      cmax=5, 
                      filt_width=3, 
                      filt_height=1.0, 
                      x_step=None, 
                      pick_threshold=2, 
                      horizontal_polarization=False, 
                      manual_picking=False,
                      plotting=False, 
                      savefig=None,
                      sta1=None,
                      sta2=None):
    """
    Function for picking the phase-velocity curve from the zero crossings of
    the frequency-domain representation of the stacked cross correlation.
    
    The picking procedure is based on drawing an ellipse around each zero
    crossing and assigning a weight according to the distance from the zero 
    crossing to the ellipse boundary. The weights are then stacked in a
    phase-velocity - frequency plane and a smoothed version of the zero-crossing 
    branches is obtained. Because of the similarity to a kernel-density estimate, 
    the elliptical shapes are called kernels in this code.
    
    This procedure reduces the influence of spurious zero crossings due to data
    noise, makes it easier to identify the well constrained parts of the phase-
    velocity curve and to obtain a smooth dispersion curve. A reference 
    dispersion curve must be given to guide the algorithm in finding the 
    correct phase-velocity branch to start the picking, because parallel 
    branches are subject to a :math:`2 \pi` ambiguity.
    
    Parameters
    ----------
    frequencies : ndarray of shape (n,)
        Frequency vector of the cross-spectrum
        
    corr_spectrum : ndarray of shape (n,)
        Real or complex-valued array containing the cross-correlation. 
        (Only the real part is used.)
        
    interstation_distance : float 
        Inter-station distance in km
    
    ref_curve: ndarray of shape (m, 2) 
        Reference phase velocity curve, where the 1st column is frequency,
        the 2nd is velocity in km/s. The final phase-velocity curve is only 
        picked within the frequency range spanned by the reference curve.
        
    freqmin, freqmax : float
        Restricts the zero crossings to the corresponding frequency range. 
        Values in the frequency vector outside this range are ignored. Default
        values are 0 and 1
        
    cmin, cmax : float (in km/s)
        Min and max (estimated) surface-wave velocities. Only zero crossings 
        corresponding to velocities within this range are considered. Default
        values are 1 and 5
        
    filt_width : int
        Controls the width of the smoothing window. Corresponds to the number 
        of zero crossings that should be within one window. Default is 3
        
    filt_height : float
        Controls the height of the smoothing window. Corresponds to the portion 
        of a cycle jump. Should never exceed 1, otherwise it will smooth over 
        more than one cycle. Default is 1
        
    x_step: float
        Controls the step width for the picking routine along with the x 
        (frequency) axis. Expressed in fractions of the expected step width 
        between two zero crossings. If not provided, it is chosen automatically   
        
    horizontal_polarization : bool
        If `True`, the zero crossings from the cross-spectrum are compared to 
        the difference function :math:`J_0 - J_2` (Bessel functions of the first 
        kind of order 0 and 2 respectively) [1]_. Should be `True` for Love and 
        radially-polarized Rayleigh waves. If `False` (default) only 
        :math:`J_0` is used
 
    manual_picking : bool
        If True, the user is required to pick the dispersion curve manually.
        The picking is carried out through an interactive plot.
    
    plotting : bool        
        If True, a control plot is created and information on the picking 
        procedure are printed. Default is False
        
    savefig : str
        Absolute path. If not `None`, and if plotting is `True`, the control 
        plot is saved on disk at the absolute path provided
        
        
    Returns
    -------
    crossings : ndarray of shape (k, 2)
        2-D array containing frequencies (1st column) and corresponding
        velocities (2nd column, in km/s) at the zero crossings used to pick the
        dispersion curve
        
    Dispersion curve : ndarray of shape (l, 2)
        Picked phase-velocity curve, where the 1st column is frequency and the 
        2nd is velocity in km/s
        
    
    References
    ----------
    .. [1] Kästle et al. 2016, Two-receiver measurements of phase velocity: 
        cross-validation of ambient-noise and earthquake-based observations, 
        GJI
    """    

    def dv_cycle_jump(frequency,velocity,interstation_distance):
        
        return np.abs(velocity-1./(1./(interstation_distance*frequency) + 
                                   1./velocity))
    
    def get_slope(picks,freq, x_step, reference_curve_func, verbose=False):
        """
        function to determine a slope estimate for the picked dispersion curve
        it tries to automatically weight between the slope of the reference
        curve and that of the previous picks. The slope is used to make a
        prediction where the next zero crossing is expected.
        
        freq: frequency (x-axis) where the slope shall be determined
        picks: previous picks
        reference_curve_function: interpolation function for the ref curve
        """
        
        # number of picks to use for prediction
        npicks = np.max([3,int(3/x_step)])
        
        # slope of the reference curve
        slope_ref = ((reference_curve_func(freq+0.02) - 
                      reference_curve_func(freq-0.02))/0.04)        
                 
        # slope and intercept of previous picks
        if len(picks)>npicks: # picks for prediction  
            curveslope, intercept, r_value, p_value, std_err = \
                linregress(np.array(picks)[-npicks:,0],
                            np.array(picks)[-npicks:,1])
        else:
            curveslope = slope_ref
           
        # compare to the slopes between the previous picks
        if len(picks) > 3:
            slope_history = np.diff(np.array(picks),axis=0)
            slope_history = slope_history[:,1]/slope_history[:,0]           
            # trying to predict how the slope is changing from previous steps
            average_slope_change = np.mean(np.diff(slope_history[-10:]))
            slope_pred = np.mean(slope_history[-10:]) + average_slope_change
        else: 
            slope_history = []
            slope_pred = slope_ref
            
        # the final slope is a weighted average between the slope predicted
        # from the previous slopes, the slope from the linear fit to the
        # previous picks and the slope of the reference curve
        slope = 0.25*slope_pred + 0.25*curveslope + 0.5*slope_ref
        
        if len(slope_history) > 3:
            # check whether slope is getting steeper with increasing frequency
            # if yes, make it closer to the reference slope
            if np.abs(slope) > np.mean(np.abs(slope_history[-3:])):
                slope = np.mean(slope_history[-3:])

        # make sure that the slope is negative (lower velocities with
        # increasing frequency
        if slope > 0 or np.isnan(slope):
            slope=0.

        return slope
        
    def get_zero_crossing_slopes(crossings,freq,reference_velocity,slope,width,
                                 reference_curve_func,interstation_distance,
                                 bad_freqs):

        """
        This function attributes a slope to each zero crossing.
        
        This has been simplified so that the slope next to the reference
        curve is always zero. The original version with varying slopes some-
        times led to a bias in the picked phase velocities.
        """
           
        crossfreq,uniqueidx = np.unique(crossings[:,0],return_inverse=True)
        freq_idx = np.where(crossfreq==freq)[0]
        cross_idx = np.where(uniqueidx==freq_idx)[0]
        cross = crossings[cross_idx]
                   
        fstep = reference_velocity/(2*interstation_distance)
        dvcycle = dv_cycle_jump(freq,reference_velocity,interstation_distance)

        # first, find the best-fitting slope for the zero crossing wich is
        # closest to the reference velocity
        closest_idx = np.abs(cross[:,1] - reference_velocity).argmin()
        vel = cross[closest_idx,1]
        if np.abs(vel-reference_velocity)>1.:
              return np.zeros(len(cross_idx)), cross_idx, reference_velocity, 0.      
        
        """
        # slope of the reference curve
        slope_ref = ((reference_curve_func(freq+0.01) - 
                      reference_curve_func(freq-0.01))/0.02)
        if slope is None:
            slope = slope_ref
        # test a couple of different slopes that may vary between the last
        # slope minus 0.1 cycle jumps and the reference curve slope 
        dslope = np.min([0.1*dvcycle,0.01])/(width/2.*fstep)
        test_slopes = np.arange(slope-dslope,0,1)
    
        if len(test_slopes)==0:# or freq in bad_freqs:
            test_slopes = [slope_ref]
        test_slopes = [0.]
        # loop through all the test slopes and check the clostest distances
        # from a line with the given slope to the next crossings
        dists = np.zeros_like(test_slopes)
        # putting more weight on the next/future crossings
        freqax = np.unique(crossfreq[(crossfreq<=freq+width/2.*fstep)*(crossfreq>=freq)])
        for i,test_slope in enumerate(test_slopes):
            
            v_predicted = vel+test_slope*(freqax-freq)
            
            v_distance = 0.
            for f,v in np.column_stack((freqax,v_predicted)):
                if f in bad_freqs:
                    continue
                cross_v = crossings[crossings[:,0]==f,1]
                if (cross_v>v).all() or (cross_v<v).all():
                    continue
                v_distance += np.min(np.abs(v-cross_v))
            
            dists[i] = v_distance
        best_slope = test_slopes[dists.argmin()]
        
        if best_slope>0:
            best_slope=0.
        """
        best_slope = 0.
        
        # for all the other zero crossings, we model the slope from the
        # 'best_slope'.
        cycle_count = (cross[:,1]-vel)/dvcycle

        dvcycle1 = dv_cycle_jump(freq-width/2.*fstep,reference_velocity,
                                  interstation_distance)
        dvcycle2 = dv_cycle_jump(freq+width/2.*fstep,reference_velocity,
                                  interstation_distance)
        
        dv = ((vel+cycle_count*dvcycle2) - 
              (vel+cycle_count*dvcycle1))
        cross_slopes = dv/(width*fstep) + best_slope
        
        # make sure the slopes are not too large (more than one cycle jump
        # over 3 frequency steps)
        maxslope = (dv_cycle_jump(freq,cross[:,1],interstation_distance) / 
                    (3.*fstep))
        mod_idx = np.abs(cross_slopes) > np.abs(maxslope)
        cross_slopes[mod_idx] = maxslope[mod_idx]*np.sign(cross_slopes[mod_idx])
        
        # reduce positive slopes that are probably wrong
        cross_slopes[cross_slopes>0] *= 0.5
        
        # at the main branch, set the previously determined optimal slope
        cross_slopes[closest_idx] = best_slope

        return cross_slopes, cross_idx, vel, best_slope
 
    def get_kernel(X,Y,freq,vel,slope,interstation_distance,
                   fstep,filt_width,filt_height,
                   return_poly_coords=False,return_weights=False):
        """
        X,Y: background grid (not regular)
        freq: central_frequency (x-axis) around which the kernel is drawn
        vel: central velocity (y-axis) around which the kernel is drawn
        slope: slope of the elliptically shaped kernel
        
        Description
        function to determine the elliptical kernels around each zero crossing.
        similar to a kernel density estimate (KDE) method, an ellipse is
        drawn around each zero crossing having a weight of 1 at the location
        of the zero crossing which decreases to 0 at its border.
        the phase velocity pick is taken where the weight of the overlapping
        elliptical kernels is maximum which gives smoother picks as compared
        to picking the (noisy) zero crossings directly.
        Shape and size of the elliptical kernels is user determined
        
        """
        # the width of the elliptical kernel
        width = fstep*filt_width
        # sample the ellipse at the frequencies where X has sample points
        freqax = np.unique(X[(X<=freq+width/2.)*(X>=freq-width/2.)])
        # theoretical width of the kernel relative to the actual width
        factor = width/(freqax[-1]-freqax[0])
        
        if freqax[-1]<=freqax[0]:
            print(freq,fstep,filt_width,freqax,np.unique(X))
            raise Exception("here")
        if len(freqax)<2:
            print(fstep,filt_width,freqax,np.unique(X))
            raise Exception("test")
        # along the frequency axis, the weights are 1 at the zero crossing
        # and decrease to 0 towards the edges of the freqax
        xweights = np.abs(np.cos((freq-freqax)/(width/2.)*np.pi/2.))
        # predicted velocities along the frequency axis
        v_predicted = vel+slope*(freqax-freq)
        # predicted distance between cycles along the frequency axis
        dv_cycle = dv_cycle_jump(freqax,vel,interstation_distance)
        dv_cycle[dv_cycle>1] = 1.
        # the height of the ellipse is maximum around the zero crossing
        # and decreases to zero towards the edges of the freqax
        heights = dv_cycle*filt_height*xweights
        
        boundary_coords = None
        if return_poly_coords:
            # boundary points of the polygon patch
            fax_bound = np.linspace(freq-width/2.,freq+width/2.,20)
            xweights_bound = np.sqrt(np.abs(np.cos((freq-fax_bound)/(width/2.)*np.pi/2.)))
            v_predicted_bound = vel+slope*(fax_bound-freq)
            dv_cycle_bound = dv_cycle_jump(fax_bound,v_predicted_bound,
                                           interstation_distance)
            dv_cycle_bound[dv_cycle_bound>.5] = 0.5
            heights_bound = dv_cycle_bound*filt_height*xweights_bound
            boundary_coords = np.vstack((
                np.column_stack((fax_bound,v_predicted_bound+heights_bound/2.)),
                np.column_stack((fax_bound,v_predicted_bound-heights_bound/2.))[::-1],
                np.array([fax_bound[0],v_predicted_bound[0]+heights_bound[0]/2.])))

                    
        poly_weight_ind = np.empty((0,),dtype=int)
        poly_weights = np.empty((0,))
        if return_weights:
            # loop over all frequencies along the freqax and get the weights
            for f_ind,f in enumerate(freqax):
                x_ind = np.where(X==f)[0]
                y = (Y[x_ind]-v_predicted[f_ind])/(heights[f_ind]/2.)
                y_ind = np.where(np.abs(y)<1.)[0]
                yweights = np.zeros(len(y))
                yweights[y_ind] = np.cos(y*np.pi/2.)[y_ind] * xweights[f_ind]
                poly_weight_ind = np.append(poly_weight_ind,x_ind)
                poly_weights = np.append(poly_weights,yweights)
           
        # compensate for boundary effects
        # this increases the weights at the boundaries and reduces the bias
        poly_weights *= (1+factor)/2.
                
        return boundary_coords,poly_weight_ind,poly_weights
        
    def update_density_field(X,Y,density,crossings,fstep,
                             filt_width,filt_height,
                             interstation_distance,
                             distortion=None,idx_plot=[]):
        
        ellipse_paths = []
        
        # add the weights to the density field for every zero crossing
        for j,(freq,vel,slope) in enumerate(crossings):
                  
            if j in idx_plot:
                return_poly_coords = True
            else:
                return_poly_coords = False
                
            poly_coords, poly_weight_ind, poly_weights = get_kernel(
                    X,Y,freq,vel,slope,interstation_distance,
                    fstep,filt_width,filt_height,
                    return_poly_coords=return_poly_coords,
                    return_weights=True)
            
            density[poly_weight_ind] += poly_weights
                
            if j in idx_plot:
                ellipse_paths.append(poly_coords)

        return density, ellipse_paths  

    def check_previous_picks(picks,picks_backup,frequency,slope,minvel,maxvel,
                             reference_curve_func,freqax_picks,
                             interstation_distance, verbose=False):
        
        # QUALITY CHECKING PREVIOUS PICKS        
        total_picks_jumped = 0
        
        # number of picks to use for prediction
        no_test_picks = np.max([5,int(4/x_step)])
        
        if len(picks_backup)>1 and len(picks)>=1:
            if (np.abs(picks_backup[-1][1] - picks[0][1]) > 
                dv_cycle_jump(picks[0][0], picks[0][1], interstation_distance)):
                picks = []
                return picks,picks_backup,slope,total_picks_jumped
            
        # CHECK 1: IS THERE A CYCLE JUMP BETWEEN THE LAST PICK AND THE ONE THREE CYCLES BEFORE?
        if len(picks) > no_test_picks:
            
            ref_dv = reference_curve_func(picks[-1][0]) - reference_curve_func(picks[-no_test_picks][0])
            
            pickarray = np.array(picks)
            slope_history = np.diff(pickarray,axis=0)
            slope_history = slope_history[:,1]/slope_history[:,0]
            testslope = np.mean(slope_history[-no_test_picks-5:-no_test_picks+1])
                        
            testpicks = pickarray[-no_test_picks:]
            dfreq,dv,_ = testpicks[-1] - testpicks[0]
            dv_cycle = np.mean([dv_cycle_jump(testpicks[-1,0],testpicks[-1,1],interstation_distance),
                                dv_cycle_jump(testpicks[0,0],testpicks[0,1],interstation_distance)])
            
            allowed_vel_reduction = -0.6*dv_cycle + np.min([
                testslope*(picks[-1][0]-picks[-no_test_picks][0]),
                ref_dv])
            
            allowed_vel_increase = 0.3*dv_cycle + slope*(picks[-1][0]-picks[-no_test_picks][0])
            dv_max = np.max([np.max(pickarray[:,1])-np.min(pickarray[:,1]),
                             np.abs(ref_dv),dv_cycle,(maxvel-minvel)/4.])
            
            # jumps to lower velocities
            if dv < allowed_vel_reduction and dv < 0.:
                if verbose:
                    print("    %.3f: cycle jump to lower velocities detected" 
                          %frequency)
                    print("      last pick:",picks[-1])
                    print("      pick before:",picks[-no_test_picks])
                    print("      removing the last picks")
                for i in range(int(no_test_picks/3)):
                    picks.remove(picks[-1])
            
            # jumps to higher velocities are more likely to get punished   
            elif (dv > allowed_vel_increase or dv > 0.2*dv_max) and dv > 0.:
                if verbose:
                    print("    %.3f: cycle jump to higher velocities detected" 
                          %frequency)
                    print("      last pick:",picks[-1])
                    print("      pick before:",picks[-no_test_picks])
                    print("      removing the last picks")  
                for i in range(int(no_test_picks/3)):
                    picks.remove(picks[-1])
          
        # CHECK 2: COUNT HOW MANY PICKS WERE NOT MADE.
        if len(picks) > 0:
            idx0 = np.where(picks[0][0]==freqax_picks)[0][0]
            idxpick = np.where(frequency==freqax_picks)[0][0]
            missing_indices = np.in1d(freqax_picks[idx0:idxpick],
                                      np.array(picks)[:,0],assume_unique=True,
                                      invert=True)
            total_picks_jumped = np.sum(missing_indices)
            # if the total number of jumped picks is too large, abort
            if ((total_picks_jumped > no_test_picks) or 
                (total_picks_jumped > no_test_picks/3. and len(picks) < no_test_picks) or
                (total_picks_jumped >= 1 and len(picks) <= 1) or
                (missing_indices[-int(np.ceil(no_test_picks/1.5)):]).all()):
                if verbose:
                    print("    %.3f: %d picks were not made, data quality too poor." 
                          %(frequency,total_picks_jumped))
                    print("    restarting")
                if len(picks_backup)*2<len(picks):
                    picks_backup = picks[:-1] # save a copy of "bad" picks, without last one
                picks = []
                return picks,picks_backup,slope,total_picks_jumped      

        return picks,picks_backup,slope,total_picks_jumped
        
    
    
    def pick_velocity(picks, frequency, densities, slope, x_step, minvel, maxvel,
                      dvcycle, reference_curve_func, pick_threshold,
                      no_start_picks=3, verbose=False):
        """
        Function to add a new phase-velocity pick to the picks list.       
        
        Parameters
        ----------
        picks : TYPE
            List of picked phase velocities, new pick will be appended to
            that list.
        frequency : TYPE
            Frequency at which the next pick will be made.
        densities : TYPE
            Array containing the result of the kernel densities.
        slope : TYPE
            Slope is used to predict the velocity of the next pick.
        x_step : TYPE
            x_step gives the stepsize between two adjacent picks relative to
            the step between two zero crossings.
        dvcycle : TYPE
            Expected velocity difference between two adjacent cycles.
        reference_curve_func : TYPE
            Function that returns the velocity of the reference curve at a
            given frequency.
        pick_threshold : TYPE
            Picks are only made at density maxima if the maximum is larger
            than pick_threshold times the adjacent minimum.
        no_start_picks : int, optional
            Number of 'first picks' for which more strict picking criteria
            apply. The default is 3.
        verbose : TYPE, optional
            Switch verbose to True or False. The default is False.
        Returns
        -------
        picks : TYPE
            List of picks.
        """
        
        if len(picks)<=no_start_picks:
            # higher pick threshold for the first few picks
            thresh_factor = 1.25
            pick_threshold *= thresh_factor
        else:
            thresh_factor = 1.
        
        velocities, weights = densities
        
        # bit of smoothing 
        # to avoid that there are small density maxima very close to each other
        weights = running_mean(weights,7)
        
        # picks are made where there is a maximum in the density array
        idxmax = find_peaks(weights)[0] #all indices of maxima
        idxmax = idxmax[weights[idxmax]>0.5*np.max(weights)]
        if len(idxmax)==0:
            if verbose:
                print("    %.3f: no maximum" %frequency)
            return picks
        # don't pick anything, if there are only maxima below the reference 
        # curve, unless the velocity difference is less than 20%
        if np.max(velocities[idxmax]) < 0.8*reference_curve_func(frequency):
            if verbose:
                print("    %.3f: too far from reference curve" %frequency)
            return picks
        
        no_start_picks /= x_step
        
        # if there are previous picks, try to predict the next pick
        if len(picks)>=no_start_picks:
            dv_ref = reference_curve_func(frequency)-reference_curve_func(picks[-1][0])
            dv_predicted = slope*(frequency-picks[-1][0])
            v_predicted = picks[-1][1]+dv_predicted
        # otherwise take the reference velocity as prediction
        elif len(picks)>=1:
            dv_ref = reference_curve_func(frequency)-reference_curve_func(picks[-1][0])
            dv_predicted = 0.
            #if len(picks)<no_start_picks:
            #    v_predicted = reference_curve_func(frequency)
            #else:
            v_predicted = picks[-1][1]+dv_ref
        else:
            v_predicted = reference_curve_func(frequency)
                
        idxpick1 = idxmax[np.abs(velocities[idxmax]-v_predicted).argmin()] #index of closest maximum
        vpick = velocities[idxpick1]
        
        # check also the second closest maximum and make sure that the two 
        # maxima are well separated so that the pick is not ambiguous
        if len(idxmax)>1:
            idxpick2 = idxmax[np.abs(velocities[idxmax]-v_predicted).argsort()[1]] # index of 2nd closest maximum
            vpick2 = velocities[idxpick2]
            
            if (np.abs((vpick2-v_predicted)/(vpick-v_predicted+1e-5)) < 1.5 or
                np.abs(vpick2-vpick) < 0.4*dvcycle or
                np.abs(vpick2-vpick) > 2.5*dvcycle):
                if verbose:
                    if np.abs(vpick2-vpick) > 2.5*dvcycle:
                        print("    %.3f: branches are too far apart to get a good pick" %(frequency))
                    else:
                        print("    %.3f: branches are too close to get a good pick" %(frequency))
                        print("        pick1=%.3f pick2=%.3f pick_predicted=%.3f slope=%.3f" 
                              %(vpick,vpick2,v_predicted,slope))
                    return picks
        
        # check the weights at the maximum
        maxamp = weights[idxpick1]
        
        # and the weights at the adjacent minima
        idxmin = find_peaks(weights*-1)[0]
        # if there are no minima
        minamp1 = minamp2 = np.min(weights[(velocities>vpick-dvcycle)*
                                           (velocities<vpick+dvcycle)])
        # if there are minima overwrite minamp1 and minamp2
        if len(idxmin)>0:
            # if there is no minimum above
            if (idxpick1>idxmin).all():
                minamp2 = weights[idxmin[-1]]
            # if there is no minimum below
            elif (idxpick1<idxmin).all():
                minamp1 = weights[idxmin[0]]
            else:
                minamp1 = weights[idxmin[idxmin>idxpick1][0]]
                minamp2 = weights[idxmin[idxmin<idxpick1][-1]]
        minamp = np.mean([minamp1,minamp2])
    
        
        if maxamp > pick_threshold * minamp and maxamp > 0.5*np.max(weights):
            
            if len(picks)>0:
            
                # quality checking current pick
                if (maxamp/np.mean(np.array(picks)[-int(4/x_step):,2]) < 0.5 and 
                    len(picks)>no_start_picks):
                    if verbose:
                        print("    %.3f: amplitude of pick too low" %frequency)
                    return picks
                
                maximum_allowed_velocity_reduction = (
                    -0.3*dvcycle + np.min([dv_ref, dv_predicted]) )
                maximum_allowed_velocity_increase = ( 
                     0.05*dvcycle + np.max([dv_ref, dv_predicted]) )
                                    
                maximum_allowed_velocity_reduction = np.min([
                    -(maxvel-minvel)/100.,maximum_allowed_velocity_reduction])
                maximum_allowed_velocity_increase = np.max([
                    (maxvel-minvel)/100.,maximum_allowed_velocity_increase])
                
                if len(picks) <= no_start_picks and frequency < 1./30:
                    # at long periods, increasing velocities are very unlikely
                    maximum_allowed_velocity_increase *= 0.05
                                
                if vpick-picks[-1][1] < maximum_allowed_velocity_reduction:
                    if verbose:
                        print("    %.3f: velocityjump to lower velocities too large" %frequency)
                        print("      veljump: %.2f; allowed jump: %.2f" %(vpick-picks[-1][1],
                                                    maximum_allowed_velocity_reduction))
                        print("      slope:",slope)
                        print("      last pick:",picks[-1])
                    return picks
                
                if vpick-picks[-1][1] > maximum_allowed_velocity_increase:
                    if verbose:
                        print("    %.3f: velocityjump to higher velocities too large" %frequency)
                        print("      veljump: %.2f; allowed jump: %.2f" %(vpick-picks[-1][1],
                                                    maximum_allowed_velocity_increase))
                    return picks
                
                
            else:
                # if the maximum is more than half a cycle away from the reference
                # we cannot be sure that it is actually the correct branch
                if np.abs(v_predicted-vpick)/dvcycle > 0.45:
                    if verbose:
                        print("    %.3f: too far from reference curve, no first pick taken" %frequency)
                        print("           velocity difference: %.2f maximum allowed difference: %.2f" %(np.abs(v_predicted-vpick),dvcycle*0.45))
                    return picks
            
            picks.append([frequency,vpick,maxamp])
       
        else:
            if verbose:
                print("    %.3f: amplitude of maximum too low, no pick taken" %frequency)
                print("      maxamp=%.2f minamp=%.2f vpick=%.2f" %(maxamp,minamp,vpick))
            
        return picks
      
    def pick_curve_manually(crossings, refcurve, sta1=None, sta2=None, dist=None):
            
        from IPython import get_ipython
        get_ipython().magic('matplotlib auto')
        title = r'$\bf{Pick}$: Left click, $\bf{Delete}$: Right click, $\bf{When\ finished}$: Enter'
        question = r'$\bf{Are\ you\ satisfied?\ Answer\ in\ the\ console\ (y/n)}$'
        while True:
            fig = plt.figure(figsize=(13, 9))
            plt.plot(*zero_crossings[:, :2].T, 
                     color='b',
                     marker='o', 
                     lw=0, 
                     label='Crossings')
            plt.plot(*refcurve.T, label='Reference', color='k')
            plt.suptitle(title, fontsize=23)
            if sta1 is not None and sta2 is not None:
                plt.title('%s - %s Dist: %.2f km'%(sta1, sta2, dist))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Phase Velocity [km/s]')
            plt.legend()
            dispcurve = np.array(plt.ginput(-1, 0))
            if not dispcurve.size:
                raise DispersionCurveException
#            dispcurve[:,1] = savgol_filter(dispcurve[:,1], 5, 2)
            plt.plot(*dispcurve.T, label='Picked', color='r', lw=2)
            plt.legend()
            plt.suptitle(question, fontsize=23)
            print('Are you satisfied (y/n)? Answer below')
            fig.canvas.draw()
            fig.canvas.flush_events()
            answer = input()
            plt.show()
            plt.close()
            if answer.lower() == 'y': return (zero_crossings[:, :2], dispcurve)
            
        
    # Main function
        
    # get the zero crossings of the cross correlation spectrum
    zero_crossings = get_zero_crossings(frequencies, 
                                        corr_spectrum, 
                                        interstation_distance, 
                                        freqmin=freqmin, 
                                        freqmax=freqmax, 
                                        cmin=cmin, 
                                        cmax=cmax, 
                                        horizontal_polarization=horizontal_polarization,
                                        return_indexes=True)
    
    if manual_picking:
        return pick_curve_manually(zero_crossings,
                                   ref_curve,
                                   dist=interstation_distance,
                                   sta1=sta1,
                                   sta2=sta2)
    
    w_axis = np.unique(zero_crossings[:,0])
    w_axis = w_axis[(np.min(ref_curve[:,0])<w_axis)*(w_axis<np.max(ref_curve[:,0]))] 
    maxfreq = np.max(w_axis)
    crossings = np.column_stack((zero_crossings[:,:2],
                                 np.zeros(len(zero_crossings))))

    # interpolation function for the reference curve
    try:
        reference_curve_func = interp1d(ref_curve[:,0],ref_curve[:,1],
                                        bounds_error=False,
                                        fill_value='extrapolate')
    except:
        raise Exception("Error: please make sure that you are using the "+
                        "latest version of SciPy (>1.0).")
    
    # identify crossings that have a bad quality based on
    # (1) amplitude ratio of spectral amplitudes of the bessel function
    # (2) absolute amplitudes of the bessel function
    # (3) spacing between zero crossings along the frequency axis
    bad_quality = np.zeros(len(w_axis),dtype=bool)
    crossamps = np.zeros(len(w_axis))
    peakamps = np.zeros(len(w_axis))
    for i in range(1,len(w_axis)-1):
        idx = np.where((frequencies>w_axis[i-1])*(frequencies<w_axis[i+1]))[0]
        idx = np.where((frequencies>w_axis[i-1])*(frequencies<w_axis[i+1]))[0]
        maxpeak = np.abs(np.max(corr_spectrum.real[idx]))
        minpeak = np.abs(np.min(corr_spectrum.real[idx]))
        crossamps[i] = np.mean([maxpeak,minpeak])
        peakamps[i] = np.max([maxpeak,minpeak])
        if maxpeak>minpeak:
            peak_ratio = maxpeak/minpeak
        else:
            peak_ratio = minpeak/maxpeak
        if peak_ratio>3.:
            bad_quality[i] = True
    crossamps[0] = crossamps[1]
    crossamps[-1] = crossamps[-2]
    peakamps[0] = peakamps[1]
    peakamps[-1] = peakamps[-2]
    expected_fstep = reference_curve_func(w_axis[:-1]+np.diff(w_axis))/(2*interstation_distance)
    peakamps = np.interp(np.append(w_axis[0],w_axis[0]+np.cumsum(expected_fstep)),w_axis,peakamps)
    peakamps = running_mean(peakamps,7)
    peakamps = np.interp(w_axis,np.append(w_axis[0],w_axis[0]+np.cumsum(expected_fstep)),peakamps)
    bad_quality[np.append(False,np.diff(w_axis)>1.5*expected_fstep)] = True
    bad_quality[crossamps < 0.5*peakamps] = True
    for i in range(1,len(bad_quality)-1):
        if bad_quality[i-1] and bad_quality[i+1]:
            bad_quality[i] = True
    
    # frequency at which to start picking:
    freq_pick_start = w_axis[~bad_quality*(w_axis>=freqmin)]
    if len(freq_pick_start)>1:
        freq_pick_start = freq_pick_start[0]
    else:
        freq_pick_start = freqmin
    
    # create a logarithmically spaced frequency axis. Final picks are inter-
    # polated and smoothed on this axis.
    npts = 2
    f_max = np.max(ref_curve[:,0])
    f_min = np.min(ref_curve[:,0])
    while True:
        if 1.05**npts * f_min > f_max:
            break
        npts += 1
    logspaced_frequencies = np.logspace(np.log10(f_min),np.log10(f_max),npts)
    df = np.abs(logspaced_frequencies[1]-logspaced_frequencies[0])
    if df>1:
        n_round = 1
    else:
        n_round = int(np.abs(np.log10(df))) + 2
    logspaced_frequencies = np.around(logspaced_frequencies,n_round)
    
    # cut the logspaced_frequencies to the range where there are crossings
    valid_range = np.where((logspaced_frequencies > np.min(w_axis))*
                           (logspaced_frequencies < np.max(w_axis)))[0]
    idx0 = valid_range[0]
    idx1 = np.min([valid_range[-1]+2,len(logspaced_frequencies)])
    logspaced_frequencies = logspaced_frequencies[idx0:idx1]
    
    if x_step is None:
        x_step = np.around(len(w_axis)/np.sum((ref_curve[:,0]>np.min(w_axis))*
                                               (ref_curve[:,0]<np.max(w_axis))),1)
        x_step = np.min([x_step,0.9])
        x_step = np.max([x_step,0.3])


    if plotting:
        print("  starting picking (x-step = %.1f)" %x_step)
    picks = []
    picks_backup=[]
    
    X=np.array([])
    Y=np.array([])
    density=np.array([])
    
    sampling = 20
    slope = None
    cross_vel = None
    cross_slope = None
    cross_idx = None
    if plotting:
        ellipse_paths = []
      
    gridaxis = []
    freq = np.min(w_axis)
    while freq <= np.max(w_axis):
        gridaxis.append(freq)
        freq += x_step*reference_curve_func(freq)/(2*interstation_distance)
    gridaxis = np.array(gridaxis)
    
    # kernel: approximately elliptical shape drawn around each zero crossing
    # that assigns a weight which is maximum at the center of the ellipse
    # (where the zero crossing is) and decreases to 0 towards the border of
    # the ellipse. The elliptical kernels are overlapping. Picks are made
    # where the weight density is maximum.

    # the procedure works with 3 nested loops
    # INNERMOST LOOP creates new gridpoints at which the kernel density is
    # evaluated. This has to be done first, before the kernels are calculated
    # and before the picks are made. This loop advances the fastest along
    # the frequency axis
    #freq_gridpoints = np.min(w_axis)
    idx_gridpoints = 0
    # INNER LOOP creates the (approximately) elliptical kernels around each
    # zero crossing. The values of all kernels are evaluated at the gridpoints
    # and summed to give the density field. This loop runs behind the inner-
    # most loop
    idx_kernel = 0
    # OUTER LOOP runs behind both the inner and the innermost loops. Once the
    # kernels are evaluated it will pick the phase velocities where the kernel
    # density field is maximum.
    #freqax_picks = [np.min(w_axis)]
    #freq_pick = np.min(w_axis)
    #idx_pick = 0
    idx_pick = 0
    
    idx_plot = []
    
    ##########################################################################
    # OUTER LOOP FOR PICKING THE VELOCITIES
    while idx_pick <= idx_gridpoints and idx_pick < len(gridaxis):
        
        freq_pick = gridaxis[idx_pick]
        
        if freq_pick > maxfreq:
            if plotting:
                print("    %.3f: Distance between the next zero crossings " +
                      "seems to be wrong. Stopping." %freq_pick)
            break
        

        ######################################################################
        # INNER LOOP TO UPDATE THE KERNEL DENSITY FIELD
        # the density field must be advanced with respect to the picking loop
        while idx_kernel < len(w_axis):
                        
            freq_kernel = w_axis[idx_kernel]
            fstep_kernel = reference_curve_func(freq_kernel)/(2*interstation_distance) 
            
            if freq_kernel > freq_pick + 0.6*filt_width*fstep_kernel or freq_kernel>gridaxis[-1]:
                break
            
            # if there is more than one zero-crossing gap, abort.
            if idx_kernel>1 and maxfreq > w_axis[idx_kernel]:
                if ((w_axis[idx_kernel]-w_axis[idx_kernel-1])/fstep_kernel > 1.5 and
                    (w_axis[idx_kernel-1]-w_axis[idx_kernel-2])/fstep_kernel > 1.5):
                    maxfreq = w_axis[idx_kernel]
            
            kernel_upper_boundary = freq_kernel + filt_width*fstep_kernel
            
            ##################################################################
            # INNERMOST LOOP THAT ADDS NEW GRIDPOINTS
            # the distance between the gridpoints is dependent on the distance
            # between zero crossings (along x-axis) and on the distance between
            # adjacent phase-velocity cyles/branches (along y-axis)
            # new gridpoints at which the density field is computed
            # the gridpoints have again to be created before the density
            # field can be computed
            while (idx_gridpoints < len(gridaxis)):
                
                freq_gridpoints = gridaxis[idx_gridpoints]
                if freq_gridpoints > kernel_upper_boundary:
                    break
                                
                # determine the sampling for the weight field in y-direction (vel)
                # we do not care about the branches that are very far away from the
                # last pick, only take those into account that are close
                if len(picks)>3:
                    dv = dv_cycle_jump(freq_gridpoints,picks[-1][1],
                                       interstation_distance)
                    dy = np.min([0.05,dv/sampling])
                    lower_vlim = np.min([picks[-1][1]-8*dv,
                                         reference_curve_func(freq_gridpoints)-dv])
                    upper_vlim = np.max([picks[-1][1]+2*dv,
                                         reference_curve_func(freq_gridpoints)+dv])
                    Ypoints = np.arange(np.max((lower_vlim,cmin)),
                                        np.min((upper_vlim,cmax)),dy)
                else:
                    dv = dv_cycle_jump(freq_gridpoints,
                                       reference_curve_func(freq_gridpoints),
                                       interstation_distance)
                    dy = np.min([0.05,dv/sampling])
                    Ypoints = np.arange(cmin,cmax,dy)
                Xpoints = np.ones(len(Ypoints))*freq_gridpoints
                
                X = np.append(X,Xpoints)
                Y = np.append(Y,Ypoints)
                density = np.append(density,np.zeros(len(Xpoints)))
                        
                idx_gridpoints += 1
                # END OF INNERMOST LOOP
            ##################################################################
                
            # the reference velocity is needed to find the optimal rotation
            # angles for the elliptical kernels
            if len(picks)<5:
                reference_velocity = reference_curve_func(freq_kernel)
                #if cross_vel is not None and idx_kernel>2:
                #    reference_velocity = cross_vel
            else:
                reference_velocity = (picks[-1][1] + np.mean([
                     slope*(freq_kernel-picks[-1][0]),
                     reference_curve_func(freq_kernel)-
                     reference_curve_func(picks[-1][0])]) )
            # find the most likely slope associated to each zero crossing
            # these slopes are used to rotate the elliptical kernels
            cross_slopes,cross_idx,cross_vel,cross_slope = (
                get_zero_crossing_slopes(
                    crossings,freq_kernel,reference_velocity,cross_slope,
                filt_width,reference_curve_func,interstation_distance,
                w_axis[bad_quality]))
            crossings[cross_idx,2] = cross_slopes
      
            # add the kernel weights
            # if the plotting option is True, some of the ellipses will be drawn
            # the idx_plot list controls for which zero crossings this will be done
            if plotting:
                idx_plot = []
                if idx_kernel%1==0:
                    idx_plot = np.abs(crossings[cross_idx,1]-reference_velocity).argmin()
                    idx_plot = [idx_plot]#[idx_plot-2,idx_plot-1,idx_plot+1]
            
            # get the elliptical kernels and update the weight field
            if not bad_quality[idx_kernel]:
                density, ell_paths = update_density_field(
                    X,Y,density,crossings[cross_idx],
                    reference_velocity/(2*interstation_distance),
                    filt_width,filt_height,
                    interstation_distance,idx_plot=idx_plot)
                
                if plotting:
                    ellipse_paths += ell_paths
                
            idx_kernel += 1
    
            # END OF INNER LOOP
        ######################################################################             
        # back to picking loop
        
        # take either the velocity of the last pick or that of the ref curve
        if len(picks)<3:
            reference_velocity = reference_curve_func(freq_pick)
        else:
            reference_velocity = picks[-1][1]
            
        # dv_cycle: velocity jump corresponding to 2pi cycle jump    
        dv_cycle = dv_cycle_jump(freq_pick,reference_velocity,
                                 interstation_distance)

        # estimate the slope of the picked phase velocity curve
        # if there are no picks yet, use the reference curve
        slope = get_slope(picks,freq_pick,x_step,
            reference_curve_func, verbose=plotting)
    
        # check if there are already picks made, if yes, do a quality check
        picks,picks_backup,slope,picks_jumped = check_previous_picks(
            picks,picks_backup,freq_pick,slope,cmin,cmax,
            reference_curve_func,gridaxis,interstation_distance,
            verbose=plotting)
        if len(picks)==0 and picks_jumped>0:
            idx_pick -= picks_jumped
            continue
                           
        # check that the parallel phase velocity branches are well separated
        # for taking the first picks. otherwise do not take a first pick
        if len(picks)<3:
            cross_freq = w_axis[np.abs(w_axis-freq_pick).argmin()]
            if (np.sum(crossings[:,0]==cross_freq) > 20 or 
                (dv_cycle < np.max([reference_curve_func(w_axis[0]) - reference_velocity,
                                    (cmax-cmin)/10.]))):
                if plotting:
                    print("    dvcycle: %.2f  ref_curve[0,1]-ref_vel : %.2f"
                          %(dv_cycle, ref_curve[0,1]-reference_velocity))
                    print("    %.3f: cycles are too close, stopping" %freq_pick)
                break # terminate picking
        
        # weights at the current frequency where the next pick will be taken
        # Y array contains the velocities along the y axis and weights the
        # corresponding weights. Picks will be taken where the weights are maximum
        pick_density = (Y[freq_pick==X],density[freq_pick==X])

        # pick next velocity, skip if the three last zero crossings had a bad quality
        if (freq_pick>=freq_pick_start and 
            not (bad_quality[np.abs(w_axis-freq_pick).argsort()][:2]).all()):
            picks = pick_velocity(picks,freq_pick,pick_density,slope,x_step,
                                  cmin,cmax,dv_cycle,reference_curve_func,
                                  pick_threshold,verbose=plotting)
        elif freq_pick>=freq_pick_start and plotting:
            print("    %.3f: bad crossing quality, skipping" %freq_pick)
         
        if len(picks)>1:
            if np.sum(bad_quality[(w_axis<=freq_pick)*(w_axis>=picks[int(len(picks)/2)][0])]) > 6:
                if plotting:
                    print("    %.3f: terminating picking, to many zero crossings with bad quality." %freq_pick)
                break
        
        idx_pick += 1
        # END OF PICKING LOOP
    ##########################################################################




    # CHECK IF THE BACKUP PICKS ("BAD" ONES DISCARDED BEFORE) HAVE THE BETTER COVERAGE
    if len(picks_backup)*2>len(picks):
        picks = picks_backup
    
    # SMOOTH PICKED PHASE VELOCITY CURVE
    picks=np.array(picks)    
    if (len(picks) > 3 and np.max(picks[:,0])-np.min(picks[:,0]) > 
                          (np.max(w_axis)-np.min(w_axis))/8.):
        # remove picks that are above the highest/fastest zero crossing
        bad_pick_idx = []
        maxv = 0.
        for freq in w_axis:
            cross = crossings[freq==crossings[:,0]]
            maxv = np.max(np.append(cross[:,1],maxv))
            if maxv > np.max(picks[:,1]):
                break
            bad_pick_idx.append(np.where((picks[:,0]<freq)*(picks[:,1]>maxv))[0])
        if len(bad_pick_idx)>0:
            bad_pick_idx = np.unique(np.hstack(bad_pick_idx))
        picks_smoothed = np.delete(picks,bad_pick_idx,axis=0)
        # smooth and interpolate picks
        if len(picks_smoothed)>2:
            picks_smoothed[:,1] = running_mean(
                picks_smoothed[:,1],np.min([len(picks_smoothed),int(3/x_step)]))
            picksfu = interp1d(picks_smoothed[:,0],picks_smoothed[:,1],
                               bounds_error=False,fill_value=np.nan)
            picks_interpolated = picksfu(logspaced_frequencies)
            smoothingpicks = picks_interpolated[~np.isnan(picks_interpolated)]
            smooth_picks = running_mean(smoothingpicks, 7)[1:]
            smooth_picks_x = logspaced_frequencies[~np.isnan(picks_interpolated)][1:]
        else:
            smooth_picks = []
            smooth_picks_x = []
    else:
        if plotting and len(picks)>1:
            print("  picked freq range: %.3f required: %.3f" %(
                np.max(picks[:,0])-np.min(picks[:,0]),
                (np.max(w_axis)-np.min(w_axis))/6.))
        smooth_picks = []
        smooth_picks_x = []
        
    if len(smooth_picks) < len(logspaced_frequencies)/5.:
        if plotting:
            print("  picked range too short")
            print("  picked crossings:",len(picks),"of a total of",len(gridaxis))
        smooth_picks = []
        smooth_picks_x = []         
        
    # PLOT THE RESULTS
    if plotting:  
        legend_dict = dict(loc='upper right', 
                           framealpha=0.9, 
                           handletextpad=0.5, 
                           borderpad=0.3, 
                           handlelength=1, 
                           markerscale=3)
        xlim = freqmin - freqmin*0.1, np.max([freq_pick+0.1, 
                                              w_axis[int(len(w_axis)/2)]])
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.15, height_ratios=(1, 2))
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(frequencies, corr_spectrum.real, 'k')
#        ax0.plot(w_axis, peakamps, 'k')
        ax0.plot(frequencies, np.zeros(len(frequencies)), 'k', linewidth=0.3)
        ax0.plot(w_axis[bad_quality], 
                 np.zeros(np.sum(bad_quality)), 
                 'ro',
                 markersize=2, 
                 label='Low quality crossing')
        ax0.legend(**legend_dict)
        ax = fig.add_subplot(gs[1])
        ax2 = ax.twiny()
        distortion = 0.0001
        dy = np.diff(Y)
        dy = np.min(dy[dy>0])
        dx = np.diff(X)
        dx = np.min(dx[dx>0])
        X_min, X_max = np.min(X), np.max(X)
        Y_min, Y_max = np.min(Y), np.max(Y)
        x = np.linspace(X_min, X_max, int(np.max([100,(X_max-X_min)/dx])))
        y = np.linspace(Y_min, Y_max, int(np.max([100,(Y_max-Y_min)/dy])))
        xplot, yplot = np.meshgrid(x, y)
        density_interpolated = griddata((X, Y*distortion),
                                        density,
                                        (xplot, yplot*distortion))
        for xi in range(len(x)):
            xtest = X[np.abs(x[xi] - X).argmin()]
            testvels = Y[X == xtest]
            density_interpolated[yplot[:, xi] > np.max(testvels), xi] = np.nan
            density_interpolated[yplot[:, xi] < np.min(testvels), xi] = np.nan
        try:
            vmax = np.nanmax(density_interpolated[xplot > picks[0, 0]])
        except:
            vmax=np.nanmax(density_interpolated)
            
        ax.pcolormesh(xplot, yplot, density_interpolated, vmin=0, vmax=vmax,
                      shading='nearest')
        for branchidx in np.unique(zero_crossings[:,2]):
            ax.plot(zero_crossings[zero_crossings[:,2] == branchidx,0],
                    zero_crossings[zero_crossings[:,2] == branchidx,1],
                    marker='o',
                    color='b',
                    ms=2,
                    linewidth=0.1)
        ax.plot(ref_curve[:,0], 
                ref_curve[:,1],
                linewidth=2,
                color='lightblue',
                label='reference curve')
        #plt.plot(tipx,tipy,'o')
        if True:
            for vertices in ellipse_paths[::1]:
                ax.plot(vertices[:,0],
                        vertices[:,1],
                        color='white',
                        linewidth=0.5)
        ax.axvline(freq_pick, linestyle='dashed', color='black')
        ax.axvline(freqmin, linestyle='dashed', color='white')
        #for testpnt in test_points:
        #    ax.plot(testpnt[:,0],testpnt[:,1],'r--',linewidth=0.9)
        for pick in picks:
            ax.plot(pick[0],pick[1],'o',color='red',ms=2)
        ax.plot([],[],'o', color='red', ms=2, label='picks')
        ax.plot(smooth_picks_x, smooth_picks,'r--',label='smoothed picks')
        ax.legend(**legend_dict)
        ax0.set_xlim(*xlim)
        ax.set_xlim(*xlim)
        ax2.set_xlim(ax.get_xlim())
        new_tick_locations = ax.get_xticks()[ax.get_xticks()<np.max(ax.get_xlim())]
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("axes", -0.1))
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.set_xticks(new_tick_locations[new_tick_locations>0.])
        ax2.set_xticklabels(np.around(1./new_tick_locations[new_tick_locations>0.],2))
        ax2.set_xlabel("Period [s]")
        ax.set_ylim(cmin, cmax)
        #ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase Velocity [km/s]")
        ax0.text(x=0.68, 
                 y=0.05, 
                 s="Distance: %.1f km" %interstation_distance,
#                 fontsize=12,
                 bbox=dict(boxstyle='square,pad=0.2',
                           fc='white',
                           ec='None', 
                           alpha=0.85),
                 va='bottom',
                 ha='left',
                 transform=ax0.transAxes)
        #ax.set_aspect('equal')
        #plt.colorbar(img, shrink=0.5)
        if savefig is not None:
            plt.savefig(savefig, bbox_inches='tight', dpi=200)
        plt.show()
        
        
    if len(smooth_picks) > 0: 
        return (zero_crossings[:, :2], 
                np.column_stack((smooth_picks_x, smooth_picks)))
    
    raise DispersionCurveException


