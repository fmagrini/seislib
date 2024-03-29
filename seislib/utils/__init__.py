"""
===========================================
Utility Functions (:mod:`seislib.utils`)
===========================================

This module provides support for several useful functions for
processing and analysing seismic data. These mainly build on
`ObsPy <https://docs.obspy.org/>`_, and include:

- Vectorized calculation of great-circle distances, azimuths, and back
  azimuths
  
- Slicing / resampling of pairs of seismograms so as to adapt them to 
  a common time window / sampling rate
  
- Pre-processing operations such as zero-padding and bandpass
  filtering
  
- Rotation of horizontal-component recordings to an arbitrary amount
  of degrees. SeisLib's implementation also takes into account not 
  uncommon ValueErrors raised by ObsPy due to the presence of slightly 
  (sub-sample) differences in the time spanned by the two traces 

- Interpolation of scattered data on equal-area grids

- Calculation of the Pearson correlation coefficient between 
  physical parameters defined on equal-area grids
  
"""
from ._utils import *

__all__ = [
    'adapt_timespan',
    'adapt_timespan_interpolate',
    'adapt_sampling_rate',
    'bandpass_gaussian',
    'azimuth_backazimuth',
    'gc_distance',
    'gaussian',
    'load_pickle',
    'next_power_of_2',
    'pearson_corrcoef',
    'remove_file',
    'resample',
    'rotate',
    'rotate_stream',
    'running_mean',
    'save_pickle',
    'scatter_to_mesh',
    'skewed_normal',
    'zeropad',
    ]
