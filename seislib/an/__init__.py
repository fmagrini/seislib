r"""
===========================================
Seismic Ambient Noise (:mod:`seislib.an`)
===========================================

As shown in several theoretical studies, the cross correlation of 
seismic ambient noise can be related to the surface-wave Green's 
function between the two points of observation [1]_. In case of a `diffuse` 
ambient wave-field recorded at two receivers on the vertical component, 
the empirical Rayleigh-wave Green's function has a relatively simple 
expression, that can be employed to constrain the velocity structure of the 
Earth's interior. In the frequency domain, this is proportional to a zeroth 
order Bessel function of the first kind (:math:`J_0`), and reads [2]_

.. math::
    \Re{\big\lbrace\rho({\bf x}_A, {\bf x}_B, \omega)\big\rbrace}
    \approx
    J_0\left(\frac{\omega \Delta}{c}\right) 
    \mbox{e}^{-\alpha \Delta},

where :math:`\Delta`, :math:`\omega`, and `c` denote inter-station 
distance, angular frequency, and phase velocity, respectively, :math:`\rho` 
the statistical expectation of the normalized cross-spectrum associated 
with the two ambient-noise recordings, and :math:`\Re{\big\lbrace \dotso \big\rbrace}` 
maps a complex number into its real part. The exponential damping term in the above
equation accounts for the (possibly frequency-dependent) attenuation of the 
Rayleigh waves propagating between the two receivers :math:`{\bf x}_A` and 
:math:`{\bf x}_B`, through the coefficient :math:`\alpha` [2]_.

In ambient-noise seismology, where continuous seismograms of relatively long duration 
(months or years) are employed, the statistical expectation of :math:`\rho` is 
replaced by an ensemble average of the cross-spectra calculated over a relatively 
large number of time windows. This contributes to approximating the condition of 
a diffuse ambient wave-field [1]_, allowing the use of the above equation to measure 
the (average) inter-station phase velocity. 

In practice, SeisLib makes use of the above equation to
calculate

- Rayleigh and Love phase velocities:
    since phase velocity is related to the `phase` of the empirical Green's function, 
    but not to its amplitude, the exponential damping term is neglected, simplifying 
    the problem of retrieving `c` from the data. This approach resulted in numerous 
    successful applications of velocity imaging and monitoring, and can nowadays be 
    considered standard in ambient-noise tomography [3]_. (See 
    :mod:`seislib.an.an_velocity`)
- Rayleigh-wave attenuation:
    where the attenuation coefficient is retrieved by nonlinear inversion based on 
    preliminary measurements of phase velocity [2]_. (See 
    :mod:`seislib.an.an_attenuation`)



References
----------

.. [1] Boschi & Weemstra, (2015), Reviews of Geophysics 
    Stationary-phase integrals in the cross correlation, Reviews of Geophysics

.. [2] Magrini & Boschi, (2021), Surface‐Wave Attenuation From Seismic Ambient Noise: 
    Numerical Validation and Application, JGR 

.. [3] Nakata et al., (2019), Seismic ambient noise, Cambridge University Press
"""


from .an_processing import *
from .an_velocity import *
from .an_attenuation import *
from .an_downloader import *

