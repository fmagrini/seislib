r"""
===========================================
Teleseismic Earthquakes (:mod:`seislib.eq`)
===========================================

To retrieve earthquake-based phase-velocity measurements, SeisLib implements 
a `two-station method` [1]_. The two-station method builds on the assumption 
that a fundamental-mode surface wave that has traveled large distances (~20°) 
from the epicenter of a strong earthquake (magnitudes :math:`\gtrsim 5.5`) can 
be decomposed into a sum of monochromatics, or plane waves. This assumption is 
particularly convenient if we consider two receivers approximately lying on the same 
great-circle path as the epicenter. As the wavefront propagates from the first 
receiver to the second, its amplitude gets modified (by inter-station attenuation 
and/or site effects) and its phase gets shifted according to inter-station distance 
:math:`\Delta` and average velocity of propagation `c`. If attenuation is small, 
the phase shift can be expressed, in the frequency domain, as 
:math:`\phi_2(\omega) - \phi_1(\omega) = \frac{\omega \Delta}{c(\omega)}` [2]_, 
where the subscripts denote the receiver and :math:`\phi` the phase.

It is understood that the observed phase delay is invariant under :math:`2\pi` 
translations, giving rise to an ambiguity in phase velocity

.. math::
	c(\omega) = \frac{\omega \Delta}{\phi_2(\omega) - \phi_1(\omega) + 2n\pi},

where `n` is integer. In other words, the delay in the arrival time of the 
wavefront at the second receiver can be explained by different average inter-station 
phase velocities. Such ambiguity needs to be solved algorithmically, and SeisLib
does it through the method :meth:`extract_dispcurve` of 
:class:`seislib.eq.eq_velocity.TwoStationMethod`.

References
----------
.. [1] Meier et al. (2004), One-dimensional models of shear wave velocity for the eastern 
    Mediterranean obtained from the inversion of Rayleigh wave phase velocities and 
    tectonic implications, GJI
    
.. [2] Magrini et al. 2020, Arrival-angle effects on two-receiver measurements 
    of phase velocity, GJI

"""
from .eq_velocity import *
from .eq_downloader import *


