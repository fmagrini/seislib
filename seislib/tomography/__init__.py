r"""
==============================================
Seismic Tomography (:mod:`seislib.tomography`)
==============================================

This module allows for inverting inter-station (or epicenter-station) 
velocity measurements for surface-wave velocity maps. SeisLib implements a 
least-square inversion scheme based on ray theory [1]_, i.e., it assumes that 
the surface waves travel, from a given point on the Earth's surface to 
another, without deviating from the great-circle path connecting them. 

By default, to discretize the Earth's surface, 
:class:`seislib.tomography.tomography.SeismicTomography` 
implements an equal-area grid (see :class:`seislib.tomography.grid.EqualAreaGrid`). 
Alternatively, a regular grid can be employed (see 
:class:`seislib.tomography.grid.EqualAreaGrid`). (The latter is particularly 
suited to tomographic applications at local scale, where the use of 
equal-area parameterizations does not have clear advantages.) The grid of choice 
can be refined, in the areas characterized by a relatively high density of
available rays, an arbitrary of times. The refinement is carried out by splitting 
a given block of the parameterization into four sub-blocks.

.. [1] Boschi & Dziewonski 1999, High-and low-resolution images of the Earth's mantle: 
    Implications of different approaches to tomographic modeling, JGR
"""
from .grid import *
from .tomography import *
