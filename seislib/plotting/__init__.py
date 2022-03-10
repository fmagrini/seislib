"""
===========================================
Plotting (:mod:`seislib.plotting`)
===========================================

SeisLib provides support for plotting the results obtained from
each of its modules, through the functions contained in 
:mod:`seislib.plotting`. :mod:`seislib.plotting` relies on
`CartoPy <https://scitools.org.uk/cartopy/docs/latest/>`_,
and allows one to

- Display geographic location of seismic receivers and epicenters

- Plot inter-station (or epicenter-station) great-circle paths
  for which velocity measurements are available. These can also 
  be colored according to the measured velocity

- Display spatial variations in any physical parameter defined on
  unstructured blocks (e.g., equal-area grids)

- Display Earth's features (e.g., coastlines) in a very simple (high-level) 
  fashion
    
- Easily add inset axes and colorbars

- Combine the above features to obtain high-quality figures.

"""
from .plotting import *
