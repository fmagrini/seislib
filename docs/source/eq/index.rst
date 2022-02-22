#######################
Teleseismic Earthquakes
#######################

**seislib** is a Python (and Cython) package that allows for obtaining seismic images of the sub-surface from the local to the global scale. It is the result of a long-term effort of our team to make efficient and open source some of the Python codes behind our seismological publications over the last few years. The library is in rapid expansion and, at present, includes:


**Seismic Ambient Noise Interferometry**
	- Automated download of continuous seismograms
	- Fast cross-correlation of continuous seismograms in the frequency domain
	- Extraction of frequency-dependent phase velocities for both Rayleigh and Love waves, based on pairs of receivers
	- Retrieval of frequency-dependent Rayleigh-wave attenuation coefficient, based on dense seismic arrays

**Surface-Wave Tomography based on Teleseismic Earthquakes**
	- Automated download of seismograms recording strong earthquakes
	- Retrieval of frequency-dependent Rayleigh and Love phase velocities, based on pairs of receivers lying on the same great-circle path as the epicentre (Two-Station Method)

**Least-Square Imaging of Lateral Variations in Surface-Wave Velocity**
	- Equal-area parameterizations, suited for data sets collected at local, regional, and global scale
	- Adaptive parameterizations, with finer resolution in the areas characterized by relatively high density of measurements
	- Linearized inversion of velocity measurements based on ray theory
	- Computational speed optimized (via Cython) for very large data sets
	- Possibility to perform L-curve analyses and resolution tests (e.g., spike, checkerboard)
	
.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 1
   :caption: Contents:


Installation
============

.. toctree::	 
   :titlesonly:
    
   installation
   
Teleseismic Earthquakes
=======================

.. toctree::	 
   :titlesonly:
    
   eq/index
   
 
How to cite us
============== 

.. toctree::	 
   :titlesonly:
    
   How to cite <cite_us>

   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
