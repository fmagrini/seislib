# **SeisLib: Multi-Scale Seismic Imaging with Python**

***SeisLib*** is a Python (and Cython) package that allows for obtaining seismic images of the sub-surface from the local to the global scale. It is the result of a long-term effort of our team to make efficient and open source some of the Python codes behind our seismological publications over the last few years. The library is in rapid expansion and, at present, includes:


## **Seismic Ambient Noise Interferometry**
*  Automated **download** of continuous seismograms
* **Fast cross-correlation** of continuous seismograms in the **frequency domain**
* Extraction of frequency-dependent **phase velocities** for both **Rayleigh and Love waves**, based on pairs of receivers
* Retrieval of frequency-dependent **Rayleigh-wave attenuation coefficient**, based on dense seismic arrays

## **Surface-Wave Tomography based on Teleseismic Earthquakes**
* Automated **download** of seismograms recording strong earthquakes
* Retrieval of frequency-dependent **Rayleigh and Love phase velocities**, based on pairs of receivers lying on the same great-circle path as the epicentre (**Two-Station Method**)

## **Least-Squares Imaging of Lateral Variations in Surface-Wave Velocity**
* **Equal-area and regular parameterizations**, suited for data sets collected at local, regional, and global scale
* **Adaptive parameterizations**, with finer resolution in the areas characterized by relatively high density of measurements
* **Linearized inversion** of velocity measurements **based on ray theory**
* **Computational speed optimized** (via Cython) for very **large data sets**
* Possibility to perform **L-curve analyses and resolution tests (e.g., spike, checkerboard)**

<p>&nbsp;</p>

# ***Documentation***

For more information on SeisLib, make sure to visit our [wiki page](https://seislib.readthedocs.io/en/latest/)!

<p>&nbsp;</p>

# ***Installation***

First, make sure you have all the **dependences** installed, i.e., ***obspy***, ***cartopy***, and ***cython***. We recommend installing such dependences using conda (see below). You will also need ***gcc*** or equivalent, to compile the cython parts of the library.

```bash
conda create -n seislib python=3.9 numpy=1.20
conda activate seislib
conda install -c conda-forge obspy
conda install -c conda-forge cartopy=0.18
conda install -c anaconda cython
```


Note that we installed Python 3.9 (rather than Python 3.10) since numpy's version 1.22 currently leads to compatibility issues. Once the above dependences have been installed, you can proceed with the installation of ***seislib***. 

```
pip install seislib
```

If you run into troubles with the above, you can try the following approach:
```
git clone https://github.com/fmagrini/seislib.git
cd seislib/seislib/tomography/_ray_theory
python setup_all.py build_ext --inplace
```
The last command will compile the Cython files. If you work on an anaconda environment, you might need to replace "python" with, e.g., "/home/your_name/anaconda3/bin/python". (You can retrieve the path to your python executable by typing "import sys; print(sys.executable)" in your Python GUI. Make sure to then add ~/seislib to your path to being able to import its modules in your Python codes.


<p>&nbsp;</p>

# ***References***
Specific to the Python package:
- Magrini, F., Lauro, S., Kästle, E. & Boschi, L., 2022. Surface-wave tomography using SeisLib: a Python package for multi-scale seismic imaging. *Geophys. J. Int.*, ggac236, https://doi.org/10.1093/gji/ggac236

Additional references depending on the use you made of SeisLib:
- Boschi, L. & Dziewonski, A.M., 1999. High- and low-resolution images of the Earth's mantle: Implications of different approaches to tomographic modeling. *J. Geophys. Res.*, 104(B11)
- Boschi, L., Magrini, F., Cammarano, F., & van der Meijde, M. 2019. On seismic ambient noise cross-correlation and surface-wave attenuation. *Geophys. J. Int.*, 219(3), 1568-1589
- Kästle, E., Soomro, R., Weemstra, C., Boschi, L. & Meier, T., 2016. Two-receiver measurements of phase velocity: cross-validation of ambient-noise and earthquake-based observations. *Geophys. J. Int.*, 207, 1493-1512
- Magrini, F., Diaferia, G., Boschi, L. & Cammarano, F., 2020. Arrival-angle effects on two-receiver measurements of phase velocity. *Geophys. J. Int.*, 220, 1838-1844
- Magrini, F. & Boschi, L., 2021. Surface-wave attenuation from seismic ambient noise: numerical validation and application. *J. Geophys. Res.*, 126, e2020JB019865
- Magrini, F., Boschi, L., Gualtieri, L., Lekić, V. & Cammarano, F., 2021. Rayleigh‑wave attenuation across the conterminous United States in the microseism frequency band. *Scientific Reports*, 11, 1-9

