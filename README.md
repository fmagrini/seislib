# **SeisLib: Seismic Imaging at Local, Regional, and Global Scale**

***seislib*** is a Python (and Cython) package that allows for obtaining seismic images of the sub-surface at both local, regional, and global scale. It is the result of a long-term effort of our team to make efficient and open source some of the Python codes behind our seismological publications over the last few years. The library is in rapid expansion and, at present, includes:

<p>&nbsp;</p>

***
## **Seismic Ambient Noise Interferometry**
*  Automated **download** of continuous seismograms
* **Fast cross-correlation** of continuous seismograms in the **frequency domain**
* Extraction of frequency-dependent **phase velocities** for both **Rayleigh and Love waves**, based on pairs of receivers
* Retrieval of frequency-dependent **Rayleigh-wave attenuation coefficient**, based on dense seismic arrays

<p>&nbsp;</p>

## **Surface-Wave Tomography based on Teleseismic Earthquakes**
* Automated **download** of seismograms recording strong earthquakes
* Retrieval of frequency-dependent **Rayleigh and Love phase velocities**, based on pairs of receivers lying on the same great-circle path as the epicentre (**Two-Station Method**)

<p>&nbsp;</p>

## **Inversion for 2-D Tomographic Images**
* **Equal-area parameterizations**, suited for data sets collected at both local, regional, and global scale
* **Adaptive parameterizations**, with finer resolution in the areas characterized by relatively high density of measurements
* **Linearized inversion** of velocity measurements **based on the ray-theory**
* **Fast computations** even in presence of large data sets, thanks to the use of **Cython**
* Possibility to perform **L-curve analyses and resolution tests (e.g., spike, checkerboard)**
   
***

<p>&nbsp;</p>



## *References*
- Boschi, L. & Dziewonski, A.M., 1999. High- and low-resolution images of the Earth's mantle: Implications of different approaches to tomographic modeling. *J. Geophys. Res.*, 104(B11)
- Boschi, L., Magrini, F., Cammarano, F., & van der Meijde, M. 2019. On seismic ambient noise cross-correlation and surface-wave attenuation. *Geophys. J. Int.*, 219(3), 1568-1589
- Kästle, E., Soomro, R., Weemstra, C., Boschi, L. & Meier, T., 2016. Two-receiver measurements of phase velocity: cross-validation of ambient-noise and earthquake-based observations. *Geophys. J. Int.*, 207, 1493-1512
- Magrini, F., Diaferia, G., Boschi, L. & Cammarano, F., 2020. Arrival-angle effects on two-receiver measurements of phase velocity. *Geophys. J. Int.*, 220, 1838-1844
- Magrini, F. & Boschi, L., 2021. Surface-wave attenuation from seismic ambient noise: numerical validation and application. *J. Geophys. Res.*, 126, e2020JB019865
- Magrini, F., Boschi, L., Gualtieri, L., Lekić, V. & Cammarano, F., 2021. Rayleigh‑wave attenuation across the conterminous United States in the microseism frequency band. *Scientific Reports*, 11, 1-9

