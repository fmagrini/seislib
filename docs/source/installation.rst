============
Installation
============

To use seislib, make sure you have all the **dependences** installed, i.e., **obspy**, 
**cartopy**, and **cython**. We recommend installing such dependences using conda.

.. code-block:: console

   ~$ conda create -n seislib python=3.9 numpy=1.20
   ~$ conda activate seislib
   ~$ conda install -c conda-forge obspy
   ~$ conda install -c conda-forge cartopy=0.18
   ~$ conda install -c anaconda cython

Note that we installed Python 3.9 (rather than Python 3.10) since numpy's version 1.22 currently leads to compatibility issues. Once the above dependences have been installed, you can proceed with the installation of **seislib**.

.. code-block:: console

   ~$ pip install seislib


.. note::

   If you run into troubles with the above, you can try the following approach:


   .. code-block:: console

      ~$ git clone https://github.com/fmagrini/seislib.git
      ~$ cd seislib/seislib/tomography/_ray_theory
      ~$ python setup_all.py build_ext --inplace

   Where the last command will compile the Cython files. If you work on an anaconda environment, 
   you might need to replace "python" with, e.g., "/home/your_name/anaconda3/bin/python". 
   (You can retrieve the path to your python executable by typing "import sys; print(sys.executable)" 
   in your Python GUI. Make sure to then add ~/seislib to your path, to being able to import 
   its modules in your Python codes.)




