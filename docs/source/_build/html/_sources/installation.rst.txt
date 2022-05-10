Installation
------------

To use seislib, make sure you have all the **dependences** installed, i.e., **obspy**, 
**cartopy**, and **cython**. Having installed these three packages, you should not 
need about other dependences such as *numpy*, *scipy*, and *matplotlib*.

Using CONDA (recommended):

.. code-block:: console

   ~$ conda install -c conda-forge obspy
   ~$ conda install -c conda-forge cartopy
   ~$ conda install -c anaconda cython>=0.29.2

Using PIP:


.. code-block:: console

   ~$ pip install obspy
   ~$ pip install cartopy
   ~$ pip install cython>=0.29.2


Once the above dependences have been installed, you can proceed with the installation of 
**seislib**. 

.. code-block:: console

   ~$ pip install seislib


.. note::

   If you run into troubles with the above, you can try the following approach:


   .. code-block:: console

      ~$ git clone https://github.com/fmagrini/seislib.git
      ~$ cd seislib/seislib/clib/
      ~$ python setup_all.py build_ext --inplace

   Where the last command will compile the Cython files. If you work on an anaconda environment, 
   you might need to replace "python" with, e.g., "/home/your_name/anaconda3/bin/python". 
   (You can retrieve the path to your python executable by typing "import sys; print(sys.executable)" 
   in your Python GUI. Make sure to then add ~/seislib to your path, to being able to import 
   its modules in your Python codes.)




