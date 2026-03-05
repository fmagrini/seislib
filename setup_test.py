#import os
#import sys
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

# https://stackoverflow.com/questions/47361418/how-to-specify-path-to-pxd-file-in-cython
# https://github.com/jkleckner/cython_example/tree/master
#import seislib
#print('######## seislib.__file__: ' + seislib.__file__)
#print('######## os.path.dirname(seislib.__file__): ' + os.path.dirname(seislib.__file__))
#seislib_include = os.path.dirname(seislib.__file__)
#print('######## os.path.dirname(os.path.dirname(seislib.__file__)): ' + os.path.dirname(os.path.dirname(seislib.__file__)))
#print('######## seislib_include: ' + seislib_include)
#sys.path.append(seislib_include)

name = 'tomography_test'
source = '%s.pyx'%name
extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"]
extra_link_args=['-fopenmp']
language = 'c++'

extension = Extension(f"{name}",
                      sources = [source],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language=language,
                include_dirs=[numpy.get_include()],
                )

setup(
    name='Test tomography_test',
    ext_modules = cythonize(extension),
)
