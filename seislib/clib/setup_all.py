#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 23:05:58 2019

@author: Fabrizio Magrini
@email1: fabrizio.magrini@uniroma3.it
@email2: fabrizio.magrini90@gmail.com

On terminal:
    
    python setup.py build_ext --inplace

for html:
    cython -a example_cython.pyx

"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import os

names = ['_math', '_spherical_geometry', '_tomography']
sources = ['%s.pyx'%name for name in names]
extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ]
extra_link_args=['-fopenmp']


for name, source in zip(names, sources):
    language = 'c++' if name!='_math' else None
    ext_modules=[Extension(name,
                           sources=[source],
                           extra_compile_args=extra_compile_args,
                           extra_link_args=extra_link_args,
                           language=language)
                 ]
        
    setup(name=name,
          cmdclass={"build_ext": build_ext},
          ext_modules=cythonize(ext_modules, language_level="3"),
          include_dirs=[numpy.get_include()]
    )


    






