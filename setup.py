#!/usr/bin/env python
"""
SeisLib: Seismic Imaging library for Python
"""

import os
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os
import platform



def readme():
	with open("README.md", "r") as f:
		return f.read()


def configuration(parent_package="", top_path=None):
	from numpy.distutils.misc_util import Configuration

	config = Configuration(None, parent_package, top_path)
	config.set_options(
		ignore_setup_xxx_py=True,
		assume_default_configuration=True,
		delegate_options_to_subpackages=True,
		quiet=True)

	# add the subpackage 'seislib' to the config (to call it's setup.py later)
	config.add_subpackage("seislib")

	return config


def get_maps():
	cwd = os.getcwd()
	src = os.path.join(cwd, 'seislib', 'colormaps')
	maps_dirs = [i for i in os.listdir(src) if os.path.isdir(os.path.join(src, i))]
	return ['colormaps/%s/*'%m for m in maps_dirs]
	

here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, 'seislib', "__version__.py")) as f:
	exec(f.read(), about)


def setup_cython_extension():
	ext_path = "seislib/tomography/_ray_theory/"
	ext_name = ext_path.replace("/", ".")
	names = ['_math', '_spherical_geometry', '_tomography']
	sources = ['%s.pyx'%name for name in names]
	extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ]
	platform_name = platform.system()
	if platform_name.lower() == 'darwin':
		versions = os.listdir('/usr/local/Cellar/gcc/')
		version = max(versions, key=lambda i: int(i.split('.')[0]))
		version_int = version.split('.')[0]
		path = '/usr/local/Cellar/gcc/%s/lib/gcc/%s'%(version, version_int)
		os.environ['CC'] = 'gcc-%s'%version_int
		os.environ['CXX'] = 'g++-%s'%version_int
		extra_link_args=['-Wl,-rpath,%s'%path]
	else:
		extra_link_args=['-fopenmp']
	ext_modules = []
	for name, source in zip(names, sources):
		language = 'c++' if name!='_math' else None
		ext_modules.append(Extension(f"{ext_name}{name}",
							sources=[f"{ext_path}{source}"],
							extra_compile_args=extra_compile_args,
							extra_link_args=extra_link_args,
							language=language,
							include_dirs=[numpy.get_include()]))
	return ext_modules

pkg_metadata = dict(
		name="seislib",
		version=about["__version__"],
		description="Multi-scale Seismic Imaging",
		long_description=readme(),
		long_description_content_type="text/markdown",
		url="https://github.com/fmagrini/seislib",
		author="Fabrizio Magrini",
		author_email="fabrizio.magrini90@gmail.com",
		license="MIT",
		packages=find_packages(),
		package_data={  
			"seislib": ["tomography/_ray_theory/*", "colormaps/*"] + get_maps()
			},
		include_package_data=True,
		python_requires=">=3.6",
		keywords="Seismic Imaging, Surface Waves, Seismic Ambient Noise, Earthquake Seismology, Tomographic Inversion",
		configuration=configuration,
		cmdclass={"build_ext": build_ext},
		ext_modules=setup_cython_extension(),
		install_requires=['obspy>=1.1.0',
						  'numpy>=1.16.0',
						  'scipy>=1.3.0',
						  'matplotlib>=3.0.2',
						  'cartopy>=0.17.0',
						  'cython>=0.29'],
		classifiers=["License :: OSI Approved :: MIT License",
					 "Programming Language :: Python :: 3",
					 "Programming Language :: Cython"]
		)

setup(**pkg_metadata)
