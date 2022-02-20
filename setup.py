#!/usr/bin/env python
"""
SeisLib: Seismic Imaging library for Python
"""

import sys
import os
from setuptools import find_packages



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


def generate_cython():
    python = sys.executable
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    os.chdir(os.path.join(cwd, 'seislib/clib'))
    print(os.getcwd())
    os.system('%s ./setup_all.py build_ext --inplace'%python)
    os.chdir(cwd)
    return


def parse_setuppy_commands():
    """Check the commands and respond appropriately.
    Return a boolean value for whether or not to run the build or not.
    
    Inspired from SciPy's setup.py : https://github.com/scipy/scipy/blob/master/setup.py
    """
    args = sys.argv[1:]

    if not args:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--url',
                     '--license', '--description', '--long-description',
                     '--platforms', '--classifiers', '--keywords',
                     '--provides', '--requires', '--obsoletes',
                     'egg_info', 'install_egg_info', 'rotate']

    for command in info_commands:
        if command in args:
            return False

    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py',
                     'build_clib', 'build_scripts', 'bdist_wheel', 'bdist_rpm',
                     'bdist_wininst', 'bdist_msi', 'bdist_mpkg',
                     'build_sphinx')

    for command in good_commands:
        if command in args:
            return True

    # The following commands are supported, but we need to show more useful messages to the user
    if "install" in args:
        return True

    return False


def get_maps():
    cwd = os.getcwd()
    src = os.path.join(cwd, 'seislib', 'colormaps')
    maps_dirs = [i for i in os.listdir(src) if os.path.isdir(os.path.join(src, i))]
    return ['colormaps/%s/*'%m for m in maps_dirs]
    

here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, 'seislib', "__version__.py")) as f:
    exec(f.read(), about)


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
            "seislib": ["clib/*", "colormaps/*"] + get_maps()
            },
        include_package_data=True,
        python_requires=">=3.6",
        keywords="Seismic Imaging, Surface Waves, Seismic Ambient Noise, Earthquake Seismology, Tomographic Inversion",
        configuration=configuration,
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









if __name__ == "__main__":
    if "--force" in sys.argv:
        run_build = True
        sys.argv.remove("--force")
    else:
        # Raise errors for unsupported commands, improve help output, etc.
        run_build = parse_setuppy_commands()

    # This import is here because it needs to be done before importing setup() from numpy.distutils
    from setuptools import setup

    if run_build:
        from numpy.distutils.core import setup
        if not 'sdist' in sys.argv:
            # Generate Cython sources, unless we're creating an sdist
            # Cython is a build dependency, and shipping generated .c files
            # can cause problems (see gh-14199)
            generate_cython()
    # Don't import numpy in other cases - non-build actions are required to succeed without NumPy,
    # for example when pip is used to install this when NumPy is not yet present in the system.

    setup(**pkg_metadata)
