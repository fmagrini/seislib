#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fabrizio Magrini
@email: fabrizio.magrini@uniroma3.it
@email2: fabrizio.magrini90@gmail.com

seispy
=============================================================================
"""

from __future__ import print_function
import sys
import os
#sys.path.insert(1, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'clib'))
from .tomography.grid import EqualAreaGrid
from . import an
from . import colormaps
from . import eq
from . import exceptions
from . import plotting
from . import tomography
from . import utils
#from .tomography.tomography import SeismicTomography



