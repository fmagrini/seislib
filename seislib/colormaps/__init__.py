"""
    Scientific colormaps created by Fabio Crameri 
    (https://www.fabiocrameri.ch/colourmaps/).
    
    Usage
    -----
    import seislib.colormaps as scm
    
    plt.imshow(data, cmap=scm.berlin)


    Available colourmaps
    ---------------------
    acton, bamako, batlow, berlin, bilbao, broc, buda, cork, davos, devon,
    grayC, hawaii, imola, lajolla, lapaz, lisbon, nuuk, oleron, oslo, roma,
    tofino, tokyo, turku, vik
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from string import ascii_letters

folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


__all__ = {i for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i)) and i[0] in ascii_letters}
for name in __all__:
    file = os.path.join(folder, name, name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name + '_r'] = LinearSegmentedColormap.from_list(name + '_r', np.flip(cm_data, axis=0))

del np, ascii_letters, LinearSegmentedColormap, os, plt
