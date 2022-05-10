"""
===========================================
Colormaps (:mod:`seislib.colormaps`)
===========================================

SeisLib comes with the beautiful scientific
colormaps created by Fabio Crameri. These can
be simply used as traditional `matplotlib` colormaps,
for example::
    
    import seislib.colormaps as scm
    import numpy as np
    import matplotlib.pyplot as plt
    
    X, Y = np.meshgrid(np.linspace(0, 10), np.linspace(0, 10))
    Z = np.sin(X) * np.cos(Y)
    plt.pcolormesh(X, Y, Z, cmap=scm.batlow)

Make sure to check `Fabio Crameri's website 
<https://www.fabiocrameri.ch/colourmaps/>`_ for more information!

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
