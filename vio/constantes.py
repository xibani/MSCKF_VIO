"""
vio/constantes.py — Constantes físicas comunes a todo el proyecto.
"""

import numpy as np

G = 9.80665                              # gravedad estándar [m/s²]
GRAVEDAD_W = np.array([0.0, 0.0, -G])    # vector gravedad en el mundo (z hacia arriba) [m/s²]