"""
vio/estado.py — Estado de la IMU: orientación, velocidad, posición y sesgos.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EstadoImu:
    """Estado "nominal" de la IMU en un instante.

    R_w_i: (3, 3) orientación (de ejes de la IMU a ejes del mundo)
    v_w:   (3,)   velocidad en el mundo [m/s]
    p_w:   (3,)   posición en el mundo [m]
    b_g:   (3,)   sesgo del giroscopio [rad/s]
    b_a:   (3,)   sesgo del acelerómetro [m/s²]
    """
    R_w_i: np.ndarray
    v_w: np.ndarray
    p_w: np.ndarray
    b_g: np.ndarray
    b_a: np.ndarray

    def copia(self):
        """Copia independiente: cambiar la copia no cambia el original."""
        return EstadoImu(self.R_w_i.copy(), self.v_w.copy(), self.p_w.copy(),
                         self.b_g.copy(), self.b_a.copy())


def estado_inicial(R_w_i, b_g=None, b_a=None):
    """Estado en reposo en el origen: v = 0, p = 0. Sesgos 0 si no se dan."""
    return EstadoImu(
        R_w_i=np.array(R_w_i, dtype=float),
        v_w=np.zeros(3),
        p_w=np.zeros(3),
        b_g=np.zeros(3) if b_g is None else np.array(b_g, dtype=float),
        b_a=np.zeros(3) if b_a is None else np.array(b_a, dtype=float),
    )