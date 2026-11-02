"""
vio/integrador_giro.py — Orientación integrando el giroscopio.

En cada intervalo entre dos muestras, la placa gira phi = w_medio · Δt
(en sus propios ejes), y la orientación se actualiza multiplicando por la
DERECHA:   R_w_i(k+1) = R_w_i(k) @ exp_so3(phi)
"""

import numpy as np

from vio import so3


def paso_giro(R_w_i, w_rad_s, b_g, dt_s):
    """Un paso de integración.

    R_w_i: (3, 3) orientación al principio del intervalo.
    w_rad_s: (3,) velocidad angular medida durante el intervalo [rad/s].
    b_g: (3,) sesgo del giroscopio [rad/s].  dt_s: duración del intervalo [s].
    Devuelve: (3, 3) orientación al final del intervalo.
    """
    phi = (w_rad_s - b_g) * dt_s          # giro en los ejes de la placa [rad]
    return R_w_i @ so3.exp_so3(phi)


class IntegradorGiro:
    """Integra muestras de giroscopio una a una, con Δt reales."""

    def __init__(self, R_inicial, b_g, t_ns, w_rad_s, cfg_int):
        """R_inicial: (3, 3). b_g: (3,) [rad/s].
        t_ns, w_rad_s: primera muestra (instante y medida).
        cfg_int: sección "integracion" de config/orientacion.yaml.
        """
        self.R = np.array(R_inicial, dtype=float)
        self.b_g = np.array(b_g, dtype=float)
        self.t_ns = t_ns
        self.w_anterior = np.array(w_rad_s, dtype=float)
        self.cada = cfg_int["reortonormalizar_cada"]
        self.dt_max_s = cfg_int["dt_max_s"]
        self.n = 0
        self.huecos = 0

    def actualizar(self, t_ns, w_rad_s):
        """Añade una muestra nueva. Devuelve: R_w_i en el instante t_ns."""
        w_rad_s = np.asarray(w_rad_s, dtype=float)
        dt_s = (t_ns - self.t_ns) / 1e9
        if dt_s > self.dt_max_s:
            self.huecos += 1

        # Regla del trapecio: la media de las dos medidas del intervalo
        # (exacta si la velocidad cambia linealmente durante el intervalo)
        w_medio = 0.5 * (self.w_anterior + w_rad_s)
        self.R = paso_giro(self.R, w_medio, self.b_g, dt_s)

        self.n += 1
        if self.n % self.cada == 0:
            self.R = so3.reortonormalizar(self.R)

        self.t_ns = t_ns
        self.w_anterior = w_rad_s
        return self.R


def integrar_secuencia(R_inicial, b_g, t_ns, w, cfg_int):
    """Integra una secuencia completa (uso sin conexión, con grabaciones).

    t_ns: (N,) int64. w: (N, 3) [rad/s].
    Devuelve: (R, huecos): R array (N, 3, 3) con R[k] = R_w_i en t_ns[k]; huecos: int.
    """
    integrador = IntegradorGiro(R_inicial, b_g, int(t_ns[0]), w[0], cfg_int)
    R = np.zeros((len(t_ns), 3, 3))
    R[0] = integrador.R
    for k in range(1, len(t_ns)):
        R[k] = integrador.actualizar(int(t_ns[k]), w[k])
    return R, integrador.huecos