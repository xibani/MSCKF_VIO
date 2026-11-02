"""
vio/propagacion.py — Orientación, velocidad y posición integrando la IMU completa.

Modelo del acelerómetro (fuerza específica, en ejes de la IMU):
    a_m = R_w_i^T @ (a_w - g_w) + b_a          con g_w = (0, 0, -g)
Despejando la aceleración en el mundo:
    a_w = R_w_i @ (a_m - b_a) + g_w
Integración en cada intervalo Δt (aceleración constante dentro del intervalo):
    p <- p + v·Δt + ½·a_w·Δt²
    v <- v + a_w·Δt
    R <- R @ exp((w_m - b_g)·Δt)
"""

import numpy as np

from vio import so3
from vio.constantes import GRAVEDAD_W
from vio.integrador_giro import paso_giro


def aceleracion_mundo(R_w_i, a_m, b_a, g_w):
    """Aceleración en el mundo [m/s²] a partir de la medida del acelerómetro."""
    return R_w_i @ (a_m - b_a) + g_w


def propagar(estado, w_m, a_m, dt_s, g_w=GRAVEDAD_W):
    """Un paso de integración. No modifica `estado`: devuelve uno nuevo.

    w_m: (3,) giroscopio [rad/s]; a_m: (3,) acelerómetro [m/s²], ambos
    representativos del intervalo. dt_s: duración del intervalo [s].
    """
    nuevo = estado.copia()

    # 1. Aceleración en el mundo, con la orientación del PRINCIPIO del intervalo
    a_w = aceleracion_mundo(estado.R_w_i, a_m, estado.b_a, g_w)

    # 2. Posición y velocidad (fórmulas del movimiento con aceleración constante)
    nuevo.p_w = estado.p_w + estado.v_w * dt_s + 0.5 * a_w * dt_s ** 2
    nuevo.v_w = estado.v_w + a_w * dt_s

    # 3. Orientación (igual que en la etapa 5)
    nuevo.R_w_i = paso_giro(estado.R_w_i, w_m, estado.b_g, dt_s)
    return nuevo


class PropagadorImu:
    """Propaga el estado muestra a muestra, con Δt reales y regla del trapecio."""

    def __init__(self, estado, t_ns, w_rad_s, a_m_s2, cfg_int, g_w=GRAVEDAD_W):
        """estado: EstadoImu inicial. t_ns, w_rad_s, a_m_s2: primera muestra.
        cfg_int: sección "integracion" de config/orientacion.yaml.
        g_w: vector gravedad en el mundo (por defecto, la estándar).
        """
        self.estado = estado.copia()
        self.t_ns = t_ns
        self.w_anterior = np.array(w_rad_s, dtype=float)
        self.a_anterior = np.array(a_m_s2, dtype=float)
        self.cada = cfg_int["reortonormalizar_cada"]
        self.dt_max_s = cfg_int["dt_max_s"]
        self.g_w = np.array(g_w, dtype=float)
        self.n = 0
        self.huecos = 0

    def actualizar(self, t_ns, w_rad_s, a_m_s2):
        """Añade una muestra. Devuelve: EstadoImu en el instante t_ns."""
        w_rad_s = np.asarray(w_rad_s, dtype=float)
        a_m_s2 = np.asarray(a_m_s2, dtype=float)
        dt_s = (t_ns - self.t_ns) / 1e9
        if dt_s > self.dt_max_s:
            self.huecos += 1

        # Medidas representativas del intervalo: media de sus dos extremos
        w_medio = 0.5 * (self.w_anterior + w_rad_s)
        a_medio = 0.5 * (self.a_anterior + a_m_s2)
        self.estado = propagar(self.estado, w_medio, a_medio, dt_s, self.g_w)

        self.n += 1
        if self.n % self.cada == 0:
            self.estado.R_w_i = so3.reortonormalizar(self.estado.R_w_i)

        self.t_ns = t_ns
        self.w_anterior = w_rad_s
        self.a_anterior = a_m_s2
        return self.estado


def propagar_secuencia(estado_inicial, t_ns, w, a, cfg_int, g_w=GRAVEDAD_W):
    """Propaga una secuencia completa (uso sin conexión).

    t_ns: (N,) int64. w: (N, 3) [rad/s]. a: (N, 3) [m/s²].
    Devuelve: dict con R (N, 3, 3), v (N, 3), p (N, 3) y huecos (int).
    """
    n = len(t_ns)
    R = np.zeros((n, 3, 3))
    v = np.zeros((n, 3))
    p = np.zeros((n, 3))

    propagador = PropagadorImu(estado_inicial, int(t_ns[0]), w[0], a[0], cfg_int, g_w)
    R[0], v[0], p[0] = estado_inicial.R_w_i, estado_inicial.v_w, estado_inicial.p_w
    for k in range(1, n):
        e = propagador.actualizar(int(t_ns[k]), w[k], a[k])
        R[k], v[k], p[k] = e.R_w_i, e.v_w, e.p_w
    return {"R": R, "v": v, "p": p, "huecos": propagador.huecos}