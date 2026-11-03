"""
vio/simulador.py — Trayectorias con verdad conocida y la IMU ideal que las mediría.

Sirve para probar los algoritmos con datos cuya respuesta correcta se sabe.
"""

import numpy as np

from vio import so3
from vio.constantes import GRAVEDAD_W


def tiempos_ns(duracion_s, frecuencia_hz):
    """Marcas de tiempo regulares desde 0. Devuelve: array (N,) int64 [ns]."""
    n = int(round(duracion_s * frecuencia_hz)) + 1
    periodo_ns = int(round(1e9 / frecuencia_hz))
    return np.arange(n, dtype=np.int64) * periodo_ns


def circulo(t_s, radio, omega, altura):
    """Círculo horizontal a velocidad constante, con la IMU mirando hacia delante.

    t_s: (N,) tiempos [s]. radio [m], omega [rad/s], altura [m].
    Posición  p(t) = (r·cos ωt, r·sen ωt, altura)
    Velocidad v(t) = dp/dt = (-r·ω·sen ωt,  r·ω·cos ωt, 0)
    Acelerac. a(t) = dv/dt = (-r·ω²·cos ωt, -r·ω²·sen ωt, 0)   (hacia el centro)
    Orientación: giro sobre z de ωt + 90°, para que el eje x de la IMU apunte
    en la dirección del movimiento. Su velocidad angular en ejes de la IMU es (0, 0, ω).
    Devuelve: dict con R_w_i (N, 3, 3), p_w, v_w, a_w, w_i (N, 3).
    """
    n = len(t_s)
    c = np.cos(omega * t_s)
    s = np.sin(omega * t_s)

    p_w = np.column_stack([radio * c, radio * s, np.full(n, altura)])
    v_w = np.column_stack([-radio * omega * s, radio * omega * c, np.zeros(n)])
    a_w = np.column_stack([-radio * omega ** 2 * c, -radio * omega ** 2 * s, np.zeros(n)])

    R_w_i = np.zeros((n, 3, 3))
    for k in range(n):
        R_w_i[k] = so3.exp_so3([0.0, 0.0, omega * t_s[k] + np.pi / 2])
    w_i = np.tile([0.0, 0.0, omega], (n, 1))

    return {"R_w_i": R_w_i, "p_w": p_w, "v_w": v_w, "a_w": a_w, "w_i": w_i}


def medidas_imu(R_w_i, a_w, w_i, g_w=GRAVEDAD_W, b_g=None, b_a=None):
    """Lo que mediría una IMU sin ruido (con sesgos constantes opcionales).

    R_w_i: (N, 3, 3). a_w: (N, 3) aceleración en el mundo. w_i: (N, 3) en ejes de la IMU.
    Devuelve: (w_m, a_m), arrays (N, 3):
        w_m = w_i + b_g
        a_m = R_w_i^T @ (a_w - g_w) + b_a
    """
    n = len(R_w_i)
    b_g = np.zeros(3) if b_g is None else np.asarray(b_g, dtype=float)
    b_a = np.zeros(3) if b_a is None else np.asarray(b_a, dtype=float)

    w_m = w_i + b_g
    a_m = np.zeros((n, 3))
    for k in range(n):
        a_m[k] = R_w_i[k].T @ (a_w[k] - g_w) + b_a
    return w_m, a_m




# =====================================================================
# Etapa 11 — Simulador VIO completo
# =====================================================================

# ---------------------------------------------------------------------
# Paso 11.1 — Trayectoria que excita todos los ejes e IMU ideal de cualquier trayectoria
# ---------------------------------------------------------------------
def pose_vio(t_s, cfg_tr):
    """Pose de la IMU en el instante t_s [s] (un número): (R_w_i (3, 3), p_w (3,)).

    Posición: el círculo de la etapa 6 más una oscilación vertical,
        p(t) = (r·cos ωt, r·sen ωt, altura + A_z·sen(ω_z·t))
    Orientación: rumbo tangente al círculo (como en la etapa 6) con balanceo y cabeceo,
        R_w_i = R_z(ωt + 90°) · R_y(cabeceo(t)) · R_x(balanceo(t))
    El mismo orden z-y-x que angulos_zyx de la etapa 4.
    """
    r, w = cfg_tr["radio_m"], cfg_tr["omega_rad_s"]
    p_w = np.array([r * np.cos(w * t_s),
                    r * np.sin(w * t_s),
                    cfg_tr["altura_m"] + cfg_tr["oscilacion_z_m"] * np.sin(cfg_tr["omega_z_rad_s"] * t_s)])

    rumbo = w * t_s + np.pi / 2
    cabeceo = cfg_tr["cabeceo_rad"] * np.sin(cfg_tr["omega_cabeceo_rad_s"] * t_s)
    balanceo = cfg_tr["balanceo_rad"] * np.sin(cfg_tr["omega_balanceo_rad_s"] * t_s)
    R_w_i = (so3.exp_so3([0.0, 0.0, rumbo]) @ so3.exp_so3([0.0, cabeceo, 0.0])
             @ so3.exp_so3([balanceo, 0.0, 0.0]))
    return R_w_i, p_w


def imu_ideal(pose, t_s, h):
    """Velocidad, aceleración y velocidad angular de CUALQUIER trayectoria.

    pose: función t [s] -> (R_w_i (3, 3), p_w (3,)).
    t_s: (N,) instantes [s]. h: paso de las diferencias finitas [s].
    Diferencias finitas centradas (error proporcional a h²):
        v_w(t) = (p(t+h) - p(t-h)) / (2h)
        a_w(t) = (p(t+h) - 2·p(t) + p(t-h)) / h²
        w_i(t) = log_so3(R(t-h)ᵀ · R(t+h)) / (2h)      (giro en ejes de la IMU)
    Devuelve: dict con R_w_i (N, 3, 3), p_w, v_w, a_w, w_i (N, 3).
    """
    n = len(t_s)
    R_w_i = np.zeros((n, 3, 3))
    p_w = np.zeros((n, 3))
    v_w = np.zeros((n, 3))
    a_w = np.zeros((n, 3))
    w_i = np.zeros((n, 3))
    for k, t in enumerate(t_s):
        R_menos, p_menos = pose(t - h)
        R_k, p_k = pose(t)
        R_mas, p_mas = pose(t + h)
        R_w_i[k], p_w[k] = R_k, p_k
        v_w[k] = (p_mas - p_menos) / (2 * h)
        a_w[k] = (p_mas - 2 * p_k + p_menos) / h ** 2
        w_i[k] = so3.log_so3(R_menos.T @ R_mas) / (2 * h)
    return {"R_w_i": R_w_i, "p_w": p_w, "v_w": v_w, "a_w": a_w, "w_i": w_i}