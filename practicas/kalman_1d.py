"""
practicas/kalman_1d.py — Filtro de Kalman de juguete: un carrito en una recta.

Estado x = [p, v]: posición [m] y velocidad [m/s].
    - Un acelerómetro ruidoso mide la aceleración u a 100 Hz (como la IMU): se usa
      para PREDECIR el estado.
    - Cada 10 pasos llega una medida ruidosa de la posición z (como la cámara): se usa
      para CORREGIR la predicción.
Como los datos son simulados, se conoce la verdad y se puede medir si el filtro es
consistente: si sus errores reales tienen el tamaño que él mismo dice en P.

Uso, desde ~/svo-pi:
    python -m practicas.kalman_1d
Guarda datos/kalman_1d.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from vio.config import RAIZ_PROYECTO, cargar_config

H = np.array([[1.0, 0.0]])      # la medida es la posición: z = H · x + ruido


# =====================================================================
# Paso 10.1 — Modelo y simulación
# =====================================================================
def matrices(dt):
    """Matrices del movimiento con aceleración constante durante dt [s].

        p_siguiente = p + v·dt + ½·u·dt²
        v_siguiente =     v    +   u·dt
    es decir, x_siguiente = F · x + B · u.
    """
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([0.5 * dt ** 2, dt])
    return F, B


def simular(cfg, rng):
    """Una simulación: verdad, lecturas del acelerómetro y medidas de posición.

    cfg: dict de config/kalman_1d.yaml. rng: generador de números aleatorios.
    Devuelve: dict con
        t_s (N+1,)           instantes [s], con N = número de pasos
        x (N+1, 2)           estado verdadero en cada instante
        u (N,)               lectura del acelerómetro entre el instante k y el k+1
        con_medida (N+1,)    bool: si en ese instante llega una medida de posición
        z (N+1,)             medida de posición (nan donde no hay)
        x0_est, P0           estimación inicial que recibe el filtro y su covarianza
    """
    sim, ruido, ini = cfg["simulacion"], cfg["ruido"], cfg["inicial"]
    dt = 1.0 / sim["frecuencia_hz"]
    n = int(round(sim["duracion_s"] * sim["frecuencia_hz"]))
    F, B = matrices(dt)
    t_s = np.arange(n + 1) * dt

    # 1. Estado inicial: el filtro cree x0_est con covarianza P0; la verdad es una
    #    muestra de esa distribución (si no, P0 mentiría desde el principio)
    x0_est = np.zeros(2)
    P0 = np.diag([ini["sigma_p0"] ** 2, ini["sigma_v0"] ** 2])
    x = np.zeros((n + 1, 2))
    x[0] = x0_est + np.linalg.cholesky(P0) @ rng.normal(0.0, 1.0, 2)

    # 2. Movimiento verdadero con la aceleración A·sen(omega·t), constante en cada paso
    a = sim["aceleracion_amplitud"] * np.sin(sim["aceleracion_omega"] * t_s[:-1])
    for k in range(n):
        x[k + 1] = F @ x[k] + B * a[k]

    # 3. Sensores: acelerómetro en cada paso, posición cada cada_n_medida pasos
    u = a + rng.normal(0.0, ruido["sigma_a"], n)
    con_medida = (np.arange(n + 1) % sim["cada_n_medida"] == 0)
    con_medida[0] = False                              # en t = 0 solo está la estimación inicial
    z = np.full(n + 1, np.nan)
    z[con_medida] = x[con_medida, 0] + rng.normal(0.0, ruido["sigma_z"], con_medida.sum())

    return {"t_s": t_s, "x": x, "u": u, "con_medida": con_medida, "z": z,
            "x0_est": x0_est, "P0": P0}