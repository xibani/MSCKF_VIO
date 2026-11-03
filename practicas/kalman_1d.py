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


# =====================================================================
# Paso 10.2 — Predicción y corrección
# =====================================================================
def ruido_proceso(B, sigma_a):
    """Q: covarianza que añade al estado el ruido del acelerómetro en un paso.

    El filtro usa u = a + n, con n de varianza sigma_a². Ese n entra en el estado
    como B·n, cuya covarianza es B · sigma_a² · Bᵀ.
    """
    return np.outer(B, B) * sigma_a ** 2


def predecir(x, P, u, F, B, Q):
    """Paso de predicción: avanzar el estado con la lectura del acelerómetro.

    x (2,), P (2, 2), u [m/s²]. Devuelve (x, P) predichos. La incertidumbre crece.
    """
    x = F @ x + B * u
    P = F @ P @ F.T + Q
    return x, P


def corregir(x, P, z, R):
    """Paso de corrección con una medida de posición z [m] de varianza R [m²].

    Devuelve (x, P, y, S): estado y covarianza corregidos, innovación y su covarianza.
    P se actualiza con la forma de Joseph, que la mantiene simétrica y positiva.
    """
    # 1. Innovación: lo que la medida dice y la predicción no esperaba
    y = z - (H @ x)[0]
    # 2. Covarianza de la innovación: incertidumbre de la predicción + la de la medida
    S = (H @ P @ H.T)[0, 0] + R
    # 3. Ganancia: cuánto de la innovación se cree, y cómo se reparte entre p y v
    K = (P @ H.T)[:, 0] / S                            # (2,)
    # 4. Estado corregido
    x = x + K * y
    # 5. Covarianza corregida (forma de Joseph)
    I_KH = np.eye(2) - np.outer(K, H[0])
    P = I_KH @ P @ I_KH.T + np.outer(K, K) * R
    return x, P, y, S


def ejecutar_filtro(sim, cfg, factor_q=1.0, factor_r=1.0):
    """Pasa el filtro por una simulación.

    factor_q, factor_r: multiplican Q y R que USA el filtro (los datos no cambian).
        Sirven para ver qué pasa cuando el filtro se equivoca sobre sus ruidos.
    Devuelve: (x_est (N+1, 2), P_est (N+1, 2, 2)).
    """
    dt = sim["t_s"][1] - sim["t_s"][0]
    F, B = matrices(dt)
    Q = ruido_proceso(B, cfg["ruido"]["sigma_a"]) * factor_q
    R = cfg["ruido"]["sigma_z"] ** 2 * factor_r

    n = len(sim["u"])
    x_est = np.zeros((n + 1, 2))
    P_est = np.zeros((n + 1, 2, 2))
    x, P = sim["x0_est"].copy(), sim["P0"].copy()
    x_est[0], P_est[0] = x, P
    for k in range(1, n + 1):
        x, P = predecir(x, P, sim["u"][k - 1], F, B, Q)
        if sim["con_medida"][k]:
            x, P, _, _ = corregir(x, P, sim["z"][k], R)
        x_est[k], P_est[k] = x, P
    return x_est, P_est


# =====================================================================
# Paso 10.3 — Consistencia: NEES
# =====================================================================
def nees(e, P):
    """NEES (Normalized Estimation Error Squared) = eᵀ · P⁻¹ · e.

    e (2,) error real (estimación - verdad), P (2, 2) covarianza que da el filtro.
    """
    return e @ np.linalg.solve(P, e)


def nees_medio(cfg, rng, factor_q=1.0, factor_r=1.0):
    """NEES medio en cada instante sobre cfg["consistencia"]["simulaciones"] simulaciones.

    Devuelve: array (N+1,).
    """
    n_sim = cfg["consistencia"]["simulaciones"]
    suma = None
    for _ in range(n_sim):
        sim = simular(cfg, rng)
        x_est, P_est = ejecutar_filtro(sim, cfg, factor_q, factor_r)
        valores = np.array([nees(x_est[k] - sim["x"][k], P_est[k]) for k in range(len(x_est))])
        suma = valores if suma is None else suma + valores
    return suma / n_sim


def intervalo_nees(dimension, n_sim, confianza):
    """Intervalo en el que debe caer el NEES medio de un filtro consistente.

    Si el filtro es consistente, cada NEES sigue una chi-cuadrado con 'dimension' grados
    de libertad; la suma de n_sim independientes, una chi-cuadrado con dimension·n_sim.
    Devuelve: (inferior, superior) del NEES MEDIO (la suma dividida por n_sim).
    """
    gl = dimension * n_sim
    cola = (1.0 - confianza) / 2.0
    return chi2.ppf(cola, gl) / n_sim, chi2.ppf(1.0 - cola, gl) / n_sim