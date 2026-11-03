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


# ---------------------------------------------------------------------
# Paso 11.2 — Ruido blanco y sesgo con paseo aleatorio
# ---------------------------------------------------------------------
def ruido_y_sesgo(n, frecuencia_hz, densidad_ruido, paseo_sesgo, sesgo_inicial, rng):
    """Ruido blanco y sesgo variable de un sensor de 3 ejes.

    densidad_ruido: N [unidad/√Hz]. Cada muestra lleva un ruido de sigma = N·√f.
    paseo_sesgo: K [unidad/s/√Hz]. En cada paso, b <- b + K·√Δt·(normal de varianza 1).
    Devuelve: (blanco (n, 3), sesgo (n, 3)).
    """
    dt = 1.0 / frecuencia_hz
    blanco = rng.normal(0.0, densidad_ruido * np.sqrt(frecuencia_hz), (n, 3))
    sesgo = np.zeros((n, 3))
    sesgo[0] = sesgo_inicial
    for k in range(1, n):
        sesgo[k] = sesgo[k - 1] + paseo_sesgo * np.sqrt(dt) * rng.normal(0.0, 1.0, 3)
    return blanco, sesgo


def imu_con_ruido(tr, cfg_imu, frecuencia_hz, rng):
    """Lo que mediría la MPU-6050 en la trayectoria tr (dict de imu_ideal).

    Devuelve: (w_m, a_m, b_g, b_a), arrays (N, 3): medidas y sesgos verdaderos.
    """
    n = len(tr["R_w_i"])
    w_ideal, a_ideal = medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"])
    blanco_g, b_g = ruido_y_sesgo(n, frecuencia_hz, cfg_imu["ruido_gir"],
                                  cfg_imu["paseo_sesgo_gir"], cfg_imu["sesgo_inicial_gir"], rng)
    blanco_a, b_a = ruido_y_sesgo(n, frecuencia_hz, cfg_imu["ruido_acc"],
                                  cfg_imu["paseo_sesgo_acc"], cfg_imu["sesgo_inicial_acc"], rng)
    return w_ideal + b_g + blanco_g, a_ideal + b_a + blanco_a, b_g, b_a


# ---------------------------------------------------------------------
# Paso 11.3 — Mundo, extrínseca y cámara
# ---------------------------------------------------------------------
def puntos_habitacion(dimensiones, n_por_superficie, rng):
    """Puntos al azar en las 4 paredes, el suelo y el techo de una habitación.

    dimensiones: (lx, ly, lz) [m]. La habitación ocupa x en [-lx/2, lx/2],
    y en [-ly/2, ly/2] y z en [0, lz].
    Devuelve: array (6·n_por_superficie, 3) [m].
    """
    lx, ly, lz = dimensiones
    n = n_por_superficie
    superficies = []
    for x_pared in (-lx / 2, lx / 2):                      # paredes x = constante
        superficies.append(np.column_stack([np.full(n, x_pared),
                                            rng.uniform(-ly / 2, ly / 2, n),
                                            rng.uniform(0.0, lz, n)]))
    for y_pared in (-ly / 2, ly / 2):                      # paredes y = constante
        superficies.append(np.column_stack([rng.uniform(-lx / 2, lx / 2, n),
                                            np.full(n, y_pared),
                                            rng.uniform(0.0, lz, n)]))
    for z_plano in (0.0, lz):                              # suelo y techo
        superficies.append(np.column_stack([rng.uniform(-lx / 2, lx / 2, n),
                                            rng.uniform(-ly / 2, ly / 2, n),
                                            np.full(n, z_plano)]))
    return np.vstack(superficies)


def extrinseca(cfg_cam):
    """T_c_i (IMU -> cámara) a partir del YAML: p_c = R_c_i · p_i + t_c_i.

    El YAML da R_i_c (columnas = ejes de la cámara en ejes de la IMU) y p_i_c (dónde
    está la cámara en ejes de la IMU). Despejando p_i = R_i_c · p_c + p_i_c:
        R_c_i = R_i_cᵀ,     t_c_i = -R_c_i · p_i_c
    """
    R_i_c = np.array(cfg_cam["R_i_c"], dtype=float)
    p_i_c = np.array(cfg_cam["p_i_c_m"], dtype=float)
    R_c_i = R_i_c.T
    return R_c_i, -R_c_i @ p_i_c


def pose_camara(R_w_i, p_w_i, R_c_i, t_c_i):
    """Pose de la cámara (R_c_w, t_c_w) a partir de la de la IMU y la extrínseca.

    Un punto del mundo en ejes de la IMU: p_i = R_w_iᵀ · (p_w - p_w_i).
    En ejes de la cámara: p_c = R_c_i · p_i + t_c_i. Agrupando términos:
        R_c_w = R_c_i · R_w_iᵀ,     t_c_w = t_c_i - R_c_w · p_w_i
    """
    R_c_w = R_c_i @ R_w_i.T
    return R_c_w, t_c_i - R_c_w @ p_w_i


def observar_puntos(R_c_w, t_c_w, puntos_w, K, tamano, profundidad_min, ruido_px, rng):
    """Qué puntos ve la cámara y dónde, con ruido.

    puntos_w: (M, 3). tamano: (ancho, alto) [px].
    Un punto se ve si está a más de profundidad_min delante de la cámara y su proyección
    SIN ruido cae dentro de la imagen. Después se añade el ruido en píxeles.
    Devuelve: (ids (V,), uv (V, 2) [px], xy (V, 2) normalizadas), V = puntos vistos.
    """
    ancho, alto = tamano
    # 1. Todos los puntos a ejes de la cámara (de una vez: cada fila es un punto)
    p_c = puntos_w @ R_c_w.T + t_c_w
    delante = p_c[:, 2] > profundidad_min
    # 2. Proyección a píxeles (solo los de delante, para no dividir por z <= 0)
    ids = np.flatnonzero(delante)
    xy = p_c[ids, :2] / p_c[ids, 2:3]
    u = K[0, 0] * xy[:, 0] + K[0, 2]
    v = K[1, 1] * xy[:, 1] + K[1, 2]
    dentro = (u >= 0) & (u < ancho) & (v >= 0) & (v < alto)
    ids, u, v = ids[dentro], u[dentro], v[dentro]
    # 3. Ruido en píxeles y vuelta a coordenadas normalizadas
    uv = np.column_stack([u, v]) + rng.normal(0.0, ruido_px, (len(ids), 2))
    xy = np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1]])
    return ids, uv, xy