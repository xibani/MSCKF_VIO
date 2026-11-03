"""
vio/triangulacion.py — Posición 3D de un punto visto desde varias poses conocidas.

Convenciones:
    Una pose es la pareja (R_c_w, t_c_w): lleva un punto del mundo a ejes de la cámara,
        p_c = R_c_w · p_w + t_c_w
    Las observaciones son coordenadas normalizadas sin distorsión (x, y) = (X/Z, Y/Z),
    las que guarda cada Pista del front-end en pista.xy.

Pasos:
    1. Triangulación lineal (DLT): una primera estimación sin necesidad de punto de partida.
    2. Refinamiento por Gauss-Newton en inversa de profundidad: el mejor punto en
       el sentido de mínimos cuadrados del error de reproyección.
    3. Comprobaciones de fiabilidad: paralaje, profundidad y error de reproyección.
"""

import numpy as np


# =====================================================================
# Paso 9.1 — Proyección y triangulación lineal (DLT)
# =====================================================================
def transformar(R_c_w, t_c_w, p_w):
    """Punto del mundo -> ejes de la cámara. p_w: (3,). Devuelve p_c (3,) [m]."""
    return R_c_w @ p_w + t_c_w


def proyectar_normalizado(R_c_w, t_c_w, p_w):
    """Punto del mundo -> coordenadas normalizadas en esa cámara.

    Devuelve: (xy (2,), z) con z la profundidad en esa cámara [m] (negativa = detrás).
    """
    p_c = transformar(R_c_w, t_c_w, p_w)
    return p_c[:2] / p_c[2], p_c[2]


def triangular_lineal(poses, xy):
    """Triangulación lineal por DLT (Direct Linear Transform).

    poses: lista de N parejas (R_c_w, t_c_w).
    xy: array (N, 2) de observaciones normalizadas.
    Devuelve: p_w (3,) [m], o None si el punto sale en el infinito.

    Cada observación (x, y) en una cámara con P = [R | t] (3x4) cumple
        x · (P[2] · X) - P[0] · X = 0
        y · (P[2] · X) - P[1] · X = 0
    con X = (X, Y, Z, W) el punto en coordenadas homogéneas: 2 ecuaciones lineales.
    """
    # 1. Apilar las 2 ecuaciones de cada observación en una matriz A (2N x 4)
    filas = []
    for (R, t), (x, y) in zip(poses, xy):
        P = np.hstack([R, t.reshape(3, 1)])        # matriz de proyección normalizada (3x4)
        filas.append(x * P[2] - P[0])
        filas.append(y * P[2] - P[1])
    A = np.array(filas)

    # 2. Resolver A · X = 0 con |X| = 1: el vector singular derecho del menor valor singular
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]

    # 3. De homogéneas a cartesianas (dividir por W). W ≈ 0 = punto en el infinito
    if abs(X[3]) < 1e-12 * np.linalg.norm(X[:3]):
        return None
    return X[:3] / X[3]


# =====================================================================
# Paso 9.2 — Refinamiento por Gauss-Newton en inversa de profundidad
# =====================================================================
def pose_relativa(R_a_w, t_a_w, R_b_w, t_b_w):
    """Pose de la cámara b respecto a la a: p_b = R_b_a · p_a + t_b_a.

    Sale de despejar p_w = R_a_wᵀ (p_a - t_a_w) y sustituir en p_b = R_b_w p_w + t_b_w.
    """
    R_b_a = R_b_w @ R_a_w.T
    t_b_a = t_b_w - R_b_a @ t_a_w
    return R_b_a, t_b_a


def a_inversa_profundidad(p_a):
    """Punto en ejes de la cámara ancla -> (alfa, beta, rho) = (X/Z, Y/Z, 1/Z)."""
    return np.array([p_a[0] / p_a[2], p_a[1] / p_a[2], 1.0 / p_a[2]])


def desde_inversa_profundidad(theta):
    """(alfa, beta, rho) -> punto en ejes de la cámara ancla: (alfa, beta, 1) / rho."""
    alfa, beta, rho = theta
    return np.array([alfa, beta, 1.0]) / rho


def prediccion_y_jacobiano(theta, poses_rel):
    """Predicción de cada observación y su jacobiano respecto a theta = (alfa, beta, rho).

    poses_rel: lista de N parejas (R_i_a, t_i_a): cada cámara respecto a la ancla.
    Devuelve: h (N, 2) predicciones normalizadas, J (2N, 3) = d h / d theta.

    En la cámara i el punto es p_i = R_i_a · (alfa, beta, 1) / rho + t_i_a. Multiplicar
    por rho no cambia su proyección, así que se trabaja con
        g = R_i_a · (alfa, beta, 1) + rho · t_i_a,      h = (g_x / g_z, g_y / g_z)
    y por la regla de la cadena  d h / d theta = (d h / d g) · (d g / d theta), con
        d g / d theta = [ R_i_a[:, 0] | R_i_a[:, 1] | t_i_a ]
        d h / d g     = [[1/g_z, 0, -g_x/g_z²], [0, 1/g_z, -g_y/g_z²]]
    Etapa 13: las N cámaras a la vez, sin bucle (la versión con bucle está en los tests).
    """
    alfa, beta, rho = theta
    R = np.array([pose[0] for pose in poses_rel])                 # (N, 3, 3)
    t = np.array([pose[1] for pose in poses_rel])                 # (N, 3)
    # 1. Predicción: fila i = R_i · (alfa, beta, 1) + rho · t_i
    g = R @ np.array([alfa, beta, 1.0]) + rho * t                 # (N, 3)
    h = g[:, :2] / g[:, 2:3]
    # 2. Jacobiano por la regla de la cadena, una matriz 2x3 por cámara
    n = len(g)
    dh_dg = np.zeros((n, 2, 3))
    dh_dg[:, 0, 0] = 1.0 / g[:, 2]
    dh_dg[:, 0, 2] = -g[:, 0] / g[:, 2] ** 2
    dh_dg[:, 1, 1] = 1.0 / g[:, 2]
    dh_dg[:, 1, 2] = -g[:, 1] / g[:, 2] ** 2
    dg_dtheta = np.stack([R[:, :, 0], R[:, :, 1], t], axis=2)    # columnas de cada (3, 3)
    J = (dh_dg @ dg_dtheta).reshape(2 * n, 3)                     # filas x0, y0, x1, y1...
    return h, J


def errores_reproyeccion_px(p_w, poses, xy, focal_px):
    """Distancia entre lo observado y la proyección de p_w en cada imagen [px].

    Etapa 13: todas las imágenes a la vez, sin bucle.
    """
    R = np.array([pose[0] for pose in poses])                     # (N, 3, 3)
    t = np.array([pose[1] for pose in poses])                     # (N, 3)
    p_c = R @ p_w + t                                             # (N, 3): el punto en cada cámara
    prediccion = p_c[:, :2] / p_c[:, 2:3]
    return np.linalg.norm(prediccion - np.asarray(xy), axis=1) * focal_px


def refinar(p_w, poses, xy, cfg_gn):
    """Gauss-Newton: el punto que minimiza la suma de errores de reproyección al cuadrado.

    p_w: (3,) punto inicial (el de triangular_lineal). La ancla es la primera pose.
    cfg_gn: sección "gauss_newton" de config/triangulacion.yaml.
    Devuelve: (p_w (3,), info) con info = {"iteraciones", "convergido"}.
    """
    # 1. Todo respecto a la cámara ancla y en inversa de profundidad
    R_a_w, t_a_w = poses[0]
    poses_rel = [pose_relativa(R_a_w, t_a_w, R, t) for R, t in poses]
    theta = a_inversa_profundidad(transformar(R_a_w, t_a_w, p_w))
    observado = np.asarray(xy, dtype=float).ravel()          # (2N,): x0, y0, x1, y1...

    convergido = False
    iteracion = 0
    for iteracion in range(1, cfg_gn["iteraciones_max"] + 1):
        # 2. Residuo r = observado - predicho, y jacobiano de la predicción
        h, J = prediccion_y_jacobiano(theta, poses_rel)
        r = observado - h.ravel()
        # 3. Ecuaciones normales: (Jᵀ J) · delta = Jᵀ r
        H = J.T @ J
        b = J.T @ r
        try:
            delta = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            break                                            # H singular: sin solución única
        # 4. Actualizar y comprobar si la corrección ya es despreciable
        theta = theta + delta
        if np.linalg.norm(delta) < cfg_gn["tolerancia"]:
            convergido = True
            break

    # 5. Volver al mundo: p_w = R_a_wᵀ (p_a - t_a_w)
    p_a = desde_inversa_profundidad(theta)
    p_w = R_a_w.T @ (p_a - t_a_w)
    return p_w, {"iteraciones": iteracion, "convergido": convergido}


# =====================================================================
# Paso 9.3 — Fiabilidad
# =====================================================================
def angulo_paralaje(poses, xy):
    """Mayor ángulo entre los rayos observados, expresados en el mundo [grados].

    Solo usa las observaciones y las orientaciones: se puede calcular ANTES de triangular.
    El rayo de la observación (x, y) en ejes de la cámara es (x, y, 1); en el mundo,
    R_c_wᵀ · (x, y, 1). Si la cámara solo ha girado, todos los rayos son paralelos: 0°.
    Etapa 13: todos los pares a la vez, sin bucles (era el 75 % del tiempo del MSCKF).
    """
    # 1. Rayos en el mundo, uno por fila: rayo_k = R_kᵀ · (x_k, y_k, 1)
    R = np.array([pose[0] for pose in poses])                     # (N, 3, 3)
    xy = np.asarray(xy, dtype=float)
    rayos_c = np.column_stack([xy, np.ones(len(xy))])             # (N, 3)
    rayos = np.einsum("kji,kj->ki", R, rayos_c)                   # (N, 3)
    rayos = rayos / np.linalg.norm(rayos, axis=1, keepdims=True)
    # 2. Ángulo de cada par (i, j) con atan2(|a x b|, a · b): matrices (N, N)
    senos = np.linalg.norm(np.cross(rayos[:, None, :], rayos[None, :, :]), axis=2)
    cosenos = rayos @ rayos.T
    return float(np.degrees(np.arctan2(senos, cosenos).max()))


def es_fiable(p_w, poses, xy, cfg_fia, focal_px):
    """¿Se puede usar este punto? Devuelve (bool, motivo)."""
    # 1. Suficiente ángulo entre los rayos
    if angulo_paralaje(poses, xy) < cfg_fia["paralaje_min_grados"]:
        return False, "paralaje insuficiente"
    # 2. Delante de todas las cámaras
    for R, t in poses:
        if transformar(R, t, p_w)[2] < cfg_fia["profundidad_min_m"]:
            return False, "detrás de alguna cámara"
    # 3. Su proyección coincide con lo observado en todas las imágenes
    if errores_reproyeccion_px(p_w, poses, xy, focal_px).max() > cfg_fia["error_max_px"]:
        return False, "error de reproyección alto"
    return True, "ok"


def triangular(poses, xy, cfg, focal_px):
    """Triangulación completa: comprobaciones baratas, DLT, Gauss-Newton y fiabilidad.

    cfg: dict de config/triangulacion.yaml.
    Devuelve: dict con p_w (o None), fiable, motivo, paralaje_grados, error_max_px, iteraciones.
    """
    fia = cfg["fiabilidad"]
    xy = np.asarray(xy, dtype=float)
    resultado = {"p_w": None, "fiable": False, "motivo": "", "paralaje_grados": 0.0,
                 "error_max_px": np.inf, "iteraciones": 0}

    # 1. Comprobaciones previas: no merece la pena triangular sin ellas
    if len(poses) < fia["observaciones_min"]:
        resultado["motivo"] = "pocas observaciones"
        return resultado
    resultado["paralaje_grados"] = angulo_paralaje(poses, xy)
    if resultado["paralaje_grados"] < fia["paralaje_min_grados"]:
        resultado["motivo"] = "paralaje insuficiente"
        return resultado

    # 2. Estimación lineal
    p_w = triangular_lineal(poses, xy)
    if p_w is None:
        resultado["motivo"] = "punto en el infinito"
        return resultado
    profundidades = [transformar(R, t, p_w)[2] for R, t in poses]
    if min(profundidades) < fia["profundidad_min_m"]:
        resultado["p_w"] = p_w
        resultado["motivo"] = "detrás de alguna cámara"
        return resultado

    # 3. Refinamiento y comprobación final
    p_w, info = refinar(p_w, poses, xy, cfg["gauss_newton"])
    resultado["p_w"] = p_w
    resultado["iteraciones"] = info["iteraciones"]
    resultado["error_max_px"] = errores_reproyeccion_px(p_w, poses, xy, focal_px).max()
    resultado["fiable"], resultado["motivo"] = es_fiable(p_w, poses, xy, fia, focal_px)
    return resultado