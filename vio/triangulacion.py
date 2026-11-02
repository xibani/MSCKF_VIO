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