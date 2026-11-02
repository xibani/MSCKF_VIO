"""
vio/so3.py — Rotaciones en 3D con matrices de rotación.

Convención del proyecto: R_a_b es la rotación que pasa un vector expresado
en el marco b a expresarlo en el marco a:   v_a = R_a_b @ v_b.

Todas las funciones son puras y trabajan con arrays de numpy:
    vector: array de forma (3,)      matriz: array de forma (3, 3)
"""

import numpy as np

# Por debajo de este ángulo [rad] se usan aproximaciones de ángulo pequeño
# (desarrollo en serie), para no dividir por números casi cero
UMBRAL_ANGULO_PEQUENO = 1e-6

# A menos de esta distancia de pi [rad], log_so3 usa la fórmula especial
UMBRAL_CERCA_DE_PI = 1e-3


# =====================================================================
# Paso 4.1 — Qué es una rotación; producto vectorial como matriz
# =====================================================================
def es_rotacion(R, tolerancia=1e-9):
    """¿Es R una matriz de rotación? (3x3, R^T R = I, det(R) = +1)

    R: array (3, 3).  tolerancia: margen para los errores de redondeo.
    Devuelve: bool.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    # 1. Columnas de longitud 1 y perpendiculares entre sí: R^T R = I
    error_ortogonal = np.abs(R.T @ R - np.eye(3)).max()
    # 2. Determinante +1 (con -1 sería un espejo, no una rotación)
    error_det = abs(np.linalg.det(R) - 1.0)
    return error_ortogonal < tolerancia and error_det < tolerancia


def hat(v):
    """Matriz antisimétrica [v]x: la que cumple hat(v) @ u == v x u (producto vectorial).

    v: array (3,).  Devuelve: array (3, 3).
    """
    x, y, z = v
    return np.array([[0.0,  -z,   y],
                     [  z, 0.0,  -x],
                     [ -y,   x, 0.0]])


def vee(M):
    """Operación inversa de hat: de matriz antisimétrica (3, 3) a vector (3,)."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


# =====================================================================
# Paso 4.2 — Exponencial: de vector de rotación a matriz
# =====================================================================
def exp_so3(phi):
    """Matriz de rotación del vector de rotación phi (fórmula de Rodrigues).

    phi: array (3,) [rad]. Dirección = eje de giro; longitud = ángulo.
    Devuelve: array (3, 3), giro de |phi| radianes alrededor de phi / |phi|.
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)          # ángulo [rad]
    Phi = hat(phi)
    Phi2 = Phi @ Phi

    # Coeficientes A = sen(t)/t y B = (1 - cos(t))/t^2
    if theta < UMBRAL_ANGULO_PEQUENO:
        # Desarrollo en serie (evita dividir entre casi 0)
        A = 1.0 - theta ** 2 / 6.0
        B = 0.5 - theta ** 2 / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta ** 2

    return np.eye(3) + A * Phi + B * Phi2


# =====================================================================
# Paso 4.3 — Logaritmo: de matriz a vector de rotación
# =====================================================================
def log_so3(R):
    """Vector de rotación de la matriz R (inversa de exp_so3).

    R: array (3, 3), matriz de rotación.
    Devuelve: array (3,) con longitud entre 0 y pi [rad].
    """
    R = np.asarray(R, dtype=float)

    # 1. Ángulo, a partir de su coseno y su seno:
    #    cos(t) = (traza(R) - 1) / 2       sen(t) = |vee(R - R^T)| / 2
    #    atan2 da el ángulo con precisión en todo el rango [0, pi]
    w = vee(R - R.T) / 2.0               # = sen(t) * eje
    cos_t = (np.trace(R) - 1.0) / 2.0
    sen_t = np.linalg.norm(w)
    theta = np.arctan2(sen_t, cos_t)

    # 2. Ángulo pequeño: theta / sen(theta) ~ 1 + theta^2 / 6
    if theta < UMBRAL_ANGULO_PEQUENO:
        return w * (1.0 + theta ** 2 / 6.0)

    # 3. Caso general: phi = theta * eje = theta / sen(theta) * w
    if theta < np.pi - UMBRAL_CERCA_DE_PI:
        return w * (theta / sen_t)

    # 4. Cerca de pi, sen(theta) ~ 0 y w no sirve para hallar el eje.
    #    Se usa la parte simétrica: (R + R^T)/2 = I + (1 - cos t)(u u^T - I)
    #    =>  u u^T = I + ((R + R^T)/2 - I) / (1 - cos t)
    uuT = np.eye(3) + ((R + R.T) / 2.0 - np.eye(3)) / (1.0 - cos_t)
    k = int(np.argmax(np.diag(uuT)))     # la componente más grande del eje: la más precisa
    u = uuT[:, k] / np.sqrt(uuT[k, k])
    # El signo del eje lo fija w (= sen(t) * u, con sen(t) >= 0)
    if np.dot(u, w) < 0:
        u = -u
    return theta * u


# =====================================================================
# Paso 4.4 — Utilidades
# =====================================================================
def reortonormalizar(R):
    """Devuelve la matriz de rotación más cercana a R.

    Tras miles de multiplicaciones, el redondeo hace que R deje de ser una
    rotación exacta. La SVD (R = U S V^T) da la más cercana: U V^T.
    """
    U, S, Vt = np.linalg.svd(R)
    R_limpia = U @ Vt
    if np.linalg.det(R_limpia) < 0:      # sería un espejo: invertir el último eje de U
        U[:, -1] = -U[:, -1]
        R_limpia = U @ Vt
    return R_limpia


def angulo_entre(R_a, R_b):
    """Ángulo [rad] de la rotación que lleva de R_a a R_b. 0 = iguales."""
    return float(np.linalg.norm(log_so3(R_a.T @ R_b)))