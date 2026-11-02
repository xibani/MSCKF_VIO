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