"""tests/test_so3.py — Tests de vio/so3.py."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from vio import so3

TOL = 1e-9


def vectores_aleatorios(n, angulo_max, semilla=0):
    """n vectores de rotación con dirección al azar y ángulo entre 0 y angulo_max."""
    rng = np.random.default_rng(semilla)
    ejes = rng.normal(size=(n, 3))
    ejes = ejes / np.linalg.norm(ejes, axis=1, keepdims=True)    # longitud 1
    angulos = rng.uniform(0.0, angulo_max, size=(n, 1))
    return ejes * angulos


# =====================================================================
# Paso 4.1
# =====================================================================
def test_es_rotacion():
    assert so3.es_rotacion(np.eye(3))
    espejo = np.diag([1.0, 1.0, -1.0])              # columnas perpendiculares, pero det = -1
    assert not so3.es_rotacion(espejo)
    estirada = np.diag([1.0, 1.0, 2.0])             # una columna de longitud 2
    assert not so3.es_rotacion(estirada)
    assert not so3.es_rotacion(np.eye(2))           # forma incorrecta


def test_hat_es_el_producto_vectorial():
    rng = np.random.default_rng(1)
    for _ in range(100):
        a, b = rng.normal(size=3), rng.normal(size=3)
        assert so3.hat(a) @ b == pytest.approx(np.cross(a, b))


def test_hat_es_antisimetrica_y_vee_la_invierte():
    v = np.array([1.0, -2.0, 3.0])
    M = so3.hat(v)
    assert np.array_equal(M.T, -M)                  # antisimétrica: M^T = -M
    assert np.array_equal(so3.vee(M), v)