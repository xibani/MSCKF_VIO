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


# =====================================================================
# Paso 4.2
# =====================================================================
def test_exp_de_cero_es_la_identidad():
    assert np.array_equal(so3.exp_so3(np.zeros(3)), np.eye(3))


def test_exp_90_grados_sobre_z():
    R = so3.exp_so3([0.0, 0.0, np.pi / 2])
    # Regla de la mano derecha: girando sobre z, x pasa a y, e y pasa a -x
    assert R @ [1, 0, 0] == pytest.approx([0, 1, 0], abs=TOL)
    assert R @ [0, 1, 0] == pytest.approx([-1, 0, 0], abs=TOL)
    assert R @ [0, 0, 1] == pytest.approx([0, 0, 1], abs=TOL)   # el eje no se mueve


def test_exp_da_siempre_rotaciones():
    for phi in vectores_aleatorios(1000, 3 * np.pi):
        assert so3.es_rotacion(so3.exp_so3(phi))


def test_exp_igual_que_scipy():
    for phi in vectores_aleatorios(1000, np.pi):
        R_scipy = Rotation.from_rotvec(phi).as_matrix()
        assert so3.exp_so3(phi) == pytest.approx(R_scipy, abs=TOL)


def test_exp_igual_que_la_exponencial_de_matrices():
    # El nombre "exp" viene de aquí: exp_so3(phi) = e^{hat(phi)} = I + A + A²/2! + ...
    for phi in vectores_aleatorios(100, np.pi):
        assert so3.exp_so3(phi) == pytest.approx(expm(so3.hat(phi)), abs=TOL)


def test_exp_inversa_y_composicion_sobre_el_mismo_eje():
    eje = np.array([1.0, 2.0, 2.0]) / 3.0           # longitud 1
    R = so3.exp_so3(0.7 * eje)
    assert R @ so3.exp_so3(-0.7 * eje) == pytest.approx(np.eye(3), abs=TOL)   # girar y deshacer
    assert R.T == pytest.approx(so3.exp_so3(-0.7 * eje), abs=TOL)             # inversa = traspuesta
    assert so3.exp_so3(0.3 * eje) @ so3.exp_so3(0.4 * eje) == pytest.approx(R, abs=TOL)


def test_exp_angulo_pequeno():
    phi = np.array([1e-9, -2e-9, 3e-9])
    R = so3.exp_so3(phi)
    assert so3.es_rotacion(R)
    assert R == pytest.approx(np.eye(3) + so3.hat(phi), abs=1e-15)