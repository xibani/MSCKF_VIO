"""tests/test_modelo_camara.py — Tests de vio/modelo_camara.py."""

import cv2
import numpy as np
import pytest

from vio import modelo_camara as mc

K = mc.matriz_K(600.0, 605.0, 322.0, 238.0)
DIST = np.array([-0.25, 0.08, 0.001, -0.0007, -0.01])     # parecida a la de una lente barata


def puntos_delante(n, semilla=0):
    """n puntos 3D al azar delante de la cámara, dentro del campo de visión."""
    rng = np.random.default_rng(semilla)
    z = rng.uniform(0.5, 3.0, n)
    x = rng.uniform(-0.45, 0.45, n) * z
    y = rng.uniform(-0.35, 0.35, n) * z
    return np.column_stack([x, y, z])


def test_proyectar_sin_distorsion_a_mano():
    # Un punto en el eje óptico cae en el centro óptico (cx, cy)
    uv = mc.proyectar(np.array([[0.0, 0.0, 2.0]]), K, np.zeros(5))
    assert uv[0] == pytest.approx([322.0, 238.0])
    # X/Z = 0.1 -> u = fx · 0.1 + cx
    uv = mc.proyectar(np.array([[0.2, 0.0, 2.0]]), K, np.zeros(5))
    assert uv[0] == pytest.approx([600.0 * 0.1 + 322.0, 238.0])


def test_proyectar_igual_que_opencv():
    p = puntos_delante(500)
    uv_cv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, DIST)
    assert mc.proyectar(p, K, DIST) == pytest.approx(uv_cv.reshape(-1, 2), abs=1e-9)


def test_normalizar_invierte_a_proyectar():
    p = puntos_delante(500, semilla=1)
    uv = mc.proyectar(p, K, DIST)
    xy = mc.normalizar(uv, K, DIST)
    # La inversa es iterativa: precisión de ~1e-6 (unos 0,001 px), de sobra para el VIO
    assert xy == pytest.approx(p[:, :2] / p[:, 2:3], abs=1e-5)


def test_guardar_y_cargar(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    monkeypatch.setattr(mc, "RAIZ_PROYECTO", tmp_path)
    mc.guardar_intrinsecos(tmp_path / "config" / "prueba.yaml", K, DIST, 640, 480, {"rms_px": 0.2})
    K2, dist2, tamano = mc.cargar_intrinsecos("prueba")
    assert np.array_equal(K2, K)
    assert np.array_equal(dist2, DIST)
    assert tamano == (640, 480)