"""tests/test_triangulacion.py — Tests de vio/triangulacion.py con cámaras y puntos sintéticos."""

import cv2
import numpy as np
import pytest

from vio import so3
from vio import triangulacion as tri
from vio.config import cargar_config

CFG = cargar_config("triangulacion")
FOCAL = 643.0                                   # focal provisional de la cámara [px]
P_VERDAD = np.array([0.3, -0.2, 5.0])           # punto a 5 m delante de las cámaras


def pose_desde_centro(c_w, phi):
    """Pose (R_c_w, t_c_w) de una cámara situada en c_w y girada phi [rad] respecto al mundo.

    Si la cámara está en c_w con orientación R_w_c, un punto p_w en ejes de la cámara es
    p_c = R_w_cᵀ (p_w - c_w), es decir R_c_w = R_w_cᵀ y t_c_w = -R_c_w · c_w.
    """
    R_c_w = so3.exp_so3(phi).T
    return R_c_w, -R_c_w @ np.asarray(c_w, dtype=float)


def camaras_en_linea(n=5, base_m=0.5, semilla=0):
    """n cámaras repartidas a lo largo de x en base_m metros, mirando hacia +z con giros pequeños."""
    rng = np.random.default_rng(semilla)
    poses = []
    for x in np.linspace(-base_m / 2, base_m / 2, n):
        poses.append(pose_desde_centro([x, 0.0, 0.0], rng.normal(0.0, 0.03, 3)))
    return poses


def observar(poses, p_w, ruido_px=0.0, rng=None):
    """Observaciones normalizadas de p_w en cada cámara, con ruido gaussiano opcional [px]."""
    xy = np.array([tri.proyectar_normalizado(R, t, p_w)[0] for R, t in poses])
    if ruido_px > 0:
        xy = xy + rng.normal(0.0, ruido_px / FOCAL, xy.shape)
    return xy


# =====================================================================
# Paso 9.1
# =====================================================================
def test_proyectar_normalizado():
    R, t = pose_desde_centro([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])      # cámara 1 m a la derecha
    xy, z = tri.proyectar_normalizado(R, t, np.array([1.5, 0.5, 2.0]))
    assert z == pytest.approx(2.0)
    assert xy == pytest.approx([0.25, 0.25])


def test_lineal_exacto():
    poses = camaras_en_linea()
    p = tri.triangular_lineal(poses, observar(poses, P_VERDAD))
    assert np.linalg.norm(p - P_VERDAD) < 1e-9


def test_lineal_igual_que_opencv():
    rng = np.random.default_rng(1)
    poses = camaras_en_linea(n=2)
    xy = observar(poses, P_VERDAD, ruido_px=1.0, rng=rng)
    p = tri.triangular_lineal(poses, xy)
    P1, P2 = (np.hstack([R, t.reshape(3, 1)]) for R, t in poses)
    X = cv2.triangulatePoints(P1, P2, xy[0].reshape(2, 1), xy[1].reshape(2, 1)).ravel()
    assert np.linalg.norm(p - X[:3] / X[3]) < 1e-9
    assert np.linalg.norm(p - P_VERDAD) > 1e-3                      # con ruido NO sale la verdad