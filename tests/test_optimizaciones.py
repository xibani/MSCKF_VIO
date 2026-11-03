"""tests/test_optimizaciones.py — Etapa 13: cada función optimizada da lo mismo que la original.

Las versiones "de referencia" son las de las etapas 9 y 12, con bucles, copiadas aquí tal
cual. Si una optimización cambia un resultado, estos tests lo detectan.
"""

import numpy as np
import pytest
from scipy.stats import chi2

from vio import msckf as ms
from vio import so3
from vio import triangulacion as tri


# =====================================================================
# Versiones de referencia (las originales, con bucles)
# =====================================================================
def angulo_paralaje_referencia(poses, xy):
    rayos = []
    for (R, _), (x, y) in zip(poses, xy):
        d = R.T @ np.array([x, y, 1.0])
        rayos.append(d / np.linalg.norm(d))
    maximo = 0.0
    for i in range(len(rayos)):
        for j in range(i + 1, len(rayos)):
            angulo = np.arctan2(np.linalg.norm(np.cross(rayos[i], rayos[j])), rayos[i] @ rayos[j])
            maximo = max(maximo, angulo)
    return np.degrees(maximo)


def errores_reproyeccion_referencia(p_w, poses, xy, focal_px):
    errores = []
    for (R, t), obs in zip(poses, xy):
        prediccion, _ = tri.proyectar_normalizado(R, t, p_w)
        errores.append(np.linalg.norm(prediccion - obs) * focal_px)
    return np.array(errores)


def prediccion_y_jacobiano_referencia(theta, poses_rel):
    alfa, beta, rho = theta
    n = len(poses_rel)
    h = np.zeros((n, 2))
    J = np.zeros((2 * n, 3))
    for i, (R, t) in enumerate(poses_rel):
        g = R @ np.array([alfa, beta, 1.0]) + rho * t
        h[i] = g[:2] / g[2]
        dh_dg = np.array([[1.0 / g[2], 0.0, -g[0] / g[2] ** 2],
                          [0.0, 1.0 / g[2], -g[1] / g[2] ** 2]])
        J[2 * i:2 * i + 2] = dh_dg @ np.column_stack([R[:, 0], R[:, 1], t])
    return h, J


def residuo_pista_referencia(poses_clones, posiciones, xy, p_w, dim):
    n = len(xy)
    r, H_x, H_f = np.zeros(2 * n), np.zeros((2 * n, dim)), np.zeros((2 * n, 3))
    for k, ((R_w_c, p_w_c), posicion) in enumerate(zip(poses_clones, posiciones)):
        h, H_clon, H_punto = ms.jacobianos_observacion(R_w_c, p_w_c, p_w)
        r[2 * k:2 * k + 2] = xy[k] - h
        columna = ms.DIM_IMU + ms.DIM_CLON * posicion
        H_x[2 * k:2 * k + 2, columna:columna + ms.DIM_CLON] = H_clon
        H_f[2 * k:2 * k + 2] = H_punto
    return r, H_x, H_f


def caso_al_azar(n, semilla):
    """n cámaras al azar mirando hacia un punto a ~4 m, y sus observaciones."""
    rng = np.random.default_rng(semilla)
    p_w = np.array([0.5, -0.3, 4.0])
    poses = []
    for _ in range(n):
        R_c_w = so3.exp_so3(rng.normal(0.0, 0.1, 3))
        c_w = rng.uniform(-0.5, 0.5, 3)
        poses.append((R_c_w, -R_c_w @ c_w))
    xy = np.array([tri.proyectar_normalizado(R, t, p_w)[0] for R, t in poses])
    return poses, xy + rng.normal(0.0, 0.002, xy.shape), p_w


# =====================================================================
# Paso 13.1
# =====================================================================
@pytest.mark.parametrize("n", [2, 5, 15])
def test_angulo_paralaje_igual_que_la_referencia(n):
    poses, xy, _ = caso_al_azar(n, n)
    assert tri.angulo_paralaje(poses, xy) == pytest.approx(angulo_paralaje_referencia(poses, xy), abs=1e-10)


def test_errores_reproyeccion_igual_que_la_referencia():
    poses, xy, p_w = caso_al_azar(12, 1)
    assert tri.errores_reproyeccion_px(p_w, poses, xy, 643.0) == pytest.approx(
        errores_reproyeccion_referencia(p_w, poses, xy, 643.0), abs=1e-10)


def test_prediccion_y_jacobiano_igual_que_la_referencia():
    poses, _, _ = caso_al_azar(10, 2)
    poses_rel = [tri.pose_relativa(*poses[0], R, t) for R, t in poses]
    theta = np.array([0.1, -0.05, 0.25])
    h, J = tri.prediccion_y_jacobiano(theta, poses_rel)
    h_ref, J_ref = prediccion_y_jacobiano_referencia(theta, poses_rel)
    assert h == pytest.approx(h_ref, abs=1e-14) and J == pytest.approx(J_ref, abs=1e-14)


def test_residuo_pista_igual_que_la_referencia():
    rng = np.random.default_rng(3)
    poses_clones = [(so3.exp_so3(rng.normal(0.0, 0.1, 3)), rng.uniform(-0.5, 0.5, 3)) for _ in range(7)]
    p_w = np.array([0.4, 0.2, 4.0])
    xy = rng.normal(0.0, 0.1, (7, 2))
    posiciones = [2, 3, 4, 5, 6, 7, 8]
    dim = 15 + 6 * 10
    for a, b in zip(ms.residuo_pista(poses_clones, posiciones, xy, p_w, dim),
                    residuo_pista_referencia(poses_clones, posiciones, xy, p_w, dim)):
        assert a == pytest.approx(b, abs=1e-14)


def test_compresion_qr_da_la_misma_correccion():
    rng = np.random.default_rng(4)
    A = rng.normal(0.0, 1.0, (45, 45))
    P = A @ A.T / 45 + 0.1 * np.eye(45)
    H = rng.normal(0.0, 1.0, (200, 45))                      # muchas más filas que columnas
    r = rng.normal(0.0, 1.0, 200)
    H_c, r_c = ms.comprimir_qr(H, r)
    assert H_c.shape == (45, 45) and r_c.shape == (45,)
    dx, P2 = ms.correccion_kalman(P, H, r, 0.3)
    dx_c, P2_c = ms.correccion_kalman(P, H_c, r_c, 0.3)
    assert dx_c == pytest.approx(dx, abs=1e-10)
    assert P2_c == pytest.approx(P2, abs=1e-10)
    # Con menos filas que columnas no hace nada
    H_pocas, r_pocas = ms.comprimir_qr(H[:10], r[:10])
    assert np.array_equal(H_pocas, H[:10])
    assert np.array_equal(r_pocas, r[:10])


def test_umbral_chi2_recordado():
    assert ms.umbral_chi2(0.95, 17) == chi2.ppf(0.95, 17)
    assert (0.95, 17) in ms._UMBRALES_CHI2                    # la segunda vez no se recalcula