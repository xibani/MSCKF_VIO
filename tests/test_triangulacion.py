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
    

# =====================================================================
# Paso 9.2
# =====================================================================
def test_pose_relativa():
    poses = camaras_en_linea()
    (R_a, t_a), (R_b, t_b) = poses[0], poses[3]
    R_b_a, t_b_a = tri.pose_relativa(R_a, t_a, R_b, t_b)
    p_a = tri.transformar(R_a, t_a, P_VERDAD)
    assert R_b_a @ p_a + t_b_a == pytest.approx(tri.transformar(R_b, t_b, P_VERDAD), abs=1e-12)


def test_inversa_profundidad_ida_y_vuelta():
    theta = tri.a_inversa_profundidad(P_VERDAD)
    assert theta == pytest.approx([0.06, -0.04, 0.2])
    assert tri.desde_inversa_profundidad(theta) == pytest.approx(P_VERDAD, abs=1e-12)


def test_jacobiano_igual_que_diferencias_finitas():
    poses = camaras_en_linea()
    poses_rel = [tri.pose_relativa(*poses[0], R, t) for R, t in poses]
    theta = np.array([0.05, -0.03, 0.25])
    _, J = tri.prediccion_y_jacobiano(theta, poses_rel)

    # Diferencias centradas: (h(theta + e) - h(theta - e)) / (2 e), una columna por incógnita
    e = 1e-6
    J_num = np.zeros_like(J)
    for k in range(3):
        d = np.zeros(3)
        d[k] = e
        h_mas, _ = tri.prediccion_y_jacobiano(theta + d, poses_rel)
        h_menos, _ = tri.prediccion_y_jacobiano(theta - d, poses_rel)
        J_num[:, k] = (h_mas.ravel() - h_menos.ravel()) / (2 * e)
    assert np.abs(J - J_num).max() < 1e-8


def test_refinar_converge_desde_lejos():
    poses = camaras_en_linea()
    xy = observar(poses, P_VERDAD)
    inicio = P_VERDAD + np.array([0.5, -0.3, 1.5])                  # 1,6 m del punto verdadero
    p, info = tri.refinar(inicio, poses, xy, CFG["gauss_newton"])
    assert info["convergido"] and info["iteraciones"] <= 10
    assert np.linalg.norm(p - P_VERDAD) < 1e-9


def test_refinar_nunca_empeora_el_error_de_reproyeccion():
    # Cámara avanzando hacia el punto: las profundidades cambian mucho de una imagen a otra
    rng = np.random.default_rng(5)
    poses = [pose_desde_centro([0.05 * k, 0.02 * k, 0.6 * k], rng.normal(0.0, 0.03, 3)) for k in range(5)]
    punto = np.array([1.0, -0.6, 6.0])
    for _ in range(200):
        xy = observar(poses, punto, ruido_px=1.0, rng=rng)
        p_lineal = tri.triangular_lineal(poses, xy)
        p_gn, _ = tri.refinar(p_lineal, poses, xy, CFG["gauss_newton"])
        coste_lineal = np.sum(tri.errores_reproyeccion_px(p_lineal, poses, xy, FOCAL) ** 2)
        coste_gn = np.sum(tri.errores_reproyeccion_px(p_gn, poses, xy, FOCAL) ** 2)
        assert coste_gn <= coste_lineal + 1e-9


def test_error_de_profundidad_con_ruido_de_1px():
    # Repetir muchas veces con 1 px de ruido y medir la dispersión de la profundidad
    rng = np.random.default_rng(2)
    poses = camaras_en_linea()
    profundidades = []
    for _ in range(1000):
        xy = observar(poses, P_VERDAD, ruido_px=1.0, rng=rng)
        p, _ = tri.refinar(tri.triangular_lineal(poses, xy), poses, xy, CFG["gauss_newton"])
        profundidades.append(p[2])
    sigma_medida = np.std(profundidades)

    # Predicción: sigma_Z = Z² · sigma_px / (f · sqrt(Σ (b_k - b_media)²))
    b = np.linspace(-0.25, 0.25, 5)
    sigma_predicha = P_VERDAD[2] ** 2 * 1.0 / (FOCAL * np.sqrt(np.sum((b - b.mean()) ** 2)))
    assert sigma_medida == pytest.approx(sigma_predicha, rel=0.15)  # ≈ 10 cm
    assert abs(np.mean(profundidades) - P_VERDAD[2]) < 0.02         # sin sesgo apreciable
    

# =====================================================================
# Paso 9.3
# =====================================================================
def test_angulo_paralaje():
    poses = [pose_desde_centro([-0.25, 0, 0], [0, 0, 0]), pose_desde_centro([0.25, 0, 0], [0, 0, 0])]
    punto = np.array([0.0, 0.0, 5.0])
    esperado = np.degrees(2 * np.arctan(0.25 / 5.0))                # 5,72°
    assert tri.angulo_paralaje(poses, observar(poses, punto)) == pytest.approx(esperado, abs=1e-9)


def test_solo_giro_no_es_fiable():
    rng = np.random.default_rng(3)
    poses = [pose_desde_centro([0.1, 0.2, 0.0], rng.normal(0, 0.1, 3)) for _ in range(5)]
    xy = observar(poses, P_VERDAD)
    assert tri.angulo_paralaje(poses, xy) < 1e-6
    r = tri.triangular(poses, xy, CFG, FOCAL)
    assert not r["fiable"] and r["motivo"] == "paralaje insuficiente"


def test_punto_detras_rechazado():
    poses = camaras_en_linea()
    detras = np.array([0.3, -0.2, -5.0])
    r = tri.triangular(poses, observar(poses, detras), CFG, FOCAL)
    assert not r["fiable"] and r["motivo"] == "detrás de alguna cámara"


def test_observacion_erronea_rechazada():
    poses = camaras_en_linea()
    xy = observar(poses, P_VERDAD)
    xy[2, 0] += 10.0 / FOCAL                                        # un seguimiento erróneo de 10 px
    r = tri.triangular(poses, xy, CFG, FOCAL)
    assert not r["fiable"] and r["motivo"] == "error de reproyección alto"


def test_triangular_punto_bueno():
    # Con 1 px de ruido, casi todos los puntos deben aceptarse (el umbral es ≈ 3 sigma)
    rng = np.random.default_rng(4)
    poses = camaras_en_linea()
    resultados = [tri.triangular(poses, observar(poses, P_VERDAD, ruido_px=1.0, rng=rng), CFG, FOCAL)
                  for _ in range(500)]
    fiables = [r for r in resultados if r["fiable"]]
    assert len(fiables) / len(resultados) > 0.97
    assert all(r["motivo"] == "ok" and r["error_max_px"] <= CFG["fiabilidad"]["error_max_px"] for r in fiables)
    assert max(r["iteraciones"] for r in fiables) <= 5
    assert np.median([np.linalg.norm(r["p_w"] - P_VERDAD) for r in fiables]) < 0.15


def test_pocas_observaciones():
    poses = camaras_en_linea(n=1)
    r = tri.triangular(poses, observar(poses, P_VERDAD), CFG, FOCAL)
    assert not r["fiable"] and r["motivo"] == "pocas observaciones"