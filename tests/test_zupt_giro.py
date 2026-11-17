"""
tests/test_zupt_giro.py — Etapa 32: ZUPT de giro puro (vio/pipeline.es_giro_puro).

Escenas sintéticas: puntos 3D delante de la cámara, a profundidades distintas, vistos antes y
después de un movimiento conocido (giro, traslación o los dos). Así se sabe la respuesta correcta.

    python -m pytest tests/test_zupt_giro.py -v
"""

import cv2
import numpy as np

from vio import modelo_camara as mc
from vio import so3
from vio.config import cargar_config
from vio.pipeline import PipelineVio, es_giro_puro, giro_que_mejor_explica

CFGS = {nombre: cargar_config(nombre) for nombre in ("vio", "frontend", "msckf", "triangulacion")}
CFG = CFGS["vio"]["giro_puro"]
K = mc.matriz_K(621.0, 621.0, 320.0, 240.0)
SIN_DIST = np.zeros(5)
CON_DIST = np.array([-0.30, 0.10, 0.001, -0.001, 0.0])

# 3° de giro en una imagen (60 °/s a 20 fps): unos 30 px en la imagen
R_GIRO = so3.exp_so3(np.radians([1.0, 2.8, 0.5]))


def escena(n=120, z_min=1.0, z_max=4.0, semilla=0):
    """Puntos 3D (n, 3) en ejes de la cámara anterior, delante y dentro del campo de visión."""
    rng = np.random.default_rng(semilla)
    z = rng.uniform(z_min, z_max, n)
    x = rng.uniform(-0.35, 0.35, n) * z
    y = rng.uniform(-0.25, 0.25, n) * z
    return np.column_stack([x, y, z])


def proyectar(P, dist):
    uv, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return uv.reshape(-1, 2)


def ver_antes_y_despues(P_ant, R_act_ant, t_act_ant, dist=SIN_DIST):
    """Píxeles de los puntos antes y después de que la cámara se mueva:
    p_act = R_act_ant · p_ant + t_act_ant (ejes de la cámara actual)."""
    P_act = P_ant @ R_act_ant.T + t_act_ant
    return proyectar(P_ant, dist), proyectar(P_act, dist)


def test_wahba_recupera_el_giro():
    P = escena()
    d_ant = P / np.linalg.norm(P, axis=1, keepdims=True)
    d_act = d_ant @ R_GIRO.T
    R = giro_que_mejor_explica(d_ant, d_act)
    assert np.degrees(np.linalg.norm(so3.log_so3(R @ R_GIRO.T))) < 1e-6
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_giro_puro_se_detecta():
    uv_ant, uv_act = ver_antes_y_despues(escena(), R_GIRO, np.zeros(3))
    solo_gira, info = es_giro_puro(uv_ant, uv_act, R_GIRO, K, SIN_DIST, CFG)
    assert solo_gira
    assert info["giro_px"] > 20.0
    assert info["resto_px"] < 0.01 and info["diferencia_grados"] < 0.01


def test_giro_puro_con_distorsion_y_ruido():
    rng = np.random.default_rng(4)
    uv_ant, uv_act = ver_antes_y_despues(escena(semilla=4), R_GIRO, np.zeros(3), CON_DIST)
    uv_act = uv_act + rng.normal(scale=0.3, size=uv_act.shape)        # 0,3 px de ruido del KLT
    solo_gira, info = es_giro_puro(uv_ant, uv_act, R_GIRO, K, CON_DIST, CFG)
    assert solo_gira
    assert info["resto_px"] < 0.6


def test_giroscopio_algo_equivocado_sigue_valiendo():
    # 0,5° de error del giroscopio (p. ej. 6 ms de error de t_d a 80 °/s): el resto se mide con
    # el giro visto, así que no cuenta como traslación; la diferencia (0,5°) queda bajo el máximo
    uv_ant, uv_act = ver_antes_y_despues(escena(), R_GIRO, np.zeros(3))
    R_medido = so3.exp_so3(np.radians([0.3, -0.4, 0.0])) @ R_GIRO
    solo_gira, info = es_giro_puro(uv_ant, uv_act, R_medido, K, SIN_DIST, CFG)
    assert solo_gira
    assert 0.4 < info["diferencia_grados"] < 0.6


def test_giro_con_traslacion_no_es_giro_puro():
    # 3 cm en una imagen (0,6 m/s) con profundidades de 1 a 4 m: los cercanos se mueven más
    # que los lejanos y ningún giro lo explica
    uv_ant, uv_act = ver_antes_y_despues(escena(), R_GIRO, np.array([0.03, 0.0, 0.0]))
    solo_gira, info = es_giro_puro(uv_ant, uv_act, R_GIRO, K, SIN_DIST, CFG)
    assert not solo_gira
    assert info["resto_px"] > CFG["resto_max_px"]


def test_traslacion_disfrazada_de_giro_la_delata_el_giroscopio():
    # Pared a profundidad uniforme (1,5 m) y 5 cm de traslación lateral en una imagen (1 m/s):
    # visto en la imagen, es casi igual que girar ~1,9° más. El resto sale pequeño, pero el giro
    # visto ya no coincide con el del giroscopio
    uv_ant, uv_act = ver_antes_y_despues(escena(z_min=1.5, z_max=1.5), R_GIRO, np.array([0.05, 0.0, 0.0]))
    solo_gira, info = es_giro_puro(uv_ant, uv_act, R_GIRO, K, SIN_DIST, CFG)
    assert info["resto_px"] < CFG["resto_max_px"]                  # la imagen sola no lo distingue
    assert info["diferencia_grados"] > CFG["diferencia_max_grados"]
    assert not solo_gira


def test_sin_giro_apreciable_no_decide():
    R_poco = so3.exp_so3(np.radians([0.0, 0.1, 0.0]))                # 0,1°: ~1 px
    uv_ant, uv_act = ver_antes_y_despues(escena(), R_poco, np.zeros(3))
    solo_gira, info = es_giro_puro(uv_ant, uv_act, R_poco, K, SIN_DIST, CFG)
    assert info["giro_px"] < CFG["giro_min_px"]
    assert not solo_gira


def test_con_pocos_puntos_no_decide():
    uv_ant, uv_act = ver_antes_y_despues(escena(n=CFG["puntos_min"] - 1), R_GIRO, np.zeros(3))
    solo_gira, _ = es_giro_puro(uv_ant, uv_act, R_GIRO, K, SIN_DIST, CFG)
    assert not solo_gira


# ---------------------------------------------------------------------------------------------
# El método del pipeline: compara con la imagen de hace imagenes_ventana imágenes
# ---------------------------------------------------------------------------------------------
def pipeline_con_historial(P_viejo, R_cam, t_cam):
    """Un PipelineVio con el historial lleno, como si la cámara se hubiera movido
    p_act = R_cam·p_viejo + t_cam desde la imagen más antigua del historial hasta ahora.
    Devuelve (pipeline, R_w_i_act) listos para llamar a solo_gira."""
    ext = cargar_config("camara_imu")
    R_c_i = np.array(ext["R_i_c"], dtype=float).T
    pipe = PipelineVio(K, SIN_DIST, R_c_i, np.zeros(3), 0, CFGS)
    uv_viejo, uv_act = ver_antes_y_despues(P_viejo, R_cam, t_cam)
    # Orientaciones de la IMU coherentes con ese giro de la cámara:
    # R_cam = R_c_i·R_w_i_actᵀ·R_w_i_viejo·R_c_iᵀ  con R_w_i_viejo = I
    R_w_i_viejo = np.eye(3)
    R_w_i_act = (R_c_i.T @ R_cam @ R_c_i).T
    for _ in range(CFG["imagenes_ventana"]):                        # solo cuenta la más antigua
        pipe.historial.append(({k: uv_viejo[k] for k in range(len(uv_viejo))}, R_w_i_viejo))
    pipe.frontend.ids = np.arange(len(uv_act))
    pipe.frontend.puntos = uv_act.astype(np.float32)
    return pipe, R_w_i_act


def test_pipeline_detecta_un_giro_puro_en_la_ventana():
    R_cam = so3.exp_so3(np.radians([2.0, 12.0, 1.0]))               # 12° en 0,25 s ≈ 50 °/s
    pipe, R_w_i_act = pipeline_con_historial(escena(), R_cam, np.zeros(3))
    solo_gira, info = pipe.solo_gira(R_w_i_act)
    assert solo_gira, info
    assert info["resto_px"] < 0.05 and info["diferencia_grados"] < 0.05


def test_pipeline_sin_historial_lleno_no_decide():
    pipe, R_w_i_act = pipeline_con_historial(escena(), R_GIRO, np.zeros(3))
    pipe.historial.popleft()
    assert not pipe.solo_gira(R_w_i_act)[0]


def test_la_ventana_evita_confundir_una_traslacion_lenta_con_un_giro():
    # Escena lejos (3 a 6 m), la cámara gira ~40 °/s y se traslada a 0,3 m/s en lateral.
    # En UNA imagen (50 ms, 1,5 cm) la traslación se confunde con un giro de unas décimas de
    # grado y pasaría por giro puro; en la ventana (0,25 s, 7,5 cm) ya no
    P = escena(z_min=3.0, z_max=6.0, semilla=7)
    v = np.array([0.3, 0.0, 0.0])                                    # m/s, en ejes de la cámara
    w = np.radians([0.0, 40.0, 0.0])                                 # rad/s
    dt_imagen = 0.05
    R_una = so3.exp_so3(w * dt_imagen)
    uv_ant, uv_act = ver_antes_y_despues(P, R_una, v * dt_imagen)
    assert es_giro_puro(uv_ant, uv_act, R_una, K, SIN_DIST, dict(CFG, giro_min_px=3.0))[0]

    T = CFG["imagenes_ventana"] * dt_imagen
    pipe, R_w_i_act = pipeline_con_historial(P, so3.exp_so3(w * T), v * T)
    solo_gira, info = pipe.solo_gira(R_w_i_act)
    assert not solo_gira, info