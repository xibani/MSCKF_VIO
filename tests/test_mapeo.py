"""tests/test_mapeo.py — Etapa 18: del MSCKF al mapa en el PC.

vio/msckf.py (puntos aceptados y clon saliente) -> vio/mapeo.py (keyframes de mapeo)
-> vio/pipeline.py -> vio/protocolo.py (paquetes v4 y v5, archivo) -> apps/pi_vio.py
-> vio/reconstruccion.py -> apps/pc_mapa.py
"""

import copy
import os
import signal
import socket
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

from vio import mapeo
from vio import msckf as ms
from vio import protocolo

# from vio import reconstruccion as rec
from vio import simulador as sim
from vio import so3
from vio.config import RAIZ_PROYECTO, cargar_config
from vio.estado import EstadoImu
from vio.pipeline import PipelineVio, leer_poses

K = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
CFG_MAPEO = cargar_config("vio")["mapeo"]
CFGS = {
    nombre: cargar_config(nombre)
    for nombre in ("vio", "frontend", "msckf", "triangulacion")
}
EXT = cargar_config("camara_imu")
R_C_I = np.array(EXT["R_i_c"], dtype=float).T
T_C_I = -R_C_I @ np.array(EXT["p_i_c_m"], dtype=float)


# =====================================================================
# Paso 17.1 — El MSCKF guarda puntos y entrega el clon saliente
# =====================================================================
@pytest.fixture(scope="module")
def simulacion():
    """El MSCKF sobre 5 s simulados (como en test_msckf), con los ids del simulador:
    el id de cada observación es el índice del punto verdadero en s["puntos"]."""
    cfg_sim = copy.deepcopy(cargar_config("simulador")["vio"])
    cfg_sim["trayectoria"]["duracion_s"] = 5.0
    cfg_sim["mundo"]["puntos_por_superficie"] = 150
    rng = np.random.default_rng(11)
    s = sim.generar_secuencia(cfg_sim, K, (640, 480), rng)
    v = s["verdad"]
    P0 = ms.covarianza_inicial(CFGS["msckf"]["inicial"])
    verdad_0 = EstadoImu(
        v["R_w_i"][0], v["v_w"][0], v["p_w"][0], v["b_g"][0], v["b_a"][0]
    )
    inicial = ms.inyectar_imu(
        verdad_0, -np.linalg.cholesky(P0) @ rng.normal(0.0, 1.0, 15)
    )
    filtro = ms.Msckf(
        inicial,
        P0,
        s["imu"]["t_ns"][0],
        s["imu"]["w"][0],
        s["imu"]["a"][0],
        s["R_c_i"],
        s["t_c_i"],
        K[0, 0],
        CFGS["msckf"],
        CFGS["triangulacion"],
    )
    res = ms.ejecutar_secuencia(filtro, s["imu"], s["obs"], s["camaras"]["t_ns"])
    return {"s": s, "filtro": filtro, "recuentos": res["recuentos"]}


def test_msckf_guarda_los_puntos_aceptados(simulacion):
    s, recuentos = simulacion["s"], simulacion["recuentos"]
    # Uno por pista aceptada: tantos como "usadas"
    assert sum(len(r["puntos_mapa"]) for r in recuentos) == sum(
        r["usadas"] for r in recuentos
    )
    puntos = [(i, p, a) for r in recuentos for i, p, a in r["puntos_mapa"]]
    assert len(puntos) > 200
    error = np.array([np.linalg.norm(p - s["puntos"][i]) for i, p, _ in puntos])
    assert np.median(error) < 0.2  # en el mundo: incluye el error de la pose
    assert (
        min(a for *_, a in puntos)
        >= CFGS["triangulacion"]["fiabilidad"]["paralaje_min_grados"]
    )


def test_msckf_entrega_cada_clon_al_salir(simulacion):
    s, recuentos, filtro = (
        simulacion["s"],
        simulacion["recuentos"],
        simulacion["filtro"],
    )
    salientes = [r["saliente"] for r in recuentos if r["saliente"] is not None]
    n_imagenes, M = len(s["camaras"]["t_ns"]), CFGS["msckf"]["ventana"]["clones_max"]
    # Cada clon sale una vez, en orden, empezando por la imagen 0; quedan M - 1 en la ventana
    assert [c["imagen"] for c in salientes] == list(range(n_imagenes - M + 1))
    assert all(
        k >= filtro.clones[0]["imagen"] for k in filtro.mapa_obs
    )  # nada viejo pendiente

    errores_relativos = []
    for c in salientes:
        vistos = set(s["obs"]["id"][s["obs"]["imagen"] == c["imagen"]])
        R_c_w, t_c_w = (
            s["camaras"]["R_c_w"][c["imagen"]],
            s["camaras"]["t_c_w"][c["imagen"]],
        )
        assert len(c["obs"]) > 10
        for id_punto, xy, z, paralaje in c["obs"]:
            assert id_punto in vistos  # solo puntos vistos en esa imagen
            z_verdad = (R_c_w @ s["puntos"][id_punto] + t_c_w)[2]
            errores_relativos.append(abs(z - z_verdad) / z_verdad)
        # Y su pose es la de esa imagen: a menos de 10 cm de la verdadera
        assert np.linalg.norm(c["p_w_c"] - (-R_c_w.T @ t_c_w)) < 0.1
    # La profundidad EN LA CÁMARA del keyframe es buena: no le afecta el error global de la pose
    assert np.median(errores_relativos) < 0.05


def test_filtrar_observaciones_y_puntos():
    cfg = dict(
        CFG_MAPEO,
        paralaje_min_grados=2.0,
        profundidad_min_m=0.1,
        profundidad_max_m=20.0,
    )
    obs = [
        (1, np.array([0.1, 0.2]), 2.0, 5.0),  # vale
        (2, np.array([0.0, 0.0]), 2.0, 1.5),  # poco paralaje
        (3, np.array([0.0, 0.0]), 0.05, 5.0),  # demasiado cerca
        (4, np.array([0.0, 0.0]), 30.0, 5.0),  # demasiado lejos
        (5, np.array([-0.3, 0.1]), 4.0, 2.0),
    ]  # vale (justo en el umbral)
    ids, xy, z = mapeo.filtrar_observaciones(obs, cfg)
    assert list(ids) == [1, 5] and ids.dtype == np.int64
    assert xy == pytest.approx(np.array([[0.1, 0.2], [-0.3, 0.1]])) and list(z) == [
        2.0,
        4.0,
    ]
    vacio = mapeo.filtrar_observaciones([], cfg)
    assert vacio[1].shape == (0, 2) and len(vacio[2]) == 0

    ids, p_w = mapeo.filtrar_puntos(
        [(7, np.array([1.0, 2.0, 3.0]), 3.0), (8, np.zeros(3), 1.0)], cfg
    )
    assert list(ids) == [7] and p_w == pytest.approx(np.array([[1.0, 2.0, 3.0]]))
    assert mapeo.filtrar_puntos([], cfg)[1].shape == (0, 3)


def test_selector_keyframes():
    cfg = dict(
        CFG_MAPEO, puntos_min=15, fraccion_profundidad=0.12, giro_min_grados=15.0
    )
    selector = mapeo.SelectorKeyframes(cfg)
    z = np.full(20, 2.5)  # umbral de distancia: 0,12·2,5 = 0,30 m
    R0, p0 = np.eye(3), np.zeros(3)
    assert selector.decidir(R0, p0, z[:10]) == (False, "pocos puntos con profundidad")
    assert selector.decidir(R0, p0, z)[0]  # el primero
    assert not selector.decidir(R0, np.array([0.29, 0.0, 0.0]), z)[0]  # poco
    assert selector.decidir(R0, np.array([0.31, 0.0, 0.0]), z)[0]  # 31 cm
    assert not selector.decidir(R0, np.array([0.31, 0.0, 0.0]), z)[
        0
    ]  # ahora la referencia es esa
    R_giro = so3.exp_so3(np.radians([0.0, 16.0, 0.0]))
    assert selector.decidir(R_giro, np.array([0.31, 0.0, 0.0]), z)[
        0
    ]  # girar 16° sin moverse
    # Con la escena más lejos, hace falta moverse más
    assert not selector.decidir(R_giro, np.array([0.70, 0.0, 0.0]), np.full(20, 5.0))[0]
    assert mapeo.angulo_entre(np.eye(3), R_giro) == pytest.approx(16.0)


def test_pipeline_entrega_keyframes(tmp_path):
    from apps.pi_vio import ADELANTO_IMU_NS, eventos_grabacion
    from herramientas.grabar_simulacion import guardar_grabacion_sintetica

    cfg_vio = copy.deepcopy(cargar_config("simulador")["vio"])
    cfg_vio["trayectoria"]["duracion_s"] = 8.0
    guardar_grabacion_sintetica(
        tmp_path, cfg_vio, K, (640, 480), EXT, np.random.default_rng(3)
    )

    pipeline = PipelineVio(K, DIST, R_C_I, T_C_I, EXT["t_d_ns"], CFGS)
    keyframes, n_puntos, max_buffer = [], 0, 0
    for evento in eventos_grabacion(tmp_path, ADELANTO_IMU_NS):
        if evento[0] == "imu":
            pipeline.nueva_imu(*evento[1:])
            continue
        r = pipeline.nueva_imagen(*evento[1:])
        if r is None:
            continue
        assert (
            "saliente" not in r["msckf"] and "puntos_mapa" not in r["msckf"]
        )  # no se mezclan
        ids, p_w = r["puntos_mapa"]
        assert p_w.shape == (len(ids), 3)
        n_puntos += len(ids)
        max_buffer = max(max_buffer, len(pipeline.grises))
        if r["keyframe"] is not None:
            keyframes.append(r["keyframe"])

    M = CFGS["msckf"]["ventana"]["clones_max"]
    assert max_buffer <= M  # solo las imágenes de la ventana
    assert (
        len(keyframes) >= 5
        and pipeline.n_keyframes == len(keyframes)
        and n_puntos > 200
    )
    assert all(b.indice > a.indice for a, b in zip(keyframes, keyframes[1:]))
    for kf in keyframes:
        assert kf.gris.shape == (480, 640) and kf.gris.dtype == np.uint8
        assert len(kf.z) >= CFG_MAPEO["puntos_min"] and (kf.z > 0).all()
        assert kf.xy.shape == (len(kf.ids), 2)
