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

from vio import reconstruccion as rec
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


# =====================================================================
# Paso 18.2 — Paquetes de mapa y archivo
# =====================================================================
def keyframe_de_prueba(n=40, semilla=0):
    """Un KeyframeMapeo inventado. La imagen es ruido suavizado: se comprime como una imagen
    real (el ruido blanco no cabe en un paquete ni con la calidad JPEG más baja)."""
    rng = np.random.default_rng(semilla)
    ruido = rng.integers(0, 256, (480, 640)).astype(np.float32)
    gris = cv2.normalize(
        cv2.GaussianBlur(ruido, (0, 0), 3), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return mapeo.KeyframeMapeo(
        t_ns=123456789012345,
        indice=77,
        R_w_c=so3.exp_so3([0.1, -0.2, 0.3]),
        p_w_c=np.array([1.5, -0.25, 0.75]),
        ids=rng.integers(0, 2**31 - 1, n),
        xy=rng.uniform(-0.5, 0.5, (n, 2)),
        z=rng.uniform(0.5, 6.0, n),
        gris=gris,
    )


def test_paquete_keyframe_ida_y_vuelta():
    kf = keyframe_de_prueba()
    dist = np.array([-0.3, 0.1, 0.001, -0.002, 0.0])
    jpeg = cv2.imencode(".jpg", kf.gris, [cv2.IMWRITE_JPEG_QUALITY, 20])[1].tobytes()
    datos = protocolo.empaquetar_keyframe(kf, K, dist, (640, 480), jpeg)
    assert datos[0] == 4 and len(
        datos
    ) == protocolo.TAMANO_CABECERA_KEYFRAME + 16 * 40 + len(jpeg)
    leido = protocolo.leer_keyframe(datos)
    assert (
        leido["t_ns"] == kf.t_ns
        and leido["indice"] == 77
        and leido["tamano"] == (640, 480)
    )
    assert leido["R_w_c"] == pytest.approx(kf.R_w_c, abs=1e-15) and leido[
        "p_w_c"
    ] == pytest.approx(kf.p_w_c)
    assert leido["K"] == pytest.approx(K) and leido["dist"] == pytest.approx(dist)
    assert list(leido["ids"]) == list(kf.ids)
    # Puntos en float32: 7 cifras significativas
    assert leido["xy"] == pytest.approx(kf.xy, rel=1e-6, abs=1e-7) and leido[
        "z"
    ] == pytest.approx(kf.z, rel=1e-6)
    assert leido["jpeg"] == jpeg


def test_paquete_keyframe_que_no_cabe():
    kf = keyframe_de_prueba(n=150)
    espacio = protocolo.espacio_jpeg_keyframe(150)
    assert (
        espacio
        == protocolo.MAX_BYTES_UDP - protocolo.TAMANO_CABECERA_KEYFRAME - 150 * 16
    )
    assert (
        len(protocolo.empaquetar_keyframe(kf, K, DIST, (640, 480), b"x" * espacio))
        == protocolo.MAX_BYTES_UDP
    )
    with pytest.raises(ValueError):
        protocolo.empaquetar_keyframe(kf, K, DIST, (640, 480), b"x" * (espacio + 1))
    with pytest.raises(ValueError):  # sin JPEG: incompleto
        protocolo.leer_keyframe(
            protocolo.empaquetar_keyframe(kf, K, DIST, (640, 480), b"")[:-1]
        )
    with pytest.raises(ValueError):  # un paquete de pose no es un keyframe
        protocolo.leer_keyframe(protocolo.empaquetar_pose(0, np.eye(3), np.zeros(3)))


def test_paquetes_de_puntos():
    rng = np.random.default_rng(1)
    n = 2 * protocolo.MAX_PUNTOS_POR_PAQUETE + 10
    ids, p_w = np.arange(n), rng.uniform(-5, 5, (n, 3))
    paquetes = protocolo.empaquetar_puntos(42, ids, p_w)
    assert len(paquetes) == 3 and all(
        len(d) <= protocolo.MAX_BYTES_UDP for d in paquetes
    )
    leidos = [protocolo.leer_puntos(d) for d in paquetes]
    assert all(t == 42 for t, _, _ in leidos)
    assert list(np.concatenate([i for _, i, _ in leidos])) == list(ids)
    assert np.vstack([p for _, _, p in leidos]) == pytest.approx(
        p_w, rel=1e-6, abs=1e-6
    )
    assert protocolo.empaquetar_puntos(0, np.zeros(0, int), np.zeros((0, 3))) == []
    with pytest.raises(ValueError):
        protocolo.leer_puntos(paquetes[2][:-3])  # cortado
    with pytest.raises(ValueError):
        protocolo.leer_puntos(protocolo.empaquetar_imagen(0, b"jpeg"))


def test_archivo_de_paquetes(tmp_path):
    ruta = tmp_path / "mapa.paquetes"
    paquetes = [b"uno", b"", bytes(range(256)) * 300, b"cuatro"]
    with open(ruta, "wb") as f:
        for datos in paquetes:
            protocolo.escribir_paquete(f, datos)
    assert list(protocolo.leer_archivo_paquetes(ruta)) == paquetes
    # La Pi se apaga mientras escribe el último: se leen los completos, sin error
    contenido = ruta.read_bytes()
    for corte in (len(contenido) - 2, len(contenido) - len(b"cuatro") - 2):
        ruta.write_bytes(contenido[:corte])
        assert list(protocolo.leer_archivo_paquetes(ruta)) == paquetes[:3]


def test_paquetes_mapa_de_pi_vio():
    from apps.pi_vio import paquetes_mapa

    kf = keyframe_de_prueba(n=30)
    r = {"t_ns": 5, "puntos_mapa": (np.array([3, 4]), np.ones((2, 3))), "keyframe": kf}
    paquetes = paquetes_mapa(r, K, DIST, CFG_MAPEO)
    assert [d[0] for d in paquetes] == [5, 4]
    leido = protocolo.leer_keyframe(paquetes[1])
    imagen = cv2.imdecode(np.frombuffer(leido["jpeg"], np.uint8), cv2.IMREAD_GRAYSCALE)
    assert imagen.shape == (480, 640)
    # Si ni con la calidad más baja cabe (ruido blanco), ese keyframe no se envía
    kf.gris = np.random.default_rng(0).integers(0, 256, (480, 640), dtype=np.uint8)
    assert [d[0] for d in paquetes_mapa(r, K, DIST, CFG_MAPEO)] == [5]
    sin_nada = {
        "t_ns": 5,
        "puntos_mapa": (np.zeros(0, int), np.zeros((0, 3))),
        "keyframe": None,
    }
    assert paquetes_mapa(sin_nada, K, DIST, CFG_MAPEO) == []


# =====================================================================
# Paso 17.3 — Reconstructor en el PC
# =====================================================================
def test_mapa_disperso_media_por_id():
    mapa = rec.MapaDisperso()
    mapa.anadir([7, 3], np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 5.0]]))
    mapa.anadir(
        [7], np.array([[3.0, 2.0, 0.0]])
    )  # el mismo punto, triangulado otra vez
    ids, p_w, cuentas = mapa.puntos()
    assert list(ids) == [3, 7] and list(cuentas) == [1, 2] and len(mapa) == 2
    assert p_w == pytest.approx(np.array([[0.0, 0.0, 5.0], [2.0, 1.0, 0.0]]))
    assert rec.MapaDisperso().puntos()[1].shape == (0, 3)


def test_reconstructor_tipos_de_paquete():
    r = rec.Reconstructor()
    kf = keyframe_de_prueba(n=20)
    assert (
        r.procesar(protocolo.empaquetar_puntos(1, np.array([1, 2]), np.ones((2, 3)))[0])
        == "puntos"
    )
    assert (
        r.procesar(protocolo.empaquetar_keyframe(kf, K, DIST, (640, 480), b"\xff\xd8x"))
        == "keyframe"
    )
    assert (
        r.procesar(protocolo.empaquetar_pose(0, np.eye(3), np.zeros(3))) == "malos"
    )  # otro puerto
    assert r.procesar(b"") == "malos" and r.procesar(b"\x05basura") == "malos"
    assert r.recuentos == {"puntos": 1, "keyframe": 1, "malos": 3}
    assert len(r.keyframes) == 1 and len(r.disperso) == 2
    assert "keyframes    1 | puntos      2" in r.resumen()


def test_puntos_keyframe_en_mundo():
    R_w_c = so3.exp_so3([0.2, 0.1, -0.4])
    p_w_c = np.array([1.0, 2.0, 0.5])
    p_w = np.array([[0.5, 4.0, 1.0], [2.0, 3.0, -0.5]])
    p_c = (p_w - p_w_c) @ R_w_c  # R_w_cᵀ·(p_w - p_w_c), fila a fila
    kf = {
        "R_w_c": R_w_c,
        "p_w_c": p_w_c,
        "xy": p_c[:, :2] / p_c[:, 2:3],
        "z": p_c[:, 2],
    }
    assert rec.puntos_keyframe_en_mundo(kf) == pytest.approx(p_w)


def test_ply_ida_y_vuelta(tmp_path):
    p_w = np.random.default_rng(2).uniform(-3, 3, (500, 3))
    colores = rec.colores_por_altura(p_w[:, 2])
    assert colores.shape == (500, 3) and colores.dtype == np.uint8
    rec.guardar_ply(tmp_path / "n.ply", p_w, colores)
    texto = (tmp_path / "n.ply").read_bytes()
    assert texto.startswith(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 500\n"
    )
    assert len(texto) == texto.index(b"end_header\n") + len(b"end_header\n") + 500 * 15
    leido, colores_leidos = rec.leer_ply(tmp_path / "n.ply")
    assert leido == pytest.approx(p_w, rel=1e-6, abs=1e-6) and np.array_equal(
        colores_leidos, colores
    )


def test_reconstructor_guardar(tmp_path):
    r = rec.Reconstructor()
    for semilla in range(3):
        kf = keyframe_de_prueba(n=20, semilla=semilla)
        kf.indice = semilla
        jpeg = cv2.imencode(".jpg", kf.gris, [cv2.IMWRITE_JPEG_QUALITY, 70])[
            1
        ].tobytes()
        r.procesar(protocolo.empaquetar_keyframe(kf, K, DIST, (640, 480), jpeg))
    r.procesar(
        protocolo.empaquetar_puntos(
            0, np.arange(50), np.random.default_rng(0).normal(0, 1, (50, 3))
        )[0]
    )
    rutas = r.guardar(tmp_path / "salida")
    assert set(rutas) == {"ply", "keyframes", "figura"}
    filas = (
        (tmp_path / "salida" / "keyframes.csv").read_text(encoding="utf-8").splitlines()
    )
    assert filas[0] == rec.CABECERA_KEYFRAMES.strip() and len(filas) == 4
    assert all(len(f.split(",")) == len(filas[0].split(",")) for f in filas[1:])
    assert rec.decodificar_gris(r.keyframes[0]).shape == (480, 640)
    assert (
        tmp_path / "salida" / "keyframes" / "000002.jpg"
    ).read_bytes() == r.keyframes[2]["jpeg"]
    assert rec.Reconstructor().guardar(tmp_path / "vacio") == {}


# =====================================================================
# De principio a fin: grabación -> pi_vio -> mapa.paquetes -> pc_mapa -> PLY
# =====================================================================
def test_mapa_de_principio_a_fin(tmp_path):
    from herramientas.grabar_simulacion import guardar_grabacion_sintetica
    from scipy.spatial import cKDTree

    cfg_vio = copy.deepcopy(cargar_config("simulador")["vio"])
    cfg_vio["trayectoria"]["duracion_s"] = 10.0
    s = guardar_grabacion_sintetica(
        tmp_path, cfg_vio, K, (640, 480), EXT, np.random.default_rng(3)
    )

    # 1. pi_vio sobre la grabación (como en la Pi, pero sin red): escribe mapa.paquetes
    entorno = dict(os.environ, PYTHONUNBUFFERED="1")
    vio = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.pi_vio",
            "--grabacion",
            str(tmp_path),
            "--sin-red",
            "--sin-calibracion-acc",
        ],
        cwd=RAIZ_PROYECTO,
        env=entorno,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert vio.returncode == 0, vio.stdout + vio.stderr
    assert (
        "keyframes de mapeo" in vio.stdout
        and (tmp_path / CFG_MAPEO["archivo"]).exists()
    )

    # 2. pc_mapa repite el archivo y guarda el mapa
    pc = subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.pc_mapa",
            "--archivo",
            str(tmp_path / CFG_MAPEO["archivo"]),
        ],
        cwd=RAIZ_PROYECTO,
        env=entorno,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert pc.returncode == 0, pc.stdout + pc.stderr
    salida = tmp_path / "mapa"
    assert (salida / "mapa.png").exists()
    p_mapa, _ = rec.leer_ply(salida / "puntos_3d.ply")
    n_keyframes = (
        len((salida / "keyframes.csv").read_text(encoding="utf-8").splitlines()) - 1
    )
    assert len(p_mapa) > 200 and n_keyframes >= 8

    # 3. Frente a la verdad: los puntos del simulador llevados al mundo del filtro (que arranca
    #    con la orientación y la posición de la primera pose estimada)
    t_est, p_est, R_est = leer_poses(tmp_path / "vio_trayectoria.csv")
    t_ver, p_ver, R_ver = leer_poses(tmp_path / "verdad.csv")
    k = int(np.searchsorted(t_ver, t_est[0]))
    R_alin = R_est[0] @ R_ver[k].T
    verdad_en_filtro = (s["puntos"] - p_ver[k]) @ R_alin.T + p_est[0]
    distancia, _ = cKDTree(verdad_en_filtro).query(p_mapa)
    assert np.median(distancia) < 0.08 and np.percentile(distancia, 90) < 0.15


@pytest.mark.skipif(
    sys.platform == "win32", reason="SIGINT a otro proceso: solo en Linux"
)
def test_pc_mapa_en_directo(tmp_path):
    # pc_mapa escuchando; le llegan paquetes por UDP; Ctrl+C (SIGINT) -> guarda el mapa
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as libre:
        libre.bind(("127.0.0.1", 0))
        puerto = libre.getsockname()[1]
    proceso = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "apps.pc_mapa",
            "--puerto",
            str(puerto),
            "--salida",
            str(tmp_path),
        ],
        cwd=RAIZ_PROYECTO,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    try:
        assert "Escuchando UDP" in proceso.stdout.readline()
        kf = keyframe_de_prueba(n=20)
        paquetes = [
            protocolo.empaquetar_keyframe(
                kf,
                K,
                DIST,
                (640, 480),
                cv2.imencode(".jpg", kf.gris, [cv2.IMWRITE_JPEG_QUALITY, 70])[
                    1
                ].tobytes(),
            ),
            *protocolo.empaquetar_puntos(1, np.arange(30), np.ones((30, 3))),
        ]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as emisor:
            for datos in paquetes:
                emisor.sendto(datos, ("127.0.0.1", puerto))
        time.sleep(1.5)
        proceso.send_signal(signal.SIGINT)
        salida, _ = proceso.communicate(timeout=30)
    finally:
        if proceso.poll() is None:
            proceso.kill()
    assert proceso.returncode == 0, salida
    assert "Mapa guardado" in salida and (tmp_path / "puntos_3d.ply").exists()
    assert (
        list(protocolo.leer_archivo_paquetes(tmp_path / CFG_MAPEO["archivo"]))
        == paquetes
    )  # la copia
    assert (
        len((tmp_path / "keyframes.csv").read_text(encoding="utf-8").splitlines()) == 2
    )
