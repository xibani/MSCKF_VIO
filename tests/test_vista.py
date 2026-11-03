"""tests/test_vista.py — Etapa 14: la imagen con sus pistas, del VIO al navegador de la tablet.

    vio/vista.py (dibujo + JPEG) -> vio/protocolo.py (paquete v3) -> apps/pi_vio.py (envío)
    -> apps/visor_web.py (recepción, /foto y vídeo MJPEG en /video)
"""

import http.client
import socket
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from apps import pi_vio, visor_web
from vio import protocolo
from vio import vista as vista_vio
from vio.config import cargar_config
from vio.frontend import Pista

CFG_VISTA = cargar_config("vio")["vista"]
VERDE, ROJO = (0, 255, 0), (0, 0, 255)          # en BGR, el orden de OpenCV


def pista_recta(id_pista, n, u0, v0, du, dv):
    """Una pista vista en n imágenes, moviéndose (du, dv) píxeles en cada una."""
    pista = Pista(id_pista)
    for k in range(n):
        pista.anadir(k, k, (u0 + du * k, v0 + dv * k), (0.0, 0.0))
    return pista


def imagen_gris():
    """Imagen 640x480 con un degradado (no uniforme, como una imagen real)."""
    u = np.arange(640)
    return np.tile((50 + u * 100 // 640).astype(np.uint8), (480, 1))


# =====================================================================
# Paso 14.1 — vio/vista.py
# =====================================================================
def test_color_por_longitud():
    assert vista_vio.color_por_longitud(0, 20) == ROJO
    assert vista_vio.color_por_longitud(10, 20) == (0, 127, 127)
    assert vista_vio.color_por_longitud(20, 20) == VERDE
    assert vista_vio.color_por_longitud(55, 20) == VERDE           # no pasa del verde


def test_dibujar_vista_sin_pistas():
    gris = imagen_gris()
    copia = gris.copy()
    vista = vista_vio.dibujar_vista(gris, None, [], CFG_VISTA)
    assert vista.shape == (480, 640, 3) and vista.dtype == np.uint8
    assert np.array_equal(gris, copia)                              # no toca la imagen original
    for canal in range(3):                                          # sin nada encima: la misma imagen
        assert np.array_equal(vista[:, :, canal], gris)


def test_dibujar_vista_con_pistas_y_texto():
    gris = imagen_gris()
    larga = pista_recta(1, 30, 100, 300, 3, -2)                     # 30 imágenes: verde
    nueva = pista_recta(2, 1, 500, 400, 0, 0)                       # recién detectada: roja
    frontend = SimpleNamespace(activas={1: larga, 2: nueva})
    vista = vista_vio.dibujar_vista(gris, frontend, ["hola"], CFG_VISTA)

    # 1. El punto de cada pista, en su posición actual y de su color
    u, v = larga.uv[-1]
    assert tuple(int(c) for c in vista[int(v), int(u)]) == VERDE
    casi_rojo = vista_vio.color_por_longitud(1, CFG_VISTA["longitud_larga"])   # vista 1 vez
    assert casi_rojo == (0, 12, 242)
    assert tuple(int(c) for c in vista[400, 500]) == casi_rojo
    # 2. La cola: solo las últimas cola_pista posiciones (la más antigua no se dibuja)
    u_cola, v_cola = larga.uv[-CFG_VISTA["cola_pista"]]
    assert tuple(int(c) for c in vista[int(v_cola), int(u_cola)]) == VERDE
    u_vieja, v_vieja = larga.uv[0]
    assert np.array_equal(vista[int(v_vieja), int(u_vieja)], [gris[int(v_vieja), int(u_vieja)]] * 3)
    # 3. El texto está arriba a la izquierda (cambia píxeles allí y no abajo)
    sin_texto = vista_vio.dibujar_vista(gris, frontend, [], CFG_VISTA)
    cambios = np.any(vista != sin_texto, axis=2)
    assert cambios[:30, :60].any() and not cambios[30:, :].any()


def test_lineas_estado():
    assert "INICIALIZANDO" in vista_vio.lineas_estado(SimpleNamespace(filtro=None), None)[0]
    pipeline = SimpleNamespace(filtro=object())
    assert "esperando" in vista_vio.lineas_estado(pipeline, None)[0]
    r = {"p_w": np.array([0.1, -0.2, 0.3]),
         "frontend": {"activos": 87, "nuevos": 4},
         "msckf": {"usadas": 6, "quieto": True},
         "tiempos_ms": {"total": 31.6}}
    lineas = vista_vio.lineas_estado(pipeline, r)
    assert lineas[0] == "p = (+0.10, -0.20, +0.30) m   QUIETO (ZUPT)"
    assert lineas[1] == "puntos 87  nuevos 4  pistas usadas 6  32 ms"


def test_codificar_jpeg_cabe_y_se_puede_leer():
    vista = vista_vio.dibujar_vista(imagen_gris(), None, ["prueba"], CFG_VISTA)
    jpeg = vista_vio.codificar_jpeg(vista, CFG_VISTA["calidad_jpeg"], 65498)
    assert jpeg is not None and len(jpeg) <= 65498
    leida = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert leida.shape == vista.shape
    assert np.abs(leida.astype(int) - vista.astype(int)).mean() < 3     # pérdida pequeña


def test_codificar_jpeg_baja_la_calidad_si_no_cabe():
    ruido = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)  # lo peor para JPEG
    tamano_70 = len(vista_vio.codificar_jpeg(ruido, 70, 10**9))
    jpeg = vista_vio.codificar_jpeg(ruido, 70, tamano_70 - 1)
    assert jpeg is not None and len(jpeg) < tamano_70                # ha bajado la calidad
    assert vista_vio.codificar_jpeg(ruido, 70, 1000) is None          # ni con calidad 20 cabe



# =====================================================================
# Paso 14.2 — vio/protocolo.py (paquete de imagen, versión 3)
# =====================================================================
def test_paquete_imagen_ida_y_vuelta():
    jpeg = bytes(range(256)) * 10
    datos = protocolo.empaquetar_imagen(123456789012, jpeg)
    assert len(datos) == protocolo.TAMANO_CABECERA_IMAGEN + len(jpeg) and datos[0] == 3
    assert protocolo.leer_imagen(datos) == (123456789012, jpeg)
    # Un JPEG del tamaño máximo cabe justo en un datagrama UDP
    assert len(protocolo.empaquetar_imagen(0, b"x" * protocolo.MAX_BYTES_JPEG)) == protocolo.MAX_BYTES_UDP


def test_paquete_imagen_rechazos():
    with pytest.raises(ValueError):
        protocolo.empaquetar_imagen(0, b"x" * (protocolo.MAX_BYTES_JPEG + 1))
    with pytest.raises(ValueError):                                   # un paquete de pose (v2) no es imagen
        protocolo.leer_imagen(protocolo.empaquetar_pose(0, np.eye(3), np.zeros(3)))
    with pytest.raises(ValueError):                                   # solo la cabecera, sin JPEG
        protocolo.leer_imagen(protocolo.empaquetar_imagen(0, b""))