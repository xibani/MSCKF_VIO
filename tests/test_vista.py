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
from vio import protocolo, so3
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
    # 1. Inicializando: cuántas muestras lleva y por qué falló el último intento
    iniciando = SimpleNamespace(filtro=None, reposo=[0] * 150, intentos=3,
                                problemas=["|a| = 8.72 m/s², lejos de g: ¿se ha movido?"])
    lineas = vista_vio.lineas_estado(iniciando, None)
    assert lineas == ["INICIALIZANDO: deja la placa QUIETA",
                        "IMU: 150 muestras en reposo, intento 4",
                        "fallo: |a| = 8.72 m/s², lejos de g: ¿se ha movido?"]
    # 2. Filtro creado pero sin imagen válida: cuántas se descartan por tiempo
    pipeline = SimpleNamespace(filtro=object(), descartadas=7)
    assert vista_vio.lineas_estado(pipeline, None) == ["VIO EN MARCHA (esperando imagen)",
                                                        "imagenes descartadas por tiempo: 7"]
    r = {"p_w": np.array([0.1, -0.2, 0.3]),
         "frontend": {"activos": 87, "nuevos": 4},
         "msckf": {"usadas": 6, "quieto": True},
         "tiempos_ms": {"total": 31.6}}
    lineas = vista_vio.lineas_estado(pipeline, r)
    assert lineas[0] == "p = (+0.10, -0.20, +0.30) m   QUIETO (ZUPT)"
    assert lineas[1] == "puntos 87  nuevos 4  pistas usadas 6  32 ms"


def test_texto_en_ascii():
    # Las fuentes de OpenCV solo tienen ASCII: "σ", "²" o las tildes saldrían como "??"
    assert vista_vio.a_ascii("σ = 0.80 °/s, |a| = 8.72 m/s², ¿se ha movido? máx") == \
        "sigma = 0.80 deg/s, |a| = 8.72 m/s2, se ha movido? max"
    assert len(vista_vio.a_ascii("x" * 200)) == vista_vio.MAX_CARACTERES


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
        



# =====================================================================
# Paso 14.3 — apps/pi_vio.py (envío)
# =====================================================================
def test_enviar_no_se_detiene_si_falla_la_red():
    emisor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    emisor.close()                                                    # sendto lanzaría OSError
    pi_vio.enviar(emisor, b"hola", ("127.0.0.1", 9))                  # no lanza nada


def test_enviar_imagen_llega_y_se_decodifica():
    receptor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receptor.bind(("127.0.0.1", 0))
    receptor.settimeout(2.0)
    emisor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        pipeline = SimpleNamespace(filtro=None, frontend=None, reposo=[], intentos=0, problemas=[])  # aún inicializando
        pi_vio.enviar_imagen(emisor, receptor.getsockname(), 42, imagen_gris(), pipeline, None, CFG_VISTA)
        datos, _ = receptor.recvfrom(65536)
        t_ns, jpeg = protocolo.leer_imagen(datos)
        assert t_ns == 42
        assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (480, 640, 3)
    finally:
        emisor.close()
        receptor.close()




# =====================================================================
# Paso 14.4 — apps/visor_web.py (recepción y servidor)
# =====================================================================
def test_visor_web_usa_el_mismo_formato_que_protocolo():
    # visor_web solo usa la biblioteca estándar: tiene su copia de las constantes
    assert visor_web.FORMATO_CABECERA_IMAGEN == protocolo.FORMATO_CABECERA_IMAGEN
    assert visor_web.VERSION_IMAGEN == protocolo.VERSION_IMAGEN
    datos = protocolo.empaquetar_imagen(987, b"\xff\xd8 jpeg \xff\xd9")
    assert visor_web.decodificar_imagen(datos) == protocolo.leer_imagen(datos)
    with pytest.raises(ValueError):
        visor_web.decodificar_imagen(protocolo.empaquetar_pose(0, np.eye(3), np.zeros(3)))


def test_bucle_imagenes_guarda_la_ultima():
    estado, parar = visor_web.EstadoCompartido(), threading.Event()
    receptor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receptor.bind(("127.0.0.1", 0))
    receptor.settimeout(0.1)
    hilo = threading.Thread(target=visor_web.bucle_imagenes, args=(receptor, estado, parar), daemon=True)
    hilo.start()
    emisor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for k in range(3):
            emisor.sendto(protocolo.empaquetar_imagen(k, b"jpeg %d" % k), receptor.getsockname())
        emisor.sendto(b"basura", receptor.getsockname())              # se ignora
        time.sleep(0.3)
        with estado.cerrojo:
            assert estado.imagenes == 3 and estado.id_jpeg == 3 and estado.jpeg == b"jpeg 2"
    finally:
        parar.set()
        hilo.join(timeout=1.0)
        emisor.close()
        receptor.close()


@pytest.fixture
def servidor_web():
    """visor_web en un puerto libre, en un hilo. Devuelve (estado, puerto)."""
    estado = visor_web.EstadoCompartido()
    anterior = visor_web.ManejadorHTTP.estado
    visor_web.ManejadorHTTP.estado = estado
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), visor_web.ManejadorHTTP)
    servidor.daemon_threads = True
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    yield estado, servidor.server_address[1]
    servidor.shutdown()
    servidor.server_close()
    visor_web.ManejadorHTTP.estado = anterior


def test_foto(servidor_web):
    estado, puerto = servidor_web
    conexion = http.client.HTTPConnection("127.0.0.1", puerto, timeout=2.0)
    conexion.request("GET", "/foto")
    assert conexion.getresponse().status == 503                     # aún no hay imagen
    conexion.close()
    with estado.cerrojo:
        estado.jpeg, estado.id_jpeg = b"\xff\xd8 una imagen \xff\xd9", 1
    conexion = http.client.HTTPConnection("127.0.0.1", puerto, timeout=2.0)
    conexion.request("GET", "/foto")
    respuesta = conexion.getresponse()
    assert respuesta.status == 200 and respuesta.getheader("Content-Type") == "image/jpeg"
    assert respuesta.read() == b"\xff\xd8 una imagen \xff\xd9"
    conexion.close()


def test_video_mjpeg(servidor_web):
    estado, puerto = servidor_web
    jpeg = b"\xff\xd8 fotograma \xff\xd9"
    with estado.cerrojo:
        estado.jpeg, estado.id_jpeg = jpeg, 1
    conexion = http.client.HTTPConnection("127.0.0.1", puerto, timeout=2.0)
    conexion.request("GET", "/video")
    respuesta = conexion.getresponse()
    assert respuesta.status == 200
    assert respuesta.getheader("Content-Type") == "multipart/x-mixed-replace; boundary=fotograma"
    # Frontera inicial, cabecera de la parte, el JPEG y la frontera que lo cierra
    esperado = (b"--fotograma\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg)
                + jpeg + b"\r\n--fotograma\r\n")
    assert respuesta.read(len(esperado)) == esperado
    conexion.close()


def test_pagina_tiene_el_panel_de_video(servidor_web):
    _, puerto = servidor_web
    conexion = http.client.HTTPConnection("127.0.0.1", puerto, timeout=2.0)
    conexion.request("GET", "/")
    html = conexion.getresponse().read().decode("utf-8")
    assert '<img id="video"' in html
    assert 'fuente.onopen' in html and '"/video?t=" + Date.now()' in html   # se reconecta solo
    conexion.close()


# =====================================================================
# Etapa 15.3 — Panel de orientación sin espejo
# =====================================================================
def proyeccion_orientacion():
    """Las dos filas de la proyección de visor_web: componente "derecha" y "arriba" de X, Y, Z."""
    E = visor_web.EJES_PANTALLA
    derecha = np.array([E["x"][0], E["y"][0], E["z"][0]])
    arriba = np.array([E["x"][1], E["y"][1], E["z"][1]])
    return derecha, arriba


def test_orientacion_sin_espejo():
    # Una proyección paralela no dice desde qué lado se mira: n = derecha × arriba es la
    # dirección hacia el observador con la que el dibujo es correcto. Si n no apunta hacia
    # donde el ojo cree estar (encima, porque z sale hacia arriba), se ve en espejo.
    derecha, arriba = proyeccion_orientacion()
    n = np.cross(derecha, arriba)
    assert n[2] > 0                   # observador por encima
    assert n[0] < 0                   # y por detrás de la placa (x de la IMU apunta hacia delante)
    # La proyección anterior (isométrica) era correcta solo vista desde DEBAJO: n_z < 0
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    assert np.cross([c, -c, 0.0], [-s, -s, 1.0])[2] < 0


def test_orientacion_movimientos():
    # Lo que tiene que verse en la tablet al mover la placa (desde detrás, en reposo rumbo 0)
    derecha, arriba = proyeccion_orientacion()

    def pantalla(v):
        return np.array([derecha @ v, arriba @ v])

    giro = np.radians(10)
    # Girar a la izquierda (visto desde arriba, +z): la flecha x (hacia delante) va a la izquierda
    assert (pantalla(so3.exp_so3([0, 0, giro])[:, 0]) - pantalla([1, 0, 0]))[0] < 0
    # Alabeo a la derecha (lado derecho abajo, +x): la flecha z se inclina a la derecha
    assert (pantalla(so3.exp_so3([giro, 0, 0])[:, 2]) - pantalla([0, 0, 1]))[0] > 0
    # Morro arriba (-y): la flecha x sube
    assert (pantalla(so3.exp_so3([0, -giro, 0])[:, 0]) - pantalla([1, 0, 0]))[1] > 0
    # En reposo: y (izquierda de la placa) a la izquierda en pantalla y z hacia arriba
    assert pantalla([0, 1, 0])[0] < 0 and pantalla([0, 0, 1])[1] > 0


def test_pagina_usa_los_ejes_de_python(servidor_web):
    import json
    _, puerto = servidor_web
    conexion = http.client.HTTPConnection("127.0.0.1", puerto, timeout=2.0)
    conexion.request("GET", "/")
    html = conexion.getresponse().read().decode("utf-8")
    conexion.close()
    assert "const EJES_PANTALLA = " + json.dumps(visor_web.EJES_PANTALLA) + ";" in html
    assert "__EJES_PANTALLA__" not in html and "proyectarIso" not in html