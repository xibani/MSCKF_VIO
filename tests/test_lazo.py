"""tests/test_lazo.py — Etapa 16: pruebas de trayectoria (lazo cerrado y recta medida).

    vio/lazo.py (métricas) · apps/visor_web.py (las mismas cifras en vivo, botón "Nueva prueba")
    apps/pi_vio.py (orden de reiniciar) · herramientas/informe_lazo.py (informe posterior)
"""

import copy
import http.client
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from apps import pi_vio, visor_web
from vio import lazo, protocolo, so3
from vio.config import RAIZ_PROYECTO, cargar_config
from vio.pipeline import CABECERA_POSES, linea_pose

CFG = cargar_config("vio")["prueba"]


def cuadrado(lado, puntos_por_lado):
    """Lazo cuadrado en el suelo que sale de (0, 0, 0) y vuelve a él. Devuelve: (N, 3)."""
    esquinas = np.array([[0, 0, 0], [lado, 0, 0], [lado, lado, 0], [0, lado, 0], [0, 0, 0]], float)
    tramos = [np.linspace(esquinas[k], esquinas[k + 1], puntos_por_lado, endpoint=False) for k in range(4)]
    return np.vstack(tramos + [esquinas[-1:]])


# =====================================================================
# Paso 16.1 — vio/lazo.py
# =====================================================================
def test_recorrido_no_se_infla_con_el_temblor():
    # 2000 poses quietas con temblor de 2 mm (como el VIO parado) y después un cuadrado de 2 m
    temblor = np.random.default_rng(0).normal(0.0, 0.002, (2000, 3))
    p = np.vstack([temblor, cuadrado(2.0, 200)])
    ingenuo = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    assert ingenuo > 10.0                                      # sumando todo: ¡más de 10 m!
    assert lazo.recorrido(p, CFG["paso_min_m"]) == pytest.approx(8.0, rel=0.02)


def test_metricas_lazo():
    p = cuadrado(2.0, 200)
    p[-1] = [0.06, -0.08, 0.0]                                 # vuelve a 10 cm de la salida
    t = np.arange(len(p)) * 0.05
    m = lazo.metricas_lazo(t, p, CFG["paso_min_m"], CFG["recorrido_min_m"])
    assert m["cierre_m"] == pytest.approx(0.10)
    assert m["cierre_pct"] == pytest.approx(100 * 0.10 / m["recorrido_m"])
    assert m["recorrido_m"] == pytest.approx(8.0, rel=0.02)
    assert m["deriva_m"] == pytest.approx([0.06, -0.08, 0.0])
    assert m["alejamiento_max_m"] == pytest.approx(2.0 * np.sqrt(2.0), rel=1e-3)   # la esquina opuesta
    assert m["duracion_s"] == pytest.approx(t[-1])
    # Si apenas se ha movido, el % no tiene sentido
    corto = lazo.metricas_lazo(t[:20], p[:20], CFG["paso_min_m"], CFG["recorrido_min_m"])
    assert corto["cierre_pct"] is None


def test_error_escala():
    p = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [5.15, 0.0, 0.0]])
    estimada, error = lazo.error_escala(p, 5.0)
    assert estimada == pytest.approx(5.15) and error == pytest.approx(3.0)




# =====================================================================
# Paso 16.2 — apps/pi_vio.py: la orden de reiniciar
# =====================================================================
def test_orden_reiniciar():
    control = pi_vio.abrir_control(0)                           # 0: un puerto libre cualquiera
    emisor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        assert not pi_vio.orden_reiniciar(control)             # nada todavía
        destino = control.getsockname()
        for _ in range(3):                                      # tres toques seguidos = una orden
            emisor.sendto(protocolo.ORDEN_REINICIAR, destino)
        emisor.sendto(b"otra cosa", destino)
        time.sleep(0.1)
        assert pi_vio.orden_reiniciar(control)
        assert not pi_vio.orden_reiniciar(control)             # ya se leyeron todas
        # Puerto ocupado: aviso y None, el VIO sigue sin control
        assert pi_vio.abrir_control(destino[1]) is None
    finally:
        emisor.close()
        control.close()


def puerto_libre():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    puerto = s.getsockname()[1]
    s.close()
    return puerto


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM: solo en Linux (en la Pi)")
def test_pi_vio_reinicia_y_abre_otro_csv(tmp_path):
    # pi_vio sobre una grabación sintética: a mitad, "reiniciar" -> VIO nuevo y vio_trayectoria_2.csv
    from herramientas.grabar_simulacion import guardar_grabacion_sintetica
    from vio import modelo_camara as mc
    cfg_sim = copy.deepcopy(cargar_config("simulador")["vio"])
    cfg_sim["trayectoria"]["duracion_s"] = 10.0
    K, _, _ = mc.cargar_intrinsecos()
    guardar_grabacion_sintetica(tmp_path, cfg_sim, K, (640, 480), cargar_config("camara_imu"),
                                np.random.default_rng(5))
    puerto = puerto_libre()
    proceso = subprocess.Popen([sys.executable, "-m", "apps.pi_vio", "--grabacion", str(tmp_path),
                                "--sin-red", "--sin-calibracion-acc", "--puerto-control", str(puerto)],
                               cwd=RAIZ_PROYECTO, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    emisor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    lineas = []
    try:
        for linea in proceso.stdout:
            lineas.append(linea)
            if "VIO EN MARCHA" in linea and len(lineas) < 50:
                emisor.sendto(protocolo.ORDEN_REINICIAR, ("127.0.0.1", puerto))
            if "Reiniciado desde el visor" in linea:
                proceso.send_signal(signal.SIGTERM)
                break
        salida = "".join(lineas) + proceso.stdout.read()
        assert proceso.wait(timeout=30) == 0, salida
    finally:
        emisor.close()
        if proceso.poll() is None:
            proceso.kill()
    assert "Reiniciado desde el visor (prueba 2)" in salida
    assert (tmp_path / "vio_trayectoria.csv").read_text(encoding="utf-8").count("\n") > 1   # la 1.ª prueba
    assert (tmp_path / "vio_trayectoria_2.csv").read_text(encoding="utf-8").startswith("#t_ns")





# =====================================================================
# Paso 16.3 — apps/visor_web.py: las mismas cifras en vivo y el botón "Nueva prueba"
# =====================================================================
def test_visor_web_copia_la_configuracion():
    # visor_web solo usa la biblioteca estándar: tiene su copia de los valores
    assert visor_web.PUERTO_CONTROL == cargar_config("vio")["control"]["puerto"]
    assert visor_web.ORDEN_REINICIAR == protocolo.ORDEN_REINICIAR
    assert visor_web.RECORRIDO_PASO_MIN_M == CFG["paso_min_m"]
    assert visor_web.RECORRIDO_MIN_M == CFG["recorrido_min_m"]
    assert visor_web.CIERRE_BUENO_PCT == CFG["cierre_bueno_pct"]
    assert visor_web.CIERRE_REGULAR_PCT == CFG["cierre_regular_pct"]


def test_visor_web_mismas_cifras_que_lazo():
    # El cálculo en vivo (sin numpy, punto a punto) da lo mismo que vio/lazo.py
    temblor = np.random.default_rng(1).normal(0.0, 0.002, (300, 3))
    p = np.vstack([temblor, cuadrado(2.0, 150)])
    p[-1] = [0.03, 0.04, 0.0]
    estado = visor_web.EstadoCompartido()
    for q in p:
        visor_web.anadir_punto(estado, [float(v) for v in q])
    vivo = visor_web.metricas_prueba(estado)
    referencia = lazo.metricas_lazo(np.arange(len(p)) * 0.05, p, CFG["paso_min_m"], CFG["recorrido_min_m"])
    assert vivo["recorrido_m"] == pytest.approx(referencia["recorrido_m"], abs=1e-9)
    assert vivo["cierre_m"] == pytest.approx(referencia["cierre_m"], abs=1e-9)
    assert vivo["cierre_pct"] == pytest.approx(referencia["cierre_pct"], abs=1e-9)
    assert vivo["alejamiento_max_m"] == pytest.approx(referencia["alejamiento_max_m"], abs=1e-9)
    assert vivo["nivel"] == "bueno"                            # 5 cm en ~8 m: 0,6 %
    # Niveles de color
    estado.trayectoria[-1] = [0.0, 0.3, 0.0]                   # 30 cm: ~3,8 %
    assert visor_web.metricas_prueba(estado)["nivel"] == "regular"
    estado.trayectoria[-1] = [0.0, 0.6, 0.0]                   # 60 cm: ~7,5 %
    assert visor_web.metricas_prueba(estado)["nivel"] == "malo"
    assert visor_web.metricas_prueba(visor_web.EstadoCompartido()) is None    # sin trayectoria


@pytest.fixture
def visor_con_control():
    """visor_web en puertos libres, con un socket haciendo de pi_vio en el puerto de control."""
    estado = visor_web.EstadoCompartido()
    pi_falsa = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pi_falsa.bind(("127.0.0.1", 0))
    pi_falsa.settimeout(2.0)
    anterior = (visor_web.ManejadorHTTP.estado, visor_web.ManejadorHTTP.puerto_control)
    visor_web.ManejadorHTTP.estado = estado
    visor_web.ManejadorHTTP.puerto_control = pi_falsa.getsockname()[1]
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), visor_web.ManejadorHTTP)
    servidor.daemon_threads = True
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield estado, servidor.server_address[1], pi_falsa
    servidor.shutdown()
    servidor.server_close()
    pi_falsa.close()
    visor_web.ManejadorHTTP.estado, visor_web.ManejadorHTTP.puerto_control = anterior


def test_boton_nueva_prueba(visor_con_control):
    estado, puerto_web, pi_falsa = visor_con_control
    with estado.cerrojo:
        for q in cuadrado(1.0, 50):
            visor_web.anadir_punto(estado, list(q))
        sesion = estado.id_sesion
    conexion = http.client.HTTPConnection("127.0.0.1", puerto_web, timeout=2.0)
    conexion.request("POST", "/reiniciar")
    assert conexion.getresponse().status == 200
    conexion.close()
    assert pi_falsa.recvfrom(64)[0] == protocolo.ORDEN_REINICIAR    # la orden llega a "pi_vio"
    with estado.cerrojo:
        assert estado.trayectoria == [] and estado.recorrido_m == 0.0
        assert estado.id_sesion == sesion + 1 and estado.reinicios == 1
        assert estado.ignorar_hasta > time.monotonic()
    # Con GET no se reinicia nada (un navegador puede pedir páginas por adelantado)
    conexion = http.client.HTTPConnection("127.0.0.1", puerto_web, timeout=2.0)
    conexion.request("GET", "/reiniciar")
    assert conexion.getresponse().status == 404
    conexion.close()


def test_tras_reiniciar_se_tiran_las_poses_en_camino():
    estado, parar = visor_web.EstadoCompartido(), threading.Event()
    receptor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receptor.bind(("127.0.0.1", 0))
    receptor.settimeout(0.05)
    hilo = threading.Thread(target=visor_web.bucle_udp, args=(receptor, estado, parar), daemon=True)
    hilo.start()
    emisor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with estado.cerrojo:
            estado.ignorar_hasta = time.monotonic() + 0.3
        emisor.sendto(protocolo.empaquetar_pose(1, np.eye(3), np.array([5.0, 5.0, 0.0])), receptor.getsockname())
        time.sleep(0.4)                                        # pasa la ventana
        emisor.sendto(protocolo.empaquetar_pose(2, np.eye(3), np.zeros(3)), receptor.getsockname())
        time.sleep(0.2)
        with estado.cerrojo:
            assert estado.trayectoria == [[0.0, 0.0, 0.0]]     # la de la prueba anterior, fuera
    finally:
        parar.set()
        hilo.join(timeout=1.0)
        emisor.close()
        receptor.close()


def test_pagina_tiene_prueba_y_vista_3d(visor_con_control):
    _, puerto_web, _ = visor_con_control
    conexion = http.client.HTTPConnection("127.0.0.1", puerto_web, timeout=2.0)
    conexion.request("GET", "/")
    html = conexion.getresponse().read().decode("utf-8")
    conexion.close()
    for trozo in ('id="boton_prueba"', 'id="m_pct"', 'id="lienzo_3d"', 'fetch("/reiniciar", { method: "POST" })',
                  "function dibujarVista3D()"):
        assert trozo in html


def test_vista_3d_sin_espejo():
    # Los ejes de la vista 3D, copiados de la página: derecha × arriba = dirección del observador
    for az, el in [(200, 25), (0, 0), (90, 60), (-45, -30)]:
        a, b = np.radians(az), np.radians(el)
        derecha = np.array([-np.sin(a), np.cos(a), 0.0])
        arriba = np.array([-np.sin(b) * np.cos(a), -np.sin(b) * np.sin(a), np.cos(b)])
        observador = np.array([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)])
        assert np.cross(derecha, arriba) == pytest.approx(observador)
        assert arriba[2] >= 0                                  # z del mundo nunca sale cabeza abajo




# =====================================================================
# Paso 16.4 — herramientas/informe_lazo.py
# =====================================================================
def guardar_trayectoria(ruta, p):
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(CABECERA_POSES)
        for k, q in enumerate(p):
            f.write(linea_pose(k * 50_000_000, so3.exp_so3([0.0, 0.0, 0.01 * k]), q))


def test_informe_lazo(tmp_path):
    from herramientas.informe_lazo import elegir_csv, informe
    p = cuadrado(2.0, 100)
    p[-1] = [0.0, 0.0, 0.12]                                   # llega 12 cm por encima de la salida
    guardar_trayectoria(tmp_path / "vio_trayectoria.csv", p)
    time.sleep(0.05)
    guardar_trayectoria(tmp_path / "vio_trayectoria_2.csv", np.linspace([0, 0, 0], [5.2, 0, 0], 200))
    assert elegir_csv(tmp_path).name == "vio_trayectoria_2.csv"           # la más reciente
    # Lazo cerrado
    m = informe(tmp_path / "vio_trayectoria.csv", CFG)
    assert m["cierre_m"] == pytest.approx(0.12) and m["cierre_pct"] == pytest.approx(1.5, rel=0.03)
    assert (tmp_path / "vio_trayectoria_informe.png").exists()
    # Recta medida de 5 m que el VIO estima en 5,2 m: +4 % de escala
    m = informe(tmp_path, CFG, distancia_real_m=5.0)
    assert m["error_escala_pct"] == pytest.approx(4.0)
    assert (tmp_path / "vio_trayectoria_2_informe.png").exists()