"""
apps/visor_web.py — Visor de la VIO en el navegador (tablet, móvil o PC).

Idea:
    pi_vio / pi_orientacion  --UDP 127.0.0.1:5005-->  visor_web (este programa, en la Pi)
    pi_vio (imagen + pistas) --UDP 127.0.0.1:5006-->  visor_web
    visor_web  --HTTP + Server-Sent Events (puerto 8000)-->  navegador de la tablet
    visor_web  --HTTP, vídeo MJPEG (/video)-->  navegador de la tablet
    navegador  --HTTP /red, /red/punto, /red/cliente-->  visor_web --nmcli-->  Wi-Fi (etapa 18)

La Pi ya no dibuja nada: este proceso solo recibe los paquetes UDP, los decodifica
y reenvía la pose como texto JSON unas 15 veces por segundo. El dibujo lo hace el
navegador de la tablet con <canvas>.

No cambia el protocolo: lee exactamente los mismos paquetes v1 ('<Bq9d', 81 bytes)
y v2 ('<Bq9d3d', 105 bytes) que vio/protocolo.py, y las imágenes de la versión 3
('<Bq' + JPEG, etapa 14). Solo usa la biblioteca estándar: el JPEG llega ya comprimido
y se reenvía tal cual, sin decodificarlo.

Uso (en la Pi, con el venv activo, desde la raíz del proyecto):
    python -m apps.visor_web
    python -m apps.pi_vio --ip 127.0.0.1
y en la tablet abre:  http://<IP de la Pi>:8000
Para que arranquen solos al encender la Pi: bash sistema/instalar_servicios.sh instalar

No se puede ejecutar a la vez que apps.pc_visor en la misma máquina: los dos
escuchan en el puerto UDP 5005.
"""

import argparse
import json
import math
import signal
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from apps import red_wifi

# ---------------------------------------------------------------------------
# Formato de los paquetes (copia literal de vio/protocolo.py, sin cambios)
# ---------------------------------------------------------------------------
FORMATO_V1 = "<Bq9d"  # versión, t_ns, R_w_i (9 float64, por filas)
FORMATO_V2 = "<Bq9d3d"  # lo mismo + p_w (3 float64)
TAM_V1 = struct.calcsize(FORMATO_V1)  # 81 bytes
TAM_V2 = struct.calcsize(FORMATO_V2)  # 105 bytes

# Máximo de puntos de trayectoria que se mandan al navegador en cada mensaje.
# Si la tablet se conecta tarde, recibe la trayectoria antigua en trozos.
MAX_PUNTOS_POR_MENSAJE = 2000

# Imágenes (etapa 14): copia literal de vio/protocolo.py
FORMATO_CABECERA_IMAGEN = "<Bq"  # versión (3) + t_ns, y detrás los bytes del JPEG
TAM_CABECERA_IMAGEN = struct.calcsize(FORMATO_CABECERA_IMAGEN)  # 9 bytes
VERSION_IMAGEN = 3
PERIODO_VIDEO_S = 0.03  # cada cuánto mira el vídeo si hay imagen nueva

# Panel de orientación: dónde cae en la pantalla cada eje del mundo, como [derecha, arriba].
# Vista "desde detrás de la placa": el observador está detrás (-X), un poco a su derecha (-Y)
# y por encima (+Z), como la ves al sujetarla con la cámara mirando al frente.
# Para que el dibujo NO salga en espejo, la dirección hacia el observador
#     n = (componente "derecha" de los 3 ejes) × (componente "arriba" de los 3 ejes)
# debe apuntar hacia donde creemos estar: n_z > 0 (encima) y n_x < 0 (detrás).
EJES_PANTALLA = {"x": [0.40, 0.30], "y": [-1.0, 0.0], "z": [0.0, 1.0]}

# Etapa 16 — Prueba de trayectoria. Copia literal de config/vio.yaml (prueba, control) y de
# vio/protocolo.py: este programa solo usa la biblioteca estándar y no lee YAML
PUERTO_CONTROL = 5007  # UDP de las órdenes a pi_vio (en la propia Pi)
ORDEN_REINICIAR = b"reiniciar"
RECORRIDO_PASO_MIN_M = (
    0.01  # el recorrido se suma a pasos de al menos 1 cm (fuera el temblor)
)
RECORRIDO_MIN_M = 0.5  # con menos recorrido, el error de cierre en % no significa nada
CIERRE_BUENO_PCT = 2.0  # colores del error de cierre (orientativos)
CIERRE_REGULAR_PCT = 5.0
IGNORAR_TRAS_REINICIO_S = (
    0.5  # poses de la prueba anterior que aún estén en camino: fuera
)


def decodificar_paquete(datos):
    """Devuelve (version, t_ns, R, p). R es una lista 3x3 y p una lista [x, y, z]
    o None si el paquete es v1. Lanza ValueError si el paquete no es válido."""
    if len(datos) == TAM_V1:
        campos = struct.unpack(FORMATO_V1, datos)
        if campos[0] != 1:
            raise ValueError("tamaño de v1 pero versión %d" % campos[0])
        p = None
    elif len(datos) == TAM_V2:
        campos = struct.unpack(FORMATO_V2, datos)
        if campos[0] != 2:
            raise ValueError("tamaño de v2 pero versión %d" % campos[0])
        p = [campos[11], campos[12], campos[13]]
    else:
        raise ValueError("tamaño inesperado: %d bytes" % len(datos))

    version = campos[0]
    t_ns = campos[1]
    r9 = campos[2:11]
    # R_w_i se envió por filas (orden C de numpy): fila 0, fila 1, fila 2
    R = [
        [r9[0], r9[1], r9[2]],
        [r9[3], r9[4], r9[5]],
        [r9[6], r9[7], r9[8]],
    ]
    return version, t_ns, R, p


def decodificar_imagen(datos):
    """Devuelve (t_ns, jpeg) de un paquete de imagen. Lanza ValueError si no lo es."""
    if len(datos) <= TAM_CABECERA_IMAGEN or datos[0] != VERSION_IMAGEN:
        raise ValueError("no es un paquete de imagen (versión 3)")
    _, t_ns = struct.unpack(FORMATO_CABECERA_IMAGEN, datos[:TAM_CABECERA_IMAGEN])
    return t_ns, datos[TAM_CABECERA_IMAGEN:]


def todos_finitos(valores):
    """True si ningún valor es NaN o infinito (JSON no admite NaN)."""
    for v in valores:
        if not math.isfinite(v):
            return False
    return True


# ---------------------------------------------------------------------------
# Estado compartido entre el hilo UDP y los hilos HTTP
# ---------------------------------------------------------------------------
class EstadoCompartido:
    def __init__(self):
        self.cerrojo = threading.Lock()
        self.version = None
        self.t_ns = None
        self.R = None  # lista 3x3 o None
        self.p = None  # [x, y, z] o None
        self.hora_ultimo = None  # time.monotonic() del último paquete válido
        self.trayectoria = []  # lista de [x, y, z]
        self.id_sesion = 0  # cambia cuando se reinicia la VIO
        self.recibidos = 0
        self.invalidos = 0
        self.no_finitos = 0
        # Etapa 14: la última imagen con sus pistas
        self.jpeg = None  # bytes del último JPEG, o None
        self.id_jpeg = 0  # aumenta con cada imagen nueva
        self.imagenes = 0  # imágenes recibidas
        # Etapa 16: la prueba de trayectoria en curso
        self.recorrido_m = 0.0  # longitud del camino (a pasos de RECORRIDO_PASO_MIN_M)
        self.ancla = None  # último punto contado en el recorrido
        self.alejamiento_max_m = 0.0
        self.ignorar_hasta = (
            0.0  # time.monotonic() hasta el que se tiran las poses (tras reiniciar)
        )
        self.reinicios = 0


def distancia(a, b):
    """Distancia entre dos puntos [x, y, z]."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def anadir_punto(estado, p):
    """Un punto más de la trayectoria, con el recorrido y el alejamiento (con el cerrojo cogido).

    Mismo cálculo que vio/lazo.py -> recorrido (allí con numpy, aquí sin él): se suma la
    distancia al "ancla" (último punto contado) solo cuando llega a RECORRIDO_PASO_MIN_M.
    """
    estado.trayectoria.append(p)
    if estado.ancla is None:
        estado.ancla = p
        return
    d = distancia(p, estado.ancla)
    if d >= RECORRIDO_PASO_MIN_M:
        estado.recorrido_m += d
        estado.ancla = p
    estado.alejamiento_max_m = max(
        estado.alejamiento_max_m, distancia(p, estado.trayectoria[0])
    )


def vaciar_trayectoria(estado):
    """Prueba nueva: trayectoria y métricas a cero (con el cerrojo cogido)."""
    estado.trayectoria = []
    estado.id_sesion += 1
    estado.recorrido_m = 0.0
    estado.ancla = None
    estado.alejamiento_max_m = 0.0


def metricas_prueba(estado):
    """Lo que se muestra en el panel "Prueba de lazo" (con el cerrojo cogido)."""
    if len(estado.trayectoria) < 2:
        return None
    cierre = distancia(estado.trayectoria[-1], estado.trayectoria[0])
    pct = (
        100.0 * cierre / estado.recorrido_m
        if estado.recorrido_m >= RECORRIDO_MIN_M
        else None
    )
    if pct is None:
        nivel = None
    elif pct <= CIERRE_BUENO_PCT:
        nivel = "bueno"
    elif pct <= CIERRE_REGULAR_PCT:
        nivel = "regular"
    else:
        nivel = "malo"
    return {
        "recorrido_m": estado.recorrido_m,
        "cierre_m": cierre,
        "cierre_pct": pct,
        "nivel": nivel,
        "alejamiento_max_m": estado.alejamiento_max_m,
        "reinicios": estado.reinicios,
    }


def bucle_udp(sock, estado, parar):
    """Hilo que recibe paquetes UDP y actualiza el estado compartido."""
    while not parar.is_set():
        # 1) Esperar un paquete (con timeout para poder terminar limpio)
        try:
            datos, _origen = sock.recvfrom(1024)
        except socket.timeout:
            continue
        except OSError:
            break

        # 2) Decodificar
        try:
            version, t_ns, R, p = decodificar_paquete(datos)
        except (ValueError, struct.error):
            with estado.cerrojo:
                estado.invalidos += 1
            continue

        # 2b) Justo después de "Nueva prueba": poses de la prueba anterior aún en camino
        if time.monotonic() < estado.ignorar_hasta:
            continue

        # 3) Descartar valores no finitos (si el filtro diverge)
        R_ok = todos_finitos(R[0] + R[1] + R[2])
        p_ok = p is not None and todos_finitos(p)

        # 4) Guardar en el estado compartido
        with estado.cerrojo:
            # Si el tiempo retrocede, se ha relanzado pi_vio (o la grabación):
            # empezamos una trayectoria nueva.
            if estado.t_ns is not None and t_ns < estado.t_ns:
                vaciar_trayectoria(estado)

            estado.version = version
            estado.t_ns = t_ns
            estado.R = R if R_ok else None
            estado.p = p if p_ok else None
            estado.hora_ultimo = time.monotonic()
            estado.recibidos += 1

            if p is not None and not p_ok:
                estado.no_finitos += 1
            if p_ok:
                anadir_punto(estado, p)


def bucle_imagenes(sock, estado, parar):
    """Hilo que recibe las imágenes (etapa 14) y guarda solo la última."""
    while not parar.is_set():
        try:
            datos, _origen = sock.recvfrom(65536)  # un JPEG cabe en un paquete
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            _t_ns, jpeg = decodificar_imagen(datos)
        except (ValueError, struct.error):
            continue
        with estado.cerrojo:
            estado.jpeg = jpeg
            estado.id_jpeg += 1
            estado.imagenes += 1


# ---------------------------------------------------------------------------
# Etapa 18: cambio de red en curso (uno a la vez, aunque haya dos tablets)
# ---------------------------------------------------------------------------
class CambioRed:
    def __init__(self):
        self.cerrojo = threading.Lock()
        self.en_curso = None  # texto del cambio que se está haciendo, o None
        self.ultimo = None  # {"descripcion", "ok", "hora"} del último cambio terminado


# ---------------------------------------------------------------------------
# Servidor HTTP: página + flujo de eventos + vídeo
# ---------------------------------------------------------------------------
class ManejadorHTTP(BaseHTTPRequestHandler):
    # Se rellenan en main() antes de arrancar el servidor
    estado = None
    periodo_envio = 1.0 / 15.0
    puerto_control = PUERTO_CONTROL
    # Etapa 18: el módulo que habla con NetworkManager (los tests lo cambian por uno falso)
    red = red_wifi
    cambio_red = CambioRed()

    def log_message(self, formato, *args):
        # Silenciar el registro de cada petición para no gastar CPU en la terminal
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.enviar_pagina()
        elif self.path == "/eventos":
            self.enviar_eventos()
        elif self.path.startswith("/video"):
            self.enviar_video()
        elif self.path.startswith("/foto"):
            self.enviar_foto()
        elif self.path == "/red":
            self.enviar_estado_red()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/reiniciar":
            self.reiniciar()
        elif self.path == "/red/punto":
            self.pedir_punto()
        elif self.path == "/red/cliente":
            self.pedir_cliente()
        else:
            self.send_error(404)

        # ----- Etapa 18: red Wi-Fi -----

    def responder_json(self, codigo, datos):
        cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(cuerpo)

    def enviar_estado_red(self):
        """GET /red: modo actual, redes guardadas y si se puede cambiar desde aquí."""
        datos = self.red.estado()
        datos["punto_creado"] = self.red.existe_punto()
        datos["puede_cambiar"] = self.red.puede_cambiar()
        with self.cambio_red.cerrojo:
            datos["cambiando"] = self.cambio_red.en_curso
            datos["ultimo_cambio"] = self.cambio_red.ultimo
        self.responder_json(200, datos)

    def empezar_cambio(self, descripcion, funcion, *argumentos):
        """Reserva el cambio (uno a la vez). Devuelve False si ya hay otro en curso."""
        with self.cambio_red.cerrojo:
            if self.cambio_red.en_curso is not None:
                return False
            self.cambio_red.en_curso = descripcion
        cambio = self.cambio_red

        def trabajo():
            # En otro hilo: la respuesta HTTP ya ha salido (la función espera RETRASO_CAMBIO_S
            # antes de tocar la red, para que llegue a la tablet antes del corte)
            try:
                ok = bool(funcion(*argumentos))
            except Exception:
                ok = False
            with cambio.cerrojo:
                cambio.en_curso = None
                cambio.ultimo = {
                    "descripcion": descripcion,
                    "ok": ok,
                    "hora": time.strftime("%H:%M:%S"),
                }
            print(
                "Cambio de red (%s): %s"
                % (descripcion, "hecho" if ok else "FALLÓ, vuelta atrás")
            )

        threading.Thread(target=trabajo, daemon=True).start()
        return True

    def comprobar_que_se_puede(self):
        """Comprobaciones comunes. Devuelve None si se puede cambiar, o (código, mensaje)."""
        if not self.red.puede_cambiar():
            return (
                403,
                "El visor no puede cambiar de red sin contraseña. En la Pi: bash sistema/red.sh permitir",
            )
        if not self.red.existe_punto():
            return (
                409,
                "Falta crear el punto de acceso. En la Pi: bash sistema/red.sh crear",
            )
        return None

    def pedir_punto(self):
        """POST /red/punto: deja la red actual (sin olvidarla) y emite VIO-Pi."""
        problema = self.comprobar_que_se_puede()
        if problema is not None:
            self.responder_json(problema[0], {"ok": False, "mensaje": problema[1]})
            return
        if not self.empezar_cambio(
            "punto de acceso " + self.red.SSID_PUNTO, self.red.cambiar_a_punto
        ):
            self.responder_json(
                409, {"ok": False, "mensaje": "Ya hay un cambio de red en curso"}
            )
            return
        url = "http://%s:%d" % (self.red.IP_PUNTO, self.red.PUERTO_WEB)
        self.responder_json(
            200,
            {
                "ok": True,
                "mensaje": "La Pi deja esta red y emite %s. Conecta la tablet a la red %s y abre:"
                % (self.red.SSID_PUNTO, self.red.SSID_PUNTO),
                "enlaces": [url],
            },
        )

    def pedir_cliente(self):
        """POST /red/cliente con {"conexion": nombre}: se conecta a esa red guardada."""
        try:
            largo = int(self.headers.get("Content-Length", "0"))
            conexion = json.loads(self.rfile.read(largo).decode("utf-8"))["conexion"]
        except (ValueError, KeyError, TypeError):
            self.responder_json(
                400,
                {"ok": False, "mensaje": 'Falta {"conexion": nombre} en la petición'},
            )
            return
        problema = self.comprobar_que_se_puede()
        if problema is not None:
            self.responder_json(problema[0], {"ok": False, "mensaje": problema[1]})
            return
        estado_red = self.red.estado()
        redes = {r["conexion"]: r for r in estado_red["redes"]}
        if conexion not in redes:
            # Solo redes guardadas: nada de pasar a nmcli un nombre cualquiera
            self.responder_json(
                400,
                {
                    "ok": False,
                    "mensaje": "La red %r no está guardada en la Pi" % conexion,
                },
            )
            return
        red = redes[conexion]
        if not self.empezar_cambio(
            "red " + red["ssid"], self.red.cambiar_a_red, conexion
        ):
            self.responder_json(
                409, {"ok": False, "mensaje": "Ya hay un cambio de red en curso"}
            )
            return
        # La IP nueva la decide el router de esa red: se da la última conocida y el nombre .local
        enlaces = []
        if red.get("ultima_ip"):
            enlaces.append("http://%s:%d" % (red["ultima_ip"], self.red.PUERTO_WEB))
        enlaces.append(estado_red["url_mdns"])
        self.responder_json(
            200,
            {
                "ok": True,
                "mensaje": "La Pi se conecta a %s (si no lo consigue en %d s, vuelve a emitir %s). "
                "Conecta la tablet a %s y prueba:"
                % (
                    red["ssid"],
                    self.red.ESPERA_CONEXION_S,
                    self.red.SSID_PUNTO,
                    red["ssid"],
                ),
                "enlaces": enlaces,
            },
        )

    def reiniciar(self):
        """Botón "Nueva prueba": orden a pi_vio por UDP y trayectoria a cero.

        Si pi_vio no está en marcha, la orden se pierde sin más (UDP): no es un error.
        """
        orden = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            orden.sendto(ORDEN_REINICIAR, ("127.0.0.1", self.puerto_control))
        except OSError:
            pass
        finally:
            orden.close()
        with self.estado.cerrojo:
            vaciar_trayectoria(self.estado)
            self.estado.ignorar_hasta = time.monotonic() + IGNORAR_TRAS_REINICIO_S
            self.estado.reinicios += 1
        cuerpo = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def enviar_pagina(self):
        cuerpo = PAGINA_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(cuerpo)

    def enviar_eventos(self):
        """Server-Sent Events: una conexión HTTP abierta por la que mandamos
        'data: {json}\\n\\n' periódicamente. El navegador la reabre solo si se corta."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        estado = self.estado
        indice_enviado = 0  # cuántos puntos de la trayectoria tiene ya este navegador
        sesion_enviada = None

        try:
            while True:
                with estado.cerrojo:
                    # Si la VIO se ha reiniciado, el navegador debe empezar de cero
                    if sesion_enviada != estado.id_sesion:
                        sesion_enviada = estado.id_sesion
                        indice_enviado = 0

                    inicio = indice_enviado
                    nuevos = estado.trayectoria[
                        inicio : inicio + MAX_PUNTOS_POR_MENSAJE
                    ]
                    indice_enviado = inicio + len(nuevos)

                    if estado.hora_ultimo is None:
                        sin_datos_s = None
                    else:
                        sin_datos_s = time.monotonic() - estado.hora_ultimo

                    mensaje = {
                        "sesion": estado.id_sesion,
                        "inicio": inicio,  # posición de 'nuevos' en la trayectoria
                        "nuevos": nuevos,
                        "version": estado.version,
                        "t_s": None if estado.t_ns is None else estado.t_ns * 1e-9,
                        "R": estado.R,
                        "p": estado.p,
                        "sin_datos_s": sin_datos_s,
                        "recibidos": estado.recibidos,
                        "invalidos": estado.invalidos,
                        "no_finitos": estado.no_finitos,
                        "imagenes": estado.imagenes,
                        "prueba": metricas_prueba(estado),
                    }

                texto = "data: " + json.dumps(mensaje) + "\n\n"
                self.wfile.write(texto.encode("utf-8"))
                self.wfile.flush()
                time.sleep(self.periodo_envio)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            # La tablet ha cerrado la pestaña o se ha perdido la Wi-Fi
            return

    def enviar_video(self):
        """Vídeo MJPEG: una respuesta que no termina, con un JPEG tras otro separados por
        una frontera. Los navegadores lo muestran directamente en un <img>."""
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=fotograma"
        )
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        estado = self.estado
        ultimo_id = -1
        try:
            self.wfile.write(b"--fotograma\r\n")
            while True:
                with estado.cerrojo:
                    jpeg, id_jpeg = estado.jpeg, estado.id_jpeg
                if (
                    jpeg is not None and id_jpeg != ultimo_id
                ):  # solo cuando hay imagen nueva
                    ultimo_id = id_jpeg
                    cabecera = (
                        "Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                        % len(jpeg)
                    )
                    self.wfile.write(cabecera.encode("ascii"))
                    self.wfile.write(jpeg)
                    # La frontera va DETRÁS de cada imagen: el navegador muestra una imagen
                    # cuando ve la frontera que la cierra (si fuera delante, la última
                    # imagen no se vería hasta que llegase la siguiente)
                    self.wfile.write(b"\r\n--fotograma\r\n")
                    self.wfile.flush()
                time.sleep(PERIODO_VIDEO_S)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def enviar_foto(self):
        """La última imagen, sola (para comprobar desde el navegador o con curl)."""
        with self.estado.cerrojo:
            jpeg = self.estado.jpeg
        if jpeg is None:
            self.send_error(503, "Aún no ha llegado ninguna imagen")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(jpeg)


def ip_local_probable():
    """IP de la interfaz por la que sale el tráfico (no envía nada realmente)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def terminar_con_sigterm():
    """Que SIGTERM ("systemctl stop") cierre el servidor igual que Ctrl+C."""

    def manejador(_senal, _marco):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, manejador)


def main():
    parser = argparse.ArgumentParser(description="Visor web de la VIO (para tablet)")
    parser.add_argument(
        "--puerto-udp",
        type=int,
        default=5005,
        help="puerto donde llegan los paquetes de pi_vio (5005)",
    )
    parser.add_argument(
        "--puerto-imagen",
        type=int,
        default=5006,
        help="puerto donde llegan las imágenes con pistas de pi_vio (5006)",
    )
    parser.add_argument(
        "--puerto-web",
        type=int,
        default=8000,
        help="puerto HTTP para el navegador (8000)",
    )
    parser.add_argument(
        "--puerto-control",
        type=int,
        default=PUERTO_CONTROL,
        help="puerto UDP de las órdenes a pi_vio (5007)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=15.0,
        help="mensajes por segundo hacia el navegador (15)",
    )
    args = parser.parse_args()
    terminar_con_sigterm()

    estado = EstadoCompartido()
    parar = threading.Event()

    # 1) Socket UDP: se crea aquí para que un error de puerto ocupado salga claro
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", args.puerto_udp))
    except OSError as e:
        print("ERROR: no puedo escuchar en UDP %d (%s)." % (args.puerto_udp, e))
        print("¿Está apps.pc_visor abierto en esta máquina? Ciérralo primero.")
        return
    sock.settimeout(0.5)

    hilo_udp = threading.Thread(
        target=bucle_udp, args=(sock, estado, parar), daemon=True
    )
    hilo_udp.start()

    # 1b) Segundo socket para las imágenes (etapa 14)
    sock_img = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock_img.bind(("0.0.0.0", args.puerto_imagen))
    except OSError as e:
        print("ERROR: no puedo escuchar en UDP %d (%s)." % (args.puerto_imagen, e))
        return
    sock_img.setsockopt(
        socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20
    )  # 1 MB: varias imágenes en espera
    sock_img.settimeout(0.5)
    hilo_img = threading.Thread(
        target=bucle_imagenes, args=(sock_img, estado, parar), daemon=True
    )
    hilo_img.start()

    # 2) Servidor HTTP (un hilo por navegador conectado)
    ManejadorHTTP.estado = estado
    ManejadorHTTP.periodo_envio = 1.0 / args.hz
    ManejadorHTTP.puerto_control = args.puerto_control
    servidor = ThreadingHTTPServer(("0.0.0.0", args.puerto_web), ManejadorHTTP)

    ip = ip_local_probable()
    print(
        "Escuchando UDP en los puertos %d (pose) y %d (imágenes)."
        % (args.puerto_udp, args.puerto_imagen)
    )
    if ip:
        print("Abre en la tablet:  http://%s:%d" % (ip, args.puerto_web))
    else:
        print(
            "Abre en la tablet:  http://<IP de la Pi>:%d   (mira 'hostname -I')"
            % args.puerto_web
        )
    print("Ctrl+C para terminar.")

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        parar.set()
        servidor.server_close()
        sock.close()
        sock_img.close()
        print("Visor web detenido.")


# ---------------------------------------------------------------------------
# Página que se sirve a la tablet (HTML + JavaScript, sin dependencias externas)
# ---------------------------------------------------------------------------
PAGINA_HTML = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visor VIO</title>
<style>
  body { margin: 0; background: #111; color: #ddd; font-family: sans-serif; }
  #cabecera { padding: 8px 12px; display: flex; gap: 16px; flex-wrap: wrap;
              align-items: center; font-size: 15px; }
  #estado { font-weight: bold; padding: 3px 10px; border-radius: 4px; }
  .ok  { background: #1b5e20; }
  .mal { background: #b71c1c; }
  .esp { background: #6d4c00; }
  #texto_pose { font-family: monospace; }
  #paneles { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
             gap: 8px; padding: 8px; }
  .panel { background: #1c1c1c; border-radius: 6px; padding: 6px; }
  .panel h2 { margin: 0 0 4px; font-size: 14px; font-weight: normal; color: #aaa; }
  canvas { width: 100%; aspect-ratio: 1 / 1; display: block; }
  #video { width: 100%; aspect-ratio: 4 / 3; display: block; background: #000; object-fit: contain; }
  .panel.ancho { grid-column: 1 / -1; }
  /* Etapa 16: prueba de lazo y vista 3D */
  #cifras { display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: center; }
  .cifra { display: flex; flex-direction: column; }
  .cifra .etq { font-size: 12px; color: #999; }
  .cifra .num { font-size: 26px; font-family: monospace; }
  .num.bueno { color: #66bb6a; }  .num.regular { color: #ffca28; }  .num.malo { color: #ef5350; }
  #boton_prueba { margin-left: auto; font-size: 17px; padding: 10px 16px; border: 0;
                  border-radius: 6px; background: #1565c0; color: #fff; }
  #boton_prueba.confirmar { background: #c62828; }
  #lienzo_3d { aspect-ratio: 4 / 3; touch-action: none; }   /* el dedo gira la vista, no la página */
  /* Etapa 18: red Wi-Fi */
  #red { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  #red_modo { font-size: 16px; margin-right: auto; }
  #red button, #red select { font-size: 16px; padding: 8px 12px; border: 0; border-radius: 6px; }
  #red button { background: #1565c0; color: #fff; }
  #red button.confirmar { background: #c62828; }
  #red button:disabled, #red select:disabled { opacity: 0.4; }
  #red_aviso { margin-top: 8px; font-size: 15px; color: #ffca28; }
  #red_aviso a { color: #90caf9; font-size: 18px; margin-right: 18px; }
</style>
</head>
<body>
<div id="cabecera">
  <span id="estado" class="esp">CONECTANDO…</span>
  <span id="texto_pose">—</span>
</div>
<div id="paneles">
  <div class="panel ancho"><h2>Cámara y pistas (rojo: recién detectada → verde: seguida 20 imágenes o más)</h2><img id="video" alt="Esperando imágenes de pi_vio…"></div>
    <div class="panel ancho"><h2>Prueba de lazo: sal de una marca, da una vuelta y vuelve a ella</h2>
    <div id="cifras">
      <div class="cifra"><span class="etq">recorrido</span><span id="m_recorrido" class="num">—</span></div>
      <div class="cifra"><span class="etq">distancia al origen</span><span id="m_cierre" class="num">—</span></div>
      <div class="cifra"><span class="etq">error de cierre</span><span id="m_pct" class="num">—</span></div>
      <div class="cifra"><span class="etq">máx. alejamiento</span><span id="m_alejamiento" class="num">—</span></div>
      <button id="boton_prueba">Nueva prueba</button>
    </div>
  </div>
  <div class="panel ancho"><h2>Vista 3D [m] — arrastra: girar · pellizco o rueda: zoom · doble toque: vista inicial</h2><canvas id="lienzo_3d"></canvas></div>
  <div class="panel"><h2>Planta X–Y [m] (flecha roja: eje x de la IMU)</h2><canvas id="lienzo_xy"></canvas></div>
  <div class="panel"><h2>Alzado X–Z [m]</h2><canvas id="lienzo_xz"></canvas></div>
  <div class="panel"><h2>Orientación vista desde detrás de la placa (x rojo, y verde, z azul)</h2><canvas id="lienzo_ori"></canvas></div>
  <!-- Etapa 18: red Wi-Fi -->
  <div class="panel ancho"><h2>Red Wi-Fi de la Pi (al cambiar se corta esta página: abre la dirección que se indique)</h2>
    <div id="red">
      <span id="red_modo">—</span>
      <button id="boton_punto">Emitir red VIO-Pi</button>
      <select id="red_lista"></select>
      <button id="boton_cliente">Conectar a esta red</button>
    </div>
    <div id="red_aviso"></div>
  </div>
</div>

<script>
// ------------------------------------------------------------------
// Estado en el navegador
// ------------------------------------------------------------------
let trayectoria = [];        // lista de [x, y, z] recibida de la Pi
let ultimo = null;           // último mensaje recibido
let horaUltimoMensaje = 0;   // Date.now() del último mensaje (para detectar Pi caída)
let hayQueDibujar = true;

// ------------------------------------------------------------------
// Recepción: Server-Sent Events (el navegador reconecta solo)
// ------------------------------------------------------------------
const fuente = new EventSource("/eventos");

fuente.onmessage = function (evento) {
  const m = JSON.parse(evento.data);

  // 'inicio' es la posición de los puntos nuevos dentro de la trayectoria de la Pi.
  // Si es 0 (conexión nueva o VIO reiniciada) se vacía la trayectoria local.
  if (m.inicio <= trayectoria.length) {
    trayectoria.length = m.inicio;
    for (const q of m.nuevos) {
      trayectoria.push(q);
    }
  }

  ultimo = m;
  horaUltimoMensaje = Date.now();
  actualizarCabecera();
  actualizarPrueba(m.prueba);
  hayQueDibujar = true;
};

fuente.onerror = function () {
  ponerEstado("SIN CONEXIÓN CON LA PI", "mal");
};

// El vídeo (etapa 14): un <img> con MJPEG NO se reconecta solo si se corta (Pi reiniciada).
// Cada vez que el flujo de eventos (re)conecta, se vuelve a pedir /video; el ?t= distinto
// obliga al navegador a abrir una conexión nueva en vez de quedarse con la imagen vieja.
fuente.onopen = function () {
  document.getElementById("video").src = "/video?t=" + Date.now();
};

// Si la Pi deja de mandar mensajes (proceso caído) lo indicamos
setInterval(function () {
  if (horaUltimoMensaje > 0 && Date.now() - horaUltimoMensaje > 2000) {
    ponerEstado("SIN CONEXIÓN CON LA PI", "mal");
  }
}, 500);

// ------------------------------------------------------------------
// Cabecera de texto
// ------------------------------------------------------------------
function ponerEstado(texto, clase) {
  const e = document.getElementById("estado");
  e.textContent = texto;
  e.className = clase;
}

function eulerZYXgrados(R) {
  // R_w_i = Rz(yaw) * Ry(pitch) * Rx(roll)
  const yaw = Math.atan2(R[1][0], R[0][0]);
  const pitch = Math.asin(Math.max(-1, Math.min(1, -R[2][0])));
  const roll = Math.atan2(R[2][1], R[2][2]);
  return [roll, pitch, yaw].map(function (a) { return a * 180 / Math.PI; });
}

function actualizarCabecera() {
  const m = ultimo;
  if (m.sin_datos_s === null) {
    ponerEstado("ESPERANDO PAQUETES DE pi_vio", "esp");
  } else if (m.sin_datos_s > 1.0) {
    ponerEstado("SIN DATOS desde hace " + m.sin_datos_s.toFixed(1) + " s", "mal");
  } else {
    ponerEstado("OK · v" + m.version, "ok");
  }

  let texto = "";
  if (m.t_s !== null) texto += "t=" + m.t_s.toFixed(2) + " s   ";
  if (m.p) {
    texto += "p=[" + m.p.map(function (v) { return v.toFixed(3); }).join(", ") + "] m   ";
  }
  if (m.R) {
    const rpy = eulerZYXgrados(m.R);
    texto += "roll/pitch/yaw=[" + rpy.map(function (v) { return v.toFixed(1); }).join(", ") + "]°   ";
  }
  texto += "paquetes=" + m.recibidos;
  if (m.invalidos > 0) texto += "  inválidos=" + m.invalidos;
  if (m.no_finitos > 0) texto += "  NaN=" + m.no_finitos;
  texto += "  puntos=" + trayectoria.length;
  if (m.imagenes !== undefined) texto += "  imágenes=" + m.imagenes;
  document.getElementById("texto_pose").textContent = texto;
}

// ------------------------------------------------------------------
// Utilidades de dibujo
// ------------------------------------------------------------------
function prepararLienzo(lienzo) {
  // Ajusta la resolución interna del canvas a la de la pantalla (nitidez)
  const dpr = window.devicePixelRatio || 1;
  const ancho = lienzo.clientWidth;
  const alto = lienzo.clientHeight;
  const anchoPx = Math.round(ancho * dpr);
  const altoPx = Math.round(alto * dpr);
  if (lienzo.width !== anchoPx || lienzo.height !== altoPx) {
    lienzo.width = anchoPx;
    lienzo.height = altoPx;
  }
  const ctx = lienzo.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, ancho, alto);
  return { ctx: ctx, ancho: ancho, alto: alto };
}

function pasoRejilla(rango) {
  // Elige un paso "redondo" (1, 2 o 5 x 10^n) para unas 6 líneas
  const bruto = rango / 6;
  const potencia = Math.pow(10, Math.floor(Math.log10(bruto)));
  const n = bruto / potencia;
  let m = 1;
  if (n > 5) m = 10;
  else if (n > 2) m = 5;
  else if (n > 1) m = 2;
  return m * potencia;
}

function linea(ctx, x0, y0, x1, y1) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

// ------------------------------------------------------------------
// Vista ortográfica de dos ejes del mundo (ia = horizontal, ib = vertical)
// ------------------------------------------------------------------
function dibujarVista(idLienzo, ia, ib) {
  const l = prepararLienzo(document.getElementById(idLienzo));
  const ctx = l.ctx, ancho = l.ancho, alto = l.alto;

  // 1) Límites: trayectoria + origen, mínimo 1 m de lado
  let minA = -0.5, maxA = 0.5, minB = -0.5, maxB = 0.5;
  for (const q of trayectoria) {
    if (q[ia] < minA) minA = q[ia];
    if (q[ia] > maxA) maxA = q[ia];
    if (q[ib] < minB) minB = q[ib];
    if (q[ib] > maxB) maxB = q[ib];
  }
  const centroA = (minA + maxA) / 2;
  const centroB = (minB + maxB) / 2;
  const rango = Math.max(maxA - minA, maxB - minB) * 1.15;   // margen del 15 %
  const escala = Math.min(ancho, alto) / rango;               // píxeles por metro (igual en los dos ejes)

  function aPx(a) { return ancho / 2 + (a - centroA) * escala; }
  function bPx(b) { return alto / 2 - (b - centroB) * escala; }

  // 2) Rejilla con etiquetas
  const paso = pasoRejilla(rango);
  const decimales = Math.max(0, -Math.floor(Math.log10(paso)));
  ctx.lineWidth = 1;
  ctx.font = "11px sans-serif";
  const primeraA = Math.ceil((centroA - rango / 2) / paso);
  const ultimaA = Math.floor((centroA + rango / 2) / paso);
  for (let k = primeraA; k <= ultimaA; k++) {
    const a = k * paso;
    const x = aPx(a);
    ctx.strokeStyle = (k === 0) ? "#666" : "#2c2c2c";
    linea(ctx, x, 0, x, alto);
    ctx.fillStyle = "#777";
    ctx.fillText(a.toFixed(decimales), x + 2, alto - 4);
  }
  const primeraB = Math.ceil((centroB - rango / 2) / paso);
  const ultimaB = Math.floor((centroB + rango / 2) / paso);
  for (let k = primeraB; k <= ultimaB; k++) {
    const b = k * paso;
    const y = bPx(b);
    ctx.strokeStyle = (k === 0) ? "#666" : "#2c2c2c";
    linea(ctx, 0, y, ancho, y);
    ctx.fillStyle = "#777";
    ctx.fillText(b.toFixed(decimales), 2, y - 2);
  }

  // 3) Trayectoria
  if (trayectoria.length > 1) {
    ctx.strokeStyle = "#4fc3f7";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(aPx(trayectoria[0][ia]), bPx(trayectoria[0][ib]));
    for (let k = 1; k < trayectoria.length; k++) {
      ctx.lineTo(aPx(trayectoria[k][ia]), bPx(trayectoria[k][ib]));
    }
    ctx.stroke();
  }

  // 4) Posición actual y eje x de la IMU proyectado en este plano
  if (ultimo && ultimo.p) {
    const x = aPx(ultimo.p[ia]);
    const y = bPx(ultimo.p[ib]);
    if (ultimo.R) {
      // Columna 0 de R_w_i = eje x de la IMU expresado en el mundo
      const da = ultimo.R[ia][0];
      const db = ultimo.R[ib][0];
      const largo = 35;   // píxeles para un vector unitario
      ctx.strokeStyle = "#ef5350";
      ctx.lineWidth = 3;
      linea(ctx, x, y, x + da * largo, y - db * largo);
    }
    ctx.fillStyle = "#ffeb3b";
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// ------------------------------------------------------------------
// Orientación: ejes de la IMU vistos desde detrás de la placa
// ------------------------------------------------------------------
// [derecha, arriba] en pantalla de cada eje del mundo (lo rellena visor_web.py con
// EJES_PANTALLA; ahí está explicado por qué así el dibujo no sale en espejo)
const EJES_PANTALLA = __EJES_PANTALLA__;

function proyectarOrientacion(v) {
  // v = vector en ejes del mundo. Devuelve [horizontal hacia la derecha, vertical hacia arriba]
  const ex = EJES_PANTALLA.x, ey = EJES_PANTALLA.y, ez = EJES_PANTALLA.z;
  return [v[0] * ex[0] + v[1] * ey[0] + v[2] * ez[0],
          v[0] * ex[1] + v[1] * ey[1] + v[2] * ez[1]];
}

function dibujarOrientacion() {
  const l = prepararLienzo(document.getElementById("lienzo_ori"));
  const ctx = l.ctx, ancho = l.ancho, alto = l.alto;
  const cx = ancho / 2, cy = alto / 2;
  const largo = Math.min(ancho, alto) * 0.38;
  ctx.font = "13px sans-serif";

  // Ejes del mundo (grises, discontinuos)
  const nombresMundo = ["X", "Y", "Z"];
  ctx.setLineDash([5, 5]);
  ctx.lineWidth = 1;
  for (let i = 0; i < 3; i++) {
    const e = [0, 0, 0];
    e[i] = 1;
    const q = proyectarOrientacion(e);
    ctx.strokeStyle = "#666";
    linea(ctx, cx, cy, cx + q[0] * largo, cy - q[1] * largo);
    ctx.fillStyle = "#888";
    ctx.fillText(nombresMundo[i], cx + q[0] * largo * 1.08, cy - q[1] * largo * 1.08);
  }
  ctx.setLineDash([]);

  // Ejes de la IMU = columnas de R_w_i
  if (ultimo && ultimo.R) {
    const R = ultimo.R;
    const colores = ["#ef5350", "#66bb6a", "#42a5f5"];
    const nombres = ["x", "y", "z"];
    ctx.lineWidth = 4;
    for (let j = 0; j < 3; j++) {
      const eje = [R[0][j], R[1][j], R[2][j]];
      const q = proyectarOrientacion(eje);
      ctx.strokeStyle = colores[j];
      linea(ctx, cx, cy, cx + q[0] * largo, cy - q[1] * largo);
      ctx.fillStyle = colores[j];
      ctx.fillText(nombres[j], cx + q[0] * largo * 1.1, cy - q[1] * largo * 1.1);
    }
  }
}

// ------------------------------------------------------------------
// Etapa 16: panel de la prueba de lazo (las cifras las calcula visor_web.py)
// ------------------------------------------------------------------
function actualizarPrueba(pr) {
  const ponerCifra = function (id, texto, nivel) {
    const e = document.getElementById(id);
    e.textContent = texto;
    e.className = "num" + (nivel ? " " + nivel : "");
  };
  if (!pr) {
    for (const id of ["m_recorrido", "m_cierre", "m_pct", "m_alejamiento"]) ponerCifra(id, "—", null);
    return;
  }
  ponerCifra("m_recorrido", pr.recorrido_m.toFixed(2) + " m", null);
  ponerCifra("m_cierre", (pr.cierre_m * 100).toFixed(1) + " cm", null);
  ponerCifra("m_pct", pr.cierre_pct === null ? "—" : pr.cierre_pct.toFixed(1) + " %", pr.nivel);
  ponerCifra("m_alejamiento", pr.alejamiento_max_m.toFixed(2) + " m", null);
}

// "Nueva prueba" pide confirmación con un segundo toque (sin ventanas emergentes)
const botonPrueba = document.getElementById("boton_prueba");
let confirmarHasta = 0;
botonPrueba.addEventListener("click", function () {
  if (Date.now() > confirmarHasta) {
    confirmarHasta = Date.now() + 3000;
    botonPrueba.textContent = "¿Seguro? Toca otra vez";
    botonPrueba.className = "confirmar";
    setTimeout(function () {
      if (Date.now() > confirmarHasta) { botonPrueba.textContent = "Nueva prueba"; botonPrueba.className = ""; }
    }, 3100);
    return;
  }
  confirmarHasta = 0;
  botonPrueba.textContent = "Nueva prueba";
  botonPrueba.className = "";
  trayectoria = [];
  hayQueDibujar = true;
  fetch("/reiniciar", { method: "POST" });
});

// ------------------------------------------------------------------
// Etapa 16: vista 3D que se gira con el dedo (proyección ortográfica)
// ------------------------------------------------------------------
// Observador en la dirección v = (cos el·cos az, cos el·sin az, sin el), mirando al centro.
// Ejes de pantalla: derecha = (-sin az, cos az, 0), arriba = v × derecha. Como
// derecha × arriba = v (hacia el observador), el dibujo no sale en espejo (etapa 15.3).
// Vista inicial: detrás de la placa (-X), algo a su derecha (-Y) y por encima.
const VISTA_INICIAL = { az: 200, el: 25, zoom: 1 };
let vista3d = Object.assign({}, VISTA_INICIAL);

function ejesVista3D() {
  const a = vista3d.az * Math.PI / 180, b = vista3d.el * Math.PI / 180;
  return {
    derecha: [-Math.sin(a), Math.cos(a), 0],
    arriba: [-Math.sin(b) * Math.cos(a), -Math.sin(b) * Math.sin(a), Math.cos(b)],
  };
}

const lienzo3d = document.getElementById("lienzo_3d");
const punteros = new Map();          // dedos (o ratón) apoyados: id -> [x, y]
let distanciaPinza = null;
lienzo3d.addEventListener("pointerdown", function (e) {
  lienzo3d.setPointerCapture(e.pointerId);
  punteros.set(e.pointerId, [e.clientX, e.clientY]);
});
lienzo3d.addEventListener("pointermove", function (e) {
  if (!punteros.has(e.pointerId)) return;
  const antes = punteros.get(e.pointerId);
  punteros.set(e.pointerId, [e.clientX, e.clientY]);
  if (punteros.size === 1) {          // un dedo: girar (0,4° por píxel)
    vista3d.az -= (e.clientX - antes[0]) * 0.4;
    vista3d.el = Math.max(-89, Math.min(89, vista3d.el + (e.clientY - antes[1]) * 0.4));
  } else if (punteros.size === 2) {   // dos dedos: zoom con la distancia entre ellos
    const p = Array.from(punteros.values());
    const d = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]);
    if (distanciaPinza) vista3d.zoom *= d / distanciaPinza;
    distanciaPinza = d;
  }
  hayQueDibujar = true;
});
function soltarPuntero(e) { punteros.delete(e.pointerId); distanciaPinza = null; }
lienzo3d.addEventListener("pointerup", soltarPuntero);
lienzo3d.addEventListener("pointercancel", soltarPuntero);
lienzo3d.addEventListener("wheel", function (e) {
  e.preventDefault();
  vista3d.zoom *= Math.exp(-e.deltaY * 0.001);
  hayQueDibujar = true;
}, { passive: false });
lienzo3d.addEventListener("dblclick", function () {
  vista3d = Object.assign({}, VISTA_INICIAL);
  hayQueDibujar = true;
});

function dibujarVista3D() {
  const l = prepararLienzo(lienzo3d);
  const ctx = l.ctx, ancho = l.ancho, alto = l.alto;

  // 1) Caja que contiene la trayectoria y el origen (mínimo 1 m de lado)
  const minimo = [-0.5, -0.5, -0.5], maximo = [0.5, 0.5, 0.5];
  for (const q of trayectoria) {
    for (let i = 0; i < 3; i++) {
      if (q[i] < minimo[i]) minimo[i] = q[i];
      if (q[i] > maximo[i]) maximo[i] = q[i];
    }
  }
  const centro = [0, 1, 2].map(function (i) { return (minimo[i] + maximo[i]) / 2; });
  const radio = Math.hypot(maximo[0] - minimo[0], maximo[1] - minimo[1], maximo[2] - minimo[2]) / 2;
  const escala = Math.min(ancho, alto) / (2 * radio * 1.1) * vista3d.zoom;   // píxeles por metro
  const ejes = ejesVista3D();
  function aPantalla(q) {
    const d = [q[0] - centro[0], q[1] - centro[1], q[2] - centro[2]];
    const h = d[0] * ejes.derecha[0] + d[1] * ejes.derecha[1] + d[2] * ejes.derecha[2];
    const v = d[0] * ejes.arriba[0] + d[1] * ejes.arriba[1] + d[2] * ejes.arriba[2];
    return [ancho / 2 + h * escala, alto / 2 - v * escala];
  }
  function segmento(a, b) {
    const pa = aPantalla(a), pb = aPantalla(b);
    linea(ctx, pa[0], pa[1], pb[0], pb[1]);
  }

  // 2) Suelo: rejilla en la cota más baja de la trayectoria (o del origen), con un paso "redondo".
  //    (No en minimo[2]: la caja tiene 1 m de lado como mínimo para el zoom, y el suelo
  //    quedaría por debajo de la trayectoria, que parecería flotar.)
  let suelo = 0;
  for (const q of trayectoria) suelo = Math.min(suelo, q[2]);
  const paso = pasoRejilla(Math.max(maximo[0] - minimo[0], maximo[1] - minimo[1]));
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#2c2c2c";
  const x0 = Math.floor(minimo[0] / paso) * paso, x1 = Math.ceil(maximo[0] / paso) * paso;
  const y0 = Math.floor(minimo[1] / paso) * paso, y1 = Math.ceil(maximo[1] / paso) * paso;
  for (let x = x0; x <= x1 + 1e-9; x += paso) segmento([x, y0, suelo], [x, y1, suelo]);
  for (let y = y0; y <= y1 + 1e-9; y += paso) segmento([x0, y, suelo], [x1, y, suelo]);

  // 3) Ejes del mundo en el origen (un paso de largo)
  const coloresEjes = ["#ef5350", "#66bb6a", "#42a5f5"];
  ctx.globalAlpha = 0.5;
  for (let i = 0; i < 3; i++) {
    const e = [0, 0, 0];
    e[i] = paso;
    ctx.strokeStyle = coloresEjes[i];
    segmento([0, 0, 0], e);
  }
  ctx.globalAlpha = 1;
  if (trayectoria.length === 0) return;

  // 4) Sombra de la trayectoria en el suelo (ayuda a ver la altura) y la trayectoria
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#3a3a3a";
  ctx.beginPath();
  trayectoria.forEach(function (q, k) {
    const s = aPantalla([q[0], q[1], suelo]);
    if (k === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]);
  });
  ctx.stroke();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#4fc3f7";
  ctx.beginPath();
  trayectoria.forEach(function (q, k) {
    const s = aPantalla(q);
    if (k === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]);
  });
  ctx.stroke();

  // 5) Lo que falta para cerrar el lazo: del punto actual al de salida (naranja, discontinuo)
  const salida = trayectoria[0], actual = trayectoria[trayectoria.length - 1];
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = "#ffa726";
  segmento(actual, salida);
  ctx.setLineDash([]);

  // 6) Salida (verde) y posición actual (amarillo) con los ejes de la IMU
  const ps = aPantalla(salida), pa = aPantalla(actual);
  ctx.fillStyle = "#66bb6a";
  ctx.beginPath(); ctx.arc(ps[0], ps[1], 6, 0, 2 * Math.PI); ctx.fill();
  if (ultimo && ultimo.R) {
    const largo = radio * 0.25;
    ctx.lineWidth = 3;
    for (let j = 0; j < 3; j++) {
      ctx.strokeStyle = coloresEjes[j];
      segmento(actual, [actual[0] + ultimo.R[0][j] * largo, actual[1] + ultimo.R[1][j] * largo,
                        actual[2] + ultimo.R[2][j] * largo]);
    }
  }
  ctx.fillStyle = "#ffeb3b";
  ctx.beginPath(); ctx.arc(pa[0], pa[1], 5, 0, 2 * Math.PI); ctx.fill();
}

// ------------------------------------------------------------------
// Etapa 18: red Wi-Fi (punto de acceso VIO-Pi <-> red guardada)
// ------------------------------------------------------------------
let redesMostradas = "";      // la lista del <select> solo se rehace si cambia (si no, se cierra sola)
let avisoDeCambio = false;    // tras pedir un cambio, el aviso con la dirección nueva se queda fijo

function textoModoRed(e) {
  if (e.modo === "punto") return "Punto de acceso: red " + e.ssid + " (" + e.ip + ")";
  if (e.modo === "cliente") return "Conectada a " + e.ssid + (e.ip ? " (" + e.ip + ")" : "");
  return "Wi-Fi desconectada";
}

function ponerAvisoRed(texto, enlaces) {
  // Con textContent y createElement: un nombre de red raro no puede meter HTML en la página
  const aviso = document.getElementById("red_aviso");
  aviso.textContent = texto + " ";
  for (const url of enlaces || []) {
    const a = document.createElement("a");
    a.href = url;
    a.textContent = url;
    aviso.appendChild(a);
  }
}

function pintarRed(e) {
  document.getElementById("red_modo").textContent = textoModoRed(e);

  const lista = document.getElementById("red_lista");
  const nuevas = JSON.stringify(e.redes);
  if (nuevas !== redesMostradas) {
    redesMostradas = nuevas;
    const elegida = lista.value;
    lista.textContent = "";
    for (const r of e.redes) {
      const op = document.createElement("option");
      op.value = r.conexion;
      op.textContent = r.ssid + (r.conexion === e.conexion ? " (actual)" : "");
      lista.appendChild(op);
    }
    if (elegida) lista.value = elegida;
  }

  const bloqueado = !e.puede_cambiar || e.cambiando !== null;
  document.getElementById("boton_punto").disabled = bloqueado || !e.punto_creado || e.modo === "punto";
  document.getElementById("boton_cliente").disabled = bloqueado || e.redes.length === 0;
  lista.disabled = bloqueado || e.redes.length === 0;

  if (avisoDeCambio) return;
  if (!e.puede_cambiar) {
    ponerAvisoRed("Para cambiar de red desde aquí, en la Pi: bash sistema/red.sh permitir", []);
  } else if (!e.punto_creado) {
    ponerAvisoRed("Falta crear el punto de acceso. En la Pi: bash sistema/red.sh crear", []);
  } else if (e.cambiando !== null) {
    ponerAvisoRed("Cambiando a " + e.cambiando + "…", []);
  } else if (e.ultimo_cambio && !e.ultimo_cambio.ok) {
    ponerAvisoRed("El cambio a " + e.ultimo_cambio.descripcion + " (" + e.ultimo_cambio.hora +
                  ") falló: la Pi volvió atrás.", []);
  } else {
    ponerAvisoRed("", []);
  }
}

function actualizarRed() {
  // Sin conexión (p. ej. justo tras cambiar de red) no pasa nada: se reintenta a los 5 s
  fetch("/red")
    .then(function (r) { return r.json(); })
    .then(pintarRed)
    .catch(function () {});
}

function pedirCambioRed(ruta, cuerpo) {
  fetch(ruta, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cuerpo) })
    .then(function (r) { return r.json(); })
    .then(function (res) {
      avisoDeCambio = res.ok;          // si salió bien, la dirección nueva se queda en pantalla
      ponerAvisoRed(res.mensaje, res.enlaces);
      document.getElementById("boton_punto").disabled = true;
      document.getElementById("boton_cliente").disabled = true;
    })
    .catch(function () { ponerAvisoRed("No se pudo pedir el cambio", []); });
}

// Los dos botones piden un segundo toque, como "Nueva prueba" (cambiar de red corta la página)
function botonConConfirmacion(id, accion) {
  const boton = document.getElementById(id);
  const textoNormal = boton.textContent;
  let hasta = 0;
  boton.addEventListener("click", function () {
    if (Date.now() > hasta) {
      hasta = Date.now() + 3000;
      boton.textContent = "¿Seguro? Toca otra vez";
      boton.className = "confirmar";
      setTimeout(function () {
        if (Date.now() > hasta) { boton.textContent = textoNormal; boton.className = ""; }
      }, 3100);
      return;
    }
    hasta = 0;
    boton.textContent = textoNormal;
    boton.className = "";
    accion();
  });
}

botonConConfirmacion("boton_punto", function () { pedirCambioRed("/red/punto", {}); });
botonConConfirmacion("boton_cliente", function () {
  pedirCambioRed("/red/cliente", { conexion: document.getElementById("red_lista").value });
});
actualizarRed();
setInterval(actualizarRed, 5000);

// ------------------------------------------------------------------
// Bucle de dibujo: solo redibuja cuando ha llegado un mensaje nuevo
// ------------------------------------------------------------------
function cuadro() {
  if (hayQueDibujar) {
    hayQueDibujar = false;
    dibujarVista3D();
    dibujarVista("lienzo_xy", 0, 1);
    dibujarVista("lienzo_xz", 0, 2);
    dibujarOrientacion();
  }
  requestAnimationFrame(cuadro);
}
window.addEventListener("resize", function () { hayQueDibujar = true; });
requestAnimationFrame(cuadro);
</script>
</body>
</html>
"""

# Los ejes del panel de orientación pasan de Python a la página (un solo sitio donde cambiarlos)
PAGINA_HTML = PAGINA_HTML.replace("__EJES_PANTALLA__", json.dumps(EJES_PANTALLA))


if __name__ == "__main__":
    main()
