"""
apps/visor_web.py — Visor de la VIO en el navegador (tablet, móvil o PC).

Idea:
    pi_vio / pi_orientacion  --UDP 127.0.0.1:5005-->  visor_web (este programa, en la Pi)
    pi_vio (imagen + pistas) --UDP 127.0.0.1:5006-->  visor_web
    visor_web  --HTTP + Server-Sent Events (puerto 8000)-->  navegador de la tablet
    visor_web  --HTTP, vídeo MJPEG (/video)-->  navegador de la tablet

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


# ---------------------------------------------------------------------------
# Formato de los paquetes (copia literal de vio/protocolo.py, sin cambios)
# ---------------------------------------------------------------------------
FORMATO_V1 = "<Bq9d"      # versión, t_ns, R_w_i (9 float64, por filas)
FORMATO_V2 = "<Bq9d3d"    # lo mismo + p_w (3 float64)
TAM_V1 = struct.calcsize(FORMATO_V1)   # 81 bytes
TAM_V2 = struct.calcsize(FORMATO_V2)   # 105 bytes

# Máximo de puntos de trayectoria que se mandan al navegador en cada mensaje.
# Si la tablet se conecta tarde, recibe la trayectoria antigua en trozos.
MAX_PUNTOS_POR_MENSAJE = 2000

# Imágenes (etapa 14): copia literal de vio/protocolo.py
FORMATO_CABECERA_IMAGEN = "<Bq"            # versión (3) + t_ns, y detrás los bytes del JPEG
TAM_CABECERA_IMAGEN = struct.calcsize(FORMATO_CABECERA_IMAGEN)   # 9 bytes
VERSION_IMAGEN = 3
PERIODO_VIDEO_S = 0.03                     # cada cuánto mira el vídeo si hay imagen nueva


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
        self.R = None              # lista 3x3 o None
        self.p = None              # [x, y, z] o None
        self.hora_ultimo = None    # time.monotonic() del último paquete válido
        self.trayectoria = []      # lista de [x, y, z]
        self.id_sesion = 0         # cambia cuando se reinicia la VIO
        self.recibidos = 0
        self.invalidos = 0
        self.no_finitos = 0
        # Etapa 14: la última imagen con sus pistas
        self.jpeg = None           # bytes del último JPEG, o None
        self.id_jpeg = 0           # aumenta con cada imagen nueva
        self.imagenes = 0          # imágenes recibidas


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

        # 3) Descartar valores no finitos (si el filtro diverge)
        R_ok = todos_finitos(R[0] + R[1] + R[2])
        p_ok = p is not None and todos_finitos(p)

        # 4) Guardar en el estado compartido
        with estado.cerrojo:
            # Si el tiempo retrocede, se ha relanzado pi_vio (o la grabación):
            # empezamos una trayectoria nueva.
            if estado.t_ns is not None and t_ns < estado.t_ns:
                estado.trayectoria = []
                estado.id_sesion += 1

            estado.version = version
            estado.t_ns = t_ns
            estado.R = R if R_ok else None
            estado.p = p if p_ok else None
            estado.hora_ultimo = time.monotonic()
            estado.recibidos += 1

            if p is not None and not p_ok:
                estado.no_finitos += 1
            if p_ok:
                estado.trayectoria.append(p)


def bucle_imagenes(sock, estado, parar):
    """Hilo que recibe las imágenes (etapa 14) y guarda solo la última."""
    while not parar.is_set():
        try:
            datos, _origen = sock.recvfrom(65536)          # un JPEG cabe en un paquete
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
# Servidor HTTP: página + flujo de eventos + vídeo
# ---------------------------------------------------------------------------
class ManejadorHTTP(BaseHTTPRequestHandler):
    # Se rellenan en main() antes de arrancar el servidor
    estado = None
    periodo_envio = 1.0 / 15.0

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
        else:
            self.send_error(404)

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
        indice_enviado = 0        # cuántos puntos de la trayectoria tiene ya este navegador
        sesion_enviada = None

        try:
            while True:
                with estado.cerrojo:
                    # Si la VIO se ha reiniciado, el navegador debe empezar de cero
                    if sesion_enviada != estado.id_sesion:
                        sesion_enviada = estado.id_sesion
                        indice_enviado = 0

                    inicio = indice_enviado
                    nuevos = estado.trayectoria[inicio:inicio + MAX_PUNTOS_POR_MENSAJE]
                    indice_enviado = inicio + len(nuevos)

                    if estado.hora_ultimo is None:
                        sin_datos_s = None
                    else:
                        sin_datos_s = time.monotonic() - estado.hora_ultimo

                    mensaje = {
                        "sesion": estado.id_sesion,
                        "inicio": inicio,          # posición de 'nuevos' en la trayectoria
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
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=fotograma")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        estado = self.estado
        ultimo_id = -1
        try:
            self.wfile.write(b"--fotograma\r\n")
            while True:
                with estado.cerrojo:
                    jpeg, id_jpeg = estado.jpeg, estado.id_jpeg
                if jpeg is not None and id_jpeg != ultimo_id:      # solo cuando hay imagen nueva
                    ultimo_id = id_jpeg
                    cabecera = "Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg)
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
    parser.add_argument("--puerto-udp", type=int, default=5005,
                        help="puerto donde llegan los paquetes de pi_vio (5005)")
    parser.add_argument("--puerto-imagen", type=int, default=5006,
                        help="puerto donde llegan las imágenes con pistas de pi_vio (5006)")
    parser.add_argument("--puerto-web", type=int, default=8000,
                        help="puerto HTTP para el navegador (8000)")
    parser.add_argument("--hz", type=float, default=15.0,
                        help="mensajes por segundo hacia el navegador (15)")
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

    hilo_udp = threading.Thread(target=bucle_udp, args=(sock, estado, parar), daemon=True)
    hilo_udp.start()

    # 1b) Segundo socket para las imágenes (etapa 14)
    sock_img = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock_img.bind(("0.0.0.0", args.puerto_imagen))
    except OSError as e:
        print("ERROR: no puedo escuchar en UDP %d (%s)." % (args.puerto_imagen, e))
        return
    sock_img.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)   # 1 MB: varias imágenes en espera
    sock_img.settimeout(0.5)
    hilo_img = threading.Thread(target=bucle_imagenes, args=(sock_img, estado, parar), daemon=True)
    hilo_img.start()

    # 2) Servidor HTTP (un hilo por navegador conectado)
    ManejadorHTTP.estado = estado
    ManejadorHTTP.periodo_envio = 1.0 / args.hz
    servidor = ThreadingHTTPServer(("0.0.0.0", args.puerto_web), ManejadorHTTP)

    ip = ip_local_probable()
    print("Escuchando UDP en los puertos %d (pose) y %d (imágenes)." % (args.puerto_udp, args.puerto_imagen))
    if ip:
        print("Abre en la tablet:  http://%s:%d" % (ip, args.puerto_web))
    else:
        print("Abre en la tablet:  http://<IP de la Pi>:%d   (mira 'hostname -I')" % args.puerto_web)
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
</style>
</head>
<body>
<div id="cabecera">
  <span id="estado" class="esp">CONECTANDO…</span>
  <span id="texto_pose">—</span>
</div>
<div id="paneles">
  <div class="panel ancho"><h2>Cámara y pistas (rojo: recién detectada → verde: seguida 20 imágenes o más)</h2><img id="video" alt="Esperando imágenes de pi_vio…"></div>
  <div class="panel"><h2>Planta X–Y [m] (flecha roja: eje x de la IMU)</h2><canvas id="lienzo_xy"></canvas></div>
  <div class="panel"><h2>Alzado X–Z [m]</h2><canvas id="lienzo_xz"></canvas></div>
  <div class="panel"><h2>Orientación: ejes de la IMU en el mundo (x rojo, y verde, z azul)</h2><canvas id="lienzo_ori"></canvas></div>
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
// Orientación: ejes de la IMU en perspectiva isométrica del mundo
// ------------------------------------------------------------------
function proyectarIso(v) {
  // Devuelve [horizontal, vertical hacia arriba]
  const c = Math.cos(Math.PI / 6);
  const s = Math.sin(Math.PI / 6);
  return [(v[0] - v[1]) * c, v[2] - (v[0] + v[1]) * s];
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
    const q = proyectarIso(e);
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
      const q = proyectarIso(eje);
      ctx.strokeStyle = colores[j];
      linea(ctx, cx, cy, cx + q[0] * largo, cy - q[1] * largo);
      ctx.fillStyle = colores[j];
      ctx.fillText(nombres[j], cx + q[0] * largo * 1.1, cy - q[1] * largo * 1.1);
    }
  }
}

// ------------------------------------------------------------------
// Bucle de dibujo: solo redibuja cuando ha llegado un mensaje nuevo
// ------------------------------------------------------------------
function cuadro() {
  if (hayQueDibujar) {
    hayQueDibujar = false;
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


if __name__ == "__main__":
    main()