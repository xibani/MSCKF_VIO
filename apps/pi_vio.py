"""
apps/pi_vio.py — El VIO completo: en vivo en la Pi, o sobre una grabación.

Uso, desde ~/svo-pi:
    python -m apps.pi_vio                                 # en vivo, al PC de config/orientacion.yaml
    python -m apps.pi_vio --ip 192.168.1.37
    python -m apps.pi_vio --grabacion datos/sim_vio        # sin sensores: una grabación EuRoC
    python -m apps.pi_vio --grabacion datos/sim_vio --sin-red
    python -m apps.pi_vio --sin-vista                     # sin enviar las imágenes al visor
    python -m apps.pi_vio --grabacion datos/sim_vio --sin-calibracion-acc   # datos sintéticos
    python -m apps.pi_vio --ip-mapa 192.168.1.37          # además, el mapa al PC (apps/pc_mapa)

El acelerómetro se corrige con config/imu_calibracion.yaml (etapa 7.2) si existe.
El botón "Nueva prueba" del visor web envía "reiniciar" al puerto de control (UDP 5007,
etapa 16): el VIO vuelve a inicializarse y la trayectoria sigue en vio_trayectoria_2.csv, _3...
En vivo: QUIETO hasta que aparezca "VIO EN MARCHA". Ctrl+C para terminar.
Al encender la Pi lo lanza systemd (sistema/instalar_servicios.sh); "systemctl stop" envía
  SIGTERM, que aquí se trata igual que Ctrl+C.
Además de la pose (UDP 5005), cada pocas imágenes envía la imagen con sus pistas dibujadas
(UDP 5006, etapa 14) para apps/visor_web.
Etapa 18: los puntos del mapa y los keyframes de mapeo se guardan en mapa.paquetes y, si hay
IP del PC (--ip-mapa, o mapeo.ip_pc en config/vio.yaml), se envían a apps/pc_mapa (UDP 5007).
La trayectoria estimada se guarda en vio_trayectoria.csv (dentro de la grabación, o en
datos/vio_AAAAMMDD_HHMMSS/ en vivo).
"""

import argparse
import multiprocessing as mp
import os
import queue
import signal
import socket
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from captura import euroc
from vio import calibracion_imu as ci
from vio import modelo_camara as mc
from vio import protocolo
from vio import vista as vista_vio
from vio.config import RAIZ_PROYECTO, cargar_config, ruta_config
from vio.pipeline import CABECERA_POSES, PipelineVio, linea_pose

ADELANTO_IMU_NS = (
    20_000_000  # en vivo la IMU llega ~20 ms antes que la imagen: igual al reproducir
)


# =====================================================================
# Fuentes de datos: las dos entregan los mismos eventos, en orden de tiempo
#     ("imu", t_ns, w, a)      ("imagen", t_ns, gris)
# =====================================================================
def eventos_grabacion(carpeta, adelanto_ns):
    """Generador: muestras de IMU e imágenes de una grabación EuRoC.

    Antes de cada imagen entrega las muestras de IMU hasta adelanto_ns después de ella,
    como ocurre en vivo (la IMU tiene menos retraso que la cámara).
    """
    t_imu, w, a = euroc.leer_imu(euroc.ruta_imu(carpeta))
    t_cam, nombres = euroc.leer_cam(euroc.ruta_cam(carpeta))
    carpeta_png = euroc.carpeta_imagenes(carpeta)
    i = 0
    for t, nombre in zip(t_cam, nombres):
        while i < len(t_imu) and t_imu[i] <= t + adelanto_ns:
            yield ("imu", int(t_imu[i]), w[i], a[i])
            i += 1
        yield (
            "imagen",
            int(t),
            cv2.imread(str(carpeta_png / nombre), cv2.IMREAD_GRAYSCALE),
        )


def eventos_en_vivo(cfg_imu, cfg_cam, cfg_vio, contadores):
    """Generador: la IMU (en su propio proceso, por una cola) y la cámara (en este proceso).

    contadores: dict que se rellena con los Value compartidos (muestras perdidas, errores I2C).
    """
    from captura import camara, proceso_imu  # solo en la Pi: picamera2, smbus2

    # 1. La IMU en otro proceso, en su núcleo, enviando las muestras por una cola
    ctx = mp.get_context("spawn")
    cola = ctx.Queue(maxsize=cfg_vio["ejecucion"]["cola_imu_max"])
    evento_listo, evento_parar = ctx.Event(), ctx.Event()
    contadores["perdidas"] = ctx.Value("q", 0)
    contadores["errores_i2c"] = ctx.Value("q", 0)
    p_imu = ctx.Process(
        target=proceso_imu.ejecutar_cola,
        args=(
            cfg_imu,
            cola,
            evento_listo,
            evento_parar,
            contadores["perdidas"],
            contadores["errores_i2c"],
        ),
    )
    p_imu.start()
    picam2 = None
    try:
        if not evento_listo.wait(timeout=cfg_vio["ejecucion"]["timeout_imu_s"]):
            raise RuntimeError(
                "la IMU no ha arrancado (prueba: python -m herramientas.probar_imu)"
            )
        try:
            os.sched_setaffinity(
                0, set(cargar_config("grabacion")["nucleos_principal"])
            )
        except OSError:
            print("Aviso: no se pudieron fijar los núcleos del proceso principal")

        # 2. Cámara con exposición fija (etapa 3); mientras tanto, la IMU ya mide el reposo
        picam2 = camara.abrir(cfg_cam)
        picam2.start()
        fijada = camara.fijar_exposicion(picam2, cfg_cam)
        print(
            f"Exposición fija: {fijada['exposicion_us']} us, ganancia {fijada['ganancia']:.2f}"
        )

        # 3. Cada imagen: primero todas las muestras que hayan llegado, después la imagen
        while True:
            gris, t_sensor_ns, _, _ = camara.capturar(picam2, cfg_cam)
            while True:
                try:
                    t_ns, w, a = cola.get_nowait()
                except queue.Empty:
                    break
                yield ("imu", t_ns, w, a)
            yield ("imagen", t_sensor_ns, gris)
    finally:
        evento_parar.set()
        p_imu.join(timeout=5.0)
        if p_imu.is_alive():
            p_imu.terminate()
        if picam2 is not None:
            picam2.stop()
            picam2.close()


# =====================================================================
# Envío al visor
# =====================================================================
def enviar(emisor, datos, destino):
    """sendto que no detiene el VIO si el visor no está escuchando (o la red falla)."""
    try:
        emisor.sendto(datos, destino)
    except OSError:
        pass


def enviar_imagen(emisor, destino, t_ns, gris, pipeline, r, cfg_vista):
    """Etapa 14: dibuja las pistas sobre la imagen y la manda al visor en JPEG."""
    frontend = (
        pipeline.frontend if pipeline.filtro is not None else None
    )  # antes de arrancar, sin pistas
    imagen = vista_vio.dibujar_vista(
        gris, frontend, vista_vio.lineas_estado(pipeline, r), cfg_vista
    )
    jpeg = vista_vio.codificar_jpeg(
        imagen, cfg_vista["calidad_jpeg"], protocolo.MAX_BYTES_JPEG
    )
    if jpeg is not None:
        enviar(emisor, protocolo.empaquetar_imagen(t_ns, jpeg), destino)


def paquetes_mapa(r, K, dist, cfg_mapeo):
    """Etapa 17: los paquetes de mapa de una imagen procesada.

    - Puntos aceptados por el MSCKF en esta imagen (versión 5), si hay.
    - El keyframe de mapeo (versión 4), si el clon que acaba de salir de la ventana lo es.
      La imagen va en JPEG con la calidad del YAML, bajándola si no cabe en el paquete.
    Devuelve: lista de bytes.
    """
    paquetes = []
    ids, p_w = r["puntos_mapa"]
    if len(ids) > 0:
        paquetes += protocolo.empaquetar_puntos(r["t_ns"], ids, p_w)
    kf = r["keyframe"]
    if kf is not None:
        jpeg = vista_vio.codificar_jpeg(
            kf.gris,
            cfg_mapeo["calidad_jpeg"],
            protocolo.espacio_jpeg_keyframe(len(kf.ids)),
        )
        if jpeg is not None:
            tamano = (kf.gris.shape[1], kf.gris.shape[0])
            paquetes.append(protocolo.empaquetar_keyframe(kf, K, dist, tamano, jpeg))
    return paquetes


# =====================================================================
# Programa principal
# =====================================================================
def abrir_control(puerto):
    """Socket UDP (solo en la propia Pi) por el que el visor web manda órdenes. None si no se puede."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", puerto))
    except OSError as e:
        print(
            f"Aviso: sin puerto de control ({puerto}: {e}): el botón 'Nueva prueba' no funcionará"
        )
        sock.close()
        return None
    sock.setblocking(False)  # mirar si hay orden no debe parar el VIO
    return sock


def orden_reiniciar(sock):
    """¿Ha llegado la orden de reiniciar? Lee TODAS las que esperan (varias pulsaciones = una)."""
    hay = False
    while True:
        try:
            datos, _ = sock.recvfrom(64)
        except BlockingIOError:
            return hay
        if datos == protocolo.ORDEN_REINICIAR:
            hay = True


def terminar_con_sigterm():
    """Que SIGTERM (lo que envía "systemctl stop") haga lo mismo que Ctrl+C.

    Por defecto SIGTERM mata el proceso sin ejecutar los finally: la cámara y la IMU no se
    cerrarían y el CSV perdería sus últimas líneas. Lanzando KeyboardInterrupt se sale por
    el mismo camino que con Ctrl+C.
    """

    def manejador(_senal, _marco):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, manejador)


def cargar_calibracion_acc(ruta, desactivada):
    """Etapa 7.2: (sesgo, escala) del acelerómetro para corregir sus medidas, o None.

    ruta: Path de config/imu_calibracion.yaml. desactivada: True con --sin-calibracion-acc
    (p. ej. en grabaciones sintéticas, que no vienen de este sensor).
    """
    if desactivada:
        print("Acelerómetro SIN corregir (--sin-calibracion-acc)")
        return None
    if not ruta.exists():
        print(
            f"AVISO: acelerómetro SIN calibrar (no existe {ruta.name}): haz la etapa 7.2"
        )
        return None
    sesgo, escala = ci.cargar_calibracion_imu(ruta=ruta)
    print(
        f"Acelerómetro corregido con {ruta.name}: sesgo {np.round(sesgo, 3)} m/s²,"
        f" escala {np.round(escala, 4)}"
    )
    return sesgo, escala


def resumen_tiempos(tiempos, presupuesto_ms):
    """Imprime media, percentil 95 y máximo de cada parte, y cuántas imágenes se pasan."""
    print("Tiempo por imagen [ms]:      media    p95    máx")
    for parte in ("imu", "frontend", "msckf", "total"):
        valores = np.array([t[parte] for t in tiempos])
        print(
            f"  {parte:10s}             {valores.mean():6.1f} {np.percentile(valores, 95):6.1f}"
            f" {valores.max():6.1f}"
        )
    total = np.array([t["total"] for t in tiempos])
    print(
        f"  imágenes por encima de {presupuesto_ms:.0f} ms: {100 * np.mean(total > presupuesto_ms):.1f} %"
    )


def main():
    cfg_red = cargar_config("orientacion")["red"]
    parser = argparse.ArgumentParser(description="VIO completo: front-end + MSCKF")
    parser.add_argument(
        "--grabacion", default=None, help="carpeta EuRoC (sin ella: en vivo)"
    )
    parser.add_argument("--ip", default=cfg_red["ip_pc"], help="IP del PC con el visor")
    parser.add_argument(
        "--sin-red", action="store_true", help="no enviar nada al visor"
    )
    parser.add_argument(
        "--sin-vista", action="store_true", help="enviar la pose, pero no las imágenes"
    )
    parser.add_argument(
        "--sin-calibracion-acc",
        action="store_true",
        help="no corregir el acelerómetro con config/imu_calibracion.yaml",
    )
    parser.add_argument(
        "--ip-mapa",
        default=None,
        help="IP del PC con apps/pc_mapa (por defecto, mapeo.ip_pc de config/vio.yaml)",
    )
    parser.add_argument(
        "--sin-mapa", action="store_true", help="ni guardar ni enviar el mapa"
    )
    parser.add_argument(
        "--puerto-control",
        type=int,
        default=cargar_config("vio")["control"]["puerto"],
        help="puerto UDP de las órdenes del visor (0: ninguno)",
    )
    args = parser.parse_args()

    terminar_con_sigterm()

    # --- 1. Configuración ---
    cfgs = {
        nombre: cargar_config(nombre)
        for nombre in ("vio", "frontend", "msckf", "triangulacion")
    }
    K, dist, _ = mc.cargar_intrinsecos()
    ext = cargar_config("camara_imu")
    R_c_i = np.array(ext["R_i_c"], dtype=float).T
    t_c_i = -R_c_i @ np.array(ext["p_i_c_m"], dtype=float)
    if ext.get("info", {}).get("provisional"):
        print(
            "AVISO: extrínseca cámara-IMU PROVISIONAL (config/camara_imu.yaml): falta Kalibr (7.4)"
        )
    if cargar_config("camara_intrinsecos").get("info", {}).get("provisional"):
        print(
            "AVISO: intrínsecos PROVISIONALES (config/camara_intrinsecos.yaml): falta la etapa 7.1"
        )
    calib_acc = cargar_calibracion_acc(
        ruta_config("imu_calibracion"), args.sin_calibracion_acc
    )

    def pipeline_nuevo():
        """El VIO desde cero (al arrancar y en cada "Nueva prueba" del visor)."""
        return PipelineVio(K, dist, R_c_i, t_c_i, ext["t_d_ns"], cfgs, calib_acc)

    pipeline = pipeline_nuevo()

    # --- 2. Fuente de datos y salida ---
    contadores = {}
    if args.grabacion is not None:
        carpeta = Path(args.grabacion)
        fuente = eventos_grabacion(carpeta, max(ext["t_d_ns"], 0) + ADELANTO_IMU_NS)
    else:
        carpeta = (
            RAIZ_PROYECTO
            / cargar_config("proyecto")["carpeta_datos"]
            / datetime.now().strftime("vio_%Y%m%d_%H%M%S")
        )
        carpeta.mkdir(parents=True)
        fuente = eventos_en_vivo(
            cargar_config("imu"), cargar_config("camara"), cfgs["vio"], contadores
        )
        print("QUIETO hasta que aparezca VIO EN MARCHA...")
    emisor = None if args.sin_red else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destino = (args.ip, cfg_red["puerto"])
    cfg_vista = cfgs["vio"]["vista"]
    destino_vista = (args.ip, cfg_vista["puerto"])
    con_vista = emisor is not None and not args.sin_vista
    salida = open(carpeta / "vio_trayectoria.csv", "w", encoding="utf-8")
    salida.write(CABECERA_POSES)
    # Etapa 18: el mapa, siempre a disco y, si hay IP, también al PC
    cfg_mapeo = cfgs["vio"]["mapeo"]
    archivo_mapa = None if args.sin_mapa else open(carpeta / cfg_mapeo["archivo"], "wb")
    ip_mapa = args.ip_mapa if args.ip_mapa is not None else cfg_mapeo["ip_pc"]
    destino_mapa = None
    if archivo_mapa is not None and emisor is not None and ip_mapa:
        destino_mapa = (ip_mapa, cfg_mapeo["puerto"])
        print(
            f"Mapa al PC: {ip_mapa}:{cfg_mapeo['puerto']} (en el PC: python -m apps.pc_mapa)"
        )
    control = abrir_control(args.puerto_control) if args.puerto_control else None
    n_prueba = 1

    # --- 3. Bucle: el mismo para las dos fuentes ---
    tiempos = []
    t_aviso_ns = None
    intentos_vistos = 0
    n_imagenes = 0
    try:
        for evento in fuente:
            if evento[0] == "imu":
                pipeline.nueva_imu(evento[1], evento[2], evento[3])
                if pipeline.intentos > intentos_vistos and pipeline.problemas:
                    intentos_vistos = pipeline.intentos
                    print(
                        "No estaba quieto:",
                        "; ".join(pipeline.problemas),
                        "-> se repite",
                    )
                    if (
                        any("|a|" in p for p in pipeline.problemas)
                        and calib_acc is None
                    ):
                        print(
                            "   (si se repite igual con la placa quieta, no es movimiento: es el"
                            " acelerómetro sin calibrar -> python -m herramientas.calibrar_acelerometro)"
                        )
                continue
            # Etapa 16: "Nueva prueba" desde el visor -> VIO desde cero y trayectoria en otro CSV
            if control is not None and orden_reiniciar(control):
                n_prueba += 1
                salida.close()
                salida = open(
                    carpeta / f"vio_trayectoria_{n_prueba}.csv", "w", encoding="utf-8"
                )
                salida.write(CABECERA_POSES)
                pipeline = pipeline_nuevo()
                t_aviso_ns, intentos_vistos = None, 0
                print(
                    f"Reiniciado desde el visor (prueba {n_prueba}): QUIETO hasta VIO EN MARCHA..."
                )
            r = pipeline.nueva_imagen(evento[1], evento[2])
            n_imagenes += 1
            if con_vista and n_imagenes % cfg_vista["cada_n_imagenes"] == 0:
                enviar_imagen(
                    emisor, destino_vista, evento[1], evento[2], pipeline, r, cfg_vista
                )
            if r is None:
                continue
            if t_aviso_ns is None:
                print(
                    f"VIO EN MARCHA. Sesgo del giroscopio [°/s]: "
                    f"{np.round(np.degrees(pipeline.filtro.estado.b_g), 3)}"
                )
                t_aviso_ns = r["t_ns"]

            # Enviar, guardar y, una vez por segundo, informar
            if emisor is not None:
                enviar(
                    emisor,
                    protocolo.empaquetar_pose(r["t_ns"], r["R_w_i"], r["p_w"]),
                    destino,
                )
            salida.write(linea_pose(r["t_ns"], r["R_w_i"], r["p_w"]))
            if archivo_mapa is not None:
                for datos in paquetes_mapa(r, K, dist, cfg_mapeo):
                    protocolo.escribir_paquete(archivo_mapa, datos)
                    if destino_mapa is not None:
                        enviar(emisor, datos, destino_mapa)
            tiempos.append(r["tiempos_ms"])
            if r["t_ns"] - t_aviso_ns >= cfgs["vio"]["ejecucion"]["aviso_cada_s"] * 1e9:
                t_aviso_ns = r["t_ns"]
                p, ms_total = r["p_w"], np.mean([t["total"] for t in tiempos[-20:]])
                linea = (
                    f"  p = ({p[0]:+6.2f}, {p[1]:+6.2f}, {p[2]:+6.2f}) m | puntos "
                    f"{r['frontend']['activos']:3d} | pistas usadas {r['msckf']['usadas']:2d}"
                    f" | keyframes {pipeline.n_keyframes:3d} | {ms_total:5.1f} ms/imagen"
                )
                if contadores:
                    linea += (
                        f" | IMU perdidas {contadores['perdidas'].value}"
                        f" | errores I2C {contadores['errores_i2c'].value}"
                    )
                print(linea)
                salida.flush()  # al disco cada segundo: por si se desenchufa la Pi
                if archivo_mapa is not None:
                    archivo_mapa.flush()
    except KeyboardInterrupt:
        print("\nTerminado")
    finally:
        fuente.close()  # en vivo: para la IMU y la cámara
        salida.close()
        if emisor is not None:
            emisor.close()
        if control is not None:
            control.close()
        if archivo_mapa is not None:
            archivo_mapa.close()

    if tiempos:
        print(
            f"{len(tiempos)} imágenes procesadas. Trayectoria: {carpeta / 'vio_trayectoria.csv'}"
        )
        resumen_tiempos(tiempos, cfgs["vio"]["ejecucion"]["presupuesto_ms"])
        if archivo_mapa is not None:
            print(
                f"{pipeline.n_keyframes} keyframes de mapeo. Mapa: {carpeta / cfg_mapeo['archivo']}"
            )
    else:
        print("El filtro no llegó a arrancar.")


if __name__ == "__main__":
    main()
