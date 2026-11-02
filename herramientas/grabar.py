"""
herramientas/grabar.py — Grabación sincronizada de IMU y cámara, en formato EuRoC.

Uso, desde ~/svo-pi:
    python -m herramientas.grabar                 # duración de config/grabacion.yaml
    python -m herramientas.grabar --duracion 90
Ctrl+C para parar antes de tiempo (lo grabado se conserva).

Protocolo: quieto desde que arranca hasta 5 s después de "GRABANDO", movimientos
suaves por todos los ejes, vuelta al punto de partida y 5 s quieto al final.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime

from captura import camara, euroc, proceso_imu
from captura.escritor import EscritorPNG
from captura.reloj import desfase_boottime_ns
from vio.config import RAIZ_PROYECTO, cargar_config


def crear_carpetas(cfg_proyecto):
    """Crea datos/AAAAMMDD_HHMMSS/mav0/{imu0, cam0/data}. Devuelve: Path de la grabación."""
    nombre = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = RAIZ_PROYECTO / cfg_proyecto["carpeta_datos"] / nombre
    euroc.ruta_imu(carpeta).parent.mkdir(parents=True)
    euroc.carpeta_imagenes(carpeta).mkdir(parents=True)
    return carpeta


def main():
    # =================================================================
    # 0. Configuración
    # =================================================================
    cfg_imu = cargar_config("imu")
    cfg_cam = cargar_config("camara")
    cfg_grab = cargar_config("grabacion")
    cfg_proyecto = cargar_config("proyecto")

    parser = argparse.ArgumentParser(description="Grabación sincronizada IMU + cámara")
    parser.add_argument("--duracion", type=float, default=cfg_grab["duracion_s"],
                        help="segundos de imágenes a grabar")
    args = parser.parse_args()

    carpeta = crear_carpetas(cfg_proyecto)
    print(f"Grabación en: {carpeta}")
    print(f"BOOTTIME - MONOTONIC: {desfase_boottime_ns()} ns (debe ser casi 0)")

    # =================================================================
    # 1. IMU en su propio proceso. Va PRIMERO: así graba también el
    #    reposo inicial mientras se ajusta la cámara.
    # =================================================================
    ctx = mp.get_context("spawn")            # proceso nuevo y limpio (ver explicación)
    evento_listo = ctx.Event()
    evento_parar = ctx.Event()
    n_muestras = ctx.Value("q", 0)           # "q" = entero de 64 bits compartido
    n_errores = ctx.Value("q", 0)
    p_imu = ctx.Process(target=proceso_imu.ejecutar,
                        args=(carpeta, cfg_imu, evento_listo, evento_parar, n_muestras, n_errores))
    p_imu.start()

    if not evento_listo.wait(timeout=cfg_grab["timeout_imu_s"]):
        print("ERROR: la IMU no ha arrancado. Prueba antes: python -m herramientas.probar_imu")
        evento_parar.set()
        p_imu.terminate()
        return
    print("IMU grabando. MANTÉN EL DISPOSITIVO QUIETO.")

    # El resto de este proceso (cámara, guardado) se queda fuera del núcleo de la IMU
    try:
        os.sched_setaffinity(0, set(cfg_grab["nucleos_principal"]))
    except OSError:
        print("Aviso: no se pudieron fijar los núcleos del proceso principal")

    # =================================================================
    # 2. Cámara + guardado en hilo aparte
    # =================================================================
    picam2 = None
    escritor = None
    fijada = None
    n_img = 0
    descartadas = 0

    try:
        picam2 = camara.abrir(cfg_cam)
        picam2.start()
        fijada = camara.fijar_exposicion(picam2, cfg_cam)
        print(f"Exposición fija: {fijada['exposicion_us']} us | ganancia {fijada['ganancia']:.2f}"
              f" | brillo relativo {fijada['brillo_relativo']:.2f}")
        if fijada["brillo_relativo"] < cfg_grab["brillo_relativo_min"]:
            print("AVISO: poca luz. La imagen saldrá oscura y ruidosa; mejor más iluminación.")

        escritor = EscritorPNG(cfg_grab["cola_max_imagenes"], cfg_cam["compresion_png"])
        escritor.iniciar()
        carpeta_png = euroc.carpeta_imagenes(carpeta)

        with open(euroc.ruta_cam(carpeta), "w", encoding="utf-8") as f_cam, \
             open(euroc.ruta_diag_cam(carpeta), "w", encoding="utf-8") as f_diag:
            f_cam.write(euroc.CABECERA_CAM)
            f_diag.write(euroc.CABECERA_DIAG_CAM)

            print(f">>> GRABANDO {args.duracion:.0f} s. Quieto 5 s más y luego muévete con suavidad."
                  " Ctrl+C para parar.")
            t_inicio_ns = None
            t_aviso = time.monotonic()

            while True:
                # 2.1 Siguiente imagen
                gris, t_sensor_ns, t_llegada_ns, md = camara.capturar(picam2, cfg_cam)
                if t_inicio_ns is None:
                    t_inicio_ns = t_sensor_ns

                # 2.2 A la cola de guardado. Si no cabe, la imagen se pierde y NO se
                #     anota en los CSV: el verificador la verá como un hueco
                nombre_png = euroc.nombre_imagen(t_sensor_ns)
                if escritor.poner(carpeta_png / nombre_png, gris):
                    f_cam.write(euroc.linea_cam(t_sensor_ns, nombre_png))
                    f_diag.write(euroc.linea_diag_cam(t_sensor_ns, t_llegada_ns,
                                                      md["ExposureTime"], md["AnalogueGain"]))
                    n_img += 1
                else:
                    descartadas += 1

                # 2.3 Progreso y fin
                transcurrido_s = (t_sensor_ns - t_inicio_ns) / 1e9
                if time.monotonic() - t_aviso >= cfg_grab["aviso_cada_s"]:
                    print(f"  {transcurrido_s:5.1f} s | imágenes {n_img} | IMU {n_muestras.value}"
                          f" | cola {escritor.pendientes()} | descartadas {descartadas}"
                          f" | errores I2C {n_errores.value}")
                    t_aviso = time.monotonic()
                if transcurrido_s >= args.duracion:
                    break

    except KeyboardInterrupt:
        print("\nParando...")

    finally:
        # =============================================================
        # 3. Cierre ordenado: la IMU se para DESPUÉS de la última imagen
        # =============================================================
        evento_parar.set()
        p_imu.join(timeout=5.0)
        if p_imu.is_alive():
            p_imu.terminate()
        if picam2 is not None:
            picam2.stop()
            picam2.close()
        if escritor is not None:
            print(f"Guardando {escritor.pendientes()} imágenes pendientes...")
            escritor.cerrar()

        meta = {
            "fecha": datetime.now().isoformat(),
            "formato": "EuRoC/ASL, tiempos en ns de CLOCK_MONOTONIC",
            "duracion_pedida_s": args.duracion,
            "imu": cfg_imu,
            "camara": cfg_cam,
            "exposicion_fijada": fijada,
            "contadores": {
                "muestras_imu": n_muestras.value,
                "errores_i2c": n_errores.value,
                "imagenes": n_img,
                "descartadas_cola_llena": descartadas,
                "escritas": escritor.escritas if escritor else 0,
                "errores_escritura": escritor.errores if escritor else 0,
            },
        }
        with open(carpeta / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"Hecho: IMU {n_muestras.value} muestras | {n_img} imágenes | "
              f"{descartadas} descartadas | {n_errores.value} errores I2C")
        print(f"Verifica con:  python -m herramientas.verificar_grabacion {carpeta}")


if __name__ == "__main__":
    main()