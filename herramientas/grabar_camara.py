"""
herramientas/grabar_camara.py — Graba solo la cámara, en formato EuRoC.

Uso, desde ~/svo-pi:
    python -m herramientas.grabar_camara                 # 10 s
    python -m herramientas.grabar_camara --duracion 20
Ctrl+C para parar antes de tiempo (lo grabado se conserva).
"""

import argparse
from datetime import datetime

import cv2

from captura import camara, euroc
from captura.reloj import desfase_boottime_ns
from vio.config import RAIZ_PROYECTO, cargar_config


def main():
    # --- 1. Argumentos ---
    parser = argparse.ArgumentParser(description="Graba la cámara en formato EuRoC")
    parser.add_argument("--duracion", type=float, default=10.0, help="segundos a grabar")
    args = parser.parse_args()

    # --- 2. Configuración y carpetas de salida ---
    cfg = cargar_config("camara")
    cfg_proyecto = cargar_config("proyecto")
    nombre = "cam_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = RAIZ_PROYECTO / cfg_proyecto["carpeta_datos"] / nombre
    carpeta_png = euroc.carpeta_imagenes(carpeta)
    carpeta_png.mkdir(parents=True)
    opciones_png = [cv2.IMWRITE_PNG_COMPRESSION, cfg["compresion_png"]]

    print(f"BOOTTIME - MONOTONIC: {desfase_boottime_ns()} ns (debe ser casi 0)")

    # --- 3. Cámara ---
    picam2 = camara.abrir(cfg)
    n = 0
    try:
        picam2.start()
        fijada = camara.fijar_exposicion(picam2, cfg)
        print(f"Exposición fija: {fijada['exposicion_us']} us, ganancia {fijada['ganancia']:.2f}")
        print(f"Grabando {args.duracion:.0f} s en {carpeta}")

        with open(euroc.ruta_cam(carpeta), "w", encoding="utf-8") as f_cam, \
             open(euroc.ruta_diag_cam(carpeta), "w", encoding="utf-8") as f_diag:
            f_cam.write(euroc.CABECERA_CAM)
            f_diag.write(euroc.CABECERA_DIAG_CAM)

            t_inicio_ns = None
            try:
                while True:
                    # 3.1 Siguiente imagen
                    gris, t_sensor_ns, t_llegada_ns, md = camara.capturar(picam2, cfg)
                    if t_inicio_ns is None:
                        t_inicio_ns = t_sensor_ns

                    # 3.2 Guardar la imagen y sus filas en los CSV
                    nombre_png = euroc.nombre_imagen(t_sensor_ns)
                    cv2.imwrite(str(carpeta_png / nombre_png), gris, opciones_png)
                    f_cam.write(euroc.linea_cam(t_sensor_ns, nombre_png))
                    f_diag.write(euroc.linea_diag_cam(t_sensor_ns, t_llegada_ns,
                                                      md["ExposureTime"], md["AnalogueGain"]))
                    n += 1

                    # 3.3 Progreso y fin
                    transcurrido_s = (t_sensor_ns - t_inicio_ns) / 1e9
                    if n % cfg["fps"] == 0:
                        print(f"  {transcurrido_s:5.1f} s | {n} imágenes")
                    if transcurrido_s >= args.duracion:
                        break
            except KeyboardInterrupt:
                print("\nInterrumpido: se conserva lo grabado")
    finally:
        picam2.stop()
        picam2.close()

    print(f"Hecho: {n} imágenes")
    print(f"Analiza con:  python -m herramientas.analizar_camara {carpeta}")


if __name__ == "__main__":
    main()