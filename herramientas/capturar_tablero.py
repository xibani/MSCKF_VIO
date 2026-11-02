"""
herramientas/capturar_tablero.py — Captura imágenes de un tablero de ajedrez para calibrar.

Uso, desde ~/svo-pi:
    python -m herramientas.capturar_tablero
Con la cámara FIJA, mueves el tablero delante de ella; en cada postura lo dejas
quieto y pulsas Enter. 'q' + Enter para terminar.
Cada captura actualiza ultima.png (con las esquinas dibujadas) para verla en VSCode.
"""

from datetime import datetime

import cv2
import numpy as np

from captura import camara
from vio import calibracion_camara as cc
from vio.config import RAIZ_PROYECTO, cargar_config


def main():
    cfg_cam = cargar_config("camara")
    cfg = cargar_config("calibracion")
    t = cfg["tablero"]
    c = cfg["camara"]

    carpeta = (RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"]
               / ("calib_cam_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    carpeta.mkdir(parents=True)
    print(f"Imágenes en: {carpeta}")

    picam2 = camara.abrir(cfg_cam)
    guardadas = []                                  # esquinas de cada imagen guardada
    try:
        picam2.start()
        fijada = camara.fijar_exposicion(picam2, cfg_cam)
        print(f"Exposición {fijada['exposicion_us']} us | ganancia {fijada['ganancia']:.2f}"
              f" | brillo relativo {fijada['brillo_relativo']:.2f}")

        while len(guardadas) < c["imagenes_objetivo"]:
            orden = input(f"[{len(guardadas)}/{c['imagenes_objetivo']}] "
                          "Tablero QUIETO y Enter para capturar ('q' + Enter para terminar): ")
            if orden.strip().lower() == "q":
                break

            # 1. Imagen nueva (la siguiente que entregue la cámara)
            gris, t_ns, _, _ = camara.capturar(picam2, cfg_cam)

            # 2. ¿Se ve el tablero completo? ¿Está nítida?
            esquinas = cc.detectar_esquinas(gris, t["esquinas_x"], t["esquinas_y"])
            n = cc.nitidez(gris)
            vista = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
            if esquinas is None:
                print("   NO: no se ven todas las esquinas. Que entre el tablero entero.")
            elif n < c["nitidez_min"]:
                print(f"   NO: imagen movida (nitidez {n:.0f} < {c['nitidez_min']:.0f}). Repite quieto.")
            else:
                cv2.imwrite(str(carpeta / f"{t_ns}.png"), gris)          # se guarda SIN dibujos
                guardadas.append(esquinas)
                cv2.drawChessboardCorners(vista, (t["esquinas_x"], t["esquinas_y"]),
                                          esquinas.reshape(-1, 1, 2), True)
                print(f"   OK (nitidez {n:.0f}). Cobertura de esquinas por zonas:")
                print(cc.cobertura(guardadas, cfg_cam["ancho"], cfg_cam["alto"], c["zonas_cobertura"]))
            cv2.imwrite(str(carpeta / "ultima.png"), vista)
    finally:
        picam2.stop()
        picam2.close()

    # ultima.png no es una imagen de calibración: se borra para no confundir
    (carpeta / "ultima.png").unlink(missing_ok=True)
    print(f"\n{len(guardadas)} imágenes guardadas en {carpeta}")
    if guardadas:
        cuenta = cc.cobertura(guardadas, cfg_cam["ancho"], cfg_cam["alto"], c["zonas_cobertura"])
        if np.any(cuenta == 0):
            print("AVISO: hay zonas de la imagen sin ninguna esquina; añade capturas ahí.")
    print(f"Calibra con:  python -m herramientas.calibrar_camara {carpeta}")


if __name__ == "__main__":
    main()