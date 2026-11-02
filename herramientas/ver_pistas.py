"""
herramientas/ver_pistas.py — Pasa el front-end por una grabación y dibuja las pistas.

Uso, desde ~/svo-pi:
    python -m herramientas.ver_pistas datos/20261102_183000
Vale cualquier grabación con mav0/cam0 (las de grabar y las de grabar_camara).
Guarda en la carpeta de la grabación:
    pistas.mp4          vídeo con las pistas dibujadas
    pistas_NNNN.png     algunas imágenes sueltas, por si el vídeo no se puede abrir
    frontend.png        estadísticas por imagen e histograma de longitudes de pista
"""

import argparse
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from captura import euroc
from vio import modelo_camara as mc
from vio.config import cargar_config
from vio.frontend import FrontEnd

COLA_DIBUJO = 10            # últimas posiciones de cada pista que se dibujan
CADA_N_INSTANTANEA = 100    # cada cuántas imágenes se guarda un PNG suelto
LONGITUD_LARGA = 20         # a partir de esta longitud la pista se dibuja en verde


def dibujar(gris, frontend):
    """Imagen en color con la cola de cada pista activa (rojo = nueva, verde = larga)."""
    vista = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    for pista in frontend.activas.values():
        cola = np.array(pista.uv[-COLA_DIBUJO:], dtype=np.int32)
        fraccion = min(pista.longitud() / LONGITUD_LARGA, 1.0)
        color = (0, int(255 * fraccion), int(255 * (1 - fraccion)))      # BGR: de rojo a verde
        cv2.polylines(vista, [cola.reshape(-1, 1, 2)], False, color, 1)
        cv2.circle(vista, tuple(int(c) for c in cola[-1]), 2, color, -1)
    return vista


def main():
    parser = argparse.ArgumentParser(description="Dibuja las pistas del front-end")
    parser.add_argument("carpeta", help="carpeta de la grabación (la que contiene mav0/)")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)

    K, dist, _ = mc.cargar_intrinsecos()
    frontend = FrontEnd(K, dist, cargar_config("frontend"))
    t_ns, nombres = euroc.leer_cam(euroc.ruta_cam(carpeta))
    carpeta_png = euroc.carpeta_imagenes(carpeta)
    fps = cargar_config("camara")["fps"]

    video = None
    estadisticas = []
    terminadas = []
    tiempos_ms = []
    for k, (t, nombre) in enumerate(zip(t_ns, nombres)):
        gris = cv2.imread(str(carpeta_png / nombre), cv2.IMREAD_GRAYSCALE)

        # 1. Front-end (cronometrado: anticipo de la etapa 13)
        t0 = time.perf_counter()
        r = frontend.procesar(gris, int(t))
        tiempos_ms.append((time.perf_counter() - t0) * 1e3)
        terminadas += r["terminadas"]
        estadisticas.append(r["estadisticas"])

        # 2. Dibujo
        vista = dibujar(gris, frontend)
        if video is None:
            video = cv2.VideoWriter(str(carpeta / "pistas.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                    fps, (gris.shape[1], gris.shape[0]))
            if not video.isOpened():
                print("AVISO: no se puede escribir el vídeo; solo se guardarán PNG sueltos")
        if video.isOpened():
            video.write(vista)
        if k % CADA_N_INSTANTANEA == 0:
            cv2.imwrite(str(carpeta / f"pistas_{k:04d}.png"), vista)
    if video is not None:
        video.release()
    terminadas += frontend.terminar_todas()

    # --- 3. Resumen ---
    activos = np.array([e["activos"] for e in estadisticas])
    longitudes = np.array([p.longitud() for p in terminadas])
    print(f"imágenes: {len(estadisticas)}")
    print(f"puntos activos por imagen: media {activos.mean():.0f}, mínimo {activos.min()}")
    print(f"pistas: {len(longitudes)} | longitud media {longitudes.mean():.1f} imágenes"
          f" | mediana {np.median(longitudes):.0f} | máxima {longitudes.max()}")
    print(f"pistas de 3 imágenes o más (las útiles para el filtro): {(longitudes >= 3).sum()}")
    print(f"perdidos por Lucas-Kanade: {sum(e['perdidos_klt'] for e in estadisticas)}"
          f" | rechazados por RANSAC: {sum(e['rechazados_ransac'] for e in estadisticas)}")
    print(f"tiempo del front-end: media {np.mean(tiempos_ms):.1f} ms, "
          f"máximo {np.max(tiempos_ms):.1f} ms por imagen")

    # --- 4. Figura ---
    fig, ejes = plt.subplots(1, 2, figsize=(15, 5))
    for clave in ("activos", "nuevos", "perdidos_klt", "rechazados_ransac"):
        ejes[0].plot([e[clave] for e in estadisticas], lw=0.8, label=clave)
    ejes[0].set_xlabel("imagen")
    ejes[0].set_ylabel("puntos")
    ejes[0].set_title("Por imagen")
    ejes[0].legend()
    ejes[1].hist(longitudes, bins=np.arange(1, longitudes.max() + 2) - 0.5)
    ejes[1].set_xlabel("longitud de la pista [imágenes]")
    ejes[1].set_ylabel("pistas")
    ejes[1].set_yscale("log")
    ejes[1].set_title("Longitud de las pistas")
    fig.tight_layout()
    fig.savefig(carpeta / "frontend.png", dpi=100)
    print(f"Resultados en {carpeta}: pistas.mp4, pistas_*.png, frontend.png")


if __name__ == "__main__":
    main()