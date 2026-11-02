"""
herramientas/calibrar_camara.py — Calcula los intrínsecos de la cámara a partir de las capturas.

Uso, desde ~/svo-pi:
    python -m herramientas.calibrar_camara datos/calib_cam_20261102_183000
Guarda el resultado en config/camara_intrinsecos.yaml y una figura en la carpeta.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vio import calibracion_camara as cc
from vio import modelo_camara as mc
from vio.config import RAIZ_PROYECTO, cargar_config


def main():
    parser = argparse.ArgumentParser(description="Calibración de intrínsecos de la cámara")
    parser.add_argument("carpeta", help="carpeta con las PNG de capturar_tablero")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)
    cfg = cargar_config("calibracion")
    t, c = cfg["tablero"], cfg["camara"]

    # --- 1. Detectar las esquinas en todas las imágenes ---
    # Solo las capturas (nombre = marca de tiempo en ns), no la figura de calibracion.png
    rutas = sorted(r for r in carpeta.glob("*.png") if r.stem.isdigit())
    nombres, lista = [], []
    alto = ancho = None
    for ruta in rutas:
        gris = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        esquinas = cc.detectar_esquinas(gris, t["esquinas_x"], t["esquinas_y"])
        if esquinas is not None:
            nombres.append(ruta.name)
            lista.append(esquinas)
            alto, ancho = gris.shape
    print(f"Tablero encontrado en {len(lista)} de {len(rutas)} imágenes")
    if len(lista) < 10:
        print("ERROR: hacen falta al menos 10 imágenes válidas (mejor 25-30).")
        return
    puntos_3d = cc.puntos_tablero(t["esquinas_x"], t["esquinas_y"], t["lado_m"])

    # --- 2. Primera calibración y descarte de las imágenes peores ---
    r = cc.calibrar(lista, puntos_3d, ancho, alto)
    mediana = np.median(r["errores_por_imagen"])
    quedan = r["errores_por_imagen"] <= c["factor_descarte"] * mediana
    for nombre, error, ok in zip(nombres, r["errores_por_imagen"], quedan):
        if not ok:
            print(f"  descartada {nombre}: error {error:.3f} px (mediana {mediana:.3f})")
    print(f"Primera pasada: RMS {r['rms']:.3f} px con {len(lista)} imágenes")

    # --- 3. Calibración final con las que quedan ---
    lista = [e for e, ok in zip(lista, quedan) if ok]
    r = cc.calibrar(lista, puntos_3d, ancho, alto)
    K, dist = r["K"], r["dist"]
    print(f"Final: RMS {r['rms']:.3f} px con {len(lista)} imágenes")
    print(f"  fx = {K[0, 0]:.2f}  fy = {K[1, 1]:.2f}  cx = {K[0, 2]:.2f}  cy = {K[1, 2]:.2f}  [px]")
    print(f"  distorsión (k1, k2, p1, p2, k3) = {np.round(dist, 5)}")
    print("  cobertura de esquinas por zonas:")
    print(cc.cobertura(lista, ancho, alto, c["zonas_cobertura"]))

    if r["rms"] <= c["rms_objetivo_px"]:
        print(f">>> CALIBRACIÓN BUENA (RMS ≤ {c['rms_objetivo_px']} px)")
    elif r["rms"] <= c["rms_max_px"]:
        print(f">>> ACEPTABLE (RMS ≤ {c['rms_max_px']} px). Mejorable con más imágenes variadas.")
    else:
        print(f">>> NO VALE (RMS > {c['rms_max_px']} px): imágenes movidas o tablero no plano.")

    # --- 4. Guardar ---
    ruta_yaml = RAIZ_PROYECTO / "config" / "camara_intrinsecos.yaml"
    info = {"rms_px": round(r["rms"], 4), "imagenes": len(lista),
            "carpeta": str(carpeta), "fecha": datetime.now().isoformat(timespec="seconds")}
    mc.guardar_intrinsecos(ruta_yaml, K, dist, ancho, alto, info)
    print(f"Guardado en {ruta_yaml}")

    # --- 5. Figura: una imagen original y sin distorsión, y el error de cada imagen ---
    ejemplo = cv2.imread(str(carpeta / nombres[0]), cv2.IMREAD_GRAYSCALE)
    sin_distorsion = cv2.undistort(ejemplo, K, dist)
    fig, ejes = plt.subplots(1, 3, figsize=(17, 5))
    ejes[0].imshow(ejemplo, cmap="gray")
    ejes[0].set_title("original")
    ejes[1].imshow(sin_distorsion, cmap="gray")
    ejes[1].set_title("sin distorsión (las rectas deben salir rectas)")
    ejes[2].bar(range(len(lista)), r["errores_por_imagen"])
    ejes[2].axhline(c["rms_objetivo_px"], color="g", ls="--", label="objetivo")
    ejes[2].set_xlabel("imagen")
    ejes[2].set_ylabel("error de reproyección [px]")
    ejes[2].legend()
    fig.tight_layout()
    fig.savefig(carpeta / "calibracion.png", dpi=90)
    print(f"Figura: {carpeta / 'calibracion.png'}")


if __name__ == "__main__":
    main()