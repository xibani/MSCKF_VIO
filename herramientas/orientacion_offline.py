"""
herramientas/orientacion_offline.py — Orientación con el giroscopio sobre una grabación.

Uso, desde ~/svo-pi:
    python -m herramientas.orientacion_offline datos/imu_20261102_183000
Vale cualquier grabación con mav0/imu0/data.csv (las de grabar_imu y las de grabar).
Guarda orientacion.png en la carpeta de la grabación.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from captura import euroc
from vio import inicializacion as ini
from vio import so3
from vio.config import cargar_config
from vio.integrador_giro import integrar_secuencia


def main():
    parser = argparse.ArgumentParser(description="Orientación con giroscopio, sin conexión")
    parser.add_argument("carpeta", help="carpeta de la grabación (la que contiene mav0/)")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)
    cfg = cargar_config("orientacion")

    # --- 1. Cargar ---
    t_ns, w, a = euroc.leer_imu(euroc.ruta_imu(carpeta))
    t_s = (t_ns - t_ns[0]) / 1e9

    # --- 2. Inicializar con el tramo de reposo del principio ---
    en_reposo = t_s < cfg["inicializacion"]["t_reposo_s"]
    reposo = ini.estimar_reposo(w[en_reposo], a[en_reposo])
    for problema in ini.comprobar_reposo(reposo, cfg["inicializacion"]):
        print("AVISO:", problema)
    R0 = ini.rotacion_desde_gravedad(reposo["a_media"])
    print(f"sesgo del giroscopio [°/s]: {np.degrees(reposo['b_g'])}")
    al0, ca0, _ = so3.angulos_zyx(R0)
    print(f"inclinación inicial: alabeo {np.degrees(al0):+.2f}°, cabeceo {np.degrees(ca0):+.2f}°")

    # --- 3. Integrar toda la grabación ---
    R, huecos = integrar_secuencia(R0, reposo["b_g"], t_ns, w, cfg["integracion"])
    print(f"muestras: {len(t_ns)} | huecos: {huecos}")

    # --- 4. Ángulos para dibujar: giroscopio integrado y acelerómetro ---
    giro = np.degrees(np.array([so3.angulos_zyx(Rk) for Rk in R]))          # (N, 3)
    acel = np.degrees(np.array([ini.inclinacion_desde_acelerometro(ak) for ak in a]))   # (N, 2)
    print(f"al final: alabeo {giro[-1, 0]:+.2f}°, cabeceo {giro[-1, 1]:+.2f}°, "
          f"guiñada {giro[-1, 2]:+.2f}°")
    print(f"giro total entre el principio y el final: "
          f"{np.degrees(so3.angulo_entre(R[0], R[-1])):.2f}°")

    # --- 5. Figura ---
    fig, ejes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    nombres = ["alabeo (giro sobre x)", "cabeceo (giro sobre y)", "guiñada (giro sobre z)"]
    for i in range(3):
        ejes[i].plot(t_s, giro[:, i], lw=1.0, label="giroscopio integrado")
        if i < 2:   # el acelerómetro solo ve la inclinación, no el rumbo
            ejes[i].plot(t_s, acel[:, i], lw=0.6, alpha=0.6,
                         label="acelerómetro (válido solo sin aceleraciones)")
        ejes[i].set_ylabel("°")
        ejes[i].set_title(nombres[i])
        ejes[i].legend(loc="upper left")
        ejes[i].grid(True)
    ejes[2].set_xlabel("tiempo [s]")
    fig.tight_layout()
    ruta_png = carpeta / "orientacion.png"
    fig.savefig(ruta_png, dpi=100)
    print(f"Figura: {ruta_png}")


if __name__ == "__main__":
    main()