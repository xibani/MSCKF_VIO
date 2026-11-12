"""
herramientas/ver_habitacion.py — Etapa 20: dibuja lo que ve la cámara en la habitación con textura.

Uso, desde la raíz del proyecto:
    python -m herramientas.ver_habitacion
    python -m herramientas.ver_habitacion --instantes 0 5 10
Usa la sección "vio" de config/simulador.yaml y los intrínsecos de config/camara_intrinsecos.yaml.
Guarda datos/habitacion.png: arriba, la imagen en cada instante de la trayectoria; abajo, la
profundidad verdadera de cada píxel (la distancia, hacia delante, a la pared que se ve).
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vio import escena
from vio import modelo_camara as mc
from vio import simulador as sim
from vio.config import RAIZ_PROYECTO, cargar_config


def main():
    parser = argparse.ArgumentParser(description="Vistas de la habitación con textura")
    parser.add_argument("--instantes", type=float, nargs="+", default=[0.0, 4.0, 8.0],
                        help="segundos de la trayectoria que se dibujan")
    args = parser.parse_args()

    cfg = cargar_config("simulador")["vio"]
    K, dist, tamano = mc.cargar_intrinsecos()
    rng = np.random.default_rng(cfg["semilla"])

    # 1. La habitación y los rayos de la cámara (una sola vez)
    habitacion = escena.crear_habitacion(cfg["mundo"]["habitacion_m"], cfg["textura"], rng)
    rayos = escena.rayos_camara(K, dist, tamano)
    R_c_i, t_c_i = sim.extrinseca(cfg["camara"])

    # 2. Una columna por instante: imagen arriba, profundidad abajo
    figura, ejes = plt.subplots(2, len(args.instantes), figsize=(5 * len(args.instantes), 7.5),
                                squeeze=False)
    for columna, t_s in enumerate(args.instantes):
        R_w_i, p_w = sim.pose_vio(t_s, cfg["trayectoria"])
        R_c_w, t_c_w = sim.pose_camara(R_w_i, p_w, R_c_i, t_c_i)
        ideal, profundidad = escena.renderizar_habitacion(habitacion, R_c_w, t_c_w, rayos)
        imagen = escena.acabar_imagen(ideal, cfg["textura"], rng)

        ejes[0, columna].imshow(imagen, cmap="gray", vmin=0, vmax=255)
        ejes[0, columna].set_title(f"t = {t_s:.1f} s")
        dibujo = ejes[1, columna].imshow(profundidad, cmap="viridis")
        ejes[1, columna].set_title(f"profundidad: de {profundidad.min():.2f} a {profundidad.max():.2f} m")
        figura.colorbar(dibujo, ax=ejes[1, columna], fraction=0.046, label="m")
        for fila in range(2):
            ejes[fila, columna].axis("off")

    figura.tight_layout()
    ruta = RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"] / "habitacion.png"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=110)
    print(f"Guardado {ruta}")


if __name__ == "__main__":
    main()