"""
herramientas/ver_rotacion.py — Dibuja los ejes de un marco girado por exp_so3(phi).

Uso, desde ~/svo-pi (phi en GRADOS, para que sea fácil de imaginar):
    python -m herramientas.ver_rotacion 0 0 90       # 90° alrededor de z
    python -m herramientas.ver_rotacion 30 0 0       # 30° alrededor de x
    python -m herramientas.ver_rotacion 45 45 0      # 63,6° alrededor de (1, 1, 0)/√2
Guarda datos/rotacion.png.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vio import so3
from vio.config import RAIZ_PROYECTO, cargar_config

COLORES = ["tab:red", "tab:green", "tab:blue"]     # x rojo, y verde, z azul (convención habitual)


def main():
    parser = argparse.ArgumentParser(description="Dibuja una rotación")
    parser.add_argument("phi_grados", type=float, nargs=3, help="vector de rotación en grados")
    args = parser.parse_args()

    # --- 1. Calcular la rotación ---
    phi = np.radians(args.phi_grados)
    R = so3.exp_so3(phi)
    angulo = np.linalg.norm(phi)
    np.set_printoptions(precision=4, suppress=True)
    print("phi [rad]:", phi)
    print(f"ángulo: {np.degrees(angulo):.2f}°")
    if angulo > 0:
        print("eje:", phi / angulo)
    print("R =\n", R)
    print("¿es rotación?", so3.es_rotacion(R))
    print("log(R) [grados]:", np.degrees(so3.log_so3(R)))

    # --- 2. Dibujar: ejes del mundo (finos, discontinuos) y ejes girados (gruesos) ---
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    for i, nombre in enumerate("xyz"):
        e = np.eye(3)[:, i]                 # eje i del mundo
        ax.plot([0, e[0]], [0, e[1]], [0, e[2]], color=COLORES[i], ls="--", lw=1)
        c = R[:, i]                         # columna i de R = eje i girado, visto desde el mundo
        ax.plot([0, c[0]], [0, c[1]], [0, c[2]], color=COLORES[i], lw=3)
        ax.text(*(1.1 * c), nombre + "'", color=COLORES[i])
    if angulo > 0:
        u = phi / angulo
        ax.plot([-u[0], u[0]], [-u[1], u[1]], [-u[2], u[2]], color="k", lw=1, label="eje de giro")
        ax.legend()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))            # mismos tamaños en los tres ejes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Giro de {np.degrees(angulo):.1f}°: discontinuos = mundo, gruesos = girados")

    ruta = RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"] / "rotacion.png"
    fig.savefig(ruta, dpi=100)
    print(f"Figura: {ruta}")


if __name__ == "__main__":
    main()