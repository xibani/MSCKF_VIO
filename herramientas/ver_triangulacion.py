"""
herramientas/ver_triangulacion.py — ¿Cuánta distancia entre imágenes hace falta para triangular?

Uso, desde ~/svo-pi:
    python -m herramientas.ver_triangulacion
Simula un punto a varias profundidades visto desde 2 posiciones de la cámara separadas
una distancia B (la "base"), con ruido de 1 px en cada observación. Para cada caso repite
la triangulación muchas veces y mide la dispersión de la profundidad, que compara con
    sigma_Z = sqrt(2) · Z² · sigma_px / (f · B)
Guarda datos/triangulacion_ruido.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vio import modelo_camara as mc
from vio import triangulacion as tri
from vio.config import RAIZ_PROYECTO, cargar_config

PROFUNDIDADES_M = [2.0, 5.0, 10.0, 20.0]
BASES_M = np.geomspace(0.02, 1.0, 12)      # de 2 cm a 1 m, repartidas en escala logarítmica
RUIDO_PX = 1.0
REPETICIONES = 200


def main():
    cfg = cargar_config("triangulacion")
    K, _, _ = mc.cargar_intrinsecos()
    focal = K[0, 0]
    rng = np.random.default_rng(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    print(f"focal {focal:.0f} px, ruido {RUIDO_PX} px, {REPETICIONES} repeticiones por caso")
    print(f"{'Z [m]':>6} {'B [m]':>6} {'paralaje [°]':>12} {'fiables':>8} "
          f"{'sigma_Z medida [m]':>19} {'predicha [m]':>13}")

    for Z in PROFUNDIDADES_M:
        punto = np.array([0.0, 0.0, Z])
        sigmas, predichas, fiables = [], [], []
        for B in BASES_M:
            # 1. Dos cámaras mirando hacia +z, separadas B a lo largo de x
            poses = []
            for x in (-B / 2, B / 2):
                poses.append((np.eye(3), -np.array([x, 0.0, 0.0])))   # R_c_w = I, t_c_w = -c_w

            # 2. Muchas triangulaciones con ruido distinto
            profundidades = []
            for _ in range(REPETICIONES):
                xy = np.array([tri.proyectar_normalizado(R, t, punto)[0] for R, t in poses])
                xy = xy + rng.normal(0.0, RUIDO_PX / focal, xy.shape)
                r = tri.triangular(poses, xy, cfg, focal)
                if r["fiable"]:
                    profundidades.append(r["p_w"][2])

            # 3. Dispersión medida frente a la predicha
            fraccion = len(profundidades) / REPETICIONES
            sigma = np.std(profundidades) if len(profundidades) > 10 else np.nan
            prediccion = np.sqrt(2) * Z ** 2 * RUIDO_PX / (focal * B)
            sigmas.append(sigma)
            predichas.append(prediccion)
            fiables.append(fraccion)
            paralaje = np.degrees(2 * np.arctan(B / 2 / Z))
            print(f"{Z:6.1f} {B:6.3f} {paralaje:12.2f} {fraccion:8.0%} {sigma:19.3f} {prediccion:13.3f}")

        linea, = ax1.loglog(BASES_M, sigmas, "o-", label=f"Z = {Z:.0f} m (medida)")
        ax1.loglog(BASES_M, predichas, "--", color=linea.get_color(), lw=0.8)
        ax2.semilogx(BASES_M, fiables, "o-", color=linea.get_color(), label=f"Z = {Z:.0f} m")

    ax1.set_xlabel("base B [m]")
    ax1.set_ylabel("desviación típica de la profundidad [m]")
    ax1.set_title(f"Error de profundidad con {RUIDO_PX} px de ruido (discontinua: fórmula)")
    ax1.grid(True, which="both")
    ax1.legend()
    ax2.set_xlabel("base B [m]")
    ax2.set_ylabel("fracción de puntos fiables")
    ax2.set_title(f"Fiables (paralaje mínimo {cfg['fiabilidad']['paralaje_min_grados']}°)")
    ax2.grid(True, which="both")
    ax2.legend()
    fig.tight_layout()

    ruta = RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"] / "triangulacion_ruido.png"
    fig.savefig(ruta, dpi=100)
    print(f"Figura: {ruta}")


if __name__ == "__main__":
    main()