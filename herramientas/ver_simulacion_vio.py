"""
herramientas/ver_simulacion_vio.py — Genera una secuencia VIO simulada y la dibuja.

Uso, desde ~/svo-pi:
    python -m herramientas.ver_simulacion_vio
Usa la sección "vio" de config/simulador.yaml y los intrínsecos de config/camara_intrinsecos.yaml.
Guarda datos/simulacion_vio.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vio import modelo_camara as mc
from vio import simulador as sim
from vio.config import RAIZ_PROYECTO, cargar_config

CADA_N_EJES = 40          # cada cuántas imágenes se dibuja el eje óptico de la cámara
IMAGEN_EJEMPLO = 100      # imagen que se dibuja como "foto" simulada
SEGUNDOS_IMU = 10.0       # segundos de IMU que se dibujan


def main():
    cfg = cargar_config("simulador")["vio"]
    K, _, tamano = mc.cargar_intrinsecos()
    rng = np.random.default_rng(cfg["semilla"])

    # --- 1. Generar la secuencia ---
    s = sim.generar_secuencia(cfg, K, tamano, rng)
    pistas, _ = sim.pistas_desde_observaciones(s["obs"])
    longitudes = np.array([p.longitud() for p in pistas])
    por_imagen = np.bincount(s["obs"]["imagen"], minlength=len(s["camaras"]["t_ns"]))
    f_imu = cfg["trayectoria"]["frecuencia_imu_hz"]
    v = s["verdad"]

    print(f"duración {cfg['trayectoria']['duracion_s']} s | IMU: {len(s['imu']['t_ns'])} muestras a {f_imu} Hz"
          f" | cámara: {len(s['camaras']['t_ns'])} imágenes a {cfg['camara']['frecuencia_hz']} Hz")
    print(f"puntos del mundo: {len(s['puntos'])} | observaciones: {len(s['obs']['id'])}"
          f" | por imagen: media {por_imagen.mean():.0f}, mínimo {por_imagen.min()}")
    print(f"pistas: {len(pistas)} | longitud mediana {np.median(longitudes):.0f} imágenes"
          f" | máxima {longitudes.max()} | de 3 o más: {(longitudes >= 3).sum()}")
    print(f"ruido por muestra: giroscopio {np.degrees(cfg['imu']['ruido_gir'] * np.sqrt(f_imu)):.3f} °/s,"
          f" acelerómetro {cfg['imu']['ruido_acc'] * np.sqrt(f_imu):.4f} m/s²")
    print(f"cambio del sesgo en la secuencia: giroscopio "
          f"{np.degrees(np.linalg.norm(v['b_g'][-1] - v['b_g'][0])):.3f} °/s,"
          f" acelerómetro {np.linalg.norm(v['b_a'][-1] - v['b_a'][0]):.4f} m/s²")

    # --- 2. Figura ---
    fig = plt.figure(figsize=(18, 10))

    # 2.1 Mundo y trayectoria en 3D
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.scatter(*s["puntos"].T, s=1, c="0.6")
    ax.plot(*v["p_w"].T, "b", lw=1, label="IMU")
    for j in range(0, len(s["camaras"]["t_ns"]), CADA_N_EJES):
        R_w_c = s["camaras"]["R_c_w"][j].T
        centro = -R_w_c @ s["camaras"]["t_c_w"][j]
        ax.quiver(*centro, *(R_w_c[:, 2] * 0.6), color="r")          # eje óptico (z de la cámara)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_box_aspect(cfg["mundo"]["habitacion_m"])
    ax.set_title("Habitación, trayectoria y eje óptico (rojo)")

    # 2.2 Una "foto" simulada: lo que ve la cámara en una imagen
    ax = fig.add_subplot(2, 3, 2)
    en_imagen = s["obs"]["imagen"] == IMAGEN_EJEMPLO
    ax.scatter(*s["obs"]["uv"][en_imagen].T, s=8)
    ax.set_xlim(0, tamano[0])
    ax.set_ylim(tamano[1], 0)                                        # v crece hacia abajo
    ax.set_aspect("equal")
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")
    ax.set_title(f"Imagen {IMAGEN_EJEMPLO}: {en_imagen.sum()} puntos vistos")

    # 2.3 Longitud de las pistas
    ax = fig.add_subplot(2, 3, 3)
    ax.hist(longitudes, bins=np.arange(1, longitudes.max() + 2) - 0.5)
    ax.set_xlabel("longitud de la pista [imágenes]")
    ax.set_ylabel("pistas")
    ax.set_title("Longitud de las pistas")

    # 2.4 y 2.5 IMU medida (primeros segundos)
    t = s["imu"]["t_ns"] / 1e9
    mostrar = t <= SEGUNDOS_IMU
    for posicion, clave, nombre, unidad in ((4, "w", "giroscopio", "rad/s"), (5, "a", "acelerómetro", "m/s²")):
        ax = fig.add_subplot(2, 3, posicion)
        for eje, color in zip("xyz", "rgb"):
            ax.plot(t[mostrar], s["imu"][clave][mostrar, "xyz".index(eje)], color, lw=0.6, label=eje)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"{nombre} [{unidad}]")
        ax.set_title(f"{nombre.capitalize()} simulado (con ruido y sesgo)")
        ax.legend()

    # 2.6 Sesgos verdaderos: paseo aleatorio
    ax = fig.add_subplot(2, 3, 6)
    for eje, color in zip(range(3), "rgb"):
        ax.plot(t, np.degrees(v["b_g"][:, eje]), color, lw=0.8)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("sesgo del giroscopio [°/s] (continua)")
    ax2 = ax.twinx()
    for eje, color in zip(range(3), "rgb"):
        ax2.plot(t, v["b_a"][:, eje], color, lw=0.8, ls="--")
    ax2.set_ylabel("sesgo del acelerómetro [m/s²] (discontinua)")
    ax.set_title("Sesgos verdaderos (paseo aleatorio)")

    fig.tight_layout()
    ruta = RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"] / "simulacion_vio.png"
    fig.savefig(ruta, dpi=90)
    print(f"Figura: {ruta}")


if __name__ == "__main__":
    main()