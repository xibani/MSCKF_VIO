"""
herramientas/grabar_simulacion.py — Una grabación EuRoC SINTÉTICA, con imágenes y verdad.

Uso, desde ~/svo-pi:
    python -m herramientas.grabar_simulacion                  # datos/sim_vio
    python -m herramientas.grabar_simulacion --duracion 15
La trayectoria de la etapa 11, quieta los primeros segundos (para inicializar en reposo),
vista por la cámara como manchas sobre fondo negro. Se escribe igual que una grabación de
herramientas/grabar, más verdad.csv con la pose verdadera: así apps/pi_vio y el front-end
se prueban de principio a fin con una respuesta conocida.
"""

import argparse
import copy
from pathlib import Path

import cv2
import numpy as np

from captura import euroc
from vio import modelo_camara as mc
from vio import simulador as sim
from vio.config import RAIZ_PROYECTO, cargar_config
from vio.pipeline import CABECERA_POSES, linea_pose


def guardar_grabacion_sintetica(carpeta, cfg_vio, K, tamano, cfg_ext, rng, dist=None):
    """Simula y escribe la grabación en `carpeta`. Devuelve: la secuencia del simulador.

    cfg_vio: sección "vio" de config/simulador.yaml. cfg_ext: dict de config/camara_imu.yaml
    (la extrínseca con la que se simula es la misma que usará el filtro).
    K, dist: la cámara con que se dibujan las imágenes. Para repetirla con apps/pi_vio deben
    ser las de config/camara_intrinsecos.yaml, que es lo que usará pi_vio para quitar la
    distorsión (dist=None: cámara sin distorsión).
    """
    carpeta = Path(carpeta)
    cfg = copy.deepcopy(cfg_vio)
    cfg["camara"]["R_i_c"] = cfg_ext["R_i_c"]
    cfg["camara"]["p_i_c_m"] = cfg_ext["p_i_c_m"]
    cfg["camara"]["ruido_px"] = 0.0                  # las imágenes son exactas: mide el front-end
    graba = cfg["grabacion_sintetica"]

    # 1. Simulación, con reposo al principio
    def pose(t):
        return sim.pose_con_reposo(t, cfg["trayectoria"], graba["reposo_s"], graba["rampa_s"])
    s = sim.generar_secuencia(cfg, K, tamano, rng, pose=pose, dist=dist)

    # 2. IMU
    euroc.ruta_imu(carpeta).parent.mkdir(parents=True, exist_ok=True)
    with open(euroc.ruta_imu(carpeta), "w", encoding="utf-8") as f:
        f.write(euroc.CABECERA_IMU)
        for t, w, a in zip(s["imu"]["t_ns"], s["imu"]["w"], s["imu"]["a"]):
            f.write(euroc.linea_imu(int(t), w, a))

    # 3. Imágenes: una mancha por cada punto visto
    carpeta_png = euroc.carpeta_imagenes(carpeta)
    carpeta_png.mkdir(parents=True, exist_ok=True)
    with open(euroc.ruta_cam(carpeta), "w", encoding="utf-8") as f:
        f.write(euroc.CABECERA_CAM)
        for j, t in enumerate(s["camaras"]["t_ns"]):
            uv = s["obs"]["uv"][s["obs"]["imagen"] == j]
            imagen = sim.renderizar(uv, tamano, graba["sigma_mancha_px"], graba["brillo_mancha"])
            nombre = euroc.nombre_imagen(int(t))
            cv2.imwrite(str(carpeta_png / nombre), imagen)
            f.write(euroc.linea_cam(int(t), nombre))

    # 4. Verdad: pose de la IMU en cada muestra
    with open(carpeta / "verdad.csv", "w", encoding="utf-8") as f:
        f.write(CABECERA_POSES)
        for t, R, p in zip(s["verdad"]["t_ns"], s["verdad"]["R_w_i"], s["verdad"]["p_w"]):
            f.write(linea_pose(t, R, p))
    return s


def main():
    parser = argparse.ArgumentParser(description="Grabación EuRoC sintética con verdad")
    parser.add_argument("--duracion", type=float, default=None, help="segundos (por defecto, el YAML)")
    parser.add_argument("--carpeta", default="sim_vio", help="nombre dentro de datos/")
    args = parser.parse_args()

    cfg_vio = cargar_config("simulador")["vio"]
    if args.duracion is not None:
        cfg_vio["trayectoria"]["duracion_s"] = args.duracion
    K, dist, tamano = mc.cargar_intrinsecos()
    carpeta = RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"] / args.carpeta
    s = guardar_grabacion_sintetica(carpeta, cfg_vio, K, tamano, cargar_config("camara_imu"),
                                    np.random.default_rng(cfg_vio["semilla"]), dist=dist)
    print(f"Grabación sintética en {carpeta}: {len(s['imu']['t_ns'])} muestras de IMU,"
          f" {len(s['camaras']['t_ns'])} imágenes, quieta los primeros "
          f"{cfg_vio['grabacion_sintetica']['reposo_s']} s")


if __name__ == "__main__":
    main()