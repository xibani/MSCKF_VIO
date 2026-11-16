"""
herramientas/nube_densa.py — Etapa 26: la nube densa de una sesión del VIO.

Uso, desde la raíz del proyecto:
    python -m herramientas.nube_densa datos/sim_textura                 # con la profundidad de la etapa 25
    python -m herramientas.nube_densa datos/sim_textura --con-verdad    # simulador: con la profundidad verdadera
    python -m herramientas.nube_densa datos/20261116_101500 --archivo mapa_2.paquetes
La carpeta es la de la sesión: la que tiene el mapa.paquetes que escribe apps/pi_vio.

Sin --con-verdad lee la profundidad de cada keyframe de <carpeta>/profundidad/<indice>.png,
que es lo que deja herramientas/ver_profundidad (etapa 25, en el PC, con Depth Anything).
Con --con-verdad usa la profundidad verdadera de la grabación simulada (mav0/prof0): no
hace falta la red, y dice hasta dónde se puede llegar con una profundidad perfecta.

Escribe en la carpeta:
    nube_densa.ply    la nube (x, y, z y el gris de cada punto), para CloudCompare o MeshLab
    nube_densa.png    planta y alzado
(nube_densa_verdad.* con --con-verdad). Con una grabación del simulador compara además la
nube con la habitación de verdad.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from captura import euroc
from herramientas.medir_mapa import leer_mapa
from vio import densificar as den
from vio.config import cargar_config
from vio.pipeline import leer_poses
from vio.reconstruccion import decodificar_gris, guardar_ply

NOMBRES_CARAS = ("pared x-", "pared x+", "pared y-", "pared y+", "suelo", "techo")


def nombre_de_salida(base, extension, con_verdad):
    """nube_densa.ply, o nube_densa_verdad.ply si se ha hecho con la profundidad verdadera."""
    return f"{base}_verdad.{extension}" if con_verdad else f"{base}.{extension}"


def cargar_vistas(keyframes, carpeta, con_verdad, t_d_ns):
    """Las vistas (keyframe + profundidad + imagen) de los keyframes que tienen profundidad.

    con_verdad: False -> <carpeta>/profundidad/<indice>.png (etapa 25);
                True  -> la profundidad verdadera de la grabación (el archivo se llama como
                         la marca de la imagen = instante del keyframe - t_d).
    Devuelve: lista de dict con kf, z (alto, ancho) [m] y gris (alto, ancho) uint8.
    """
    vistas = []
    for kf in keyframes:
        if con_verdad:
            z = euroc.leer_profundidad(carpeta, kf["t_ns"] - t_d_ns)
        else:
            ruta = Path(carpeta) / "profundidad" / f"{kf['indice']:06d}.png"
            z = euroc.mm_a_profundidad(cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)) if ruta.exists() else None
        if z is not None:
            vistas.append({"kf": kf, "z": z, "gris": decodificar_gris(kf)})
    return vistas


def transformacion_de_la_sesion(carpeta):
    """(R, t) para pasar del mundo del VIO al del simulador, o None si la sesión no es del
    simulador (le falta verdad.csv) o no tiene vio_trayectoria.csv."""
    carpeta = Path(carpeta)
    if not (carpeta / "verdad.csv").exists() or not (carpeta / "vio_trayectoria.csv").exists():
        return None
    estimada = leer_poses(carpeta / "vio_trayectoria.csv")
    verdad = leer_poses(carpeta / "verdad.csv")
    return den.transformacion_a_la_verdad(*estimada, *verdad)


def informe(resultado, evaluacion, cerca_m):
    """El texto del informe. evaluacion: de densificar.evaluar_nube, o None. cerca_m: a qué
    distancia de una superficie se da un punto por bueno (evaluacion.cerca_m del YAML)."""
    lineas = [f"{'keyframe':>8s} {'píxeles':>8s} {'sin dato':>8s} {'en borde':>8s} {'sin confirmar':>13s} {'usados':>7s}"]
    for v in resultado["por_vista"]:
        lineas.append(f"{v['indice']:8d} {v['candidatos']:8d} {v['sin_dato']:8d} {v['en_borde']:8d}"
                      f" {v['sin_confirmar']:13d} {v['usados']:7d}")
    candidatos = sum(v["candidatos"] for v in resultado["por_vista"])
    usados = sum(v["usados"] for v in resultado["por_vista"])
    lineas.append("")
    lineas.append(f"{len(resultado['por_vista'])} keyframes, {candidatos} píxeles, {usados} usados"
                  f" ({100 * usados / max(candidatos, 1):.0f} %)")
    lineas.append(f"Nube densa: {len(resultado['p_w'])} puntos")
    if evaluacion is not None and evaluacion["n"] > 0:
        lineas.append(f"Frente a la habitación de verdad: error mediano {100 * evaluacion['error_mediano']:.1f} cm,"
                      f" el 90 % a menos de {100 * evaluacion['error_p90']:.1f} cm,"
                      f" {100 * evaluacion['cerca']:.0f} % de los puntos a menos de {100 * cerca_m:.0f} cm")
        por_cara = ", ".join(f"{nombre} {100 * c:.0f} %"
                             for nombre, c in zip(NOMBRES_CARAS, evaluacion["cubierto_por_cara"]))
        lineas.append(f"Superficie de la habitación cubierta: {100 * evaluacion['cubierto']:.0f} % ({por_cara})")
    return "\n".join(lineas)


def dibujar(ruta, p_w, gris, keyframes):
    """Figura con dos vistas de la nube: planta (x, y) y alzado (x, z), cada punto con su gris,
    y las posiciones de los keyframes en rojo."""
    fig, (planta, alzado) = plt.subplots(1, 2, figsize=(14, 6.5))
    camaras = np.array([kf["p_w_c"] for kf in keyframes])
    for ax, (a, b), titulo in ((planta, (0, 1), "Planta (x, y)"), (alzado, (0, 2), "Alzado (x, z)")):
        if len(p_w):
            ax.scatter(p_w[:, a], p_w[:, b], s=1, c=gris, cmap="gray", vmin=0, vmax=255)
        ax.plot(camaras[:, a], camaras[:, b], "r.-", lw=0.8, ms=4, label=f"{len(keyframes)} keyframes")
        ax.set_title(f"{titulo}: {len(p_w)} puntos")
        ax.set_xlabel("xyz"[a] + " [m]")
        ax.set_ylabel("xyz"[b] + " [m]")
        ax.set_aspect("equal")
        ax.set_facecolor("#203040")                      # fondo oscuro: los puntos claros se ven
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Nube densa de una sesión del VIO")
    parser.add_argument("carpeta", help="carpeta de la sesión (la que tiene mapa.paquetes)")
    parser.add_argument("--archivo", default=cargar_config("vio")["mapeo"]["archivo"],
                        help="nombre del archivo de paquetes dentro de la carpeta")
    parser.add_argument("--con-verdad", action="store_true",
                        help="simulador: usar la profundidad verdadera en lugar de la de la etapa 25")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)
    cfg = cargar_config("densificar")

    mapa = leer_mapa(carpeta / args.archivo)
    print(mapa.resumen())
    vistas = cargar_vistas(mapa.keyframes, carpeta, args.con_verdad, cargar_config("camara_imu")["t_d_ns"])
    if not vistas:
        print("Ningún keyframe tiene profundidad. Sin --con-verdad hace falta haber ejecutado antes"
              " herramientas.ver_profundidad; con --con-verdad, una grabación del simulador con textura.")
        return
    print(f"{len(vistas)} de {len(mapa.keyframes)} keyframes con profundidad"
          f" ({'la verdadera' if args.con_verdad else 'la de la etapa 25'})")
    resultado = den.densificar(vistas, cfg)

    evaluacion = None
    transformacion = transformacion_de_la_sesion(carpeta)
    if transformacion is not None:
        R, t = transformacion
        dimensiones = cargar_config("simulador")["vio"]["mundo"]["habitacion_m"]
        evaluacion = den.evaluar_nube(resultado["p_w"] @ R.T + t, dimensiones, cfg["evaluacion"])
    print(informe(resultado, evaluacion, cfg["evaluacion"]["cerca_m"]))

    ruta_ply = carpeta / nombre_de_salida("nube_densa", "ply", args.con_verdad)
    ruta_png = carpeta / nombre_de_salida("nube_densa", "png", args.con_verdad)
    gris = resultado["gris"]
    guardar_ply(ruta_ply, resultado["p_w"], np.column_stack([gris, gris, gris]))
    dibujar(ruta_png, resultado["p_w"], gris, [v["kf"] for v in vistas])
    print(f"Nube: {ruta_ply}\nFigura: {ruta_png}")


if __name__ == "__main__":
    main()
