"""
herramientas/ver_planos.py — Etapa 27: los planos de la nube densa de una sesión.

Uso, desde la raíz del proyecto:
    python -m herramientas.ver_planos datos/sim_textura                 # sobre nube_densa.ply
    python -m herramientas.ver_planos datos/sim_textura --con-verdad    # sobre nube_densa_verdad.ply
    python -m herramientas.ver_planos datos/20261116_101500 --archivo mapa_2.paquetes
La carpeta es la de la sesión. Necesita la nube que escribe herramientas.nube_densa (etapa 26)
y el mapa.paquetes (de él salen las posiciones de las cámaras, para saber hacia dónde es
"dentro" de la habitación).

Escribe en la carpeta:
    planos.csv    una fila por plano: tipo, normal, d, centro, ejes, extremos, superficie...
    planos.ply    la nube con cada plano de un color (gris: puntos sin plano)
    planos.png    planta con las paredes dibujadas, y alzado
(planos_verdad.* con --con-verdad). Con una grabación del simulador compara además cada
plano con la cara de verdad de la habitación.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from herramientas.medir_mapa import leer_mapa
from herramientas.nube_densa import NOMBRES_CARAS, nombre_de_salida, transformacion_de_la_sesion
from vio import planos as pl
from vio.config import cargar_config
from vio.reconstruccion import guardar_ply, leer_ply

CABECERA_CSV = ("#plano,tipo,n_x,n_y,n_z,d,c_x,c_y,c_z,e1_x,e1_y,e1_z,e2_x,e2_y,e2_z,"
                "a_min,a_max,b_min,b_max,superficie_m2,grosor_m,puntos\n")
GRIS_SIN_PLANO = (110, 110, 110)


def colores_de_planos(n_planos):
    """Un color RGB (0-255) para cada plano, de la paleta "tab20" de matplotlib (20 colores
    bien distintos; si hay más planos, se repiten). Devuelve: (n_planos, 3) uint8."""
    paleta = matplotlib.colormaps["tab20"]
    return np.array([[int(255 * c) for c in paleta(i % 20)[:3]] for i in range(n_planos)], dtype=np.uint8).reshape(-1, 3)


def guardar_csv(ruta, planos):
    """planos.csv: todo lo que describe a cada plano, para las etapas siguientes."""
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(CABECERA_CSV)
        for i, q in enumerate(planos):
            numeros = (list(q["n"]) + [q["d"]] + list(q["centro"]) + list(q["ejes"][0]) + list(q["ejes"][1])
                       + list(q["extremos"]) + [q["superficie"], q["grosor"]])
            f.write(f"{i},{q['tipo']}," + ",".join(f"{x:.6f}" for x in numeros) + f",{q['puntos']}\n")


def informe(planos, medidas, n_puntos, comparaciones, dimensiones):
    """El texto del informe.

    medidas: de planos.medidas_de_habitacion. comparaciones y dimensiones: de la comparación
    con la habitación de verdad (None si la sesión no es del simulador).
    """
    cabecera = f"{'plano':>5s} {'tipo':<18s} {'puntos':>7s} {'superficie':>10s} {'grosor':>7s}   {'normal':<22s}"
    if comparaciones is not None:
        cabecera += f" {'cara de verdad':<14s} {'ángulo':>7s} {'error':>8s}"
    lineas = [cabecera]
    for i, q in enumerate(planos):
        normal = f"({q['n'][0]:+.2f}, {q['n'][1]:+.2f}, {q['n'][2]:+.2f})"
        linea = (f"{i:5d} {q['tipo']:<18s} {q['puntos']:7d} {q['superficie']:7.1f} m² {100 * q['grosor']:4.1f} cm"
                 f"   {normal:<22s}")
        if comparaciones is not None:
            c = comparaciones[i]
            linea += f" {NOMBRES_CARAS[c['cara']]:<14s} {c['error_angulo']:6.2f}° {100 * c['error_m']:5.1f} cm"
        lineas.append(linea)
    en_planos = sum(q["puntos"] for q in planos)
    lineas.append("")
    lineas.append(f"{len(planos)} planos con {en_planos} de {n_puntos} puntos ({100 * en_planos / max(n_puntos, 1):.0f} %)")

    # Las medidas de la habitación. En el simulador, la verdad de cada una sale de las caras
    # que se han emparejado: dos paredes sobre las caras 0 y 1 están separadas lx, etc.
    if medidas["altura"] is not None:
        linea = f"Altura (suelo-techo): {medidas['altura']:.3f} m"
        if dimensiones is not None:
            linea += f" (verdad {dimensiones[2]:.3f} m, error {100 * (medidas['altura'] / dimensiones[2] - 1):+.1f} %)"
        lineas.append(linea)
    for i, j, separacion in medidas["opuestas"]:
        linea = f"Paredes {i} y {j}, enfrentadas: {separacion:.3f} m"
        if comparaciones is not None:
            caras = sorted([comparaciones[i]["cara"], comparaciones[j]["cara"]])
            if caras in ([0, 1], [2, 3]):
                verdad = dimensiones[caras[0] // 2]
                linea += f" (verdad {verdad:.3f} m, error {100 * (separacion / verdad - 1):+.1f} %)"
        lineas.append(linea)
    if medidas["esquinas"]:
        angulos = np.array([angulo for _, _, angulo in medidas["esquinas"]])
        lineas.append(f"Esquinas entre paredes: {len(angulos)}, de {angulos.min():.1f}° a {angulos.max():.1f}°"
                      f" (se apartan de 90° como mucho {np.abs(angulos - 90.0).max():.1f}°)")
    return "\n".join(lineas)


def dibujar(ruta, p, planos, camaras):
    """Planta (x, y) y alzado (x, z): los puntos de cada plano con su color, los puntos sin
    plano en gris y, en la planta, cada pared como un segmento a lo largo de su eje horizontal."""
    colores = colores_de_planos(len(planos)) / 255.0
    sin_plano = np.ones(len(p), dtype=bool)
    for q in planos:
        sin_plano[q["indices"]] = False
    fig, (planta, alzado) = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, (a, b), titulo in ((planta, (0, 1), "Planta (x, y)"), (alzado, (0, 2), "Alzado (x, z)")):
        ax.scatter(p[sin_plano, a], p[sin_plano, b], s=1, color="0.6")
        for i, q in enumerate(planos):
            ax.scatter(p[q["indices"], a], p[q["indices"], b], s=1, color=colores[i])
        ax.plot(camaras[:, a], camaras[:, b], "k.-", lw=0.8, ms=3)
        ax.set_title(titulo)
        ax.set_xlabel("xyz"[a] + " [m]")
        ax.set_ylabel("xyz"[b] + " [m]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    for i, q in enumerate(planos):
        if q["tipo"] != "pared":
            continue
        # El segmento de la pared: del extremo a_min al a_max, a lo largo de su eje horizontal
        inicio = q["centro"] + q["extremos"][0] * q["ejes"][0]
        fin = q["centro"] + q["extremos"][1] * q["ejes"][0]
        planta.plot([inicio[0], fin[0]], [inicio[1], fin[1]], "k-", lw=2.5)
        planta.annotate(str(i), q["centro"][:2] + 0.25 * q["n"][:2], ha="center", va="center", fontsize=11,
                        bbox={"boxstyle": "circle", "facecolor": colores[i], "edgecolor": "k"})
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Planos de la nube densa de una sesión")
    parser.add_argument("carpeta", help="carpeta de la sesión (la que tiene mapa.paquetes y la nube)")
    parser.add_argument("--archivo", default=cargar_config("vio")["mapeo"]["archivo"],
                        help="nombre del archivo de paquetes dentro de la carpeta")
    parser.add_argument("--con-verdad", action="store_true",
                        help="usar la nube hecha con la profundidad verdadera (nube_densa_verdad.ply)")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)
    cfg = cargar_config("planos")

    ruta_nube = carpeta / nombre_de_salida("nube_densa", "ply", args.con_verdad)
    if not ruta_nube.exists():
        print(f"No existe {ruta_nube}: ejecuta antes herramientas.nube_densa"
              f"{' --con-verdad' if args.con_verdad else ''}.")
        return
    p, _ = leer_ply(ruta_nube)
    mapa = leer_mapa(carpeta / args.archivo)
    if not mapa.keyframes:
        print("El mapa no tiene ningún keyframe: no se sabe dónde han estado las cámaras.")
        return
    camaras = np.array([kf["p_w_c"] for kf in mapa.keyframes])
    punto_dentro = camaras.mean(axis=0)
    print(f"Nube: {len(p)} puntos. Cámaras: {len(camaras)} keyframes")

    planos = pl.buscar_planos(p, punto_dentro, cfg, np.random.default_rng(cfg["ransac"]["semilla"]))
    medidas = pl.medidas_de_habitacion(planos, cfg["tipos"]["tolerancia_grados"])
    comparaciones, dimensiones = None, None
    transformacion = transformacion_de_la_sesion(carpeta)
    if transformacion is not None:
        dimensiones = cargar_config("simulador")["vio"]["mundo"]["habitacion_m"]
        comparaciones = pl.comparar_con_la_verdad(planos, *transformacion, dimensiones)
    print(informe(planos, medidas, len(p), comparaciones, dimensiones))

    ruta_csv = carpeta / nombre_de_salida("planos", "csv", args.con_verdad)
    ruta_ply = carpeta / nombre_de_salida("planos", "ply", args.con_verdad)
    ruta_png = carpeta / nombre_de_salida("planos", "png", args.con_verdad)
    guardar_csv(ruta_csv, planos)
    colores = np.tile(np.array(GRIS_SIN_PLANO, dtype=np.uint8), (len(p), 1))
    for color, q in zip(colores_de_planos(len(planos)), planos):
        colores[q["indices"]] = color
    guardar_ply(ruta_ply, p, colores)
    dibujar(ruta_png, p, planos, camaras)
    print(f"Planos: {ruta_csv}\nNube coloreada: {ruta_ply}\nFigura: {ruta_png}")


if __name__ == "__main__":
    main()
