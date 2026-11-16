"""
herramientas/buscar_bucles.py — Etapa 28: los bucles de una sesión del VIO.

Uso, desde la raíz del proyecto:
    python -m herramientas.buscar_bucles datos/sim_textura30                       # LightGlue + profundidad de la etapa 25 (PC)
    python -m herramientas.buscar_bucles datos/sim_textura30 --emparejador orb     # sin redes para emparejar
    python -m herramientas.buscar_bucles datos/sim_textura30 --emparejador orb --con-verdad   # sin ninguna red (vale en la Pi)
    python -m herramientas.buscar_bucles datos/20261116_101500 --archivo mapa_2.paquetes
La carpeta es la de la sesión: la que tiene el mapa.paquetes que escribe apps/pi_vio.

Necesita la profundidad de cada keyframe: la de <carpeta>/profundidad/ (etapa 25) o, con
--con-verdad, la verdadera de una grabación simulada.

Escribe en la carpeta:
    bucles.csv        una fila por bucle: los dos keyframes y la pose de uno visto desde el otro
    bucles.png        planta con los keyframes y una línea por bucle
    bucle_mejor.jpg   las dos imágenes del bucle con más puntos, con sus parejas
(bucles_verdad.* y bucle_mejor_verdad.jpg con --con-verdad). Con una grabación del simulador
compara además cada bucle con la pose relativa verdadera.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from herramientas.medir_mapa import leer_mapa
from herramientas.nube_densa import cargar_vistas, nombre_de_salida
from vio import bucles as bu
from vio.config import cargar_config
from vio.emparejador import dibujar_parejas
from vio.pipeline import leer_poses

CABECERA_CSV = ("#i,j,indice_i,indice_j,r00,r01,r02,r10,r11,r12,r20,r21,r22,p_x,p_y,p_z,"
                "parejas,encajan,desacuerdo_m,desacuerdo_grados,vio_m,vio_grados\n")
LINEAS_DE_TABLA = 12


def crear_emparejador(nombre, cfg):
    """El emparejador pedido: "lightglue" (necesita PyTorch) u "orb" (sin redes).

    Los import van aquí dentro para que con orb no haga falta tener PyTorch.
    """
    if nombre == "orb":
        from vio.emparejador_orb import EmparejadorOrb

        return EmparejadorOrb(cfg["orb"])
    if nombre == "lightglue":
        from vio.emparejador import EmparejadorLightGlue

        return EmparejadorLightGlue(cargar_config("lightglue"))
    raise ValueError(f"emparejador desconocido: {nombre}")


def guardar_bucles(ruta, bucles):
    """bucles.csv: lo que necesita la etapa 29 de cada bucle."""
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(CABECERA_CSV)
        for b in bucles:
            numeros = list(b["R"].ravel()) + list(b["p"])
            calidad = [b["desacuerdo_m"], b["desacuerdo_grados"], b["vio_m"], b["vio_grados"]]
            f.write(f"{b['i']},{b['j']},{b['indice_i']},{b['indice_j']},"
                    + ",".join(f"{x:.9f}" for x in numeros) + f",{b['parejas']},{b['encajan']},"
                    + ",".join(f"{x:.6f}" for x in calidad) + "\n")


def leer_bucles(ruta):
    """Inversa de guardar_bucles. Devuelve: lista de dict con i, j, indice_i, indice_j, R (3, 3),
    p (3,), parejas, encajan, desacuerdo_m, desacuerdo_grados, vio_m y vio_grados."""
    bucles = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("#") or not linea.strip():
                continue
            c = linea.strip().split(",")
            numeros = [float(x) for x in c[4:16]]
            bucles.append({"i": int(c[0]), "j": int(c[1]), "indice_i": int(c[2]), "indice_j": int(c[3]),
                           "R": np.array(numeros[:9]).reshape(3, 3), "p": np.array(numeros[9:]),
                           "parejas": int(c[16]), "encajan": int(c[17]),
                           "desacuerdo_m": float(c[18]), "desacuerdo_grados": float(c[19]),
                           "vio_m": float(c[20]), "vio_grados": float(c[21])})
    return bucles


def verdad_de_la_sesion(carpeta, vistas, cfg_ext):
    """Las poses verdaderas de la cámara en cada vista, o None si la sesión no es del simulador.

    cfg_ext: config/camara_imu.yaml (la extrínseca con la que se hizo la grabación).
    """
    if not (Path(carpeta) / "verdad.csv").exists():
        return None
    t_ver, p_ver, R_ver = leer_poses(Path(carpeta) / "verdad.csv")
    instantes = [vista["kf"]["t_ns"] for vista in vistas]
    return bu.poses_verdaderas_de_camara(instantes, t_ver, p_ver, R_ver, np.array(cfg_ext["R_i_c"], dtype=float),
                                         np.array(cfg_ext["p_i_c_m"], dtype=float))


def informe(bucles, recuentos, errores):
    """El texto del informe. errores: de bucles.errores_de_bucles, o None."""
    lineas = [f"Candidatos: {recuentos['candidatos']}. Bucles: {recuentos['bucles']}."
              f" Descartados: {recuentos['pocas parejas']} por pocas parejas, {recuentos['sin pose']} sin pose,"
              f" {recuentos['no coinciden los dos sentidos']} porque no coinciden los dos sentidos"]
    if not bucles:
        return "\n".join(lineas)
    cabecera = f"{'kf i':>5s} {'kf j':>5s} {'parejas':>7s} {'encajan':>7s} {'entre sentidos':>16s} {'frente al VIO':>16s}"
    if errores is not None:
        cabecera += f" {'error de verdad':>16s}"
    lineas.append(cabecera)
    for k, b in enumerate(bucles[:LINEAS_DE_TABLA]):
        linea = (f"{b['indice_i']:5d} {b['indice_j']:5d} {b['parejas']:7d} {b['encajan']:7d}"
                 f" {100 * b['desacuerdo_m']:6.1f} cm {b['desacuerdo_grados']:5.2f}°"
                 f" {100 * b['vio_m']:6.1f} cm {b['vio_grados']:5.2f}°")
        if errores is not None:
            linea += f" {100 * errores[k][0]:6.1f} cm {errores[k][1]:5.2f}°"
        lineas.append(linea)
    if len(bucles) > LINEAS_DE_TABLA:
        lineas.append(f"  ... y {len(bucles) - LINEAS_DE_TABLA} más (todos, en el CSV)")
    lineas.append("")
    vio_m = np.array([b["vio_m"] for b in bucles])
    vio_grados = np.array([b["vio_grados"] for b in bucles])
    lineas.append(f"Frente al VIO: diferencia mediana {100 * np.median(vio_m):.1f} cm y {np.median(vio_grados):.2f}°,"
                  f" máxima {100 * vio_m.max():.1f} cm y {vio_grados.max():.2f}°")
    if errores is not None:
        e = np.array(errores).reshape(-1, 2)
        lineas.append(f"Frente a la verdad: error mediano {100 * np.median(e[:, 0]):.1f} cm y {np.median(e[:, 1]):.2f}°,"
                      f" el 90 % a menos de {100 * np.percentile(e[:, 0], 90):.1f} cm,"
                      f" máximo {100 * e[:, 0].max():.1f} cm y {e[:, 1].max():.2f}°")
    return "\n".join(lineas)


def dibujar(ruta, vistas, bucles):
    """Planta (x, y) con la posición de cada keyframe, numerados por orden, y un segmento
    por bucle entre los dos keyframes que une."""
    camaras = np.array([vista["kf"]["p_w_c"] for vista in vistas])
    fig, ax = plt.subplots(figsize=(8, 8))
    for b in bucles:
        ax.plot(camaras[[b["i"], b["j"]], 0], camaras[[b["i"], b["j"]], 1], color="tab:green", lw=1.0, alpha=0.6)
    ax.plot(camaras[:, 0], camaras[:, 1], "-", color="0.6", lw=0.8)
    ax.scatter(camaras[:, 0], camaras[:, 1], c=np.arange(len(camaras)), cmap="viridis", s=25, zorder=3)
    for k in range(0, len(camaras), 5):
        ax.annotate(str(k), camaras[k, :2], textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title(f"{len(vistas)} keyframes (color: orden) y {len(bucles)} bucles (verde)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    cfg = cargar_config("bucles")
    parser = argparse.ArgumentParser(description="Bucles de una sesión del VIO")
    parser.add_argument("carpeta", help="carpeta de la sesión (la que tiene mapa.paquetes)")
    parser.add_argument("--archivo", default=cargar_config("vio")["mapeo"]["archivo"],
                        help="nombre del archivo de paquetes dentro de la carpeta")
    parser.add_argument("--con-verdad", action="store_true",
                        help="simulador: usar la profundidad verdadera en lugar de la de la etapa 25")
    parser.add_argument("--emparejador", choices=("lightglue", "orb"), default=cfg["emparejador"],
                        help="lightglue: SuperPoint + LightGlue (PC); orb: clásico, sin redes")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)
    cfg_ext = cargar_config("camara_imu")

    mapa = leer_mapa(carpeta / args.archivo)
    print(mapa.resumen())
    vistas = cargar_vistas(mapa.keyframes, carpeta, args.con_verdad, cfg_ext["t_d_ns"])
    if not vistas:
        print("Ningún keyframe tiene profundidad. Sin --con-verdad hace falta haber ejecutado antes"
              " herramientas.ver_profundidad; con --con-verdad, una grabación del simulador con textura.")
        return
    emparejador = crear_emparejador(args.emparejador, cfg)
    print(f"{len(vistas)} keyframes con profundidad. Emparejador: {args.emparejador} ({emparejador.dispositivo})")
    bucles, recuentos = bu.buscar_bucles(vistas, emparejador, cfg)

    errores = None
    poses_verdad = verdad_de_la_sesion(carpeta, vistas, cfg_ext)
    if poses_verdad is not None:
        errores = bu.errores_de_bucles(bucles, poses_verdad)
    print(informe(bucles, recuentos, errores))

    ruta_csv = carpeta / nombre_de_salida("bucles", "csv", args.con_verdad)
    ruta_png = carpeta / nombre_de_salida("bucles", "png", args.con_verdad)
    guardar_bucles(ruta_csv, bucles)
    dibujar(ruta_png, vistas, bucles)
    print(f"Bucles: {ruta_csv}\nFigura: {ruta_png}")
    if bucles:
        mejor = max(bucles, key=lambda b: b["encajan"])
        ruta_jpg = carpeta / nombre_de_salida("bucle_mejor", "jpg", args.con_verdad)
        imagen = dibujar_parejas(vistas[mejor["i"]]["gris"], vistas[mejor["j"]]["gris"], mejor["uv_i"], mejor["uv_j"])
        cv2.imwrite(str(ruta_jpg), imagen)
        print(f"Parejas del bucle {mejor['indice_i']}-{mejor['indice_j']}: {ruta_jpg}")


if __name__ == "__main__":
    main()
