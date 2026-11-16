"""
herramientas/cerrar_bucles.py — Etapa 29: corrige los keyframes con los bucles y rehace la nube.

Uso, desde la raíz del proyecto:
    python -m herramientas.cerrar_bucles datos/sim_textura30                  # con bucles.csv y la profundidad de la etapa 25
    python -m herramientas.cerrar_bucles datos/sim_textura30 --con-verdad     # con bucles_verdad.csv y la profundidad verdadera
    python -m herramientas.cerrar_bucles datos/sim_textura30 --con-verdad --deriva-giro 0.3 --deriva-escala 1.03
    python -m herramientas.cerrar_bucles datos/20261116_101500 --archivo mapa_2.paquetes
La carpeta es la de la sesión. Necesita el CSV de bucles que escribe herramientas.buscar_bucles
(etapa 28), hecho con las mismas opciones (--con-verdad o no).

--deriva-giro G y --deriva-escala E estropean a propósito las poses del VIO antes de
corregirlas (G grados de giro de más y un factor E en el avance, en cada paso entre
keyframes): el VIO del simulador casi no deriva, y así se ve qué arregla el grafo.

Escribe en la carpeta:
    keyframes_corregidos.csv      la pose corregida de cada keyframe y cuánto se ha movido
    grafo.png                     planta con las poses de antes y de después, y los bucles
    nube_densa_corregida.ply/png  la nube densa rehecha con las poses corregidas (etapa 26)
(con _verdad en el nombre si se usa --con-verdad). Con una grabación del simulador compara
además las poses y las nubes, de antes y de después, con la verdad.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from herramientas import nube_densa
from herramientas.buscar_bucles import leer_bucles, verdad_de_la_sesion
from herramientas.medir_mapa import leer_mapa
from vio import densificar as den
from vio import grafo as gr
from vio import so3
from vio.config import cargar_config
from vio.reconstruccion import guardar_ply

CABECERA_CSV = "#indice,t_ns,p_x,p_y,p_z,r00,r01,r02,r10,r11,r12,r20,r21,r22,movido_m,movido_grados\n"


def vistas_con_poses(vistas, poses):
    """Las mismas vistas con otra pose en cada keyframe (los demás datos, los mismos)."""
    nuevas = []
    for vista, (R, p) in zip(vistas, poses):
        kf = dict(vista["kf"], R_w_c=R, p_w_c=p)
        nuevas.append({"kf": kf, "z": vista["z"], "gris": vista["gris"]})
    return nuevas


def cambios_de_poses(antes, despues):
    """Cuánto se ha movido cada pose. Devuelve: (N, 2): [metros, grados]."""
    filas = []
    for (R_a, p_a), (R_d, p_d) in zip(antes, despues):
        angulo = np.degrees(so3.angulo_entre(R_a, R_d))
        filas.append([np.linalg.norm(p_d - p_a), angulo])
    return np.array(filas).reshape(-1, 2)


def guardar_keyframes(ruta, vistas, poses, movido):
    """keyframes_corregidos.csv: una fila por keyframe con su pose corregida (R_w_c, p_w_c)."""
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(CABECERA_CSV)
        for vista, (R, p), (metros, grados) in zip(vistas, poses, movido):
            numeros = list(p) + list(R.ravel())
            f.write(f"{vista['kf']['indice']},{vista['kf']['t_ns']}," + ",".join(f"{x:.9f}" for x in numeros)
                    + f",{metros:.6f},{grados:.6f}\n")


def linea_de_errores(nombre, errores):
    """Una línea del informe con errores (N, 2) en [m, grados]: mediana y máximo."""
    return (f"  {nombre:<8s} mediana {100 * np.median(errores[:, 0]):5.1f} cm y {np.median(errores[:, 1]):5.2f}°,"
            f" máximo {100 * errores[:, 0].max():5.1f} cm y {errores[:, 1].max():5.2f}°")


def informe(n_nudos, aristas, historia, bucles_antes, bucles_despues, movido, poses_antes, poses_despues,
            nube_antes, nube_despues):
    """El texto del informe. Los cuatro últimos (errores de poses (N, 2) y evaluaciones de
    la nube de densificar.evaluar_nube, antes y después) son None fuera del simulador."""
    n_bucles = sum(a["bucle"] for a in aristas)
    lineas = [f"Grafo: {n_nudos} keyframes, {len(aristas) - n_bucles} aristas de odometría y {n_bucles} de bucle",
              f"Optimización: {len(historia) - 1} pasos, coste {historia[0]:.1f} -> {historia[-1]:.1f}"]
    if len(bucles_antes):
        lineas.append("Lo que se apartan los keyframes de lo que miden los bucles:")
        lineas.append(linea_de_errores("antes", bucles_antes))
        lineas.append(linea_de_errores("después", bucles_despues))
    lineas.append(f"Lo que se han movido los keyframes: mediana {100 * np.median(movido[:, 0]):.1f} cm,"
                  f" máximo {100 * movido[:, 0].max():.1f} cm y {movido[:, 1].max():.2f}°")
    if poses_antes is not None:
        lineas.append("Error de las poses frente a la verdad:")
        lineas.append(linea_de_errores("antes", poses_antes))
        lineas.append(linea_de_errores("después", poses_despues))
    if nube_antes is not None:
        lineas.append("Nube densa frente a la habitación de verdad:")
        for nombre, e in (("antes", nube_antes), ("después", nube_despues)):
            lineas.append(f"  {nombre:<8s} {e['n']:6d} puntos, error mediano {100 * e['error_mediano']:5.1f} cm,"
                          f" el 90 % a menos de {100 * e['error_p90']:5.1f} cm, cubierto {100 * e['cubierto']:.0f} %")
    return "\n".join(lineas)


def dibujar(ruta, antes, despues, aristas, verdad):
    """Planta (x, y): las posiciones de antes (gris) y de después (azul), los bucles (verde,
    entre las posiciones de después) y, si se sabe, la verdad (negro a trazos).

    antes, despues: listas de (R, p). verdad: (N, 3) posiciones verdaderas ya en el mundo del
    VIO, o None.
    """
    p_antes = np.array([p for _, p in antes])
    p_despues = np.array([p for _, p in despues])
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for a in aristas:
        if a["bucle"]:
            ax.plot(p_despues[[a["i"], a["j"]], 0], p_despues[[a["i"], a["j"]], 1], color="tab:green", lw=0.8, alpha=0.5)
    if verdad is not None:
        ax.plot(verdad[:, 0], verdad[:, 1], "k--", lw=1.2, label="verdad")
    ax.plot(p_antes[:, 0], p_antes[:, 1], ".-", color="0.55", lw=1.0, ms=5, label="antes")
    ax.plot(p_despues[:, 0], p_despues[:, 1], ".-", color="tab:blue", lw=1.2, ms=5, label="después")
    ax.set_title(f"{len(antes)} keyframes y {sum(a['bucle'] for a in aristas)} bucles (verde)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Corrige los keyframes con los bucles y rehace la nube")
    parser.add_argument("carpeta", help="carpeta de la sesión (la que tiene mapa.paquetes y el CSV de bucles)")
    parser.add_argument("--archivo", default=cargar_config("vio")["mapeo"]["archivo"],
                        help="nombre del archivo de paquetes dentro de la carpeta")
    parser.add_argument("--con-verdad", action="store_true",
                        help="simulador: profundidad verdadera y bucles_verdad.csv")
    parser.add_argument("--deriva-giro", type=float, default=0.0,
                        help="prueba: grados de giro de más en cada paso entre keyframes")
    parser.add_argument("--deriva-escala", type=float, default=1.0,
                        help="prueba: factor por el que se multiplica el avance de cada paso")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)
    cfg = cargar_config("grafo")
    cfg_densificar = cargar_config("densificar")
    cfg_ext = cargar_config("camara_imu")

    ruta_bucles = carpeta / nube_densa.nombre_de_salida("bucles", "csv", args.con_verdad)
    if not ruta_bucles.exists():
        print(f"No existe {ruta_bucles}: ejecuta antes herramientas.buscar_bucles"
              f"{' --con-verdad' if args.con_verdad else ''}.")
        return
    mapa = leer_mapa(carpeta / args.archivo)
    vistas = nube_densa.cargar_vistas(mapa.keyframes, carpeta, args.con_verdad, cfg_ext["t_d_ns"])
    bucles = leer_bucles(ruta_bucles)
    for b in bucles:
        if b["j"] >= len(vistas) or vistas[b["i"]]["kf"]["indice"] != b["indice_i"] \
                or vistas[b["j"]]["kf"]["indice"] != b["indice_j"]:
            print(f"{ruta_bucles.name} no corresponde a estos keyframes: vuelve a ejecutar herramientas.buscar_bucles.")
            return

    # 1. El grafo
    antes = [(vista["kf"]["R_w_c"], vista["kf"]["p_w_c"]) for vista in vistas]
    if args.deriva_giro != 0.0 or args.deriva_escala != 1.0:
        antes = gr.anadir_deriva(antes, args.deriva_giro, args.deriva_escala)
        print(f"PRUEBA: poses estropeadas a propósito ({args.deriva_giro}° y x{args.deriva_escala} por paso)")
    aristas = gr.aristas_de_odometria(antes, cfg) + gr.aristas_de_bucles(bucles, cfg)
    despues, historia = gr.optimizar(antes, aristas, cfg)
    movido = cambios_de_poses(antes, despues)

    # 2. La nube densa, con las poses de antes y con las de después
    nube_antes = den.densificar(vistas_con_poses(vistas, antes), cfg_densificar)
    nube_despues = den.densificar(vistas_con_poses(vistas, despues), cfg_densificar)

    # 3. En el simulador: comparar con la verdad
    errores_antes = errores_despues = evaluacion_antes = evaluacion_despues = verdad_en_vio = None
    transformacion = nube_densa.transformacion_de_la_sesion(carpeta)
    poses_verdad = verdad_de_la_sesion(carpeta, vistas, cfg_ext)
    if transformacion is not None and poses_verdad is not None:
        R, t = transformacion
        errores_antes = gr.errores_de_poses(antes, poses_verdad, R, t)
        errores_despues = gr.errores_de_poses(despues, poses_verdad, R, t)
        dimensiones = cargar_config("simulador")["vio"]["mundo"]["habitacion_m"]
        evaluacion_antes = den.evaluar_nube(nube_antes["p_w"] @ R.T + t, dimensiones, cfg_densificar["evaluacion"])
        evaluacion_despues = den.evaluar_nube(nube_despues["p_w"] @ R.T + t, dimensiones, cfg_densificar["evaluacion"])
        verdad_en_vio = (np.array([p for _, p in poses_verdad]) - t) @ R      # Rᵀ·(p - t)
    print(informe(len(antes), aristas, historia, gr.residuos_de_bucles(antes, aristas),
                  gr.residuos_de_bucles(despues, aristas), movido, errores_antes, errores_despues,
                  evaluacion_antes, evaluacion_despues))

    # 4. A disco
    ruta_csv = carpeta / nube_densa.nombre_de_salida("keyframes_corregidos", "csv", args.con_verdad)
    ruta_png = carpeta / nube_densa.nombre_de_salida("grafo", "png", args.con_verdad)
    ruta_ply = carpeta / nube_densa.nombre_de_salida("nube_densa_corregida", "ply", args.con_verdad)
    ruta_nube_png = carpeta / nube_densa.nombre_de_salida("nube_densa_corregida", "png", args.con_verdad)
    guardar_keyframes(ruta_csv, vistas, despues, movido)
    dibujar(ruta_png, antes, despues, aristas, verdad_en_vio)
    gris = nube_despues["gris"]
    guardar_ply(ruta_ply, nube_despues["p_w"], np.column_stack([gris, gris, gris]))
    nube_densa.dibujar(ruta_nube_png, nube_despues["p_w"], gris, [v["kf"] for v in vistas_con_poses(vistas, despues)])
    print(f"Keyframes: {ruta_csv}\nFigura: {ruta_png}\nNube: {ruta_ply}\nFigura de la nube: {ruta_nube_png}")


if __name__ == "__main__":
    main()
