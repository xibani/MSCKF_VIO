"""
herramientas/medir_mapa.py — Etapa 24: mide el mapa disperso de una sesión del VIO.

Uso, desde la raíz del proyecto:
    python -m herramientas.medir_mapa datos/sim_textura                      # simulador
    python -m herramientas.medir_mapa datos/20261116_101500 --pared-m 1.20   # con la placa
    python -m herramientas.medir_mapa datos/vio_XXXX --archivo mapa_2.paquetes
La carpeta es la de la sesión: la que tiene el mapa.paquetes que escribe apps/pi_vio (la de
la grabación, si se usó --grabacion).

De cada keyframe de mapeo dice cuántos puntos con profundidad trae y a qué distancia están.
Y lo compara con lo que haya:
    - Simulador (la grabación trae mav0/prof0): con la profundidad verdadera de cada punto.
      "escala" 1,05 quiere decir que el mapa sale un 5 % más grande que la realidad.
    - Con la placa y --pared-m D: con la distancia D [m], medida con cinta, de la cámara a
      una pared plana a la que se ha estado mirando de frente durante la grabación.
Guarda medidas_mapa.png en la carpeta.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from captura import euroc
from vio import medidas_mapa as mm
from vio import protocolo
from vio.config import cargar_config
from vio.reconstruccion import Reconstructor


def leer_mapa(ruta):
    """El mapa de un archivo de paquetes. Devuelve: Reconstructor (keyframes y mapa disperso)."""
    mapa = Reconstructor()
    for datos in protocolo.leer_archivo_paquetes(ruta):
        mapa.procesar(datos)
    return mapa


def medir_keyframes(keyframes, carpeta, t_d_ns):
    """Las medidas de cada keyframe.

    carpeta: la de la grabación, para buscar la profundidad verdadera (solo en el simulador).
    t_d_ns: desfase cámara-IMU. El keyframe lleva el instante del filtro (marca + t_d) y el
        archivo de profundidad se llama como la marca de la imagen: marca = instante - t_d.
    Devuelve: lista de dict, uno por keyframe, con t_ns, indice, z (N,), z_verdad (N,) o None,
    y los resúmenes "distancias" y "verdad" (este, None si no hay verdad).
    """
    medidas = []
    for kf in keyframes:
        medida = {"t_ns": kf["t_ns"], "indice": kf["indice"], "z": kf["z"], "z_verdad": None,
                  "distancias": mm.resumen_profundidades(kf["z"]), "verdad": None}
        profundidad = euroc.leer_profundidad(carpeta, kf["t_ns"] - t_d_ns)
        if profundidad is not None:
            medida["z_verdad"] = mm.muestrear(profundidad, mm.pixeles_de_keyframe(kf))
            medida["verdad"] = mm.comparar_con_verdad(kf["z"], medida["z_verdad"])
        medidas.append(medida)
    return medidas


def informe(medidas, pared_m):
    """El texto del informe: una línea por keyframe y el resumen de todos juntos."""
    hay_verdad = any(m["verdad"] is not None for m in medidas)
    cabecera = f"{'keyframe':>8s} {'t [s]':>7s} {'puntos':>6s} {'z mediana':>10s} {'z 10-90 %':>15s}"
    if hay_verdad:
        cabecera += f" {'escala':>7s} {'dispersión':>10s} {'dentro 10 %':>11s}"
    if pared_m is not None:
        cabecera += f" {'error pared':>11s}"
    lineas = [cabecera]
    t0 = medidas[0]["t_ns"]
    for m in medidas:
        d = m["distancias"]
        linea = (f"{m['indice']:8d} {(m['t_ns'] - t0) / 1e9:7.2f} {d['n']:6d} {d['mediana']:8.2f} m"
                 f" {d['p10']:6.2f} - {d['p90']:5.2f} m")
        if hay_verdad:
            v = m["verdad"]
            linea += (f" {v['escala']:7.3f} {100 * v['dispersion']:8.1f} % {100 * v['dentro_10']:9.0f} %"
                      if v is not None else f" {'-':>7s} {'-':>10s} {'-':>11s}")
        if pared_m is not None:
            linea += f" {mm.error_frente_a_distancia(m['z'], pared_m):+9.1f} %"
        lineas.append(linea)

    # Todos los keyframes juntos
    z = np.concatenate([m["z"] for m in medidas])
    lineas.append("")
    lineas.append(f"{len(medidas)} keyframes, {len(z)} profundidades"
                  f" ({len(z) / len(medidas):.0f} por keyframe de media)")
    if hay_verdad:
        z_con = np.concatenate([m["z"] for m in medidas if m["z_verdad"] is not None])
        z_verdad = np.concatenate([m["z_verdad"] for m in medidas if m["z_verdad"] is not None])
        v = mm.comparar_con_verdad(z_con, z_verdad)
        lineas.append(f"Frente a la verdad ({v['n']} puntos): escala {v['escala']:.3f}"
                      f" ({100 * (v['escala'] - 1):+.1f} %), dispersión {100 * v['dispersion']:.1f} %,"
                      f" {100 * v['dentro_10']:.0f} % de los puntos a menos del 10 %")
    if pared_m is not None:
        errores = np.array([mm.error_frente_a_distancia(m["z"], pared_m) for m in medidas])
        errores = errores[np.isfinite(errores)]
        lineas.append(f"Frente a la pared a {pared_m:.2f} m: error mediano {np.median(errores):+.1f} %"
                      f" (de {errores.min():+.1f} % a {errores.max():+.1f} % según el keyframe)")
    return "\n".join(lineas)


def dibujar(ruta, medidas, pared_m):
    """Figura: las profundidades de cada keyframe y, si hay verdad, estimada frente a verdadera."""
    hay_verdad = any(m["z_verdad"] is not None for m in medidas)
    fig, ejes = plt.subplots(1, 2 if hay_verdad else 1, figsize=(13 if hay_verdad else 8, 5), squeeze=False)
    # 1. Por keyframe: mediana con su intervalo 10-90 %
    ax = ejes[0, 0]
    t = np.array([(m["t_ns"] - medidas[0]["t_ns"]) / 1e9 for m in medidas])
    mediana = np.array([m["distancias"]["mediana"] for m in medidas])
    p10 = np.array([m["distancias"]["p10"] for m in medidas])
    p90 = np.array([m["distancias"]["p90"] for m in medidas])
    ax.errorbar(t, mediana, yerr=[mediana - p10, p90 - mediana], fmt="o", capsize=3, label="mediana y 10-90 %")
    if pared_m is not None:
        ax.axhline(pared_m, color="r", lw=1, label=f"pared medida: {pared_m:.2f} m")
    ax.set_xlabel("t desde el primer keyframe [s]")
    ax.set_ylabel("profundidad de los puntos [m]")
    ax.set_title("Profundidades de cada keyframe")
    ax.grid(alpha=0.3)
    ax.legend()
    # 2. Con verdad: cada punto, estimada frente a verdadera
    if hay_verdad:
        ax = ejes[0, 1]
        z = np.concatenate([m["z"] for m in medidas if m["z_verdad"] is not None])
        z_verdad = np.concatenate([m["z_verdad"] for m in medidas if m["z_verdad"] is not None])
        ax.plot(z_verdad, z, ".", ms=3)
        tope = float(np.nanmax(z_verdad)) * 1.05
        ax.plot([0, tope], [0, tope], "k", lw=0.8, label="sin error")
        ax.plot([0, tope], [0, 1.1 * tope], "k:", lw=0.8, label="±10 %")
        ax.plot([0, tope], [0, 0.9 * tope], "k:", lw=0.8)
        ax.set_xlim(0, tope)
        ax.set_ylim(0, tope * 1.2)
        ax.set_xlabel("profundidad verdadera [m]")
        ax.set_ylabel("profundidad del VIO [m]")
        ax.set_title("Punto a punto")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Mide el mapa disperso de una sesión del VIO")
    parser.add_argument("carpeta", help="carpeta de la sesión (la que tiene mapa.paquetes)")
    parser.add_argument("--archivo", default=cargar_config("vio")["mapeo"]["archivo"],
                        help="nombre del archivo de paquetes dentro de la carpeta")
    parser.add_argument("--pared-m", type=float, default=None,
                        help="distancia medida [m] de la cámara a la pared que se ha mirado de frente")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)

    mapa = leer_mapa(carpeta / args.archivo)
    print(mapa.resumen())
    if not mapa.keyframes:
        print("El mapa no tiene ningún keyframe: no hay nada que medir.")
        return
    medidas = medir_keyframes(mapa.keyframes, carpeta, cargar_config("camara_imu")["t_d_ns"])
    print(informe(medidas, args.pared_m))
    dibujar(carpeta / "medidas_mapa.png", medidas, args.pared_m)
    print(f"Figura: {carpeta / 'medidas_mapa.png'}")


if __name__ == "__main__":
    main()
