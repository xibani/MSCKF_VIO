"""
herramientas/informe_lazo.py — Informe de una prueba de trayectoria con datos reales (etapa 16).

Uso, desde ~/proyectos/MSCKF_VIO:
    python -m herramientas.informe_lazo datos/vio_20261106_101500              # la última prueba de la carpeta
    python -m herramientas.informe_lazo datos/vio_20261106_101500/vio_trayectoria_2.csv
    python -m herramientas.informe_lazo <carpeta o csv> --distancia-real 5.0   # prueba de la recta medida
Imprime las métricas y guarda <nombre del csv>_informe.png al lado del CSV.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # sin ventana: la figura se guarda en un PNG
import matplotlib.pyplot as plt
import numpy as np

from vio import lazo
from vio.config import cargar_config
from vio.pipeline import leer_poses


def elegir_csv(ruta):
    """El CSV de la prueba: el que se da o, si es una carpeta, el más reciente de ella."""
    ruta = Path(ruta)
    if ruta.is_file():
        return ruta
    candidatos = sorted(ruta.glob("vio_trayectoria*.csv"), key=lambda r: r.stat().st_mtime)
    if not candidatos:
        raise FileNotFoundError(f"no hay vio_trayectoria*.csv en {ruta}")
    return candidatos[-1]


def dibujar(ruta_png, t_s, p, m, titulo):
    """Planta, alzado y distancia al punto de salida a lo largo del tiempo."""
    figura, ejes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (i, j, nombre) in zip(ejes[:2], [(0, 1, "Planta X–Y"), (0, 2, "Alzado X–Z")]):
        ax.plot(p[:, i], p[:, j], color="tab:blue", lw=1.5, label="trayectoria")
        ax.plot([p[-1, i], p[0, i]], [p[-1, j], p[0, j]], "--", color="tab:orange", label="cierre")
        ax.plot(p[0, i], p[0, j], "o", color="tab:green", ms=9, label="salida")
        ax.plot(p[-1, i], p[-1, j], "o", color="tab:red", ms=7, label="llegada")
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.set_title(nombre)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    ax = ejes[2]
    ax.plot(t_s, np.linalg.norm(p - p[0], axis=1), color="tab:blue")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("distancia al punto de salida [m]")
    ax.set_title("Alejamiento (debe volver a 0 al cerrar el lazo)")
    ax.grid(True, alpha=0.3)
    texto = (f"recorrido {m['recorrido_m']:.2f} m\n"
             f"cierre {m['cierre_m'] * 100:.1f} cm"
             + (f" ({m['cierre_pct']:.2f} %)" if m["cierre_pct"] is not None else ""))
    ax.text(0.98, 0.97, texto, transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.8})
    figura.suptitle(titulo)
    figura.tight_layout()
    figura.savefig(ruta_png, dpi=110)
    plt.close(figura)


def informe(ruta, cfg_prueba, distancia_real_m=None):
    """Métricas de una prueba y su figura. Devuelve: dict de vio/lazo.metricas_lazo (+ escala)."""
    ruta_csv = elegir_csv(ruta)
    t_ns, p, _ = leer_poses(ruta_csv)
    if len(t_ns) < 2:
        raise ValueError(f"{ruta_csv} no tiene poses: ¿llegó a arrancar el VIO?")
    t_s = (t_ns - t_ns[0]) / 1e9
    m = lazo.metricas_lazo(t_s, p, cfg_prueba["paso_min_m"], cfg_prueba["recorrido_min_m"])

    print(f"Prueba: {ruta_csv}")
    print(f"  duración:            {m['duracion_s']:.1f} s ({len(t_ns)} poses)")
    print(f"  recorrido:           {m['recorrido_m']:.2f} m")
    print(f"  máx. alejamiento:    {m['alejamiento_max_m']:.2f} m")
    d = m["deriva_m"]
    print(f"  final - inicio:      ({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}) m")
    if m["cierre_pct"] is None:
        print(f"  cierre:              {m['cierre_m'] * 100:.1f} cm (recorrido < "
              f"{cfg_prueba['recorrido_min_m']} m: el % no significa nada)")
    else:
        print(f"  error de cierre:     {m['cierre_m'] * 100:.1f} cm = {m['cierre_pct']:.2f} % del recorrido")
    titulo = f"{ruta_csv.parent.name}/{ruta_csv.name}"
    if distancia_real_m is not None:
        estimada, error = lazo.error_escala(p, distancia_real_m)
        m["distancia_estimada_m"], m["error_escala_pct"] = estimada, error
        print(f"  recta medida:        {distancia_real_m:.3f} m reales, {estimada:.3f} m estimados"
              f" -> error de escala {error:+.2f} %")
        titulo += f" — recta de {distancia_real_m:.2f} m: error de escala {error:+.1f} %"
    ruta_png = ruta_csv.with_name(ruta_csv.stem + "_informe.png")
    dibujar(ruta_png, t_s, p, m, titulo)
    print(f"Figura: {ruta_png}")
    return m


def main():
    parser = argparse.ArgumentParser(description="Informe de una prueba de trayectoria")
    parser.add_argument("ruta", help="carpeta datos/vio_... (usa la prueba más reciente) o un CSV")
    parser.add_argument("--distancia-real", type=float, default=None,
                        help="prueba de la recta: distancia medida con cinta entre las dos marcas [m]")
    args = parser.parse_args()
    informe(args.ruta, cargar_config("vio")["prueba"], args.distancia_real)


if __name__ == "__main__":
    main()