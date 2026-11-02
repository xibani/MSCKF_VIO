"""
herramientas/analizar_camara.py — Regularidad, latencia y exposición de una grabación de cámara.

Uso, desde ~/svo-pi:
    python -m herramientas.analizar_camara datos/cam_20261102_183000
Imprime las métricas y guarda analisis_camara.png dentro de la carpeta.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from captura import euroc
from captura.verificar import analizar_intervalos, analizar_latencia, variacion_pct
from vio.config import cargar_config


def main():
    parser = argparse.ArgumentParser(description="Analiza una grabación de cámara")
    parser.add_argument("carpeta", help="carpeta de la grabación (la que contiene mav0/)")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)

    cfg_cam = cargar_config("camara")
    cfg_ver = cargar_config("verificacion")

    # --- 1. Cargar ---
    t_ns, nombres = euroc.leer_cam(euroc.ruta_cam(carpeta))
    diag = euroc.leer_diag_cam(euroc.ruta_diag_cam(carpeta))
    carpeta_png = euroc.carpeta_imagenes(carpeta)
    faltan = [n for n in nombres if not (carpeta_png / n).exists()]

    # --- 2. Analizar ---
    r = analizar_intervalos(t_ns, cfg_cam["fps"], cfg_ver["factor_hueco"])
    lat = analizar_latencia(diag["t_sensor_ns"], diag["t_llegada_ns"])
    var_exp = variacion_pct(diag["exposicion_us"])

    print(f"imágenes:          {r['n']} ({len(faltan)} sin PNG en disco)")
    print(f"duración:          {r['duracion_s']:.2f} s")
    print(f"frecuencia real:   {r['frecuencia_hz']:.3f} Hz (nominal {cfg_cam['fps']})")
    print(f"Δt medio:          {r['dt_medio_ms']:.3f} ms")
    print(f"jitter (σ de Δt):  {r['jitter_ms']:.3f} ms")
    print(f"Δt mín / máx:      {r['dt_min_ms']:.3f} / {r['dt_max_ms']:.3f} ms")
    print(f"perdidas:          {r['perdidas']}")
    print(f"latencia:          media {lat['media_ms']:.1f} ms "
          f"(mín {lat['min_ms']:.1f}, máx {lat['max_ms']:.1f})")
    print(f"exposición:        {diag['exposicion_us'].min():.0f} - {diag['exposicion_us'].max():.0f} us"
          f" (variación {var_exp:.2f} %)")

    # --- 3. Criterio de la etapa 2 ---
    crit = cfg_ver["camara"]
    comprobaciones = [
        (f"imágenes perdidas ≤ {crit['perdidas_max']}",
         r["perdidas"] + len(faltan) <= crit["perdidas_max"]),
        ("marcas siempre crecientes", r["no_crecientes"] == 0),
        (f"latencia entre 0 y {crit['latencia_max_ms']:.0f} ms (mismo reloj)",
         lat["min_ms"] > 0 and lat["max_ms"] < crit["latencia_max_ms"]),
        (f"exposición constante (variación ≤ {crit['exposicion_variacion_max_pct']} %)",
         var_exp <= crit["exposicion_variacion_max_pct"]),
    ]
    print()
    for texto, ok in comprobaciones:
        print(f"[{'OK' if ok else 'NO'}] {texto}")

    # --- 4. Figura ---
    t_rel_s = (diag["t_sensor_ns"] - diag["t_sensor_ns"][0]) / 1e9
    latencia_ms = (diag["t_llegada_ns"] - diag["t_sensor_ns"]) / 1e6
    fig, ejes = plt.subplots(2, 1, figsize=(11, 7))

    ejes[0].hist(r["dt_s"] * 1e3, bins=50)
    ejes[0].set_yscale("log")
    ejes[0].set_xlabel("Δt [ms]")
    ejes[0].set_title("Intervalos entre imágenes")

    ejes[1].plot(t_rel_s, latencia_ms, ".", ms=3)
    ejes[1].set_xlabel("tiempo [s]")
    ejes[1].set_ylabel("ms")
    ejes[1].set_title("Latencia: llegada al programa - SensorTimestamp")

    fig.tight_layout()
    ruta_png = carpeta / "analisis_camara.png"
    fig.savefig(ruta_png, dpi=100)
    print(f"\nFigura: {ruta_png}")


if __name__ == "__main__":
    main()