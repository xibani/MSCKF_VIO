"""
herramientas/analizar_imu.py — Regularidad y señales de una grabación de IMU.

Uso, desde ~/svo-pi:
    python -m herramientas.analizar_imu datos/imu_20261030_183000
Imprime las métricas y guarda analisis_imu.png dentro de la carpeta de la grabación.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")     # sin pantalla: dibuja directamente a un archivo
import matplotlib.pyplot as plt
import numpy as np

from captura import euroc
from captura.verificar import analizar_intervalos
from vio.config import cargar_config


def main():
    parser = argparse.ArgumentParser(description="Analiza una grabación de IMU")
    parser.add_argument("carpeta", help="carpeta de la grabación (la que contiene mav0/)")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)

    cfg_imu = cargar_config("imu")
    cfg_ver = cargar_config("verificacion")

    # --- 1. Cargar y analizar ---
    t_ns, w, a = euroc.leer_imu(euroc.ruta_imu(carpeta))
    r = analizar_intervalos(t_ns, cfg_imu["frecuencia_hz"], cfg_ver["factor_hueco"])

    print(f"muestras:          {r['n']}")
    print(f"duración:          {r['duracion_s']:.2f} s")
    print(f"frecuencia real:   {r['frecuencia_hz']:.3f} Hz (nominal {cfg_imu['frecuencia_hz']})")
    print(f"Δt medio:          {r['dt_medio_ms']:.3f} ms")
    print(f"jitter (σ de Δt):  {r['jitter_ms']:.3f} ms")
    print(f"Δt mín / máx:      {r['dt_min_ms']:.3f} / {r['dt_max_ms']:.3f} ms")
    print(f"huecos:            {r['huecos']}")
    print(f"perdidas:          {r['perdidas']} ({r['perdidas_pct']:.3f} %)")
    print(f"no crecientes:     {r['no_crecientes']}")

    # --- 2. Criterio de la etapa 1 ---
    crit = cfg_ver["imu"]
    ok_perdidas = r["perdidas_pct"] <= crit["perdidas_max_pct"]
    ok_jitter = r["jitter_ms"] <= crit["jitter_max_ms"]
    ok_orden = r["no_crecientes"] == 0
    print(f"\n[{'OK' if ok_perdidas else 'NO'}] perdidas ≤ {crit['perdidas_max_pct']} %")
    print(f"[{'OK' if ok_jitter else 'NO'}] jitter ≤ {crit['jitter_max_ms']} ms")
    print(f"[{'OK' if ok_orden else 'NO'}] marcas siempre crecientes")

    # --- 3. Figura: histograma de Δt y señales ---
    t_rel_s = (t_ns - t_ns[0]) / 1e9
    fig, ejes = plt.subplots(3, 1, figsize=(11, 10))

    ejes[0].hist(r["dt_s"] * 1e3, bins=100)
    ejes[0].set_yscale("log")      # escala log: los casos raros también se ven
    ejes[0].set_xlabel("Δt [ms]")
    ejes[0].set_title("Intervalos entre muestras")

    for i, eje in enumerate("xyz"):
        ejes[1].plot(t_rel_s, np.degrees(w[:, i]), lw=0.6, label=eje)
        ejes[2].plot(t_rel_s, a[:, i], lw=0.6, label=eje)
    ejes[1].set_ylabel("°/s")
    ejes[1].set_title("Giroscopio")
    ejes[2].set_ylabel("m/s²")
    ejes[2].set_title("Acelerómetro")
    ejes[2].set_xlabel("tiempo [s]")
    ejes[1].legend()
    ejes[2].legend()

    fig.tight_layout()
    ruta_png = carpeta / "analisis_imu.png"
    fig.savefig(ruta_png, dpi=100)
    print(f"\nFigura: {ruta_png}")


if __name__ == "__main__":
    main()