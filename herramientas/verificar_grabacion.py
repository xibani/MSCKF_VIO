"""
herramientas/verificar_grabacion.py — Comprueba una grabación de herramientas/grabar.py.

Uso, desde ~/svo-pi:
    python -m herramientas.verificar_grabacion datos/20261102_183000
Imprime las métricas, el criterio de la etapa 3 y guarda verificacion.png en la carpeta.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from captura import euroc
from captura.verificar import (analizar_cobertura, analizar_intervalos, analizar_latencia,
                               analizar_reposo, variacion_pct)
from vio.config import cargar_config


def imprimir_intervalos(titulo, r, f_nominal):
    print(f"\n=== {titulo} ===")
    print(f"  muestras:          {r['n']}")
    print(f"  duración:          {r['duracion_s']:.2f} s")
    print(f"  frecuencia real:   {r['frecuencia_hz']:.3f} Hz (nominal {f_nominal})")
    print(f"  jitter (σ de Δt):  {r['jitter_ms']:.3f} ms")
    print(f"  Δt mín / máx:      {r['dt_min_ms']:.3f} / {r['dt_max_ms']:.3f} ms")
    print(f"  perdidas:          {r['perdidas']} ({r['perdidas_pct']:.3f} %)")


def main():
    parser = argparse.ArgumentParser(description="Verifica una grabación IMU + cámara")
    parser.add_argument("carpeta", help="carpeta de la grabación (la que contiene mav0/)")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)

    cfg_imu = cargar_config("imu")
    cfg_cam = cargar_config("camara")
    cfg_ver = cargar_config("verificacion")
    crit_g = cfg_ver["grabacion"]

    # --- 1. Cargar ---
    t_imu, w, a = euroc.leer_imu(euroc.ruta_imu(carpeta))
    t_cam, nombres = euroc.leer_cam(euroc.ruta_cam(carpeta))
    diag = euroc.leer_diag_cam(euroc.ruta_diag_cam(carpeta))
    with open(carpeta / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    carpeta_png = euroc.carpeta_imagenes(carpeta)
    faltan = [n for n in nombres if not (carpeta_png / n).exists()]

    # --- 2. Analizar ---
    factor = cfg_ver["factor_hueco"]
    r_imu = analizar_intervalos(t_imu, cfg_imu["frecuencia_hz"], factor)
    r_cam = analizar_intervalos(t_cam, cfg_cam["fps"], factor)
    lat = analizar_latencia(diag["t_sensor_ns"], diag["t_llegada_ns"])
    var_exp = variacion_pct(diag["exposicion_us"])
    cob = analizar_cobertura(t_imu, t_cam)
    rep = analizar_reposo(t_imu, w, a, crit_g["t_reposo_s"])

    imprimir_intervalos("IMU", r_imu, cfg_imu["frecuencia_hz"])
    imprimir_intervalos("CÁMARA", r_cam, cfg_cam["fps"])
    print(f"  PNG en disco:      {len(nombres) - len(faltan)} de {len(nombres)}")
    print(f"  latencia:          media {lat['media_ms']:.1f} ms "
          f"(mín {lat['min_ms']:.1f}, máx {lat['max_ms']:.1f})")
    print(f"  exposición:        variación {var_exp:.2f} %")
    print(f"  contadores:        {meta['contadores']}")

    print("\n=== SINCRONIZACIÓN ===")
    print(f"  la IMU empieza {cob['margen_ini_s']:.2f} s antes de la 1ª imagen "
          f"y acaba {cob['margen_fin_s']:.2f} s después de la última")

    print(f"\n=== REPOSO INICIAL (primeros {crit_g['t_reposo_s']:.1f} s, {rep['n']} muestras) ===")
    am, b, s = rep["a_media"], rep["sesgo_gir_dps"], rep["ruido_gir_dps"]
    print(f"  aceleración media: x={am[0]:+.3f} y={am[1]:+.3f} z={am[2]:+.3f} m/s²"
          f"  (|a| = {rep['norma_a']:.3f}, eje hacia arriba {rep['eje_arriba']})")
    print(f"  sesgo giroscopio:  x={b[0]:+.3f} y={b[1]:+.3f} z={b[2]:+.3f} °/s")
    print(f"  σ giroscopio:      x={s[0]:.4f} y={s[1]:.4f} z={s[2]:.4f} °/s")

    # --- 3. Criterio de la etapa 3 ---
    crit_i, crit_c = cfg_ver["imu"], cfg_ver["camara"]
    comprobaciones = [
        (f"duración de imágenes ≥ {crit_g['duracion_min_s']:.0f} s",
         r_cam["duracion_s"] >= crit_g["duracion_min_s"]),
        (f"imágenes perdidas ≤ {crit_c['perdidas_max']}",
         r_cam["perdidas"] + len(faltan) <= crit_c["perdidas_max"]),
        (f"IMU perdidas ≤ {crit_i['perdidas_max_pct']} %",
         r_imu["perdidas_pct"] <= crit_i["perdidas_max_pct"]),
        (f"jitter IMU ≤ {crit_i['jitter_max_ms']} ms", r_imu["jitter_ms"] <= crit_i["jitter_max_ms"]),
        ("marcas siempre crecientes", r_imu["no_crecientes"] == 0 and r_cam["no_crecientes"] == 0),
        ("mismo reloj (latencia de la cámara positiva y acotada)",
         lat["min_ms"] > 0 and lat["max_ms"] < crit_c["latencia_max_ms"]),
        (f"exposición constante (≤ {crit_c['exposicion_variacion_max_pct']} %)",
         var_exp <= crit_c["exposicion_variacion_max_pct"]),
        ("la IMU cubre todo el tramo de imágenes",
         cob["margen_ini_s"] > 0 and cob["margen_fin_s"] > 0),
        (f"reposo inicial (σ giroscopio ≤ {crit_g['ruido_gir_max_dps']} °/s)",
         rep["ruido_gir_dps"].max() <= crit_g["ruido_gir_max_dps"]),
    ]
    print("\n=== CRITERIO DE LA ETAPA 3 ===")
    for texto, ok in comprobaciones:
        print(f"  [{'OK' if ok else 'NO'}] {texto}")
    todo_ok = all(ok for _, ok in comprobaciones)
    print("  >>> ETAPA 3 SUPERADA" if todo_ok else "  >>> Etapa 3 NO superada")

    # --- 4. Figura ---
    t0 = t_imu[0]
    t_rel = (t_imu - t0) / 1e9
    ini_cam = (t_cam[0] - t0) / 1e9
    fin_cam = (t_cam[-1] - t0) / 1e9
    fig, ejes = plt.subplots(2, 2, figsize=(13, 8))

    ejes[0, 0].hist(r_imu["dt_s"] * 1e3, bins=100)
    ejes[0, 0].set_title("Δt IMU [ms]")
    ejes[0, 0].set_yscale("log")
    ejes[0, 1].hist(r_cam["dt_s"] * 1e3, bins=50)
    ejes[0, 1].set_title("Δt cámara [ms]")
    ejes[0, 1].set_yscale("log")

    for i, eje in enumerate("xyz"):
        ejes[1, 0].plot(t_rel, np.degrees(w[:, i]), lw=0.6, label=eje)
        ejes[1, 1].plot(t_rel, a[:, i], lw=0.6, label=eje)
    for ej, titulo in [(ejes[1, 0], "Giroscopio [°/s]"), (ejes[1, 1], "Acelerómetro [m/s²]")]:
        ej.axvline(ini_cam, color="k", ls="--", lw=0.8)   # primera imagen
        ej.axvline(fin_cam, color="k", ls="--", lw=0.8)   # última imagen
        ej.set_title(titulo + " — discontinuas: primera y última imagen")
        ej.set_xlabel("s desde la primera muestra de IMU")
        ej.legend()

    fig.tight_layout()
    ruta_png = carpeta / "verificacion.png"
    fig.savefig(ruta_png, dpi=100)
    print(f"\nFigura: {ruta_png}")


if __name__ == "__main__":
    main()