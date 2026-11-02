"""
herramientas/posicion_offline.py — Posición integrando solo la IMU, sobre una grabación.

Uso, desde ~/svo-pi:
    python -m herramientas.posicion_offline datos/imu_20261102_183000
    python -m herramientas.posicion_offline datos/imu_20261102_183000 --g-medida
--g-medida usa como gravedad la |a| medida en reposo en lugar de 9,80665.
--calibrado corrige antes el acelerómetro con config/imu_calibracion.yaml (paso 7.2).
Guarda posicion.png en la carpeta de la grabación.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from captura import euroc
from vio import calibracion_imu as ci
from vio import inicializacion as ini
from vio.config import cargar_config
from vio.constantes import G, GRAVEDAD_W
from vio.estado import estado_inicial
from vio.propagacion import propagar_secuencia

INSTANTES_INFORME_S = [1, 2, 5, 10, 20, 30]     # segundos en los que se imprime la deriva


def main():
    parser = argparse.ArgumentParser(description="Posición solo con la IMU, sin conexión")
    parser.add_argument("carpeta", help="carpeta de la grabación (la que contiene mav0/)")
    parser.add_argument("--g-medida", action="store_true",
                        help="usar |a| del reposo como gravedad")
    parser.add_argument("--calibrado", action="store_true",
                        help="corregir el acelerómetro con config/imu_calibracion.yaml")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)
    cfg = cargar_config("orientacion")

    # --- 1. Cargar e inicializar con el reposo inicial ---
    t_ns, w, a = euroc.leer_imu(euroc.ruta_imu(carpeta))
    if args.calibrado:
        sesgo_a, escala_a = ci.cargar_calibracion_imu()
        a = ci.corregir_acelerometro(a, sesgo_a, escala_a)
        print(f"Acelerómetro corregido: sesgo {np.round(sesgo_a, 4)} | escala {np.round(escala_a, 5)}")
    t_s = (t_ns - t_ns[0]) / 1e9
    en_reposo = t_s < cfg["inicializacion"]["t_reposo_s"]
    reposo = ini.estimar_reposo(w[en_reposo], a[en_reposo])
    for problema in ini.comprobar_reposo(reposo, cfg["inicializacion"]):
        print("AVISO:", problema)
    R0 = ini.rotacion_desde_gravedad(reposo["a_media"])
    e0 = estado_inicial(R0, b_g=reposo["b_g"])      # sesgo de acelerómetro desconocido: 0

    g_usada = reposo["norma_a"] if args.g_medida else G
    g_w = np.array([0.0, 0.0, -g_usada]) if args.g_medida else GRAVEDAD_W
    error_g = reposo["norma_a"] - g_usada
    print(f"|a| en reposo: {reposo['norma_a']:.4f} m/s² | gravedad usada: {g_usada:.4f} m/s²"
          f" | diferencia: {error_g:+.4f} m/s²")

    # --- 2. Propagar ---
    r = propagar_secuencia(e0, t_ns, w, a, cfg["integracion"], g_w)
    p = r["p"]

    # --- 3. Informe: deriva medida y la que predice el error de |a| ---
    print("\n   t [s] | posición x, y, z [m]           | distancia [m] | predicción ½·Δg·t² [m]")
    for t_inf in INSTANTES_INFORME_S:
        if t_inf > t_s[-1]:
            break
        k = int(np.searchsorted(t_s, t_inf))
        prediccion = 0.5 * error_g * t_inf ** 2
        print(f"  {t_inf:6.0f} | {p[k, 0]:+8.3f} {p[k, 1]:+8.3f} {p[k, 2]:+8.3f} |"
              f" {np.linalg.norm(p[k]):13.3f} | {prediccion:+10.3f} (en z)")

    # --- 4. Figura ---
    fig, ejes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i, eje in enumerate("xyz"):
        ejes[0].plot(t_s, p[:, i], label=eje)
        ejes[1].plot(t_s, r["v"][:, i], label=eje)
    ejes[0].set_ylabel("posición [m]")
    ejes[0].set_title("Posición integrando solo la IMU (la placa estaba quieta)")
    ejes[1].set_ylabel("velocidad [m/s]")
    ejes[1].set_xlabel("tiempo [s]")
    for ej in ejes:
        ej.grid(True)
        ej.legend()
    fig.tight_layout()
    ruta_png = carpeta / "posicion.png"
    fig.savefig(ruta_png, dpi=100)
    print(f"\nFigura: {ruta_png}")


if __name__ == "__main__":
    main()