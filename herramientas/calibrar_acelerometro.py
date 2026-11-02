"""
herramientas/calibrar_acelerometro.py — Sesgo y escala del acelerómetro en 6 posiciones.

Uso, desde ~/svo-pi:
    python -m herramientas.calibrar_acelerometro
Te pide que pongas cada eje de la IMU hacia arriba y hacia abajo (6 posiciones).
Apoya el montaje contra algo recto (una caja, un libro) para que el eje quede vertical.
Guarda el resultado en config/imu_calibracion.yaml.
"""

from datetime import datetime

import numpy as np
from smbus2 import SMBus

from captura import mpu6050
from vio import calibracion_imu as ci
from vio.config import RAIZ_PROYECTO, cargar_config
from vio.constantes import G


def medir_posicion(muestras, f_hz, cfg_a, nombre, eje, signo):
    """Pide una posición, la mide y la comprueba. Repite hasta que sea válida.

    Devuelve: a_media (3,) [m/s²].
    """
    n_asentar = int(cfg_a["t_asentar_s"] * f_hz)
    n_medir = int(cfg_a["t_por_posicion_s"] * f_hz)
    while True:
        input(f"\nPon el eje {nombre} de la IMU hacia ARRIBA, suelta y pulsa Enter...")
        mpu6050.tomar_muestras(muestras, n_asentar)          # descartar: la mano aún se mueve
        print(f"   midiendo {cfg_a['t_por_posicion_s']:.0f} s, no la toques...")
        _, w, a = mpu6050.tomar_muestras(muestras, n_medir)

        # 1. ¿Estaba quieta?
        sigma_gir = np.degrees(w.std(axis=0)).max()
        if sigma_gir > cfg_a["sigma_gir_max_dps"]:
            print(f"   REPITE: se ha movido (σ giroscopio {sigma_gir:.2f} °/s)")
            continue

        # 2. ¿Es la posición pedida?
        a_media = a.mean(axis=0)
        ok, coseno = ci.comprobar_posicion(a_media, eje, signo, cfg_a["coseno_min"])
        if not ok:
            arriba = int(np.argmax(np.abs(a_media)))
            print(f"   REPITE: hacia arriba está {'+' if a_media[arriba] > 0 else '-'}{'xyz'[arriba]},"
                  f" no {nombre} (coseno {coseno:.3f})")
            continue

        angulo = np.degrees(np.arccos(min(coseno, 1.0)))
        print(f"   OK: a = {np.round(a_media, 4)} m/s² | inclinación respecto a la vertical {angulo:.1f}°")
        return a_media


def main():
    cfg_imu = cargar_config("imu")
    cfg_a = cargar_config("calibracion")["acelerometro"]
    contador = {"errores_i2c": 0}

    with SMBus(cfg_imu["bus_i2c"]) as bus:
        ajustes = mpu6050.configurar(bus, cfg_imu)
        muestras = mpu6050.leer_continuo(bus, cfg_imu, ajustes, contador)
        f_hz = ajustes["frecuencia_hz"]

        # --- 1. Medir las 6 posiciones ---
        medias = {}
        for nombre, eje, signo in ci.POSICIONES:
            medias[nombre] = medir_posicion(muestras, f_hz, cfg_a, nombre, eje, signo)

    # --- 2. Sesgo y escala ---
    sesgo, escala = ci.sesgo_y_escala(medias, G)
    print("\n=== RESULTADO ===")
    for i, letra in enumerate("xyz"):
        print(f"  eje {letra}: sesgo {sesgo[i]:+.4f} m/s² | escala {escala[i]:.5f}"
              f" ({100 * (escala[i] - 1):+.2f} %)")

    # --- 3. Comprobación: |a| antes y después de corregir, en cada posición ---
    print("\n  posición | |a| sin corregir | |a| corregida   (g = 9.80665)")
    for nombre, a in medias.items():
        corregida = ci.corregir_acelerometro(a, sesgo, escala)
        print(f"     {nombre}    |     {np.linalg.norm(a):.4f}     |   {np.linalg.norm(corregida):.4f}")

    # --- 4. Guardar ---
    ruta = RAIZ_PROYECTO / "config" / "imu_calibracion.yaml"
    info = {"fecha": datetime.now().isoformat(timespec="seconds"),
            "medias_por_posicion": {k: [round(float(x), 5) for x in v] for k, v in medias.items()},
            "errores_i2c": contador["errores_i2c"]}
    ci.guardar_calibracion_imu(ruta, sesgo, escala, info)
    print(f"\nGuardado en {ruta}")


if __name__ == "__main__":
    main()