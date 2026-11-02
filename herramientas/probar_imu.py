"""
herramientas/probar_imu.py — Configura la IMU e imprime unas cuantas muestras.

Uso, desde ~/svo-pi:
    python -m herramientas.probar_imu
"""

import math
import time

from smbus2 import SMBus

from captura import mpu6050
from vio.config import cargar_config

N_MUESTRAS = 20     # cuántas muestras imprimir
PAUSA_S = 0.2       # pausa entre muestras, para que dé tiempo a leerlas


def main():
    cfg = cargar_config("imu")

    # "with" cierra el bus al terminar, aunque haya un error
    with SMBus(cfg["bus_i2c"]) as bus:
        ajustes = mpu6050.configurar(bus, cfg)
        print(f"IMU configurada: {ajustes['frecuencia_hz']:.0f} Hz, "
              f"±{cfg['rango_gir_dps']} °/s, ±{cfg['rango_acc_g']} g")
        print(" n |      giroscopio [°/s]      |   acelerómetro [m/s²]    |  |a|  | temp")

        for i in range(N_MUESTRAS):
            bloque = mpu6050.leer_bloque(bus, cfg["direccion"])
            w, a, temp = mpu6050.convertir_bloque(bloque, ajustes["escala_acc"],
                                                  ajustes["escala_gir"])
            norma_a = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
            print(f"{i:2d} | {math.degrees(w[0]):+7.2f} {math.degrees(w[1]):+7.2f} "
                  f"{math.degrees(w[2]):+7.2f} | {a[0]:+7.3f} {a[1]:+7.3f} {a[2]:+7.3f} | "
                  f"{norma_a:5.2f} | {temp:4.1f} °C")
            time.sleep(PAUSA_S)


if __name__ == "__main__":
    main()