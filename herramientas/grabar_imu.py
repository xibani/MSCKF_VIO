"""
herramientas/grabar_imu.py — Graba solo la IMU, en formato EuRoC.

Uso, desde ~/svo-pi:
    python -m herramientas.grabar_imu                 # 30 s
    python -m herramientas.grabar_imu --duracion 60
Ctrl+C para parar antes de tiempo (lo grabado se conserva).
"""

import argparse
from datetime import datetime

from smbus2 import SMBus

from captura import euroc, mpu6050
from vio.config import RAIZ_PROYECTO, cargar_config


def main():
    # --- 1. Argumentos de la línea de comandos ---
    parser = argparse.ArgumentParser(description="Graba la IMU en formato EuRoC")
    parser.add_argument("--duracion", type=float, default=30.0, help="segundos a grabar")
    args = parser.parse_args()

    # --- 2. Configuración y carpeta de salida ---
    cfg_imu = cargar_config("imu")
    cfg_proyecto = cargar_config("proyecto")
    nombre = "imu_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = RAIZ_PROYECTO / cfg_proyecto["carpeta_datos"] / nombre
    ruta_csv = euroc.ruta_imu(carpeta)
    ruta_csv.parent.mkdir(parents=True)     # crea mav0/imu0/ y las carpetas intermedias

    # --- 3. Grabación ---
    contador = {"errores_i2c": 0}
    n = 0
    with SMBus(cfg_imu["bus_i2c"]) as bus, open(ruta_csv, "w", encoding="utf-8") as f:
        ajustes = mpu6050.configurar(bus, cfg_imu)
        muestras_por_segundo = round(ajustes["frecuencia_hz"])
        f.write(euroc.CABECERA_IMU)
        print(f"Grabando {args.duracion:.0f} s en {carpeta}")

        t_inicio_ns = None
        try:
            for t_ns, w, a, temp in mpu6050.leer_continuo(bus, cfg_imu, ajustes, contador):
                if t_inicio_ns is None:
                    t_inicio_ns = t_ns
                f.write(euroc.linea_imu(t_ns, w, a))
                n += 1

                transcurrido_s = (t_ns - t_inicio_ns) / 1e9
                if n % muestras_por_segundo == 0:
                    print(f"  {transcurrido_s:5.1f} s | {n} muestras | "
                          f"errores I2C: {contador['errores_i2c']}")
                if transcurrido_s >= args.duracion:
                    break
        except KeyboardInterrupt:
            print("\nInterrumpido: se conserva lo grabado")

    print(f"Hecho: {n} muestras, {contador['errores_i2c']} errores I2C")
    print(f"Analiza con:  python -m herramientas.analizar_imu {carpeta}")


if __name__ == "__main__":
    main()