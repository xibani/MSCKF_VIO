"""
apps/pi_orientacion.py — La Pi calcula su orientación con el giroscopio y la envía al PC.

Uso, desde ~/svo-pi, con la placa QUIETA al arrancar:
    python -m apps.pi_orientacion                    # IP del PC de config/orientacion.yaml
    python -m apps.pi_orientacion --ip 192.168.1.37
Ctrl+C para terminar.
"""

import argparse
import socket

import numpy as np
from smbus2 import SMBus

from captura import mpu6050
from vio import inicializacion as ini
from vio import protocolo, so3
from vio.config import cargar_config
from vio.integrador_giro import IntegradorGiro


def leer_reposo(muestras, n):
    """Lee n muestras del generador. Devuelve: (t_ns, w (n, 3), a (n, 3))."""
    t, w, a = [], [], []
    for _ in range(n):
        t_ns, w_k, a_k, _ = next(muestras)
        t.append(t_ns)
        w.append(w_k)
        a.append(a_k)
    return np.array(t, dtype=np.int64), np.array(w), np.array(a)


def main():
    cfg_imu = cargar_config("imu")
    cfg = cargar_config("orientacion")

    parser = argparse.ArgumentParser(description="Orientación con giroscopio, enviada al PC")
    parser.add_argument("--ip", default=cfg["red"]["ip_pc"], help="IP del PC")
    args = parser.parse_args()
    destino = (args.ip, cfg["red"]["puerto"])

    emisor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)    # UDP
    contador = {"errores_i2c": 0}

    with SMBus(cfg_imu["bus_i2c"]) as bus:
        ajustes = mpu6050.configurar(bus, cfg_imu)
        muestras = mpu6050.leer_continuo(bus, cfg_imu, ajustes, contador)
        f_imu = ajustes["frecuencia_hz"]

        # --- 1. Reposo inicial: sesgo e inclinación ---
        print(f"QUIETO {cfg['inicializacion']['t_reposo_s']:.0f} s...")
        n_reposo = int(cfg["inicializacion"]["t_reposo_s"] * f_imu)
        t_rep, w_rep, a_rep = leer_reposo(muestras, n_reposo)
        reposo = ini.estimar_reposo(w_rep, a_rep)
        problemas = ini.comprobar_reposo(reposo, cfg["inicializacion"])
        if problemas:
            for p in problemas:
                print("ERROR:", p)
            print("Vuelve a lanzarlo con la placa quieta.")
            return
        R0 = ini.rotacion_desde_gravedad(reposo["a_media"])
        print(f"Sesgo del giroscopio [°/s]: {np.round(np.degrees(reposo['b_g']), 3)}")

        # --- 2. Integrar y enviar ---
        integrador = IntegradorGiro(R0, reposo["b_g"], int(t_rep[-1]), w_rep[-1],
                                    cfg["integracion"])
        enviar_cada = max(1, round(f_imu / cfg["red"]["frecuencia_envio_hz"]))
        print(f"Enviando a {destino[0]}:{destino[1]} a "
              f"{f_imu / enviar_cada:.0f} Hz. Ctrl+C para terminar.")
        n = 0
        try:
            for t_ns, w, a, _ in muestras:
                R = integrador.actualizar(t_ns, w)
                n += 1
                if n % enviar_cada == 0:
                    emisor.sendto(protocolo.empaquetar(t_ns, R), destino)
                if n % round(f_imu) == 0:        # una vez por segundo
                    al, ca, gu = np.degrees(so3.angulos_zyx(R))
                    print(f"  alabeo {al:+7.1f}° | cabeceo {ca:+7.1f}° | guiñada {gu:+7.1f}°"
                          f" | huecos {integrador.huecos} | errores I2C {contador['errores_i2c']}")
        except KeyboardInterrupt:
            print("\nTerminado")
        finally:
            emisor.close()


if __name__ == "__main__":
    main()