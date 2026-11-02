"""
herramientas/ver_simulacion.py — Círculo simulado: verdad frente a IMU propagada.

Uso, desde ~/svo-pi:
    python -m herramientas.ver_simulacion
Compara tres casos con la misma trayectoria: IMU perfecta, IMU con un sesgo de
acelerómetro sin corregir, e IMU con un sesgo de giroscopio sin corregir.
Guarda datos/simulacion_circulo.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vio import simulador as sim
from vio.config import RAIZ_PROYECTO, cargar_config
from vio.estado import EstadoImu
from vio.propagacion import propagar_secuencia

# Sesgos SIN corregir de los casos 2 y 3 (valores típicos de una MPU-6050 mal calibrada)
SESGO_ACC = np.array([0.05, 0.0, 0.0])          # m/s², en el eje x de la IMU
SESGO_GIR = np.radians([0.1, 0.0, 0.0])         # 0,1 °/s sobre x: inclina la placa


def main():
    cfg_c = cargar_config("simulador")["circulo"]
    cfg_int = cargar_config("orientacion")["integracion"]

    # --- 1. Trayectoria verdadera ---
    t_ns = sim.tiempos_ns(cfg_c["duracion_s"], cfg_c["frecuencia_hz"])
    t_s = t_ns / 1e9
    tr = sim.circulo(t_s, cfg_c["radio_m"], cfg_c["omega_rad_s"], cfg_c["altura_m"])
    e0 = EstadoImu(tr["R_w_i"][0].copy(), tr["v_w"][0].copy(), tr["p_w"][0].copy(),
                   np.zeros(3), np.zeros(3))      # el filtro cree que no hay sesgos

    # --- 2. Tres IMU distintas ---
    casos = {
        "IMU perfecta": {},
        f"sesgo acel. {SESGO_ACC[0]} m/s² sin corregir": {"b_a": SESGO_ACC},
        f"sesgo gir. {np.degrees(SESGO_GIR[0]):.1f} °/s (eje x) sin corregir": {"b_g": SESGO_GIR},
    }
    resultados = {}
    for nombre, sesgos in casos.items():
        w_m, a_m = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"], **sesgos)
        r = propagar_secuencia(e0, t_ns, w_m, a_m, cfg_int)
        error = np.linalg.norm(r["p"] - tr["p_w"], axis=1)
        resultados[nombre] = (r, error)
        print(f"{nombre:40s} error final {error[-1]:7.3f} m")

    # --- 3. Figura: planta (x-y) y error frente al tiempo ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(tr["p_w"][:, 0], tr["p_w"][:, 1], "k", lw=3, label="verdad")
    for nombre, (r, error) in resultados.items():
        ax1.plot(r["p"][:, 0], r["p"][:, 1], lw=1.2, label=nombre)
        ax2.plot(t_s, error, lw=1.2, label=nombre)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Vista desde arriba")
    ax1.legend()
    ax2.set_yscale("log")
    ax2.set_xlabel("tiempo [s]")
    ax2.set_ylabel("error de posición [m]")
    ax2.set_title("Error (escala logarítmica)")
    ax2.grid(True, which="both")
    ax2.legend()
    fig.tight_layout()

    ruta = RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"] / "simulacion_circulo.png"
    fig.savefig(ruta, dpi=100)
    print(f"Figura: {ruta}")


if __name__ == "__main__":
    main()