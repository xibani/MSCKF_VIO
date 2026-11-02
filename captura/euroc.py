"""
captura/euroc.py — Escritura y lectura de datos en formato EuRoC/ASL.

Estructura de una grabación:
    <carpeta>/mav0/imu0/data.csv    una fila por muestra de IMU
"""

from pathlib import Path

import numpy as np

CABECERA_IMU = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")


def ruta_imu(carpeta):
    """Ruta del CSV de la IMU dentro de una grabación. Devuelve: Path."""
    return Path(carpeta) / "mav0" / "imu0" / "data.csv"


def linea_imu(t_ns, w_rad_s, a_m_s2):
    """Una fila del CSV de IMU: tiempo, giroscopio y acelerómetro (en ese orden)."""
    return (f"{t_ns},{w_rad_s[0]:.7f},{w_rad_s[1]:.7f},{w_rad_s[2]:.7f},"
            f"{a_m_s2[0]:.6f},{a_m_s2[1]:.6f},{a_m_s2[2]:.6f}\n")


def leer_imu(ruta_csv):
    """Lee un CSV de IMU en formato EuRoC.

    Devuelve: (t_ns, w, a)
        t_ns: array (N,) int64, marcas de tiempo [ns]
        w:    array (N, 3) float, giroscopio [rad/s]
        a:    array (N, 3) float, acelerómetro [m/s²]
    El tiempo se lee como entero: pasarlo por float perdería nanosegundos.
    """
    tiempos = []
    filas = []
    with open(ruta_csv, encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("#") or not linea.strip():
                continue                      # cabecera o línea vacía
            campos = linea.strip().split(",")
            tiempos.append(int(campos[0]))
            filas.append([float(c) for c in campos[1:7]])

    t_ns = np.array(tiempos, dtype=np.int64)
    datos = np.array(filas, dtype=np.float64)
    return t_ns, datos[:, 0:3], datos[:, 3:6]