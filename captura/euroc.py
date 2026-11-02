"""
captura/euroc.py — Escritura y lectura de datos en formato EuRoC/ASL.

Estructura de una grabación:
    <carpeta>/mav0/imu0/data.csv           una fila por muestra de IMU
    <carpeta>/mav0/cam0/data.csv           una fila por imagen: tiempo y nombre
    <carpeta>/mav0/cam0/data/<t_ns>.png    las imágenes
    <carpeta>/mav0/cam0/diagnostico.csv    datos extra de cada imagen (formato propio)
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


# =====================================================================
# Cámara
# =====================================================================
CABECERA_CAM = "#timestamp [ns],filename\n"
CABECERA_DIAG_CAM = "t_sensor_ns,t_llegada_ns,exposicion_us,ganancia\n"


def ruta_cam(carpeta):
    """Ruta del CSV de imágenes dentro de una grabación. Devuelve: Path."""
    return Path(carpeta) / "mav0" / "cam0" / "data.csv"


def carpeta_imagenes(carpeta):
    """Carpeta donde se guardan las PNG. Devuelve: Path."""
    return Path(carpeta) / "mav0" / "cam0" / "data"


def ruta_diag_cam(carpeta):
    """Ruta del CSV de diagnóstico de la cámara. Devuelve: Path."""
    return Path(carpeta) / "mav0" / "cam0" / "diagnostico.csv"


def nombre_imagen(t_ns):
    """Nombre de archivo de una imagen: su marca de tiempo en ns."""
    return f"{t_ns}.png"


def linea_cam(t_ns, nombre):
    """Una fila del CSV de imágenes."""
    return f"{t_ns},{nombre}\n"


def linea_diag_cam(t_sensor_ns, t_llegada_ns, exposicion_us, ganancia):
    """Una fila del CSV de diagnóstico."""
    return f"{t_sensor_ns},{t_llegada_ns},{exposicion_us},{ganancia:.4f}\n"


def leer_cam(ruta_csv):
    """Lee el CSV de imágenes.

    Devuelve: (t_ns, nombres)
        t_ns:    array (N,) int64 [ns]
        nombres: lista de N nombres de archivo
    """
    tiempos = []
    nombres = []
    with open(ruta_csv, encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("#") or not linea.strip():
                continue
            t, nombre = linea.strip().split(",")
            tiempos.append(int(t))
            nombres.append(nombre)
    return np.array(tiempos, dtype=np.int64), nombres


def leer_diag_cam(ruta_csv):
    """Lee el CSV de diagnóstico de la cámara.

    Devuelve: dict de arrays (N,): t_sensor_ns y t_llegada_ns (int64),
        exposicion_us y ganancia (float).
    """
    columnas = {"t_sensor_ns": [], "t_llegada_ns": [], "exposicion_us": [], "ganancia": []}
    with open(ruta_csv, encoding="utf-8") as f:
        f.readline()                                  # saltar la cabecera
        for linea in f:
            if not linea.strip():
                continue
            t_s, t_l, exp, gan = linea.strip().split(",")
            columnas["t_sensor_ns"].append(int(t_s))
            columnas["t_llegada_ns"].append(int(t_l))
            columnas["exposicion_us"].append(float(exp))
            columnas["ganancia"].append(float(gan))
    return {
        "t_sensor_ns": np.array(columnas["t_sensor_ns"], dtype=np.int64),
        "t_llegada_ns": np.array(columnas["t_llegada_ns"], dtype=np.int64),
        "exposicion_us": np.array(columnas["exposicion_us"]),
        "ganancia": np.array(columnas["ganancia"]),
    }