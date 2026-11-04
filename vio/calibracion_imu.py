"""
vio/calibracion_imu.py — Calibración del acelerómetro en 6 posiciones.

Modelo por eje (sin acoplamiento entre ejes):   a_medida = escala · a_verdadera + sesgo
Con el eje i apuntando hacia ARRIBA la verdadera es +g, y hacia ABAJO, -g:
    arriba = escala·g + sesgo        abajo = -escala·g + sesgo
    =>  sesgo  = (arriba + abajo) / 2
        escala = (arriba - abajo) / (2·g)
"""

import numpy as np
import yaml

from vio.config import RAIZ_PROYECTO

# (nombre, eje, signo): qué eje de la IMU apunta hacia arriba en cada posición
POSICIONES = [("+x", 0, +1), ("-x", 0, -1),
              ("+y", 1, +1), ("-y", 1, -1),
              ("+z", 2, +1), ("-z", 2, -1)]


def comprobar_posicion(a_media, eje, signo, coseno_min):
    """¿Apunta de verdad el eje esperado hacia arriba?

    coseno_min: coseno mínimo del ángulo entre el eje y la vertical (0,98 ≈ 11°).
    Devuelve: (ok, coseno).
    """
    coseno = signo * a_media[eje] / np.linalg.norm(a_media)
    return coseno >= coseno_min, float(coseno)


def sesgo_y_escala(medias, g):
    """Sesgo y escala de cada eje a partir de las medias de las 6 posiciones.

    medias: dict nombre -> array (3,) media del acelerómetro en esa posición.
    Devuelve: (sesgo (3,) [m/s²], escala (3,) [adimensional]).
    """
    sesgo = np.zeros(3)
    escala = np.zeros(3)
    for i, letra in enumerate("xyz"):
        arriba = medias["+" + letra][i]
        abajo = medias["-" + letra][i]
        sesgo[i] = (arriba + abajo) / 2.0
        escala[i] = (arriba - abajo) / (2.0 * g)
    return sesgo, escala


def corregir_acelerometro(a, sesgo, escala):
    """Medida corregida: (a - sesgo) / escala. a: (3,) o (N, 3)."""
    return (np.asarray(a) - sesgo) / escala


def guardar_calibracion_imu(ruta, sesgo, escala, info):
    """Guarda el resultado en YAML."""
    datos = {"acc_sesgo_m_s2": [float(x) for x in sesgo],
             "acc_escala": [float(x) for x in escala],
             "info": info}
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# Calibración del acelerómetro: generado por herramientas/calibrar_acelerometro.py\n")
        yaml.safe_dump(datos, f, allow_unicode=True, sort_keys=False)


def cargar_calibracion_imu(nombre="imu_calibracion", ruta=None):
    """Lee config/<nombre>.yaml (o el archivo `ruta`, si se da). Devuelve: (sesgo (3,), escala (3,))."""
    if ruta is None:
        ruta = RAIZ_PROYECTO / "config" / f"{nombre}.yaml"
    with open(ruta, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return np.array(d["acc_sesgo_m_s2"]), np.array(d["acc_escala"])