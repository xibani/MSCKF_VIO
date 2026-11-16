"""
vio/medidas_mapa.py — Etapa 24: ¿cuánto se parece el mapa disperso a la realidad?

El VIO deja, con cada keyframe de mapeo (etapa 18), la profundidad de unos cuantos puntos:
a cuántos metros de la cámara, medidos a lo largo de su eje (la z en ejes de la cámara),
está lo que se ve en esos píxeles. Son pocas decenas de puntos por keyframe, pero están en
METROS, y de ellos va a depender la escala de todo lo que venga después (etapa 25).

Aquí se comparan con dos referencias:
    - En el simulador, la profundidad verdadera de cada píxel (mav0/prof0, etapa 24).
    - Con la placa, una distancia medida con cinta: la de la cámara a una pared plana a la
      que se mira de frente. Todos los puntos de una pared así tienen la misma z, estén en
      el centro de la imagen o en una esquina.

Las funciones de este módulo son cuentas sobre arrays: no leen archivos.
"""

import numpy as np

from vio import modelo_camara as mc

A_DESVIACION = 1.4826       # desviación típica ≈ 1,4826 · MAD, si los errores son gaussianos


def pixeles_de_keyframe(kf):
    """Dónde cae, en la imagen ORIGINAL (con su distorsión), cada punto de un keyframe.

    kf: dict de protocolo.leer_keyframe (xy normalizadas sin distorsión, K, dist).
    Devuelve: (N, 2) [px].
    """
    n = len(kf["z"])
    if n == 0:
        return np.zeros((0, 2))
    # Un punto con coordenadas normalizadas (x, y) es el punto 3D (x, y, 1) en la cámara
    return mc.proyectar(np.column_stack([kf["xy"], np.ones(n)]), kf["K"], kf["dist"])


def muestrear(mapa, uv):
    """El valor de un mapa (alto, ancho) en posiciones con decimales, por interpolación bilineal.

    uv: (N, 2) [px]. Fuera de la imagen, o si alguno de los 4 píxeles vecinos no tiene dato
    (NaN), el resultado es NaN.
    Devuelve: (N,).
    """
    alto, ancho = mapa.shape
    valores = np.full(len(uv), np.nan)
    for k in range(len(uv)):
        u, v = uv[k]
        u0, v0 = int(np.floor(u)), int(np.floor(v))
        if u0 < 0 or v0 < 0 or u0 + 1 >= ancho or v0 + 1 >= alto:
            continue
        a, b = u - u0, v - v0                                    # parte decimal: de 0 a 1
        valores[k] = ((1 - a) * (1 - b) * mapa[v0, u0] + a * (1 - b) * mapa[v0, u0 + 1]
                      + (1 - a) * b * mapa[v0 + 1, u0] + a * b * mapa[v0 + 1, u0 + 1])
    return valores


def resumen_profundidades(z):
    """Mediana y percentiles 10 y 90 de las profundidades de un keyframe [m].

    Devuelve: dict con n, mediana, p10 y p90 (NaN si no hay puntos).
    """
    if len(z) == 0:
        return {"n": 0, "mediana": np.nan, "p10": np.nan, "p90": np.nan}
    p10, mediana, p90 = np.percentile(z, [10, 50, 90])
    return {"n": len(z), "mediana": float(mediana), "p10": float(p10), "p90": float(p90)}


def comparar_con_verdad(z, z_verdad):
    """Compara profundidades estimadas con las verdaderas, punto a punto.

    z, z_verdad: (N,) [m]. Los puntos sin verdad (NaN) no cuentan.
    Se trabaja con la razón z / z_verdad, que no depende de lo lejos que esté cada punto:
        escala       mediana de la razón. 1,05 = el mapa sale un 5 % más grande que la realidad
        dispersion   desviación típica robusta de la razón alrededor de la escala (0,03 = 3 %)
        dentro_10    fracción de puntos con la razón entre 0,9 y 1,1
    Devuelve: dict con n, escala, dispersion y dentro_10 (NaN si no hay puntos).
    """
    con_verdad = np.isfinite(z_verdad) & (z_verdad > 0)
    n = int(con_verdad.sum())
    if n == 0:
        return {"n": 0, "escala": np.nan, "dispersion": np.nan, "dentro_10": np.nan}
    razon = z[con_verdad] / z_verdad[con_verdad]
    escala = float(np.median(razon))
    mad = float(np.median(np.abs(razon - escala)))           # mediana de las desviaciones
    return {"n": n, "escala": escala, "dispersion": A_DESVIACION * mad,
            "dentro_10": float(np.mean(np.abs(razon - 1.0) < 0.1))}


def error_frente_a_distancia(z, distancia_m):
    """Error de la profundidad mediana de un keyframe frente a una distancia medida [%].

    Positivo: el mapa pone la pared más lejos de lo que está. NaN si no hay puntos.
    """
    if len(z) == 0:
        return np.nan
    return 100.0 * (float(np.median(z)) - distancia_m) / distancia_m
