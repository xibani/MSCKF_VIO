"""
captura/camara.py — Imágenes en gris de la cámara OV5647 con Picamera2.

Parte 1: funciones puras (se prueban sin cámara).
Parte 2: funciones que reciben el objeto Picamera2.
"""

import time

from captura.reloj import ahora_ns

MODO_MANUAL = 1   # valor "Manual" de los controles ExposureTimeMode y AnalogueGainMode


# =====================================================================
# Parte 1 — Funciones puras
# =====================================================================
def periodo_us(fps):
    """Duración de cada imagen en microsegundos (int). 20 fps -> 50000 us."""
    return int(round(1e6 / fps))


def plano_y(yuv, ancho, alto):
    """Extrae la imagen en gris (plano Y) de un array YUV420.

    yuv: array (alto*3/2, ancho_o_mayor) uint8, tal como lo entrega Picamera2:
         primero las `alto` filas de Y, después U y V.
    Devuelve: array (alto, ancho) uint8, copia independiente del búfer.
    """
    return yuv[:alto, :ancho].copy()


def exposicion_fija(exp_us, ganancia, exp_max_us, ganancia_max):
    """Calcula exposición y ganancia fijas a partir de las automáticas.

    El brillo de la imagen es proporcional a exposición · ganancia. Si la
    exposición supera el máximo, se recorta y se sube la ganancia en la misma
    proporción; después se limita la ganancia.
    Devuelve: (exp_us, ganancia, brillo_relativo)
        brillo_relativo: brillo final / brillo automático (1.0 = igual de claro).
    """
    exp_nueva = exp_us
    ganancia_nueva = ganancia

    # 1. Limitar la exposición compensando con ganancia
    if exp_nueva > exp_max_us:
        ganancia_nueva = ganancia_nueva * exp_nueva / exp_max_us
        exp_nueva = exp_max_us

    # 2. Limitar la ganancia (si se recorta aquí, la imagen queda más oscura)
    if ganancia_nueva > ganancia_max:
        ganancia_nueva = ganancia_max

    brillo_relativo = (exp_nueva * ganancia_nueva) / (exp_us * ganancia)
    return int(exp_nueva), float(ganancia_nueva), brillo_relativo


def controles_manuales(disponibles, exp_us, ganancia, ganancias_color):
    """Controles que fijan exposición, ganancia y balance de blancos.

    disponibles: nombres de los controles que ofrece la cámara
                 (las claves de picam2.camera_controls).
    ganancias_color: (rojo, azul) del balance de blancos actual, o None.
    Las versiones recientes de libcamera usan ExposureTimeMode y
    AnalogueGainMode; las antiguas, AeEnable. Se usa lo que haya.
    Devuelve: dict para picam2.set_controls().
    """
    controles = {"ExposureTime": exp_us, "AnalogueGain": ganancia}

    # 1. Desactivar la exposición automática
    if "ExposureTimeMode" in disponibles and "AnalogueGainMode" in disponibles:
        controles["ExposureTimeMode"] = MODO_MANUAL
        controles["AnalogueGainMode"] = MODO_MANUAL
    elif "AeEnable" in disponibles:
        controles["AeEnable"] = False
    else:
        raise RuntimeError("la cámara no ofrece ningún control para fijar la exposición")

    # 2. Congelar el balance de blancos (cambiaría el brillo del plano Y)
    if "AwbEnable" in disponibles:
        controles["AwbEnable"] = False
    if "ColourGains" in disponibles and ganancias_color is not None:
        controles["ColourGains"] = (float(ganancias_color[0]), float(ganancias_color[1]))

    return controles