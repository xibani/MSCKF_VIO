"""
vio/escena.py — Etapa 20: una habitación con textura, vista por la cámara.

Hasta ahora el simulador dibujaba cada punto como una mancha sobre fondo negro (etapa 13).
A FAST y a Lucas-Kanade les basta, pero una red como SuperPoint está entrenada con fotos:
necesita una imagen "de verdad", con superficies que tengan dibujo. Aquí se pinta un dibujo
al azar en las 4 paredes, el suelo y el techo, y se calcula qué ve la cámara desde una pose.

Cómo se dibuja una imagen (trazado de rayos, del píxel al mundo):
    1. De cada píxel sale un rayo: la recta de todos los puntos 3D que caen en ese píxel.
    2. Se busca dónde choca el rayo con la habitación (qué cara y a qué distancia).
    3. El brillo del píxel es el del dibujo de esa cara en el punto del choque.
Es el camino inverso de la proyección de la etapa 2 (del mundo al píxel).

La habitación es la misma del simulador: x en [-lx/2, lx/2], y en [-ly/2, ly/2], z en [0, lz].
Caras, numeradas:  0: x = -lx/2    1: x = +lx/2    2: y = -ly/2    3: y = +ly/2
                   4: z = 0 (suelo)                5: z = lz (techo)
"""

import cv2
import numpy as np

from vio import modelo_camara as mc

N_CARAS = 6


# =====================================================================
# Paso 20.1 — El dibujo de cada cara
# =====================================================================
def tamano_de_cara(cara, dimensiones):
    """(ancho, alto) en metros de una cara. Sus dos ejes (a, b) son:

    caras 0 y 1 (x constante):  a = y + ly/2,  b = z
    caras 2 y 3 (y constante):  a = x + lx/2,  b = z
    caras 4 y 5 (z constante):  a = x + lx/2,  b = y + ly/2
    Así a y b van siempre de 0 al tamaño de la cara.
    """
    lx, ly, lz = dimensiones
    if cara in (0, 1):
        return ly, lz
    if cara in (2, 3):
        return lx, lz
    return lx, ly


def textura_aleatoria(ancho_m, alto_m, cfg, rng):
    """Un dibujo al azar para una superficie de ancho_m x alto_m.

    cfg: sección "textura" de config/simulador.yaml.
    Tiene dos ingredientes:
      - Manchas suaves de varios tamaños: como una pared con sombras y desconchones. Dan
        variación de brillo en todas partes, pero pocas esquinas.
      - Rectángulos de gris uniforme (como cuadros o carteles) de tamaños al azar: cada uno
        aporta 4 esquinas nítidas, y al solaparse, muchas más. Son lo que detectan FAST y
        SuperPoint.
    Al final se suaviza un poco: un borde perfectamente afilado no existe en una foto real,
    y al verlo de lejos produciría parpadeos (aliasing).
    Devuelve: array (alto_px, ancho_px) float32, con valores entre 0 y 255.
    """
    px_m = cfg["px_por_m"]
    ancho, alto = int(round(ancho_m * px_m)), int(round(alto_m * px_m))

    # 1. Manchas suaves: una cuadrícula de valores al azar, ampliada hasta el tamaño final.
    #    Cada tamaño de celda da manchas de ese tamaño; se suman con pesos iguales.
    dibujo = np.zeros((alto, ancho), dtype=np.float32)
    for celda_m in cfg["manchas_m"]:
        columnas = max(int(np.ceil(ancho_m / celda_m)), 2)
        filas = max(int(np.ceil(alto_m / celda_m)), 2)
        valores = rng.uniform(-1.0, 1.0, (filas, columnas)).astype(np.float32)
        dibujo += cv2.resize(valores, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    dibujo = cfg["gris_medio"] + cfg["contraste_manchas"] * dibujo / len(cfg["manchas_m"])

    # 2. Rectángulos: posición, tamaño y gris al azar
    n_rectangulos = int(round(cfg["rectangulos_por_m2"] * ancho_m * alto_m))
    lado_min, lado_max = cfg["lado_rectangulo_m"]
    for _ in range(n_rectangulos):
        w = int(rng.uniform(lado_min, lado_max) * px_m)
        h = int(rng.uniform(lado_min, lado_max) * px_m)
        c0 = int(rng.uniform(0, ancho - 1))
        f0 = int(rng.uniform(0, alto - 1))
        dibujo[f0:f0 + h, c0:c0 + w] = rng.uniform(cfg["gris_min"], cfg["gris_max"])

    # 3. Suavizado (sigma en píxeles del dibujo)
    dibujo = cv2.GaussianBlur(dibujo, (0, 0), cfg["suavizado_px"])
    return np.clip(dibujo, 0.0, 255.0)


def crear_habitacion(dimensiones, cfg, rng):
    """La habitación con un dibujo distinto en cada cara.

    Devuelve: dict con dimensiones (lx, ly, lz), px_por_m y texturas (lista de 6 arrays).
    """
    texturas = []
    for cara in range(N_CARAS):
        ancho_m, alto_m = tamano_de_cara(cara, dimensiones)
        texturas.append(textura_aleatoria(ancho_m, alto_m, cfg, rng))
    return {"dimensiones": tuple(dimensiones), "px_por_m": cfg["px_por_m"], "texturas": texturas}