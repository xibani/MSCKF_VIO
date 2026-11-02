"""
vio/frontend.py — Detección y seguimiento de puntos de imagen en imagen.

Por cada imagen nueva:
    1. Seguir los puntos de la imagen anterior (Lucas-Kanade piramidal).
    2. Descartar los seguimientos dudosos (ida y vuelta, bordes).
    3. Descartar los que no encajan con un movimiento rígido de la cámara (RANSAC).
    4. Las pistas de los puntos perdidos se dan por terminadas y se entregan.
    5. Detectar puntos nuevos (FAST) en las zonas con pocos puntos.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from vio import modelo_camara as mc


# =====================================================================
# Paso 8.1 — Detección repartida por la imagen
# =====================================================================
def celda_de(u, v, ancho, alto, celdas_x, celdas_y):
    """Celda (fila, columna) de la rejilla en la que cae el píxel (u, v)."""
    columna = min(int(u / ancho * celdas_x), celdas_x - 1)
    fila = min(int(v / alto * celdas_y), celdas_y - 1)
    return fila, columna


def detectar(gris, puntos_existentes, cfg_det, margen_px):
    """Busca esquinas nuevas, repartidas y lejos de los puntos que ya se siguen.

    gris: imagen (alto, ancho) uint8.
    puntos_existentes: array (N, 2) [px] de los puntos que ya se siguen.
    cfg_det: sección "deteccion" de config/frontend.yaml.
    margen_px: franja junto al borde donde no se detecta (la misma que descarta el
        seguimiento: un punto ahí se perdería en la imagen siguiente).
    Devuelve: array (M, 2) float32 [px] con los puntos nuevos.
    """
    alto, ancho = gris.shape
    cx, cy = cfg_det["celdas_x"], cfg_det["celdas_y"]
    d_min = cfg_det["distancia_min_px"]

    # 1. Máscara: no buscar junto al borde ni alrededor de los puntos que ya existen
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    mascara[margen_px:alto - margen_px, margen_px:ancho - margen_px] = 255
    for u, v in puntos_existentes:
        cv2.circle(mascara, (int(round(u)), int(round(v))), d_min, 0, -1)   # -1 = círculo relleno

    # 2. Detector FAST
    fast = cv2.FastFeatureDetector_create(threshold=cfg_det["fast_umbral"], nonmaxSuppression=True)
    candidatos = fast.detect(gris, mascara)

    # 3. Cuántos puntos tiene ya cada celda
    ocupacion = np.zeros((cy, cx), dtype=int)
    for u, v in puntos_existentes:
        ocupacion[celda_de(u, v, ancho, alto, cx, cy)] += 1

    # 4. Elegir de mejor a peor respuesta, respetando el máximo por celda y el total
    candidatos = sorted(candidatos, key=lambda k: k.response, reverse=True)
    hueco = cfg_det["max_puntos"] - len(puntos_existentes)
    nuevos = []
    for k in candidatos:
        if len(nuevos) >= hueco:
            break
        u, v = k.pt
        celda = celda_de(u, v, ancho, alto, cx, cy)
        if ocupacion[celda] >= cfg_det["max_por_celda"]:
            continue
        # Lejos también de los nuevos ya elegidos (la máscara solo tiene los antiguos)
        if any((u - un) ** 2 + (v - vn) ** 2 < d_min ** 2 for un, vn in nuevos):
            continue
        nuevos.append((u, v))
        ocupacion[celda] += 1

    return np.array(nuevos, dtype=np.float32).reshape(-1, 2)


# =====================================================================
# Paso 8.2 — Seguimiento con Lucas-Kanade y comprobación de ida y vuelta
# =====================================================================
def seguir(gris_ant, gris_act, puntos, cfg_seg):
    """Sigue los puntos de la imagen anterior a la actual.

    puntos: array (N, 2) [px] en la imagen anterior.
    Devuelve: (puntos_act (N, 2), validos (N,) bool, error_ida_vuelta (N,) [px]).
    """
    n = len(puntos)
    if n == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, bool), np.zeros(0)

    parametros = dict(
        winSize=(cfg_seg["ventana_px"], cfg_seg["ventana_px"]),
        maxLevel=cfg_seg["niveles_piramide"],
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                  cfg_seg["iteraciones"], cfg_seg["epsilon"]))
    p0 = puntos.reshape(-1, 1, 2).astype(np.float32)

    # 1. Ida: de la imagen anterior a la actual
    p1, estado_ida, _ = cv2.calcOpticalFlowPyrLK(gris_ant, gris_act, p0, None, **parametros)
    # 2. Vuelta: de la actual a la anterior, partiendo de lo encontrado
    p0_vuelta, estado_vuelta, _ = cv2.calcOpticalFlowPyrLK(gris_act, gris_ant, p1, None, **parametros)

    p1 = p1.reshape(-1, 2)
    error_ida_vuelta = np.linalg.norm(p0_vuelta.reshape(-1, 2) - puntos, axis=1)

    # 3. Válidos: los dos sentidos encontrados, vuelta al mismo sitio y dentro de la imagen
    alto, ancho = gris_act.shape
    m = cfg_seg["margen_borde_px"]
    dentro = ((p1[:, 0] >= m) & (p1[:, 0] < ancho - m) &
              (p1[:, 1] >= m) & (p1[:, 1] < alto - m))
    validos = ((estado_ida.ravel() == 1) & (estado_vuelta.ravel() == 1) &
               (error_ida_vuelta < cfg_seg["ida_vuelta_max_px"]) & dentro)
    return p1, validos, error_ida_vuelta