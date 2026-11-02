"""tests/test_frontend.py — Tests de vio/frontend.py con imágenes sintéticas."""

import cv2
import numpy as np
import pytest

from vio import frontend as fe
from vio import modelo_camara as mc
from vio import so3
from vio.config import cargar_config

CFG = cargar_config("frontend")
MARGEN = CFG["seguimiento"]["margen_borde_px"]
K = mc.matriz_K(643.0, 643.0, 320.0, 240.0)
DIST = np.zeros(5)


def textura(alto=480, ancho=640, semilla=0):
    """Imagen con manchas aleatorias suavizadas: llena de esquinas que seguir."""
    rng = np.random.default_rng(semilla)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (alto, ancho)).astype(np.float32), (0, 0), 3)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def desplazar(img, du, dv):
    """La misma imagen movida (du, dv) píxeles (con interpolación: admite decimales)."""
    M = np.float32([[1, 0, du], [0, 1, dv]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


# =====================================================================
# Paso 8.1
# =====================================================================
def test_detectar_reparte_los_puntos():
    d = CFG["deteccion"]
    p = fe.detectar(textura(), np.zeros((0, 2)), d, MARGEN)
    assert 0 < len(p) <= d["max_puntos"]
    # Ninguna celda con más del máximo
    ocupacion = np.zeros((d["celdas_y"], d["celdas_x"]), dtype=int)
    for u, v in p:
        ocupacion[fe.celda_de(u, v, 640, 480, d["celdas_x"], d["celdas_y"])] += 1
    assert ocupacion.max() <= d["max_por_celda"]
    assert (ocupacion > 0).sum() >= 0.9 * ocupacion.size          # casi todas las celdas usadas
    # Ningún par de puntos más cerca de la distancia mínima
    distancias = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(distancias, np.inf)
    assert distancias.min() >= d["distancia_min_px"]
    # Ninguno en la franja del borde
    assert (p >= MARGEN).all() and (p[:, 0] < 640 - MARGEN).all() and (p[:, 1] < 480 - MARGEN).all()


def test_detectar_respeta_los_puntos_existentes():
    d = CFG["deteccion"]
    existentes = fe.detectar(textura(), np.zeros((0, 2)), d, MARGEN)[:40]
    nuevos = fe.detectar(textura(), existentes, d, MARGEN)
    assert len(existentes) + len(nuevos) <= d["max_puntos"]
    distancias = np.linalg.norm(nuevos[:, None, :] - existentes[None, :, :], axis=2)
    assert distancias.min() >= d["distancia_min_px"] - 1.0        # -1: el círculo se dibuja en píxeles enteros