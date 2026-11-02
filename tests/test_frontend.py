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
    

# =====================================================================
# Paso 8.2
# =====================================================================
def test_seguir_mide_un_desplazamiento_subpixel():
    img = textura()
    p = fe.detectar(img, np.zeros((0, 2)), CFG["deteccion"], MARGEN)
    p1, validos, _ = fe.seguir(img, desplazar(img, 3.3, -2.6), p, CFG["seguimiento"])
    assert validos.mean() > 0.9
    error = p1[validos] - p[validos] - np.array([3.3, -2.6])
    assert np.median(np.abs(error)) < 0.05                       # mejor que 1/20 de píxel


def test_seguir_descarta_los_que_salen_de_la_imagen():
    img = textura()
    p = fe.detectar(img, np.zeros((0, 2)), CFG["deteccion"], MARGEN)
    p1, validos, _ = fe.seguir(img, desplazar(img, 40.0, 0.0), p, CFG["seguimiento"])
    fuera = p[:, 0] + 40.0 >= 640                # su posición verdadera ya no está en la imagen
    assert fuera.any()
    assert not validos[fuera].any()
    # Ningún punto aceptado puede estar en el margen del borde
    assert (p1[validos, 0] < 640 - CFG["seguimiento"]["margen_borde_px"]).all()


def test_ida_y_vuelta_detecta_una_zona_que_cambia():
    img = textura()
    p = fe.detectar(img, np.zeros((0, 2)), CFG["deteccion"], MARGEN)
    img2 = desplazar(img, 2.0, 1.0)
    img2[100:300, 200:400] = textura(semilla=7)[100:300, 200:400]   # algo tapa esa zona
    _, validos, _ = fe.seguir(img, img2, p, CFG["seguimiento"])
    dentro = (p[:, 0] > 215) & (p[:, 0] < 385) & (p[:, 1] > 115) & (p[:, 1] < 285)
    fuera = (p[:, 0] < 180) | (p[:, 0] > 420) | (p[:, 1] < 80) | (p[:, 1] > 320)
    assert validos[dentro].mean() < 0.2
    assert validos[fuera].mean() > 0.9
    

# =====================================================================
# Paso 8.3
# =====================================================================
def vistas_sinteticas(n=200, n_malos=40, semilla=0):
    """Puntos 3D vistos desde dos posiciones de cámara + n_malos correspondencias falsas."""
    rng = np.random.default_rng(semilla)
    P = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), rng.uniform(3, 8, n)])
    R = so3.exp_so3([0.02, -0.05, 0.01])
    t = np.array([0.3, 0.05, 0.0])                       # la cámara se desplaza 30 cm
    P2 = (R @ P.T).T + t
    xy1 = P[:, :2] / P[:, 2:3]
    xy2 = P2[:, :2] / P2[:, 2:3]
    malos = rng.choice(n, n_malos, replace=False)
    xy2[malos] += rng.uniform(-0.05, 0.05, (n_malos, 2))  # entre 0 y ~30 px de error
    return xy1, xy2, malos


def test_rechazo_geometrico_encuentra_los_malos():
    xy1, xy2, malos = vistas_sinteticas()
    desplazamiento = np.linalg.norm(xy2 - xy1, axis=1) * K[0, 0]
    encajan = fe.rechazo_geometrico(xy1, xy2, desplazamiento, CFG["geometria"], K[0, 0])
    buenos = np.setdiff1d(np.arange(len(xy1)), malos)
    error_px = np.linalg.norm(xy2[malos] - xy1[malos], axis=1)    # (solo para documentar)
    assert encajan[buenos].mean() > 0.98                          # casi ningún bueno rechazado
    assert (~encajan[malos]).mean() > 0.9                         # casi todos los malos fuera
    assert error_px.max() > 0


def test_rechazo_geometrico_no_actua_sin_movimiento():
    xy1, xy2, _ = vistas_sinteticas()
    poco = np.full(len(xy1), 0.3)                                 # mediana < paralaje_min_px
    assert fe.rechazo_geometrico(xy1, xy2, poco, CFG["geometria"], K[0, 0]).all()