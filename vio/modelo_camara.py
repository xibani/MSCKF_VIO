"""
vio/modelo_camara.py — Modelo de cámara pinhole con distorsión radial-tangencial.

Cadena completa, de un punto 3D en ejes de la cámara a un píxel:
    1. Proyección:   (X, Y, Z)  ->  (x, y) = (X/Z, Y/Z)        coordenadas normalizadas
    2. Distorsión:   (x, y)     ->  (xd, yd)                   lo que hace la lente
    3. Intrínsecos:  (xd, yd)   ->  (u, v) = (fx·xd + cx, fy·yd + cy)   píxeles
Coeficientes de distorsión en el orden de OpenCV: (k1, k2, p1, p2, k3).
"""

import cv2
import numpy as np
import yaml

from vio.config import RAIZ_PROYECTO


def matriz_K(fx, fy, cx, cy):
    """Matriz de intrínsecos (3, 3)."""
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]])


def distorsionar(xy, dist):
    """Aplica la distorsión de la lente a coordenadas normalizadas.

    xy: array (N, 2) de coordenadas normalizadas sin distorsión.
    dist: (k1, k2, p1, p2, k3).
    Devuelve: array (N, 2) distorsionado.
    """
    k1, k2, p1, p2, k3 = dist
    x = xy[:, 0]
    y = xy[:, 1]
    r2 = x * x + y * y                                    # distancia al centro, al cuadrado
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3  # estira o encoge según la distancia
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.column_stack([xd, yd])


def proyectar(p_c, K, dist):
    """Puntos 3D en ejes de la cámara -> píxeles.

    p_c: array (N, 3), con Z > 0 (delante de la cámara).
    Devuelve: array (N, 2) [px].
    """
    xy = p_c[:, :2] / p_c[:, 2:3]                         # 1. proyección
    xy_d = distorsionar(xy, dist)                         # 2. distorsión
    u = K[0, 0] * xy_d[:, 0] + K[0, 2]                    # 3. intrínsecos
    v = K[1, 1] * xy_d[:, 1] + K[1, 2]
    return np.column_stack([u, v])


def normalizar(uv, K, dist):
    """Píxeles -> coordenadas normalizadas sin distorsión (inversa de 2 y 3).

    Quitar la distorsión no tiene fórmula directa: OpenCV la resuelve por iteraciones
    (precisión de ~1e-6 en coordenadas normalizadas, unos 0,001 px).
    uv: array (N, 2) [px]. Devuelve: array (N, 2).
    """
    puntos = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(puntos, K, np.asarray(dist, dtype=np.float64)).reshape(-1, 2)


def guardar_intrinsecos(ruta, K, dist, ancho, alto, info):
    """Guarda la calibración en YAML. info: dict con datos extra (rms, fecha...)."""
    datos = {
        "ancho": int(ancho),
        "alto": int(alto),
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "dist_k1_k2_p1_p2_k3": [float(d) for d in np.ravel(dist)[:5]],
        "info": info,
    }
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# Intrínsecos de la cámara: generado por herramientas/calibrar_camara.py\n")
        yaml.safe_dump(datos, f, allow_unicode=True, sort_keys=False)


def cargar_intrinsecos(nombre="camara_intrinsecos"):
    """Lee config/<nombre>.yaml. Devuelve: (K (3, 3), dist (5,), (ancho, alto))."""
    with open(RAIZ_PROYECTO / "config" / f"{nombre}.yaml", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    K = matriz_K(d["fx"], d["fy"], d["cx"], d["cy"])
    return K, np.array(d["dist_k1_k2_p1_p2_k3"]), (d["ancho"], d["alto"])