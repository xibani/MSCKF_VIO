"""
vio/calibracion_camara.py — Calibración de intrínsecos con un tablero de ajedrez (Zhang).

Idea: el tablero es plano y conocemos la posición exacta de cada esquina en él.
Viéndolo desde muchas posturas, solo hay una K y una distorsión que expliquen
dónde aparecen todas las esquinas en todas las imágenes.
"""

import cv2
import numpy as np

# Criterio de parada del refinamiento subpíxel: 30 iteraciones o movimiento < 0,001 px
CRITERIO_SUBPIXEL = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
VENTANA_SUBPIXEL = (5, 5)          # mitad del lado de la ventana de búsqueda [px]


def puntos_tablero(esquinas_x, esquinas_y, lado_m):
    """Posición 3D de las esquinas interiores en el plano del tablero (Z = 0).

    Orden: fila a fila, igual que las devuelve findChessboardCorners.
    Devuelve: array (esquinas_x · esquinas_y, 3) [m].
    """
    puntos = []
    for fila in range(esquinas_y):
        for columna in range(esquinas_x):
            puntos.append([columna * lado_m, fila * lado_m, 0.0])
    return np.array(puntos, dtype=np.float32)


def detectar_esquinas(gris, esquinas_x, esquinas_y):
    """Busca el tablero y refina sus esquinas con precisión subpíxel.

    gris: imagen (alto, ancho) uint8.
    Devuelve: array (N, 2) float32 con las esquinas [px], o None si no se encuentra.
    """
    banderas = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    encontrado, esquinas = cv2.findChessboardCorners(gris, (esquinas_x, esquinas_y), banderas)
    if not encontrado:
        return None
    esquinas = cv2.cornerSubPix(gris, esquinas, VENTANA_SUBPIXEL, (-1, -1), CRITERIO_SUBPIXEL)
    return esquinas.reshape(-1, 2)


def nitidez(gris):
    """Varianza del laplaciano: alta = bordes nítidos; baja = imagen movida o desenfocada."""
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def calibrar(lista_esquinas, puntos_3d, ancho, alto):
    """Calibra la cámara a partir de las esquinas detectadas en varias imágenes.

    lista_esquinas: lista de arrays (N, 2), uno por imagen.
    puntos_3d: array (N, 3) de puntos_tablero.
    Devuelve: dict con rms (px), K (3, 3), dist (5,), rvecs, tvecs (postura de
        cada imagen) y errores_por_imagen (px, error medio de cada imagen).
    """
    obj = [puntos_3d] * len(lista_esquinas)
    img = [e.reshape(-1, 1, 2).astype(np.float32) for e in lista_esquinas]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, (ancho, alto), None, None)

    # Error de cada imagen: distancia media entre esquina detectada y reproyectada
    errores = []
    for k in range(len(img)):
        reproyectadas, _ = cv2.projectPoints(puntos_3d, rvecs[k], tvecs[k], K, dist)
        diferencias = reproyectadas.reshape(-1, 2) - lista_esquinas[k]
        errores.append(float(np.sqrt((diferencias ** 2).sum(axis=1).mean())))

    return {"rms": float(rms), "K": K, "dist": dist.ravel()[:5], "rvecs": rvecs,
            "tvecs": tvecs, "errores_por_imagen": np.array(errores)}


def cobertura(lista_esquinas, ancho, alto, zonas):
    """Cuántas esquinas han caído en cada zona de una rejilla zonas x zonas.

    Una buena calibración necesita esquinas también cerca de los bordes, donde
    la distorsión es mayor. Devuelve: array (zonas, zonas) de enteros.
    """
    cuenta = np.zeros((zonas, zonas), dtype=int)
    for esquinas in lista_esquinas:
        for u, v in esquinas:
            columna = min(int(u / ancho * zonas), zonas - 1)
            fila = min(int(v / alto * zonas), zonas - 1)
            cuenta[fila, columna] += 1
    return cuenta