"""tests/test_calibracion_camara.py — Tests de vio/calibracion_camara.py (sin cámara)."""

import cv2
import numpy as np
import pytest

from vio import calibracion_camara as cc
from vio import modelo_camara as mc

NX, NY, LADO = 9, 6, 0.025


def imagen_tablero(lado_px=40, margen_px=60):
    """Tablero de (NX+1) x (NY+1) cuadros dibujado a mano, con sus esquinas exactas.

    Un cuadro que ocupa los píxeles [a, a + lado) tiene su borde en la coordenada
    continua a - 0.5 (los centros de los píxeles están en los enteros).
    """
    ancho = 2 * margen_px + (NX + 1) * lado_px
    alto = 2 * margen_px + (NY + 1) * lado_px
    img = np.full((alto, ancho), 255, dtype=np.uint8)
    for fila in range(NY + 1):
        for col in range(NX + 1):
            if (fila + col) % 2 == 0:
                y0, x0 = margen_px + fila * lado_px, margen_px + col * lado_px
                img[y0:y0 + lado_px, x0:x0 + lado_px] = 0
    esquinas = []
    for fila in range(1, NY + 1):
        for col in range(1, NX + 1):
            esquinas.append([margen_px + col * lado_px - 0.5, margen_px + fila * lado_px - 0.5])
    return img, np.array(esquinas)


def test_puntos_tablero():
    p = cc.puntos_tablero(NX, NY, LADO)
    assert p.shape == (54, 3)
    assert p[0] == pytest.approx([0, 0, 0])
    assert p[1] == pytest.approx([LADO, 0, 0])           # el siguiente, en la misma fila
    assert p[NX] == pytest.approx([0, LADO, 0])          # primero de la segunda fila
    assert np.all(p[:, 2] == 0)                          # todos en el plano Z = 0


def test_detecta_el_tablero_con_precision_subpixel():
    img, esperadas = imagen_tablero()
    # Perspectiva suave, como si se viera algo inclinado
    origen = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
    destino = np.float32([[20, 10], [img.shape[1] - 5, 30], [img.shape[1] - 30, img.shape[0] - 5],
                          [5, img.shape[0] - 20]])
    H = cv2.getPerspectiveTransform(origen, destino)
    girada = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]), borderValue=255)
    esperadas_h = cv2.perspectiveTransform(esperadas.reshape(-1, 1, 2), H).reshape(-1, 2)

    esquinas = cc.detectar_esquinas(girada, NX, NY)
    assert esquinas is not None and esquinas.shape == (54, 2)
    # findChessboardCorners puede empezar por cualquiera de los dos extremos
    if np.linalg.norm(esquinas[0] - esperadas_h[0]) > np.linalg.norm(esquinas[0] - esperadas_h[-1]):
        esquinas = esquinas[::-1]
    assert np.abs(esquinas - esperadas_h).max() < 0.2   # menos de 0,2 px


def test_no_encuentra_tablero_en_una_imagen_lisa():
    assert cc.detectar_esquinas(np.full((480, 640), 128, dtype=np.uint8), NX, NY) is None


def test_nitidez_baja_al_desenfocar():
    img, _ = imagen_tablero()
    borrosa = cv2.GaussianBlur(img, (0, 0), 3.0)
    assert cc.nitidez(borrosa) < cc.nitidez(img) / 10


def test_calibrar_recupera_la_camara_verdadera():
    # Cámara "verdadera" y 25 posturas del tablero: se proyectan las esquinas con ruido
    K_v = mc.matriz_K(600.0, 605.0, 322.0, 238.0)
    dist_v = np.array([-0.25, 0.08, 0.001, -0.0007, 0.0])
    puntos = cc.puntos_tablero(NX, NY, LADO)
    centro = puntos.mean(axis=0)
    rng = np.random.default_rng(0)
    lista = []
    while len(lista) < 25:
        rvec = rng.uniform(-0.5, 0.5, 3)                  # inclinaciones de hasta ~30°
        tvec = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.08), rng.uniform(0.35, 0.6)])
        R, _ = cv2.Rodrigues(rvec)
        p_c = (R @ (puntos - centro).T).T + tvec          # tablero centrado delante de la cámara
        uv = mc.proyectar(p_c, K_v, dist_v) + rng.normal(0.0, 0.1, (len(puntos), 2))
        if (uv[:, 0] > 5).all() and (uv[:, 0] < 635).all() and (uv[:, 1] > 5).all() and (uv[:, 1] < 475).all():
            lista.append(uv.astype(np.float32))
    r = cc.calibrar(lista, (puntos - centro).astype(np.float32), 640, 480)

    # RMS = raíz de la media de la distancia² en 2D: con σ = 0,1 px por coordenada, 0,1·√2
    assert r["rms"] == pytest.approx(0.1 * np.sqrt(2), rel=0.1)
    assert r["K"][0, 0] == pytest.approx(600.0, rel=0.01)
    assert r["K"][1, 1] == pytest.approx(605.0, rel=0.01)
    assert r["K"][0, 2] == pytest.approx(322.0, abs=3.0)
    assert r["K"][1, 2] == pytest.approx(238.0, abs=3.0)
    assert r["dist"][0] == pytest.approx(-0.25, abs=0.03)
    assert len(r["errores_por_imagen"]) == 25


def test_cobertura():
    esquinas = [np.array([[10.0, 10.0], [630.0, 470.0], [320.0, 240.0], [320.0, 241.0]])]
    c = cc.cobertura(esquinas, 640, 480, 3)
    assert c[0, 0] == 1 and c[2, 2] == 1 and c[1, 1] == 2 and c.sum() == 4