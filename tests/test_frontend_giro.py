"""
tests/test_frontend_giro.py — Etapa 31: Lucas-Kanade ayudado por el giroscopio (vio/frontend.py).

Un giro puro de la cámara mueve la imagen según la homografía H = K·R·K⁻¹ (sin distorsión),
igual para cualquier profundidad. Con eso se fabrica la imagen "después del giro" y se sabe
exactamente dónde debe acabar cada punto.

    python -m pytest tests/test_frontend_giro.py -v
"""

import copy

import cv2
import numpy as np

from vio import frontend as fe
from vio import modelo_camara as mc
from vio import so3
from vio.config import cargar_config

CFG = cargar_config("frontend")
K = mc.matriz_K(621.0, 621.0, 320.0, 240.0)
SIN_DIST = np.zeros(5)
CON_DIST = np.array([-0.30, 0.10, 0.001, -0.001, 0.0])     # del orden de la de una OV5647


def textura(alto=480, ancho=640, semilla=0):
    """Imagen con manchas aleatorias suavizadas: llena de esquinas que seguir."""
    rng = np.random.default_rng(semilla)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (alto, ancho)).astype(np.float32), (0, 0), 3)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def girar_imagen(img, R_act_ant):
    """La imagen que vería la cámara tras girar R_act_ant (cámara sin distorsión).
    Píxel de un punto: antes K·d, después K·R·d -> H = K·R·K⁻¹ lleva los de antes a los de después."""
    H = K @ R_act_ant @ np.linalg.inv(K)
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT), H


def aplicar_H(H, puntos):
    p = np.column_stack([puntos, np.ones(len(puntos))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def puntos_que_siguen_dentro(img, H, margen):
    """Esquinas de img cuya posición tras el giro sigue dentro de la imagen, con margen."""
    p = fe.detectar(img, np.zeros((0, 2)), CFG["deteccion"], CFG["seguimiento"]["margen_borde_px"])
    destino = aplicar_H(H, p)
    alto, ancho = img.shape
    dentro = ((destino[:, 0] > margen) & (destino[:, 0] < ancho - margen) &
              (destino[:, 1] > margen) & (destino[:, 1] < alto - margen))
    return p[dentro], destino[dentro]


# 10° de guiñada y 3° de cabeceo: unos 115 px en la imagen en una imagen (50 ms a 200 °/s, o
# 100 ms a 100 °/s). Con ventana de 21 px y 3 niveles, Lucas-Kanade solo no llega.
R_GRANDE = so3.exp_so3(np.radians([3.0, 10.0, 0.0]))


def test_prediccion_sin_distorsion_es_la_homografia():
    p = np.array([[100.0, 80.0], [320.0, 240.0], [600.0, 400.0], [50.0, 450.0]])
    prediccion, delante = fe.predecir_con_giro(p, R_GRANDE, K, SIN_DIST)
    H = K @ R_GRANDE @ np.linalg.inv(K)
    assert delante.all()
    assert np.abs(prediccion - aplicar_H(H, p)).max() < 1e-3


def test_prediccion_con_distorsion_ida_y_vuelta():
    # Girar con R y después con Rᵀ debe devolver cada píxel a su sitio (con la distorsión)
    p = np.array([[60.0, 50.0], [320.0, 240.0], [580.0, 420.0], [200.0, 300.0]])
    ida, delante = fe.predecir_con_giro(p, R_GRANDE, K, CON_DIST)
    vuelta, _ = fe.predecir_con_giro(ida, R_GRANDE.T, K, CON_DIST)
    assert delante.all()
    # Quitar la distorsión es iterativo (cv2.undistortPoints): en las esquinas, con distorsión
    # fuerte, deja unas centésimas de píxel
    assert np.abs(vuelta - p).max() < 0.05


def test_prediccion_detras_de_la_camara():
    # Media vuelta: la dirección queda detrás y se deja la posición de partida
    p = np.array([[320.0, 240.0]])
    prediccion, delante = fe.predecir_con_giro(p, so3.exp_so3(np.radians([0.0, 180.0, 0.0])), K, SIN_DIST)
    assert not delante[0]
    assert np.allclose(prediccion, p)


def test_giro_grande_klt_solo_pierde_los_puntos_y_con_giro_no():
    img = textura(semilla=1)
    img2, H = girar_imagen(img, R_GRANDE)
    p, verdad = puntos_que_siguen_dentro(img, H, margen=30)
    assert len(p) > 40

    # Solo Lucas-Kanade: o los pierde, o acaba en otro sitio
    p1, validos, _ = fe.seguir(img, img2, p, CFG["seguimiento"])
    bien_solo = validos & (np.linalg.norm(p1 - verdad, axis=1) < 0.5)
    assert bien_solo.mean() < 0.3

    # Con la predicción del giro
    p1g, validos_g, _ = fe.seguir(img, img2, p, CFG["seguimiento"], giro=(R_GRANDE, K, SIN_DIST))
    error = np.linalg.norm(p1g - verdad, axis=1)
    assert validos_g.mean() > 0.9
    # Lucas-Kanade modela una traslación de la ventana; la homografía además la deforma un poco
    # (escala y cizalla): queda ~0,1 px. Sin el giro, ni siquiera llegaba a medio píxel
    assert np.median(error[validos_g]) < 0.25


def test_giro_con_error_del_giroscopio_lo_corrige_lucas_kanade():
    # El giroscopio se equivoca 0,5° (unos 5 px): la predicción no es exacta pero cae dentro
    # de la ventana y Lucas-Kanade llega al sitio correcto
    img = textura(semilla=2)
    img2, H = girar_imagen(img, R_GRANDE)
    p, verdad = puntos_que_siguen_dentro(img, H, margen=30)
    R_medido = so3.exp_so3(np.radians([0.3, -0.4, 0.0])) @ R_GRANDE
    p1, validos, _ = fe.seguir(img, img2, p, CFG["seguimiento"], giro=(R_medido, K, SIN_DIST))
    assert validos.mean() > 0.9
    assert np.median(np.linalg.norm(p1 - verdad, axis=1)[validos]) < 0.25


def test_frontend_usa_el_giro_solo_si_esta_activado_y_se_lo_dan():
    img = textura(semilla=3)
    img2, _ = girar_imagen(img, R_GRANDE)

    # Activado y con giro: sigue casi todos, el giro explica ~115 px y Lucas-Kanade corrige muy poco
    frontend = fe.FrontEnd(K, SIN_DIST, CFG)
    assert frontend.usa_giro
    frontend.procesar(img, 0)
    est = frontend.procesar(img2, 50_000_000, R_GRANDE)["estadisticas"]
    assert est["con_giro"] == 1
    assert 100.0 < est["giro_px"] < 130.0
    assert est["resto_px"] < 0.5
    seguidos_con_giro = est["seguidos"]

    # Activado pero sin giro (p. ej. la primera imagen del filtro): como antes
    frontend = fe.FrontEnd(K, SIN_DIST, CFG)
    frontend.procesar(img, 0)
    est = frontend.procesar(img2, 50_000_000)["estadisticas"]
    assert est["con_giro"] == 0
    seguidos_sin_giro = est["seguidos"]

    # Desactivado en la configuración: aunque le den el giro, no lo usa
    cfg = copy.deepcopy(CFG)
    cfg["seguimiento"]["usar_giro"] = False
    frontend = fe.FrontEnd(K, SIN_DIST, cfg)
    assert not frontend.usa_giro
    frontend.procesar(img, 0)
    est = frontend.procesar(img2, 50_000_000, R_GRANDE)["estadisticas"]
    assert est["con_giro"] == 0

    assert seguidos_con_giro > 2 * max(seguidos_sin_giro, 1)