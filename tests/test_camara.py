"""tests/test_camara.py — Tests de las funciones puras de captura/camara.py (sin cámara)."""

import numpy as np
import pytest

from captura import camara


def test_periodo_us():
    assert camara.periodo_us(20) == 50_000
    assert camara.periodo_us(30) == 33_333


def test_plano_y_sin_relleno():
    # YUV420 de 640x480: 480 filas de Y (valor 100) + 240 filas de U y V (valor 7)
    yuv = np.full((720, 640), 7, dtype=np.uint8)
    yuv[:480, :] = 100
    gris = camara.plano_y(yuv, 640, 480)
    assert gris.shape == (480, 640)
    assert gris.dtype == np.uint8
    assert np.all(gris == 100)


def test_plano_y_con_relleno_por_fila():
    # Algunas cámaras entregan filas más largas que la imagen (relleno a la derecha)
    yuv = np.zeros((720, 704), dtype=np.uint8)
    yuv[:480, :640] = 100
    yuv[:480, 640:] = 255                    # relleno: no debe aparecer en la imagen
    gris = camara.plano_y(yuv, 640, 480)
    assert gris.shape == (480, 640)
    assert np.all(gris == 100)


def test_plano_y_es_una_copia():
    yuv = np.zeros((720, 640), dtype=np.uint8)
    gris = camara.plano_y(yuv, 640, 480)
    yuv[0, 0] = 50                           # la cámara reutiliza su búfer...
    assert gris[0, 0] == 0                   # ...y nuestra imagen no cambia


def test_exposicion_dentro_de_limites_no_cambia():
    exp, gan, brillo = camara.exposicion_fija(5000, 2.0, exp_max_us=8000, ganancia_max=8.0)
    assert (exp, gan) == (5000, 2.0)
    assert brillo == pytest.approx(1.0)


def test_exposicion_larga_se_compensa_con_ganancia():
    exp, gan, brillo = camara.exposicion_fija(20000, 1.5, exp_max_us=8000, ganancia_max=8.0)
    assert exp == 8000
    assert gan == pytest.approx(1.5 * 20000 / 8000)       # 3.75
    assert brillo == pytest.approx(1.0)                   # mismo brillo


def test_ganancia_limitada_oscurece():
    exp, gan, brillo = camara.exposicion_fija(32000, 4.0, exp_max_us=8000, ganancia_max=8.0)
    assert exp == 8000
    assert gan == 8.0                                     # habría hecho falta 16
    assert brillo == pytest.approx(0.5)                   # la imagen queda a la mitad de brillo


def test_controles_libcamera_reciente():
    disponibles = {"ExposureTime": None, "AnalogueGain": None, "ExposureTimeMode": None,
                   "AnalogueGainMode": None, "AwbEnable": None, "ColourGains": None}
    c = camara.controles_manuales(disponibles, 8000, 3.75, (1.8, 1.5))
    assert c == {"ExposureTime": 8000, "AnalogueGain": 3.75,
                 "ExposureTimeMode": camara.MODO_MANUAL, "AnalogueGainMode": camara.MODO_MANUAL,
                 "AwbEnable": False, "ColourGains": (1.8, 1.5)}


def test_controles_libcamera_antiguo():
    disponibles = {"ExposureTime": None, "AnalogueGain": None, "AeEnable": None,
                   "AwbEnable": None, "ColourGains": None}
    c = camara.controles_manuales(disponibles, 8000, 3.75, None)
    assert c == {"ExposureTime": 8000, "AnalogueGain": 3.75,
                 "AeEnable": False, "AwbEnable": False}


def test_controles_sin_forma_de_fijar_exposicion():
    with pytest.raises(RuntimeError):
        camara.controles_manuales({"ExposureTime": None}, 8000, 1.0, None)