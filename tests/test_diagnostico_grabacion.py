"""
tests/test_diagnostico_grabacion.py — Los detectores de herramientas/diagnostico_grabacion.py
encuentran fallos sintéticos conocidos (hueco, saturación, IMU congelada, frame repetido, desplazamiento).

    python -m pytest tests/test_diagnostico_grabacion.py -v
"""

import cv2
import numpy as np

from herramientas.diagnostico_grabacion import (agrupar_intervalos, analizar_imagenes, detectar_huecos,
                                                detectar_saturacion, rachas_identicas, velocidad_en_ventana)


def test_hueco_en_timestamps():
    t = np.arange(100, dtype=np.int64) * 5_000_000          # IMU a 200 Hz
    t[50:] += 300_000_000                                    # se pierden 300 ms entre la muestra 49 y la 50
    _, huecos, dt_mediano, no_monotonos = detectar_huecos(t, 2.5)
    assert len(huecos) == 1
    assert huecos[0][0] == 49
    assert abs(huecos[0][2] - 0.305) < 1e-9
    assert abs(dt_mediano - 0.005) < 1e-12
    assert no_monotonos == 0


def test_timestamps_hacia_atras():
    t = np.array([0, 10, 20, 20, 15, 30], dtype=np.int64)
    _, _, _, no_monotonos = detectar_huecos(t, 2.5)
    assert no_monotonos == 2


def test_saturacion_del_giroscopo():
    w = np.zeros((10, 3))
    w[4, 2] = np.radians(250.0)                              # fondo de escala en z
    w[7, 0] = -np.radians(249.0)                             # 99.6 % del fondo de escala: también cuenta
    marcado = detectar_saturacion(w, np.radians(250.0), 0.98)
    assert list(np.flatnonzero(marcado)) == [4, 7]


def test_imu_congelada():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(30, 6))
    v[10:18] = v[10]                                         # 8 muestras idénticas
    assert rachas_identicas(v, 5) == [(10, 8)]


def test_agrupar_intervalos():
    t = np.arange(10) * 0.05
    marcado = np.array([0, 1, 1, 0, 0, 0, 0, 1, 0, 0], dtype=bool)
    intervalos = agrupar_intervalos(t, marcado, 0.1)
    assert len(intervalos) == 2
    assert np.allclose(intervalos, [[0.05, 0.10], [0.35, 0.35]])


def test_velocidad_en_ventana():
    t = np.arange(0, 3, 0.1)
    p = np.zeros((len(t), 3))
    p[:, 0] = 2.0 * t                                        # 2 m/s en x
    v = velocidad_en_ventana(t, p, 0.5)
    assert np.allclose(v[:-1], 2.0)


def test_imagenes_desplazamiento_y_frame_repetido(tmp_path):
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((260, 360)) * 255).astype(np.uint8), (0, 0), 1.5)
    f0 = base[20:240, 20:340]
    f1 = base[20:240, 22:342]                                # la escena se desplaza 2 px en x
    f2 = f1.copy()                                           # cámara congelada
    nombres = ["0.png", "1.png", "2.png"]
    for nombre, f in zip(nombres, [f0, f1, f2]):
        cv2.imwrite(str(tmp_path / nombre), f)
    res = analizar_imagenes(tmp_path, nombres, 300, 0.01, 10, 21, 3, 1.0)
    assert res["esquinas"][0] > 100
    assert res["seguidos"][1] > 0.8 * res["esquinas"][0]
    assert abs(res["flujo_px"][1] - 2.0) < 0.2
    assert res["diferencia"][2] == 0
    assert res["flujo_px"][2] < 0.05
    assert res["ilegibles"] == []