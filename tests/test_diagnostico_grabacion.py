"""
tests/test_diagnostico_grabacion.py — Los detectores de herramientas/diagnostico_grabacion.py
encuentran fallos sintéticos conocidos (hueco, saturación, IMU congelada, frame repetido, desplazamiento,
tramo sin correcciones, inicio de la divergencia) y el CSV de recuentos de vio/pipeline.py se lee igual
que se escribe.

    python -m pytest tests/test_diagnostico_grabacion.py -v
"""

import cv2
import numpy as np

from herramientas.diagnostico_grabacion import (agrupar_intervalos, analizar_imagenes, detectar_huecos,
                                                detectar_saturacion, inicio_divergencia, primer_sostenido,
                                                rachas_identicas, sumas_por_segundo, tramos_sin_correccion,
                                                velocidad_en_ventana)
from vio.pipeline import CABECERA_RECUENTOS, leer_recuentos, linea_recuentos


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


def test_pico_suelto_de_velocidad_no_cuenta():
    t = np.arange(0, 20, 0.05)
    v = np.full(len(t), 0.2)
    v[(t > 5) & (t < 5.3)] = 3.0                             # pico de 0,3 s: ruido, no divergencia
    v[t > 12] = 2.0                                          # a partir de 12 s, sostenida
    assert primer_sostenido(t, v > 1.0, 2.0) == t[np.flatnonzero(t > 12)[0]]


def test_inicio_de_la_divergencia():
    # Se mueve por la habitación (distancia ~1 m) y a partir de t = 10 s deriva como ½·a·t²
    t = np.arange(0, 30, 0.05)
    distancia = 1.0 + 0.3 * np.sin(t)
    deriva = t > 10
    distancia[deriva] = (1.0 + 0.3 * np.sin(10.0)) + 0.5 * 0.4 * (t[deriva] - 10) ** 2
    t_dist, t_inicio = inicio_divergencia(t, distancia, 3.0, 15.0, 0.2)
    assert 13.0 < t_dist < 13.6                              # 0,84 + ½·0,4·Δt² = 3 m -> Δt ≈ 3,3 s
    assert abs(t_inicio - 10.0) < 0.1                        # el último mínimo (los valles de antes,
                                                             # a 0,7 m, son más hondos y no cuentan)
    assert inicio_divergencia(t, np.ones(len(t)), 3.0, 15.0, 0.2) == (None, None)


def test_tramos_sin_correccion():
    t = np.arange(0, 10, 0.05)
    usadas = np.full(len(t), 5)
    usadas[(t >= 2) & (t < 2.4)] = 0                         # 0,4 s: más corto que el mínimo
    usadas[(t >= 6) & (t < 9)] = 0                           # 3 s solo con IMU
    tramos = tramos_sin_correccion(t, usadas, 1.0)
    assert len(tramos) == 1
    assert np.allclose(tramos[0], [6.0, 8.95])


def test_sumas_por_segundo():
    t = np.array([0.1, 0.5, 1.2, 2.9])
    centros, sumas = sumas_por_segundo(t, np.array([1, 2, 3, 4]))
    assert np.allclose(centros, [0.5, 1.5, 2.5]) and list(sumas) == [3, 3, 4]


def test_csv_de_recuentos_ida_y_vuelta(tmp_path):
    P = np.diag(np.arange(1, 16) * 1e-4)
    r = {"t_ns": 123456789012, "v_w": np.array([3.0, 4.0, 0.0]), "frontend": {"activos": 87},
         "msckf": {"usadas": 0, "cortas": 2, "sin_paralaje": 1, "reproyeccion": 9, "detras": 0,
                   "chi2": 4, "quieto": True, "filas": 0, "giro_puro": False}}
    ruta = tmp_path / "vio_recuentos.csv"
    ruta.write_text(CABECERA_RECUENTOS + linea_recuentos(r, P), encoding="utf-8")
    d = leer_recuentos(ruta)
    assert d["t_ns"][0] == 123456789012 and d["t_ns"].dtype == np.int64
    assert d["activos"][0] == 87 and d["reproyeccion"][0] == 9 and d["chi2"][0] == 4
    assert d["quieto"][0] == 1
    assert d["giro_puro"][0] == 0
    assert abs(d["v_m_s"][0] - 5.0) < 1e-9
    assert abs(d["sigma_p_m"][0] - np.sqrt((7 + 8 + 9) * 1e-4)) < 1e-6   # P[6:9, 6:9]