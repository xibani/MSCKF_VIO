"""tests/test_calibracion_imu.py — Tests de vio/calibracion_imu.py."""

import numpy as np
import pytest

from vio import calibracion_imu as ci
from vio import so3
from vio.constantes import G

SESGO = np.array([0.12, -0.25, 0.30])           # m/s²
ESCALA = np.array([0.99, 1.01, 0.97])


def medir(eje, signo, inclinacion_rad=0.0):
    """Lo que mediría un acelerómetro con SESGO y ESCALA en esa posición (sin ruido)."""
    arriba = np.zeros(3)
    arriba[eje] = signo                         # dirección "arriba" en ejes de la IMU
    if inclinacion_rad:
        otro = (eje + 1) % 3                    # una inclinación pequeña sobre otro eje
        eje_giro = np.zeros(3)
        eje_giro[otro] = inclinacion_rad
        arriba = so3.exp_so3(eje_giro) @ arriba
    return ESCALA * (G * arriba) + SESGO


def medias_6_posiciones(inclinacion_rad=0.0):
    return {nombre: medir(eje, signo, inclinacion_rad) for nombre, eje, signo in ci.POSICIONES}


def test_recupera_sesgo_y_escala_exactos():
    sesgo, escala = ci.sesgo_y_escala(medias_6_posiciones(), G)
    assert sesgo == pytest.approx(SESGO, abs=1e-12)
    assert escala == pytest.approx(ESCALA, abs=1e-12)


def test_tolerante_a_posiciones_algo_inclinadas():
    # 3° de inclinación: error de escala ~ (1 - cos 3°) ≈ 0,14 %
    sesgo, escala = ci.sesgo_y_escala(medias_6_posiciones(np.radians(3.0)), G)
    assert escala == pytest.approx(ESCALA, rel=0.002)
    assert sesgo == pytest.approx(SESGO, abs=0.03)


def test_corregida_mide_g_en_todas_las_posiciones():
    medias = medias_6_posiciones()
    sesgo, escala = ci.sesgo_y_escala(medias, G)
    for nombre, a in medias.items():
        assert np.linalg.norm(ci.corregir_acelerometro(a, sesgo, escala)) == pytest.approx(G)


def test_corregir_vale_para_muchas_muestras():
    a = np.tile(medir(2, +1), (100, 1))         # (100, 3)
    corregidas = ci.corregir_acelerometro(a, SESGO, ESCALA)
    assert corregidas.shape == (100, 3)
    assert corregidas[0] == pytest.approx([0.0, 0.0, G])


def test_comprobar_posicion():
    ok, coseno = ci.comprobar_posicion(medir(1, -1), eje=1, signo=-1, coseno_min=0.98)
    assert ok and coseno > 0.99
    ok, _ = ci.comprobar_posicion(medir(1, -1), eje=1, signo=+1, coseno_min=0.98)
    assert not ok                               # eje correcto, sentido equivocado
    ok, _ = ci.comprobar_posicion(medir(0, +1), eje=1, signo=-1, coseno_min=0.98)
    assert not ok                               # otro eje


def test_guardar_y_cargar(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    monkeypatch.setattr(ci, "RAIZ_PROYECTO", tmp_path)
    ci.guardar_calibracion_imu(tmp_path / "config" / "prueba.yaml", SESGO, ESCALA, {"nota": "x"})
    sesgo, escala = ci.cargar_calibracion_imu("prueba")
    assert np.array_equal(sesgo, SESGO) and np.array_equal(escala, ESCALA)