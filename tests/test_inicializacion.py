"""tests/test_inicializacion.py — Tests de vio/inicializacion.py."""

import numpy as np
import pytest

from vio import inicializacion as ini
from vio import so3
from vio.config import cargar_config

G = ini.G


def test_estimar_reposo():
    rng = np.random.default_rng(0)
    sesgo = np.radians([0.5, -0.3, 0.1])
    w = sesgo + rng.normal(0.0, np.radians(0.05), (400, 3))
    a = np.array([0.0, 0.0, G]) + rng.normal(0.0, 0.02, (400, 3))
    r = ini.estimar_reposo(w, a)
    assert r["b_g"] == pytest.approx(sesgo, abs=np.radians(0.01))
    assert r["sigma_gir"] == pytest.approx(np.radians([0.05] * 3), abs=np.radians(0.01))
    assert r["norma_a"] == pytest.approx(G, abs=0.01)


def test_comprobar_reposo():
    cfg = cargar_config("orientacion")["inicializacion"]
    quieto = {"sigma_gir": np.radians([0.05] * 3), "norma_a": 9.85}
    assert ini.comprobar_reposo(quieto, cfg) == []
    movido = {"sigma_gir": np.radians([2.0, 0.05, 0.05]), "norma_a": 9.85}
    assert len(ini.comprobar_reposo(movido, cfg)) == 1
    acelerando = {"sigma_gir": np.radians([0.05] * 3), "norma_a": 12.0}
    assert len(ini.comprobar_reposo(acelerando, cfg)) == 1


def test_inclinacion_casos_a_mano():
    # Plana: z hacia arriba
    assert ini.inclinacion_desde_acelerometro([0.0, 0.0, G]) == pytest.approx((0.0, 0.0))
    # Alabeo de +90° (girada sobre x): el eje +y apunta hacia arriba
    assert ini.inclinacion_desde_acelerometro([0.0, G, 0.0]) == pytest.approx((np.pi / 2, 0.0))
    # Cabeceo de +90° (girada sobre y): el eje -x apunta hacia arriba
    assert ini.inclinacion_desde_acelerometro([-G, 0.0, 0.0]) == pytest.approx((0.0, np.pi / 2))


def test_rotacion_desde_gravedad_recupera_la_inclinacion():
    rng = np.random.default_rng(1)
    for _ in range(500):
        alabeo = rng.uniform(-np.pi, np.pi)
        cabeceo = rng.uniform(-1.5, 1.5)
        guinada = rng.uniform(-np.pi, np.pi)             # el rumbo verdadero: no se puede ver
        R_verdadera = (so3.exp_so3([0, 0, guinada]) @ so3.exp_so3([0, cabeceo, 0])
                       @ so3.exp_so3([alabeo, 0, 0]))
        a = R_verdadera.T @ np.array([0.0, 0.0, G])      # lo que mediría el acelerómetro

        R = ini.rotacion_desde_gravedad(a)
        assert so3.es_rotacion(R)
        assert R @ a == pytest.approx([0.0, 0.0, G], abs=1e-9)    # gravedad hacia -z del mundo
        al, ca, gu = so3.angulos_zyx(R)
        assert (al, ca) == pytest.approx((alabeo, cabeceo), abs=1e-9)
        assert gu == pytest.approx(0.0, abs=1e-12)                # rumbo 0 por definición


def test_rotacion_desde_gravedad_boca_abajo_y_vertical():
    for a in ([0.0, 0.0, -G], [-G, 0.0, 0.0], [G, 0.0, 0.0], [0.0, -G, 0.0]):
        R = ini.rotacion_desde_gravedad(np.array(a))
        assert so3.es_rotacion(R)
        assert R @ np.array(a) == pytest.approx([0.0, 0.0, G], abs=1e-9)