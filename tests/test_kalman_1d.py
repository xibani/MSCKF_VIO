"""tests/test_kalman_1d.py — Tests del filtro de Kalman de juguete (practicas/kalman_1d.py)."""

import copy

import numpy as np
import pytest

from practicas import kalman_1d as kf
from vio.config import cargar_config

CFG = cargar_config("kalman_1d")
# Para los tests de consistencia bastan 5 s de simulación (la mitad de tiempo de cálculo)
CFG_CORTA = copy.deepcopy(CFG)
CFG_CORTA["simulacion"]["duracion_s"] = 5.0


# =====================================================================
# Paso 10.1
# =====================================================================
def test_matrices_aceleracion_constante():
    # Con aceleración constante, el modelo discreto es EXACTO: p = ½·a·t², v = a·t
    dt, a, n = 0.01, 2.0, 300
    F, B = kf.matrices(dt)
    x = np.zeros(2)
    for _ in range(n):
        x = F @ x + B * a
    t = n * dt
    assert x == pytest.approx([0.5 * a * t ** 2, a * t], abs=1e-9)


def test_simular():
    rng = np.random.default_rng(0)
    sim = kf.simular(CFG, rng)
    n = int(CFG["simulacion"]["duracion_s"] * CFG["simulacion"]["frecuencia_hz"])
    assert sim["x"].shape == (n + 1, 2) and sim["u"].shape == (n,)
    # Una medida cada 10 pasos, ninguna en t = 0
    assert sim["con_medida"].sum() == n // CFG["simulacion"]["cada_n_medida"]
    assert not sim["con_medida"][0]
    assert np.isnan(sim["z"][~sim["con_medida"]]).all()
    # El ruido de las medidas tiene la desviación configurada
    errores = []
    for _ in range(20):
        s = kf.simular(CFG, rng)
        errores.append(s["z"][s["con_medida"]] - s["x"][s["con_medida"], 0])
    assert np.std(np.concatenate(errores)) == pytest.approx(CFG["ruido"]["sigma_z"], rel=0.1)