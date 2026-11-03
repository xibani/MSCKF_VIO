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


# =====================================================================
# Paso 10.2
# =====================================================================
def test_prediccion_y_correccion_con_numeros():
    F, B = kf.matrices(0.1)
    Q = kf.ruido_proceso(B, 0.5)
    x, P = kf.predecir(np.array([1.0, 2.0]), np.eye(2), 3.0, F, B, Q)
    assert x == pytest.approx([1.0 + 0.2 + 0.015, 2.3])            # p + v·dt + ½·u·dt², v + u·dt
    assert P == pytest.approx(F @ F.T + np.outer(B, B) * 0.25)
    # Corrección: con P = I y R = 1, la ganancia de la posición es 1/2
    x2, P2, y, S = kf.corregir(np.array([0.0, 0.0]), np.eye(2), 4.0, 1.0)
    assert y == 4.0 and S == 2.0
    assert x2 == pytest.approx([2.0, 0.0])
    assert P2 == pytest.approx(np.array([[0.5, 0.0], [0.0, 1.0]]))


def test_prediccion_aumenta_P_y_correccion_la_reduce():
    F, B = kf.matrices(0.01)
    Q = kf.ruido_proceso(B, 0.05)
    P = np.diag([0.01, 0.01])
    x = np.zeros(2)
    for _ in range(10):
        P_ant = P
        x, P = kf.predecir(x, P, 0.0, F, B, Q)
        assert np.trace(P) > np.trace(P_ant)
    assert P[0, 1] > 0                                              # p y v ya están relacionados
    _, P_corr, _, _ = kf.corregir(x, P, 0.0, 0.02 ** 2)
    assert P_corr[0, 0] < P[0, 0]
    # La velocidad NO se mide, pero su incertidumbre también baja: gracias a P[0, 1]
    assert P_corr[1, 1] < P[1, 1]


def test_joseph_igual_que_forma_simple():
    P = np.array([[0.04, 0.01], [0.01, 0.09]])
    R = 0.02 ** 2
    _, P_joseph, _, S = kf.corregir(np.zeros(2), P, 1.0, R)
    K = (P @ kf.H.T)[:, 0] / S
    P_simple = (np.eye(2) - np.outer(K, kf.H[0])) @ P
    assert P_joseph == pytest.approx(P_simple, abs=1e-15)


def test_P_simetrica_y_positiva_siempre():
    sim = kf.simular(CFG, np.random.default_rng(1))
    _, P_est = kf.ejecutar_filtro(sim, CFG)
    for P in P_est:
        assert np.abs(P - P.T).max() < 1e-15
        assert np.linalg.eigvalsh(P).min() > 0


def test_R_enorme_ignora_las_medidas():
    sim = kf.simular(CFG, np.random.default_rng(2))
    x_r, P_r = kf.ejecutar_filtro(sim, CFG, factor_r=1e20)
    solo_prediccion = dict(sim, con_medida=np.zeros_like(sim["con_medida"]))   # sin ninguna medida
    x_p, P_p = kf.ejecutar_filtro(solo_prediccion, CFG)
    assert np.abs(x_r - x_p).max() < 1e-9
    assert np.abs(P_r - P_p).max() < 1e-9


def test_R_casi_cero_sigue_a_la_medida():
    sim = kf.simular(CFG, np.random.default_rng(3))
    x_est, P_est = kf.ejecutar_filtro(sim, CFG, factor_r=1e-12)
    m = sim["con_medida"]
    assert np.abs(x_est[m, 0] - sim["z"][m]).max() < 1e-6
    assert P_est[m, 0, 0].max() < 1e-12