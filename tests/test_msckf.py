"""tests/test_msckf.py — Tests del MSCKF (vio/msckf.py), con el simulador de la etapa 11."""

import copy

import numpy as np
import pytest
from scipy.stats import chi2

from vio import msckf as ms
from vio import simulador as sim
from vio import so3
from vio.config import cargar_config
from vio.estado import EstadoImu
from vio.propagacion import propagar

CFG = cargar_config("msckf")
CFG_SIM = cargar_config("simulador")["vio"]
K = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])


def estado_ejemplo():
    """Un estado cualquiera (no trivial) para los tests de jacobianos."""
    return EstadoImu(so3.exp_so3([0.3, -0.2, 1.0]), np.array([0.5, -0.2, 0.1]), np.array([1.0, 2.0, 0.5]),
                     np.array([0.01, -0.02, 0.005]), np.array([0.1, -0.05, 0.2]))


def covarianza_al_azar(n, semilla):
    """Una matriz simétrica y positiva cualquiera (n x n)."""
    A = np.random.default_rng(semilla).normal(0.0, 1.0, (n, n))
    return A @ A.T / n + 0.1 * np.eye(n)


# =====================================================================
# Paso 12.1
# =====================================================================
def test_inyectar_y_diferencia_son_inversas():
    e = estado_ejemplo()
    dx = np.random.default_rng(0).normal(0.0, 0.01, 15)
    assert ms.diferencia_imu(ms.inyectar_imu(e, dx), e) == pytest.approx(dx, abs=1e-12)


def test_matrices_error():
    e = estado_ejemplo()
    w_hat, a_hat = np.array([0.3, -0.5, 0.8]), np.array([1.0, 0.5, 9.5])
    F, G = ms.matrices_error(e.R_w_i, w_hat, a_hat)
    assert F.shape == (15, 15) and G.shape == (15, 12)
    assert F[ms.TH, ms.TH] == pytest.approx(-so3.hat(w_hat))
    assert F[ms.V, ms.TH] == pytest.approx(-e.R_w_i @ so3.hat(a_hat))
    assert F[ms.P, ms.V] == pytest.approx(np.eye(3))
    assert np.abs(F[ms.BG]).max() == 0 and np.abs(F[ms.BA]).max() == 0    # los sesgos no dependen de nada


def test_phi_igual_que_diferencias_finitas():
    # Perturbar el estado, propagar un paso y ver cómo cambia el error: columna a columna de Phi
    e = estado_ejemplo()
    w_m, a_m = np.array([0.3, -0.5, 0.8]), np.array([1.0, 0.5, 9.5])
    discrepancias = []
    for dt in (0.005, 0.0025):
        F, G = ms.matrices_error(e.R_w_i, w_m - e.b_g, a_m - e.b_a)
        Phi, _ = ms.transicion(F, G, np.zeros((12, 12)), dt)
        nominal = propagar(e, w_m, a_m, dt)
        eps = 1e-6
        Phi_num = np.zeros((15, 15))
        for k in range(15):
            d = np.zeros(15)
            d[k] = eps
            mas = propagar(ms.inyectar_imu(e, d), w_m, a_m, dt)
            menos = propagar(ms.inyectar_imu(e, -d), w_m, a_m, dt)
            Phi_num[:, k] = (ms.diferencia_imu(mas, nominal) - ms.diferencia_imu(menos, nominal)) / (2 * eps)
        discrepancias.append(np.abs(Phi - Phi_num).max())
        assert np.abs(Phi_num - np.eye(15)).max() > 100 * discrepancias[-1]   # el test no es trivial
    # Coinciden salvo términos en dt²: al dividir dt por 2, la discrepancia baja a la cuarta parte
    assert discrepancias[0] < 2e-4
    assert discrepancias[1] / discrepancias[0] == pytest.approx(0.25, abs=0.02)


def test_propagar_covarianza_con_clones():
    P = covarianza_al_azar(15 + 12, 1)                         # IMU + 2 clones
    e = estado_ejemplo()
    F, G = ms.matrices_error(e.R_w_i, np.array([0.3, -0.5, 0.8]), np.array([1.0, 0.5, 9.5]))
    Phi, Qd = ms.transicion(F, G, ms.ruido_continuo(CFG["imu"]), 0.005)
    P2 = ms.propagar_covarianza(P, Phi, Qd)
    assert np.abs(P2 - P2.T).max() < 1e-15 and np.linalg.eigvalsh(P2).min() > 0
    assert P2[15:, 15:] == pytest.approx(P[15:, 15:])         # los clones no cambian
    assert P2[:15, 15:] == pytest.approx(Phi @ P[:15, 15:])   # su relación con la IMU, sí
    assert np.trace(P2[:15, :15]) > np.trace(Phi @ P[:15, :15] @ Phi.T)   # Q_d añade incertidumbre


def test_nees_solo_imu():
    # Solo IMU: el error real de [δθ, δv, δp] debe tener el tamaño que dice P (NEES ≈ 9).
    # P0 muy pequeña: así el error lo produce el ruido de la IMU y se pone a prueba Q_d
    tr_cfg = CFG_SIM["trayectoria"]
    t_ns = sim.tiempos_ns(3.0, 200)
    tr = sim.imu_ideal(lambda t: sim.pose_vio(t, tr_cfg), t_ns / 1e9, tr_cfg["paso_derivadas_s"])
    P0 = ms.covarianza_inicial(CFG["inicial"]) * 1e-6
    Qc = ms.ruido_continuo(CFG["imu"])
    rng = np.random.default_rng(0)
    n_sim = 30
    suma = 0.0
    for _ in range(n_sim):
        w_m, a_m, b_g, b_a = sim.imu_con_ruido(tr, CFG_SIM["imu"], 200, rng)
        verdad_0 = EstadoImu(tr["R_w_i"][0], tr["v_w"][0], tr["p_w"][0], b_g[0], b_a[0])
        e = ms.inyectar_imu(verdad_0, -(np.linalg.cholesky(P0) @ rng.normal(0.0, 1.0, 15)))
        P = P0.copy()
        for k in range(1, len(t_ns)):
            dt = (t_ns[k] - t_ns[k - 1]) / 1e9
            w_medio, a_medio = 0.5 * (w_m[k - 1] + w_m[k]), 0.5 * (a_m[k - 1] + a_m[k])
            F, G = ms.matrices_error(e.R_w_i, w_medio - e.b_g, a_medio - e.b_a)
            Phi, Qd = ms.transicion(F, G, Qc, dt)
            P = ms.propagar_covarianza(P, Phi, Qd)
            e = propagar(e, w_medio, a_medio, dt)
        verdad = EstadoImu(tr["R_w_i"][-1], tr["v_w"][-1], tr["p_w"][-1], b_g[-1], b_a[-1])
        error = ms.diferencia_imu(verdad, e)[:9]
        suma += error @ np.linalg.solve(P[:9, :9], error)
    nees = suma / n_sim
    assert chi2.ppf(0.0005, 9 * n_sim) / n_sim < nees < chi2.ppf(0.9995, 9 * n_sim) / n_sim