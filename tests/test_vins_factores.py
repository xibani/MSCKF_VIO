"""tests/test_vins_factores.py — Etapa 34: jacobianos de vins/factores.py frente a diferencias
finitas en la variedad (perturbando con vio.msckf.inyectar_imu), y pérdidas robustas."""

import numpy as np
import pytest

from vio import so3
from vio.config import cargar_config
from vio.estado import EstadoImu
from vio.msckf import inyectar_imu
from vins import factores as fa
from vins import preintegracion as pi

CFG_SIM = cargar_config("simulador")["vio"]
R_I_C = np.array(CFG_SIM["camara"]["R_i_c"], dtype=float)
P_I_C = np.array(CFG_SIM["camara"]["p_i_c_m"], dtype=float)
EPS = 1e-6


def estado_al_azar(rng):
    """Un estado cualquiera, no trivial."""
    return EstadoImu(so3.exp_so3(rng.normal(0.0, 1.0, 3)), rng.normal(0.0, 1.0, 3),
                     rng.normal(0.0, 2.0, 3), rng.normal(0.0, 0.01, 3), rng.normal(0.0, 0.1, 3))


def jacobiano_numerico(funcion, estado, n_salida):
    """∂funcion/∂(error de estado) por diferencias centradas en la variedad. (n_salida, 15)"""
    J = np.zeros((n_salida, pi.DIM))
    for k in range(pi.DIM):
        d = np.zeros(pi.DIM)
        d[k] = EPS
        J[:, k] = (funcion(inyectar_imu(estado, d)) - funcion(inyectar_imu(estado, -d))) / (2 * EPS)
    return J


def preintegracion_al_azar(rng, b_g, b_a, duracion_s=0.5, f_hz=200):
    """Preintegración de medidas cualesquiera (giro y aceleración que cambian)."""
    pre = pi.Preintegracion(b_g, b_a, CFG_SIM["imu"])
    n = int(duracion_s * f_hz)
    w0, a0 = rng.normal(0.0, 0.8, 3), rng.normal(0.0, 2.0, 3) + [0.0, 0.0, 9.8]
    for k in range(n):
        t = k / f_hz
        pre.anadir(1.0 / f_hz, w0 + 0.5 * np.sin(3 * t), a0 + np.cos(2 * t))
    return pre


# =====================================================================
# Factor IMU
# =====================================================================
@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_factor_imu_jacobianos(semilla):
    rng = np.random.default_rng(semilla)
    e_i, e_j = estado_al_azar(rng), estado_al_azar(rng)
    # La preintegración se hizo con OTROS sesgos: así se prueba también la corrección de
    # primer orden (J_r(J_R_bg·δb) en ∂r_R/∂b_g)
    pre = preintegracion_al_azar(rng, e_i.b_g + rng.normal(0.0, 0.005, 3), e_i.b_a + rng.normal(0.0, 0.05, 3))
    f = fa.FactorImu(0, 1, pre)
    _, J_i, J_j = f.evaluar(e_i, e_j)

    J_i_num = jacobiano_numerico(lambda e: f.evaluar(e, e_j)[0], e_i, pi.DIM)
    J_j_num = jacobiano_numerico(lambda e: f.evaluar(e_i, e)[0], e_j, pi.DIM)
    assert J_i == pytest.approx(J_i_num, abs=1e-6)
    assert J_j == pytest.approx(J_j_num, abs=1e-6)


def test_factor_imu_residuo_nulo_con_la_prediccion():
    # Si el estado j es exactamente la predicción de la preintegración, el residuo es 0
    rng = np.random.default_rng(5)
    e_i = estado_al_azar(rng)
    pre = preintegracion_al_azar(rng, e_i.b_g, e_i.b_a)
    e_j = pre.predecir(e_i)
    r, _, _ = fa.FactorImu(0, 1, pre).evaluar(e_i, e_j)
    assert np.abs(r).max() < 1e-12
    W = fa.FactorImu(0, 1, pre).informacion()
    assert np.all(np.linalg.eigvalsh(W) > 0)


# =====================================================================
# Factor visual (punto de 3 parámetros θ = (α, β, ρ)) y factor del ancla
# =====================================================================
def escena_visual(rng):
    """Ancla h, keyframe j cercano y un punto delante de las dos cámaras."""
    e_h = estado_al_azar(rng)
    e_j = e_h.copia()
    e_j.R_w_i = e_h.R_w_i @ so3.exp_so3(rng.normal(0.0, 0.1, 3))
    e_j.p_w = e_h.p_w + rng.normal(0.0, 0.2, 3)
    # Punto a 2–4 m delante de la cámara h, cerca del eje óptico
    P_ch = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 1.0]) * rng.uniform(2.0, 4.0)
    theta = np.array([P_ch[0] / P_ch[2], P_ch[1] / P_ch[2], 1.0 / P_ch[2]])
    c = fa.punto_en_camara_j(e_h, e_j, theta, R_I_C, P_I_C)
    assert c["P_cj"][2] > 0.5
    xy_j = c["P_cj"][:2] / c["P_cj"][2] + rng.normal(0.0, 0.002, 2)
    return e_h, e_j, theta, xy_j


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_factor_visual_jacobianos(semilla):
    rng = np.random.default_rng(semilla)
    e_h, e_j, theta, xy_j = escena_visual(rng)
    f = fa.FactorVisual(0, 1, xy_j)
    _, J_h, J_j, J_p, _ = f.evaluar(e_h, e_j, theta, R_I_C, P_I_C)

    J_h_num = jacobiano_numerico(lambda e: f.evaluar(e, e_j, theta, R_I_C, P_I_C)[0], e_h, 2)
    J_j_num = jacobiano_numerico(lambda e: f.evaluar(e_h, e, theta, R_I_C, P_I_C)[0], e_j, 2)
    J_p_num = np.zeros((2, 3))
    for k in range(3):
        d = np.zeros(3)
        d[k] = EPS
        J_p_num[:, k] = (f.evaluar(e_h, e_j, theta + d, R_I_C, P_I_C)[0]
                         - f.evaluar(e_h, e_j, theta - d, R_I_C, P_I_C)[0]) / (2 * EPS)
    assert J_h == pytest.approx(J_h_num, abs=1e-7)
    assert J_j == pytest.approx(J_j_num, abs=1e-7)
    assert J_p == pytest.approx(J_p_num, abs=1e-7)
    # Velocidad y sesgos no intervienen
    for J in (J_h, J_j):
        assert np.all(J[:, pi.V] == 0) and np.all(J[:, pi.BG] == 0) and np.all(J[:, pi.BA] == 0)


def test_factor_visual_residuo_nulo_sin_ruido():
    rng = np.random.default_rng(7)
    e_h, e_j, theta, _ = escena_visual(rng)
    c = fa.punto_en_camara_j(e_h, e_j, theta, R_I_C, P_I_C)
    r, *_ = fa.FactorVisual(0, 1, c["P_cj"][:2] / c["P_cj"][2]).evaluar(e_h, e_j, theta, R_I_C, P_I_C)
    assert np.abs(r).max() < 1e-14


def test_factor_ancla():
    # En la cámara h, P_ch = [α, β, 1]/ρ se proyecta en (α, β) sea cual sea ρ
    f = fa.FactorAncla(0, [0.1, -0.2])
    r, J = f.evaluar(np.array([0.12, -0.21, 0.4]))
    assert r == pytest.approx([0.02, -0.01], abs=1e-15)
    assert J == pytest.approx(np.array([[1, 0, 0], [0, 1, 0]]))


# =====================================================================
# Prior de gauge
# =====================================================================
def test_prior_gauge_jacobiano_y_direcciones():
    rng = np.random.default_rng(3)
    e = estado_al_azar(rng)
    R0 = e.R_w_i @ so3.exp_so3(rng.normal(0.0, 0.05, 3))
    g = fa.FactorPriorGauge(0, R0, e.p_w + 0.1, 1e-3, 1e-3)
    _, J = g.evaluar(e)
    assert J == pytest.approx(jacobiano_numerico(lambda x: g.evaluar(x)[0], e, 4), abs=1e-7)

    # Un giro de gauge (alrededor de z del mundo) cambia solo el residuo de rumbo, en ψ
    psi = 0.3
    e_rot = e.copia()
    e_rot.R_w_i = so3.exp_so3([0.0, 0.0, psi]) @ e.R_w_i
    g0 = fa.FactorPriorGauge(0, e.R_w_i, e.p_w, 1e-3, 1e-3)
    assert g0.evaluar(e_rot)[0] == pytest.approx([psi, 0.0, 0.0, 0.0], abs=1e-12)
    # Y una inclinación pura (alrededor de x del mundo) no toca el rumbo
    e_inc = e.copia()
    e_inc.R_w_i = so3.exp_so3([0.2, 0.0, 0.0]) @ e.R_w_i
    assert abs(g0.evaluar(e_inc)[0][0]) < 1e-12


# =====================================================================
# Pérdidas robustas
# =====================================================================
@pytest.mark.parametrize("tipo", ["ninguna", "huber", "cauchy"])
@pytest.mark.parametrize("s", [0.3, 2.0, 10.0, 100.0])
def test_peso_es_la_derivada_del_coste(tipo, s):
    c = 2.0
    num = (fa.coste_robusto(s + 1e-6, tipo, c) - fa.coste_robusto(s - 1e-6, tipo, c)) / 2e-6
    assert fa.peso_robusto(s, tipo, c) == pytest.approx(num, rel=1e-6)
    assert fa.coste_robusto(s, tipo, c) <= s + 1e-12          # nunca pesa más que la cuadrática