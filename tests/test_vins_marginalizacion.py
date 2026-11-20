"""tests/test_vins_marginalizacion.py — Etapa 35: vins/marginalizacion.py y los priores.

    1. jacobianos de los priores (marginalización y estado inicial) frente a diferencias finitas
    2. el prior reproduce H* y b* (J_pᵀJ_p = H*, J_pᵀr0 = b*)
    3. problema lineal: el paso de Gauss-Newton con el prior es EXACTAMENTE el del problema completo
    4. problema no lineal: marginalizar en el óptimo y optimizar el resto lleva al mismo óptimo

Usa el escenario de la etapa 34 (10 keyframes del simulador) de tests/test_vins_optimizador.py.
"""

import numpy as np
import pytest

from vio import so3
from vio.msckf import diferencia_imu, inyectar_imu
from vins import factores as fa
from vins import marginalizacion as mg
from vins import optimizador as op
from vins import preintegracion as pi
from tests.test_vins_optimizador import K, CFG_VINS, cfg_test, construir_problema, errores, escenario, perturbar
from tests.test_vins_factores import estado_al_azar

EPS_REL = CFG_VINS["marginalizacion"]["eps_autovalor_relativo"]


def jacobianos_numericos(prior, estados, eps=1e-6):
    """∂r/∂(error de cada keyframe del prior), por diferencias centradas en la variedad."""
    r0, _ = prior.evaluar_bloques(estados)
    bloques = []
    for k in range(len(estados)):
        J = np.zeros((len(r0), pi.DIM))
        for c in range(pi.DIM):
            d = np.zeros(pi.DIM)
            d[c] = eps
            mas = [inyectar_imu(e, d) if q == k else e for q, e in enumerate(estados)]
            menos = [inyectar_imu(e, -d) if q == k else e for q, e in enumerate(estados)]
            J[:, c] = (prior.evaluar_bloques(mas)[0] - prior.evaluar_bloques(menos)[0]) / (2 * eps)
        bloques.append(J)
    return bloques


@pytest.fixture(scope="module")
def esc():
    return escenario(np.random.default_rng(10))


@pytest.fixture(scope="module")
def optimo(esc):
    """El problema completo de 10 keyframes, optimizado A FONDO: con las tolerancias del YAML
    (bajada relativa < 1e-6) queda un paso de Gauss-Newton de ~5e-5 m sin dar, y ese gradiente
    residual pasaría a b*; aquí se exige un óptimo de verdad (paso restante ~1e-8)."""
    estados, thetas = perturbar(esc, np.random.default_rng(2))
    prob = construir_problema(esc, estados, thetas)
    cfg = cfg_test()
    cfg["optimizador"]["tolerancia_coste_rel"] = 1e-14
    cfg["optimizador"]["tolerancia_paso"] = 1e-12
    opt = op.OptimizadorLote(cfg, K[0, 0])
    opt.optimizar(prob, iteraciones_max=30)
    return prob, opt


def partir(prob, k_m=0):
    """Separa los factores que tocan al keyframe k_m (y a los puntos anclados en él) del resto.

    Devuelve: (sub, resto, puntos_m) — dos ProblemaLote que comparten estados y puntos.
    """
    puntos_m = {l for l, pt in enumerate(prob.puntos) if pt.ancla == k_m}
    imu_m = [f for f in prob.factores_imu if k_m in (f.i, f.j)]
    imu_r = [f for f in prob.factores_imu if k_m not in (f.i, f.j)]
    vis_m = [f for f in prob.factores_visuales if f.punto in puntos_m]
    vis_r = [f for f in prob.factores_visuales if f.punto not in puntos_m]
    pri_m = [p for p in prob.priores if k_m in p.indices]
    pri_r = [p for p in prob.priores if k_m not in p.indices]
    sub = op.ProblemaLote(prob.estados, prob.puntos, imu_m, vis_m, None, prob.R_i_c, prob.p_i_c, priores=pri_m)
    resto = op.ProblemaLote(prob.estados, prob.puntos, imu_r, vis_r, None, prob.R_i_c, prob.p_i_c, priores=pri_r)
    resto.thetas = prob.thetas.copy()
    sub.thetas = prob.thetas.copy()
    return sub, resto, puntos_m


# =====================================================================
# 1. Jacobianos de los priores
# =====================================================================
def test_prior_marginal_jacobianos():
    rng = np.random.default_rng(0)
    x_lin = [estado_al_azar(rng) for _ in range(3)]
    J = rng.normal(0.0, 1.0, (40, 45))
    prior = mg.FactorPriorMarginal([0, 1, 2], x_lin, J, rng.normal(0.0, 1.0, 40))
    # Lejos del punto de linealización, para que J_r⁻¹ no sea la identidad
    estados = [inyectar_imu(e, rng.normal(0.0, 0.2, pi.DIM)) for e in x_lin]
    _, Js = prior.evaluar_bloques(estados)
    for J_a, J_n in zip(Js, jacobianos_numericos(prior, estados)):
        assert J_a == pytest.approx(J_n, abs=1e-6)
    # En x_lin, el residuo es r0
    assert prior.evaluar_bloques(x_lin)[0] == pytest.approx(prior.r0, abs=1e-14)


def test_prior_estado_jacobiano():
    rng = np.random.default_rng(1)
    x0 = estado_al_azar(rng)
    prior = fa.FactorPriorEstado(0, x0, np.full(15, 0.1))
    e = inyectar_imu(x0, rng.normal(0.0, 0.3, pi.DIM))
    _, (J,) = prior.evaluar_bloques([e])
    assert J == pytest.approx(jacobianos_numericos(prior, [e])[0], abs=1e-7)
    assert prior.evaluar_bloques([x0])[0] == pytest.approx(np.zeros(15), abs=1e-14)


# =====================================================================
# 2. El prior reproduce H* y b*
# =====================================================================
def test_prior_reproduce_h_y_b(esc, optimo):
    prob, opt = optimo
    sub, _, _ = partir(prob)
    H_e, b_e, idx_r = mg.marginalizar(opt.linealizar(sub), [0], prob.n_estados, EPS_REL)
    prior = mg.prior_desde_schur(H_e, b_e, idx_r, prob.estados, EPS_REL)

    assert idx_r[0] == 1                                       # el IMU con el siguiente siempre
    assert prior.J.T @ prior.J == pytest.approx(H_e, rel=1e-8, abs=1e-8 * np.abs(H_e).max())
    assert prior.J.T @ prior.r0 == pytest.approx(b_e, abs=1e-8 * max(1.0, np.abs(b_e).max()))
    assert np.all(np.linalg.eigvalsh(H_e) > -1e-9 * np.abs(H_e).max())


# =====================================================================
# 3. Problema lineal: paso de Gauss-Newton idéntico
# =====================================================================
def test_paso_lineal_igual_con_y_sin_marginalizar(esc):
    """Linealizado en un punto cualquiera (no el óptimo), el problema es cuadrático: el paso
    de Gauss-Newton de los keyframes que quedan con [resto + prior] es el mismo que el del
    problema completo (identidad de Schur), salvo redondeo."""
    estados, thetas = perturbar(esc, np.random.default_rng(3), sigma_p=0.02, sigma_th_grados=0.5)
    prob = construir_problema(esc, estados, thetas)
    opt = op.OptimizadorLote(cfg_test(), K[0, 0])

    # a) Completo
    dx_c, _ = opt.resolver_schur(opt.linealizar(prob), mu=0.0)

    # b) Marginalizar el keyframe 0 (y sus puntos) en el MISMO punto y resolver el resto
    sub, resto, _ = partir(prob)
    H_e, b_e, idx_r = mg.marginalizar(opt.linealizar(sub), [0], prob.n_estados, EPS_REL)
    resto.priores.append(mg.prior_desde_schur(H_e, b_e, idx_r, prob.estados, EPS_REL))
    lin = opt.linealizar(resto)
    # El keyframe 0 ya no tiene factores: se quita de las ecuaciones antes de resolver
    quedan = np.arange(pi.DIM, pi.DIM * prob.n_estados)
    lin_r = {"H_xx": lin["H_xx"][np.ix_(quedan, quedan)], "H_xl": lin["H_xl"][quedan],
             "H_ll": lin["H_ll"], "b_x": lin["b_x"][quedan], "b_l": lin["b_l"]}
    dx_r, _ = opt.resolver_schur(lin_r, mu=0.0)

    assert np.abs(lin["H_xx"][: pi.DIM]).max() == 0.0          # de verdad no queda nada en el 0
    assert np.abs(dx_r - dx_c[pi.DIM:]).max() < 1e-7 * np.abs(dx_c).max()


# =====================================================================
# 4. Problema no lineal: el mismo óptimo
# =====================================================================
def test_marginalizar_en_el_optimo_conserva_el_optimo(esc, optimo):
    """En el óptimo del problema completo, el gradiente respecto al keyframe 0 y a sus
    puntos es nulo, así que b* recoge exactamente el gradiente de los factores marginalizados
    sobre el resto: el óptimo de [resto + prior] es el mismo. Se arranca perturbado."""
    prob, opt = optimo
    sub, resto, puntos_m = partir(prob)
    H_e, b_e, idx_r = mg.marginalizar(opt.linealizar(sub), [0], prob.n_estados, EPS_REL)
    prior = mg.prior_desde_schur(H_e, b_e, idx_r, prob.estados, EPS_REL)

    # Problema reducido: sin el keyframe 0 ni sus puntos (índices desplazados en 1)
    quedan_pts = [l for l in range(prob.n_puntos) if l not in puntos_m]
    nuevo_l = {l: q for q, l in enumerate(quedan_pts)}
    puntos = [op.Punto(prob.puntos[l].ancla - 1, prob.puntos[l].xy_ancla, prob.thetas[l]) for l in quedan_pts]
    imu = [fa.FactorImu(f.i - 1, f.j - 1, f.pre) for f in resto.factores_imu]
    vis = [fa.FactorVisual(nuevo_l[f.punto], f.j - 1, f.xy_j) for f in resto.factores_visuales]
    prior.indices = [k - 1 for k in prior.indices]
    rng = np.random.default_rng(4)
    arranque = [inyectar_imu(e, np.r_[rng.normal(0, 0.01, 3), rng.normal(0, 0.05, 3), rng.normal(0, 0.05, 3),
                                      np.zeros(6)]) for e in prob.estados[1:]]
    red = op.ProblemaLote(arranque, puntos, imu, vis, None, prob.R_i_c, prob.p_i_c, priores=[prior])
    red.thetas = prob.thetas[quedan_pts] * [1.0, 1.0, 1.1]
    informe = opt.optimizar(red, iteraciones_max=20)

    assert informe["convergido"]
    assert errores(red.estados, prob.estados[1:])[0] < 1e-6       # posiciones iguales (< 1 µm)
    assert errores(red.estados, prob.estados[1:])[1] < 1e-6       # orientaciones (grados)
    assert np.abs(red.thetas - prob.thetas[quedan_pts]).max() < 1e-6