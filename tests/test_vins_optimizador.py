"""tests/test_vins_optimizador.py — Etapa 34: optimización en lote (vins/optimizador.py) en
una ventana de keyframes del simulador de la etapa 11.

Escenario: la trayectoria de pose_vio (círculo + oscilación + balanceo/cabeceo) con la IMU
del simulador (ruido + paseo de los sesgos), 10 keyframes cada 0,5 s (cada 10 imágenes a
20 Hz) y los puntos de la habitación vistos por al menos dos keyframes.

Todas las observaciones llevan ruido, también la del ANCLA de cada punto: desde la etapa 35
el punto tiene 3 parámetros (α, β, ρ) y esa observación es un residuo más (FactorAncla).
Con el modelo de 1 parámetro de VINS (rayo del ancla exacto), en este mismo escenario el χ²/gdl
salía ~2,1 y el error máximo de posición 3 cm.
"""

import copy

import numpy as np
import pytest

from vio import simulador as sim
from vio import so3
from vio.config import cargar_config
from vio.estado import EstadoImu
from vins import factores as fa
from vins import marginalizacion as mg
from vins import optimizador as op
from vins import preintegracion as pi

CFG_SIM = cargar_config("simulador")["vio"]
CFG_VINS = cargar_config("vins")
K = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])
TAMANO = (640, 480)
T0_S = 2.0              # inicio de la ventana dentro de la trayectoria [s]


# ---------------------------------------------------------------------
# Escenario
# ---------------------------------------------------------------------
def escenario(rng, n_kf=10, muestras_por_kf=100, max_puntos=None):
    """Ventana de n_kf keyframes con su verdad, preintegraciones y observaciones.

    Devuelve: dict con verdad (lista de EstadoImu), intervalos (lista, por tramo, de
    (dt, w, a)), puntos_w (M, 3), obs (lista de (punto, kf, xy)), anclas (M,), xy_anclas (M, 2)
    (observaciones con ruido en el ancla), thetas_verdad (M, 3), R_i_c, p_i_c.
    """
    cfg_tr, cfg_imu, cfg_cam = CFG_SIM["trayectoria"], CFG_SIM["imu"], CFG_SIM["camara"]
    f = cfg_tr["frecuencia_imu_hz"]

    # 1. Trayectoria e IMU con ruido y sesgo variable
    n = (n_kf - 1) * muestras_por_kf + 1
    t_s = T0_S + np.arange(n) / f
    tr = sim.imu_ideal(lambda t: sim.pose_vio(t, cfg_tr), t_s, cfg_tr["paso_derivadas_s"])
    w_m, a_m, b_g, b_a = sim.imu_con_ruido(tr, cfg_imu, f, rng)

    # 2. Estados verdaderos en los keyframes e intervalos de IMU entre ellos
    k_kf = np.arange(n_kf) * muestras_por_kf
    verdad = [EstadoImu(tr["R_w_i"][k].copy(), tr["v_w"][k].copy(), tr["p_w"][k].copy(),
                        b_g[k].copy(), b_a[k].copy()) for k in k_kf]
    intervalos = []
    for a, b in zip(k_kf[:-1], k_kf[1:]):
        intervalos.append([(t_s[k] - t_s[k - 1], 0.5 * (w_m[k - 1] + w_m[k]), 0.5 * (a_m[k - 1] + a_m[k]))
                           for k in range(a + 1, b + 1)])

    # 3. Puntos de la habitación y su proyección (exacta) en cada keyframe
    puntos = sim.puntos_habitacion(CFG_SIM["mundo"]["habitacion_m"], CFG_SIM["mundo"]["puntos_por_superficie"], rng)
    R_c_i, t_c_i = sim.extrinseca(cfg_cam)
    visto = np.zeros((n_kf, len(puntos)), dtype=bool)
    xy = np.zeros((n_kf, len(puntos), 2))
    prof = np.zeros((n_kf, len(puntos)))
    for kf, e in enumerate(verdad):
        R_c_w, t_c_w = sim.pose_camara(e.R_w_i, e.p_w, R_c_i, t_c_i)
        p_c = puntos @ R_c_w.T + t_c_w
        z = p_c[:, 2]
        delante = z > cfg_cam["profundidad_min_m"]
        xy[kf, delante] = p_c[delante, :2] / z[delante, None]
        u = K[0, 0] * xy[kf, :, 0] + K[0, 2]
        v = K[1, 1] * xy[kf, :, 1] + K[1, 2]
        visto[kf] = delante & (u >= 0) & (u < TAMANO[0]) & (v >= 0) & (v < TAMANO[1])
        prof[kf] = z

    # 4. Puntos vistos por al menos dos keyframes (todos, o max_puntos al azar)
    elegidos = np.flatnonzero(visto.sum(axis=0) >= 2)
    if max_puntos is not None and len(elegidos) > max_puntos:
        elegidos = rng.choice(elegidos, size=max_puntos, replace=False)
    sigma_xy = cfg_cam["ruido_px"] / K[0, 0]
    anclas, xy_anclas, thetas, obs = [], [], [], []
    for l, id_p in enumerate(elegidos):
        kfs = np.flatnonzero(visto[:, id_p])
        h = kfs[0]
        anclas.append(h)
        xy_anclas.append(xy[h, id_p] + rng.normal(0.0, sigma_xy, 2))      # con ruido
        thetas.append([xy[h, id_p, 0], xy[h, id_p, 1], 1.0 / prof[h, id_p]])  # verdad
        for j in kfs[1:]:
            obs.append((l, j, xy[j, id_p] + rng.normal(0.0, sigma_xy, 2)))

    R_i_c = np.array(cfg_cam["R_i_c"], dtype=float)
    return {"verdad": verdad, "intervalos": intervalos, "puntos_w": puntos[elegidos], "obs": obs,
            "anclas": np.array(anclas), "xy_anclas": np.array(xy_anclas), "thetas_verdad": np.array(thetas),
            "R_i_c": R_i_c, "p_i_c": np.array(cfg_cam["p_i_c_m"], dtype=float)}


def construir_problema(esc, estados_ini, thetas_ini, ruido_imu=None):
    """ProblemaLote con las preintegraciones hechas con los sesgos INICIALES (los estimados)."""
    ruido_imu = CFG_SIM["imu"] if ruido_imu is None else ruido_imu
    factores_imu = []
    for i, tramo in enumerate(esc["intervalos"]):
        pre = pi.Preintegracion(estados_ini[i].b_g, estados_ini[i].b_a, ruido_imu)
        for dt, w, a in tramo:
            pre.anadir(dt, w, a)
        factores_imu.append(fa.FactorImu(i, i + 1, pre))
    puntos = [op.Punto(int(h), xy_h, np.array(theta, dtype=float))
              for h, xy_h, theta in zip(esc["anclas"], esc["xy_anclas"], thetas_ini)]
    visuales = [fa.FactorVisual(l, int(j), xy_j) for l, j, xy_j in esc["obs"]]
    # Gauge: el keyframe 0 define el mundo (aquí, en su pose verdadera)
    e0 = esc["verdad"][0]
    gauge = fa.FactorPriorGauge(0, e0.R_w_i, e0.p_w, CFG_VINS["gauge"]["sigma_rumbo_rad"],
                                CFG_VINS["gauge"]["sigma_posicion_m"])
    return op.ProblemaLote(estados_ini, puntos, factores_imu, visuales, gauge, esc["R_i_c"], esc["p_i_c"])


def cfg_test(perdida="ninguna"):
    """vins.yaml con el ruido de píxel del simulador (para que el modelo sea el de los datos)."""
    cfg = copy.deepcopy(CFG_VINS)
    cfg["visual"]["ruido_px"] = CFG_SIM["camara"]["ruido_px"]
    cfg["visual"]["perdida"] = perdida
    return cfg


def perturbar(esc, rng, sigma_p=0.10, sigma_th_grados=2.0, sigma_v=0.10, rel_rho=0.30):
    """Estado inicial: la verdad perturbada (poses ±10 cm y 2°, velocidades ±0,1 m/s,
    profundidades ±30 %), sesgos a cero (como si no se conocieran) y el rayo de cada punto =
    su observación en el ancla (como al triangular)."""
    estados = []
    for e in esc["verdad"]:
        estados.append(EstadoImu(e.R_w_i @ so3.exp_so3(rng.normal(0.0, np.radians(sigma_th_grados), 3)),
                                 e.v_w + rng.normal(0.0, sigma_v, 3),
                                 e.p_w + rng.normal(0.0, sigma_p, 3),
                                 np.zeros(3), np.zeros(3)))
    thetas = np.column_stack([esc["xy_anclas"],
                              esc["thetas_verdad"][:, 2] * (1.0 + rng.uniform(-rel_rho, rel_rho, len(esc["xy_anclas"])))])
    return estados, thetas


def errores(estados, verdad):
    """Error máximo de posición [m] y de orientación [°] en la ventana."""
    e_p = max(np.linalg.norm(e.p_w - v.p_w) for e, v in zip(estados, verdad))
    e_th = max(np.degrees(so3.angulo_entre(e.R_w_i, v.R_w_i)) for e, v in zip(estados, verdad))
    return e_p, e_th


@pytest.fixture(scope="module")
def esc():
    return escenario(np.random.default_rng(10))


# =====================================================================
# Tests
# =====================================================================
def test_escenario_razonable(esc):
    assert len(esc["verdad"]) == 10
    assert len(esc["puntos_w"]) >= 300
    assert len(esc["obs"]) >= 1.5 * len(esc["puntos_w"])   # observaciones fuera del ancla


def test_schur_igual_que_resolver_h_completa(esc):
    estados, thetas = perturbar(esc, np.random.default_rng(1))
    prob = construir_problema(esc, estados, thetas)
    opt = op.OptimizadorLote(cfg_test(), K[0, 0])
    lin = opt.linealizar(prob)
    for mu in (0.0, 1e-3):
        dx_s, dl_s = opt.resolver_schur(lin, mu)
        dx_c, dl_c = opt.resolver_completo(lin, mu)
        assert np.abs(dx_s - dx_c).max() < 1e-8 * max(1.0, np.abs(dx_c).max())
        assert np.abs(dl_s - dl_c).max() < 1e-8 * max(1.0, np.abs(dl_c).max())


def test_sin_gauge_quedan_cuatro_direcciones_sin_observar(esc):
    """Con el prior de gauge casi nulo, el sistema reducido tiene 4 autovalores ~0
    (x, y, z y rumbo) y el resto bien separados."""
    prob = construir_problema(esc, [e.copia() for e in esc["verdad"]], esc["thetas_verdad"])
    prob.prior_gauge.W = prob.prior_gauge.W * 1e-30
    opt = op.OptimizadorLote(cfg_test(), K[0, 0])
    lin = opt.linealizar(prob)
    H_red, _, _ = mg.marginalizar(lin, [], prob.n_estados, 1e-12, todos=True)   # solo Schur de los puntos
    # Escalar por la diagonal: los bloques tienen unidades muy distintas
    d = 1.0 / np.sqrt(np.diag(H_red))
    autovalores = np.sort(np.linalg.eigvalsh(H_red * d[:, None] * d[None, :]))
    assert autovalores[3] < 1e-6 * autovalores[4]


def test_converge_a_la_verdad_desde_una_perturbacion(esc):
    estados, thetas = perturbar(esc, np.random.default_rng(2))
    prob = construir_problema(esc, estados, thetas)
    e_p0, e_th0 = errores(prob.estados, esc["verdad"])
    rho_v = esc["thetas_verdad"][:, 2]
    rho0 = np.median(np.abs(prob.thetas[:, 2] / rho_v - 1.0))

    informe = op.OptimizadorLote(cfg_test(), K[0, 0]).optimizar(prob)
    e_p, e_th = errores(prob.estados, esc["verdad"])
    rho = np.median(np.abs(prob.thetas[:, 2] / rho_v - 1.0))

    assert e_p0 > 0.1 and e_th0 > 2.0                   # de verdad empezaba lejos
    assert informe["convergido"] and informe["iteraciones"] < 10
    assert e_p < 0.03 and e_th < 0.5                    # criterio de salida: cm y < 0,5°
    # Profundidades: de ~15 % de error (mediana) a menos del 3 % (1 px a 2–3 m con 0,4 m de base)
    assert rho < 0.03 and rho < rho0 / 5
    # Sesgo del giroscopio: bien observado en 4,5 s de giros
    assert np.linalg.norm(prob.estados[0].b_g - esc["verdad"][0].b_g) < 2e-3


def test_arranques_distintos_llegan_al_mismo_minimo(esc):
    """Lo que se le pide al OPTIMIZADOR (separado de la precisión del estimador): desde dos
    perturbaciones distintas, el mismo mínimo."""
    cfg = cfg_test()
    cfg["optimizador"]["tolerancia_coste_rel"] = 1e-12     # hasta el fondo: es lo que se prueba
    opt = op.OptimizadorLote(cfg, K[0, 0])
    resultados = []
    for semilla in (5, 6):
        estados, thetas = perturbar(esc, np.random.default_rng(semilla))
        prob = construir_problema(esc, estados, thetas)
        resultados.append((opt.optimizar(prob, iteraciones_max=30), prob))
    (inf_a, a), (inf_b, b) = resultados
    assert inf_a["coste_final"] == pytest.approx(inf_b["coste_final"], rel=1e-6)
    assert errores(a.estados, b.estados)[0] < 1e-4      # posiciones iguales a 0,1 mm
    assert np.abs(a.thetas - b.thetas).max() < 1e-4


def test_chi2_por_grado_de_libertad_coherente_con_el_ruido(esc):
    """En el óptimo, Σ rᵀWr ≈ gdl si el modelo de ruido es el de los datos. Con ~1200
    gdl, la desviación típica de χ²/gdl es √(2/gdl) ≈ 0,04: se admite ±0,15."""
    estados, thetas = perturbar(esc, np.random.default_rng(3))
    prob = construir_problema(esc, estados, thetas)
    informe = op.OptimizadorLote(cfg_test(), K[0, 0]).optimizar(prob)
    razon = informe["chi2"] / informe["gdl"]
    assert informe["gdl"] > 1000
    assert 0.85 < razon < 1.15, f"χ²/gdl = {razon:.3f} con {informe['gdl']} gdl"


def test_perdida_huber_converge_igual_sin_outliers(esc):
    estados, thetas = perturbar(esc, np.random.default_rng(4))
    prob = construir_problema(esc, estados, thetas)
    # Con pesos que cambian en cada iteración (IRLS) converge algo más despacio: 9–12 pasos
    informe = op.OptimizadorLote(cfg_test("huber"), K[0, 0]).optimizar(prob, iteraciones_max=20)
    e_p, e_th = errores(prob.estados, esc["verdad"])
    assert informe["convergido"] and e_p < 0.03 and e_th < 0.5