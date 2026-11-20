"""tests/test_vins_preintegracion.py — Etapa 33: tests de vins/preintegracion.py.

Los cuatro tests del criterio de salida:
    1. preintegrar + aplicar al estado en i == propagar muestra a muestra
    2. corrección de primer orden ≈ reintegrar (error de segundo orden en δb)
    3. jacobianos de sesgo frente a diferencias finitas
    4. covarianza frente a Monte Carlo con el ruido del simulador
más los de apoyo (J_r, umbral de reintegración, ruido del YAML, bloque de sesgos).
"""

import copy

import numpy as np
import pytest

from vio import simulador as sim
from vio import so3
from vio.config import cargar_config
from vio.estado import EstadoImu
from vio.propagacion import propagar
from vins import preintegracion as pi

CFG_SIM = cargar_config("simulador")["vio"]
CFG_VINS = cargar_config("vins")["preintegracion"]

T0_S = 3.0            # inicio del tramo dentro de la trayectoria del simulador [s]
DURACION_S = 1.0      # duración del tramo (unas 5 imágenes a 20 Hz: un intervalo entre keyframes)


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def tramo_simulado(duracion_s, rng, jitter=True):
    """Trayectoria del simulador (etapa 11) en [T0, T0 + duración], con Δt irregulares.

    Devuelve: (t_s (N,), tr dict de imu_ideal).
    """
    f = CFG_SIM["trayectoria"]["frecuencia_imu_hz"]
    n = int(round(duracion_s * f)) + 1
    t_s = T0_S + np.arange(n) / f
    if jitter:
        # ±20 % del periodo en cada muestra (menos en la primera): Δt reales no uniformes
        t_s[1:] += rng.uniform(-0.2, 0.2, n - 1) / f
    pose = lambda t: sim.pose_vio(t, CFG_SIM["trayectoria"])
    tr = sim.imu_ideal(pose, t_s, CFG_SIM["trayectoria"]["paso_derivadas_s"])
    return t_s, tr


def intervalos(t_s, w, a):
    """De muestras a intervalos (dt, media de w, media de a), como Msckf.anadir_imu."""
    lista = []
    for k in range(1, len(t_s)):
        lista.append((t_s[k] - t_s[k - 1], 0.5 * (w[k - 1] + w[k]), 0.5 * (a[k - 1] + a[k])))
    return lista


def preintegrar(lista, b_g, b_a, ruido=None, con_covarianza=True):
    """Preintegración de una lista de intervalos."""
    ruido = CFG_SIM["imu"] if ruido is None else ruido
    pre = pi.Preintegracion(b_g, b_a, ruido, con_covarianza)
    for dt, w, a in lista:
        pre.anadir(dt, w, a)
    return pre


def error_medida(dR_ref, dv_ref, dp_ref, dR, dv, dp):
    """Error (9,) [δθ, δv, δp] de una medida respecto a otra de referencia."""
    return np.concatenate([so3.log_so3(dR.T @ dR_ref), dv_ref - dv, dp_ref - dp])


B_G = np.array([0.012, -0.008, 0.015])   # sesgos de ejemplo (del orden de una MPU-6050)
B_A = np.array([0.15, -0.10, 0.20])


# =====================================================================
# Paso 33.1 — Jacobiano derecho
# =====================================================================
@pytest.mark.parametrize("phi", [[0.3, -0.5, 0.8], [1e-8, 2e-8, -1e-8], [2.0, 1.0, -1.5]])
def test_jacobiano_derecho_frente_a_diferencias_finitas(phi):
    phi = np.array(phi)
    Jr = pi.jacobiano_derecho(phi)
    eps = 1e-6
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        # Exp(φ + δ) = Exp(φ)·Exp(J_r·δ)  =>  J_r[:, k] ≈ Log(Exp(φ)ᵀ·Exp(φ + δ_k)) / ε
        mas = so3.log_so3(so3.exp_so3(phi).T @ so3.exp_so3(phi + d))
        menos = so3.log_so3(so3.exp_so3(phi).T @ so3.exp_so3(phi - d))
        assert (mas - menos) / (2 * eps) == pytest.approx(Jr[:, k], abs=1e-8)
    assert pi.jacobiano_derecho_inv(phi) @ Jr == pytest.approx(np.eye(3), abs=1e-10)


# =====================================================================
# Test 1 — Preintegrar y aplicar == propagar muestra a muestra
# =====================================================================
def test_preintegrar_y_aplicar_igual_que_propagar():
    rng = np.random.default_rng(1)
    t_s, tr = tramo_simulado(DURACION_S, rng)
    # Medidas CON ruido y sesgo variable: la igualdad es algebraica, no depende de la entrada
    w_m, a_m, _, _ = sim.imu_con_ruido(tr, CFG_SIM["imu"], CFG_SIM["trayectoria"]["frecuencia_imu_hz"], rng)
    lista = intervalos(t_s, w_m, a_m)

    # Estado en i: el verdadero, con sesgos estimados cualesquiera (constantes en el tramo)
    e_i = EstadoImu(tr["R_w_i"][0].copy(), tr["v_w"][0].copy(), tr["p_w"][0].copy(), B_G.copy(), B_A.copy())

    # a) Muestra a muestra con vio/propagacion.propagar (lo que hace el MSCKF)
    e = e_i.copia()
    for dt, w, a in lista:
        e = propagar(e, w, a, dt)

    # b) Preintegración con los mismos sesgos, aplicada al estado en i
    pre = preintegrar(lista, B_G, B_A)
    e_j = pre.predecir(e_i)

    assert pre.dt_total == pytest.approx(t_s[-1] - t_s[0], abs=1e-12)
    assert so3.angulo_entre(e_j.R_w_i, e.R_w_i) < 1e-9
    assert np.abs(e_j.v_w - e.v_w).max() < 1e-9
    assert np.abs(e_j.p_w - e.p_w).max() < 1e-9


# =====================================================================
# Test 2 — Corrección de primer orden frente a reintegrar
# =====================================================================
def test_correccion_primer_orden_error_de_segundo_orden():
    rng = np.random.default_rng(2)
    t_s, tr = tramo_simulado(DURACION_S, rng)
    w_m, a_m = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"], b_g=B_G, b_a=B_A)
    lista = intervalos(t_s, w_m, a_m)
    pre = preintegrar(lista, B_G, B_A)

    # Dirección del cambio de sesgo: del orden de los umbrales de reintegración
    dbg = np.array([0.6, -0.8, 0.0]) * CFG_VINS["umbral_reintegrar_bg_rad_s"]
    dba = np.array([0.0, 0.6, -0.8]) * CFG_VINS["umbral_reintegrar_ba_m_s2"]

    errores, cambios = [], []
    for s in (1.0, 0.5, 0.25):
        b_g, b_a = B_G + s * dbg, B_A + s * dba
        exacta = preintegrar(lista, b_g, b_a)                 # reintegrar con el sesgo nuevo
        corr = pre.corregida(b_g, b_a)                        # primer orden
        e = error_medida(exacta.dR, exacta.dv, exacta.dp, *corr)
        c = error_medida(exacta.dR, exacta.dv, exacta.dp, pre.dR, pre.dv, pre.dp)  # sin corregir
        errores.append(np.linalg.norm(e))
        cambios.append(np.linalg.norm(c))

    # Segundo orden: al dividir δb entre 2, el error se divide entre ~4
    assert 3.5 < errores[0] / errores[1] < 4.5
    assert 3.5 < errores[1] / errores[2] < 4.5
    # Y la corrección se lleva casi todo el cambio (> 99 % con los umbrales del YAML)
    assert errores[0] < 0.01 * cambios[0]


# =====================================================================
# Test 3 — Jacobianos de sesgo frente a diferencias finitas
# =====================================================================
def test_jacobianos_de_sesgo_frente_a_diferencias_finitas():
    rng = np.random.default_rng(3)
    t_s, tr = tramo_simulado(DURACION_S, rng)
    w_m, a_m, _, _ = sim.imu_con_ruido(tr, CFG_SIM["imu"], CFG_SIM["trayectoria"]["frecuencia_imu_hz"], rng)
    lista = intervalos(t_s, w_m, a_m)
    pre = preintegrar(lista, B_G, B_A)
    J = pre.jacobiano_sesgos()                                # (9, 6)

    eps = 1e-6
    J_num = np.zeros((9, 6))
    for k in range(6):
        d = np.zeros(6)
        d[k] = eps
        mas = preintegrar(lista, B_G + d[:3], B_A + d[3:], con_covarianza=False)
        menos = preintegrar(lista, B_G - d[:3], B_A - d[3:], con_covarianza=False)
        # δθ en la variedad: Log(ΔR̂ᵀ·ΔR(b ± ε)), igual que el error de la medida
        e_mas = error_medida(mas.dR, mas.dv, mas.dp, pre.dR, pre.dv, pre.dp)
        e_menos = error_medida(menos.dR, menos.dv, menos.dp, pre.dR, pre.dv, pre.dp)
        J_num[:, k] = (e_mas - e_menos) / (2 * eps)

    escala = np.abs(J).max()
    assert np.abs(J - J_num).max() < 1e-6 * escala
    assert np.abs(pre.J_R_bg).max() > 0.5            # no trivial: ~Δt = 1 s
    assert np.abs(J[0:3, 3:6]).max() == 0.0          # ΔR no depende de b_a


# =====================================================================
# Test 4 — Covarianza frente a Monte Carlo
# =====================================================================
def test_covarianza_frente_a_monte_carlo():
    """N realizaciones del ruido del simulador (blanco + paseo de los sesgos) sobre el mismo
    tramo. Error de cada una: medida verdadera (IMU ideal, sin sesgo) ⊟ medida preintegrada
    con el sesgo verdadero en i; más el cambio de sesgo de i a j.

    Criterio (el "< 10 %" del plan):
      - la σ de cada una de las 15 componentes, a menos del 10 % de la predicha
        (con N = 1000 el error estadístico de una σ es ~2,2 %);
      - NEES medio = media de eᵀ·Σ⁻¹·e a menos del 10 % de 15 (comprueba también las
        correlaciones; error estadístico ~1,6 %).
    """
    n_mc = 1000
    duracion = 0.5
    rng = np.random.default_rng(4)
    f = CFG_SIM["trayectoria"]["frecuencia_imu_hz"]
    cfg_imu = copy.deepcopy(CFG_SIM["imu"])
    t_s, tr = tramo_simulado(duracion, rng, jitter=False)

    # Medida verdadera: IMU sin ruido ni sesgo
    w_id, a_id = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"])
    verdad = preintegrar(intervalos(t_s, w_id, a_id), np.zeros(3), np.zeros(3), con_covarianza=False)

    # Covarianza predicha: la misma para todas las realizaciones (solo depende de las medidas
    # y del sesgo de linealización, de forma muy débil); se calcula con la IMU ideal + sesgo
    w_b, a_b = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"],
                               b_g=cfg_imu["sesgo_inicial_gir"], b_a=cfg_imu["sesgo_inicial_acc"])
    pre_ref = preintegrar(intervalos(t_s, w_b, a_b), cfg_imu["sesgo_inicial_gir"],
                          cfg_imu["sesgo_inicial_acc"], ruido=cfg_imu)
    Sigma = pre_ref.cov

    errores = np.zeros((n_mc, pi.DIM))
    for r in range(n_mc):
        w_m, a_m, b_g, b_a = sim.imu_con_ruido(tr, cfg_imu, f, rng)
        pre = preintegrar(intervalos(t_s, w_m, a_m), b_g[0], b_a[0], ruido=cfg_imu, con_covarianza=False)
        errores[r, 0:9] = error_medida(verdad.dR, verdad.dv, verdad.dp, pre.dR, pre.dv, pre.dp)
        errores[r, pi.BG] = b_g[-1] - b_g[0]
        errores[r, pi.BA] = b_a[-1] - b_a[0]

    # σ por componente
    sigma_mc = errores.std(axis=0)
    sigma_pred = np.sqrt(np.diag(Sigma))
    relativo = sigma_mc / sigma_pred - 1.0
    assert np.abs(relativo).max() < 0.10, f"σ_MC/σ_pred − 1 = {np.round(relativo, 3)}"

    # NEES medio
    nees = np.array([e @ np.linalg.solve(Sigma, e) for e in errores])
    assert abs(nees.mean() / pi.DIM - 1.0) < 0.10, f"NEES medio = {nees.mean():.2f} (esperado {pi.DIM})"


# =====================================================================
# Apoyo
# =====================================================================
def test_bloque_de_sesgos_es_el_paseo_aleatorio():
    rng = np.random.default_rng(5)
    t_s, tr = tramo_simulado(DURACION_S, rng)
    w_m, a_m = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"])
    pre = preintegrar(intervalos(t_s, w_m, a_m), np.zeros(3), np.zeros(3))
    T = pre.dt_total
    assert pre.cov[pi.BG, pi.BG] == pytest.approx(CFG_SIM["imu"]["paseo_sesgo_gir"] ** 2 * T * np.eye(3), rel=1e-12)
    assert pre.cov[pi.BA, pi.BA] == pytest.approx(CFG_SIM["imu"]["paseo_sesgo_acc"] ** 2 * T * np.eye(3), rel=1e-12)
    assert np.all(np.linalg.eigvalsh(pre.cov) > 0)


def test_actualizar_sesgo_reintegra_solo_por_encima_del_umbral():
    rng = np.random.default_rng(6)
    t_s, tr = tramo_simulado(DURACION_S, rng)
    w_m, a_m = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"], b_g=B_G, b_a=B_A)
    lista = intervalos(t_s, w_m, a_m)
    pre = preintegrar(lista, B_G, B_A)
    u_g, u_a = CFG_VINS["umbral_reintegrar_bg_rad_s"], CFG_VINS["umbral_reintegrar_ba_m_s2"]

    # Por debajo: no reintegra y el punto de linealización no cambia
    assert not pre.actualizar_sesgo(B_G + [0.5 * u_g, 0, 0], B_A, u_g, u_a)
    assert pre.b_g_lin == pytest.approx(B_G)

    # Por encima: reintegra y queda igual que una preintegración nueva con ese sesgo
    b_a_nuevo = B_A + [0, 0, 2.0 * u_a]
    assert pre.actualizar_sesgo(B_G, b_a_nuevo, u_g, u_a)
    nueva = preintegrar(lista, B_G, b_a_nuevo)
    assert pre.b_a_lin == pytest.approx(b_a_nuevo)
    assert pre.dv == pytest.approx(nueva.dv, abs=1e-14)
    assert pre.cov == pytest.approx(nueva.cov, abs=1e-20)


def test_ruido_desde_config_infla_las_densidades():
    ruido = pi.ruido_desde_config(CFG_VINS)
    medido = cargar_config(CFG_VINS["fichero_ruido"])
    for clave in ("ruido_gir", "ruido_acc", "paseo_sesgo_gir", "paseo_sesgo_acc"):
        assert ruido[clave] == pytest.approx(CFG_VINS["factor_inflado"] * medido[clave])


def test_intervalos_nulos_se_ignoran():
    pre = pi.Preintegracion(B_G, B_A, CFG_SIM["imu"])
    pre.anadir(0.0, [1.0, 2.0, 3.0], [0.0, 0.0, 9.8])
    assert pre.dt_total == 0.0 and len(pre.medidas) == 0
    assert np.array_equal(pre.dR, np.eye(3))


# =====================================================================
# Etapa 35 — Fusión de dos preintegraciones
# =====================================================================
@pytest.mark.parametrize("mismo_sesgo", [True, False])
def test_fusion_igual_que_preintegrar_todo_seguido(mismo_sesgo):
    rng = np.random.default_rng(8)
    t_s, tr = tramo_simulado(DURACION_S, rng)
    w_m, a_m, _, _ = sim.imu_con_ruido(tr, CFG_SIM["imu"], CFG_SIM["trayectoria"]["frecuencia_imu_hz"], rng)
    lista = intervalos(t_s, w_m, a_m)
    corte = 77                                           # dos tramos de longitud distinta

    primera = preintegrar(lista[:corte], B_G, B_A)
    # El segundo tramo puede venir linealizado en otro sesgo (p. ej. el estimado en j)
    b_g2, b_a2 = (B_G, B_A) if mismo_sesgo else (B_G + 0.003, B_A - 0.02)
    segunda = preintegrar(lista[corte:], b_g2, b_a2)
    fus = primera.fusionada(segunda)
    seguida = preintegrar(lista, B_G, B_A)

    assert fus.dt_total == pytest.approx(seguida.dt_total, abs=1e-12)
    assert so3.angulo_entre(fus.dR, seguida.dR) < 1e-12
    assert fus.dv == pytest.approx(seguida.dv, abs=1e-12)
    assert fus.dp == pytest.approx(seguida.dp, abs=1e-12)
    assert fus.jacobiano_sesgos() == pytest.approx(seguida.jacobiano_sesgos(), abs=1e-12)
    escala = np.abs(seguida.cov).max()
    assert np.abs(fus.cov - seguida.cov).max() < 1e-10 * escala
    assert len(fus.medidas) == len(lista)
    # No modifica las de entrada
    assert len(primera.medidas) == corte and np.array_equal(segunda.b_g_lin, b_g2)