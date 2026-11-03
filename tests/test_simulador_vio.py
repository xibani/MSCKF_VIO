"""tests/test_simulador_vio.py — Tests del simulador VIO completo (etapa 11, vio/simulador.py)."""

import copy

import numpy as np
import pytest

from vio import simulador as sim
from vio import so3
from vio.config import cargar_config

CFG = cargar_config("simulador")["vio"]
TR = CFG["trayectoria"]
K = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])
TAMANO = (640, 480)


def pose_circulo(t):
    """El círculo de la etapa 6 como función t -> (R_w_i, p_w)."""
    c = sim.circulo(np.array([t]), 1.0, 1.0, 0.0)
    return c["R_w_i"][0], c["p_w"][0]


# =====================================================================
# Paso 11.1
# =====================================================================
def test_imu_ideal_del_circulo_por_diferencias():
    # El círculo tiene solución exacta: las diferencias finitas deben reproducirla
    t_s = sim.tiempos_ns(10.0, 200) / 1e9
    exacto = sim.circulo(t_s, 1.0, 1.0, 0.0)
    numerico = sim.imu_ideal(pose_circulo, t_s, TR["paso_derivadas_s"])
    assert np.abs(numerico["v_w"] - exacto["v_w"]).max() < 1e-6
    assert np.abs(numerico["a_w"] - exacto["a_w"]).max() < 1e-6
    assert np.abs(numerico["w_i"] - exacto["w_i"]).max() < 1e-9


def test_trayectoria_vio_excita_todos_los_ejes():
    t_s = sim.tiempos_ns(TR["duracion_s"], 50) / 1e9
    tr = sim.imu_ideal(lambda t: sim.pose_vio(t, TR), t_s, TR["paso_derivadas_s"])
    assert all(so3.es_rotacion(R) for R in tr["R_w_i"])
    # Dentro de la habitación, a más de 1 m de las paredes, el suelo y el techo
    lx, ly, lz = CFG["mundo"]["habitacion_m"]
    assert (np.abs(tr["p_w"][:, 0]) < lx / 2 - 1).all() and (np.abs(tr["p_w"][:, 1]) < ly / 2 - 1).all()
    assert (tr["p_w"][:, 2] > 1).all() and (tr["p_w"][:, 2] < lz - 1).all()
    # Giro y aceleración variando en los tres ejes
    #   (sobre z el giro es casi constante, el del rumbo: basta con que no sea nulo)
    maximo_w = np.abs(tr["w_i"]).max(axis=0)
    rango_a = tr["a_w"].max(axis=0) - tr["a_w"].min(axis=0)
    assert (maximo_w > 0.1).all(), maximo_w
    assert (rango_a > 0.1).all(), rango_a


def test_propagar_imu_ideal_error_baja_a_la_mitad():
    from vio.estado import EstadoImu
    from vio.propagacion import propagar_secuencia
    cfg_int = cargar_config("orientacion")["integracion"]
    errores = []
    for f_hz in (100, 200):
        t_ns = sim.tiempos_ns(5.0, f_hz)
        tr = sim.imu_ideal(lambda t: sim.pose_vio(t, TR), t_ns / 1e9, TR["paso_derivadas_s"])
        w_m, a_m = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"])
        inicial = EstadoImu(tr["R_w_i"][0].copy(), tr["v_w"][0].copy(), tr["p_w"][0].copy(),
                            np.zeros(3), np.zeros(3))
        r = propagar_secuencia(inicial, t_ns, w_m, a_m, cfg_int)
        errores.append(np.linalg.norm(r["p"][-1] - tr["p_w"][-1]))
    # Método de primer orden: el error es proporcional a Δt
    assert errores[1] / errores[0] == pytest.approx(0.5, abs=0.05)


# =====================================================================
# Paso 11.2
# =====================================================================
def test_ruido_blanco_tiene_sigma_N_raiz_f():
    rng = np.random.default_rng(0)
    N, f = 3.9e-3, 200
    blanco, _ = sim.ruido_y_sesgo(100_000, f, N, 0.0, np.zeros(3), rng)
    assert blanco.std(axis=0) == pytest.approx(np.full(3, N * np.sqrt(f)), rel=0.02)


def test_paseo_aleatorio_del_sesgo():
    rng = np.random.default_rng(1)
    Kp, f, n = 4.0e-4, 200, 1001                     # 5 s de simulación
    b0 = np.array([0.05, -0.04, 0.08])
    # 1. Cada paso suma un ruido de sigma = K·√Δt
    _, sesgo = sim.ruido_y_sesgo(20_000, f, 0.0, Kp, b0, rng)
    assert sesgo[0] == pytest.approx(b0)
    assert np.diff(sesgo, axis=0).std() == pytest.approx(Kp * np.sqrt(1 / f), rel=0.02)
    # 2. Tras T segundos, el sesgo se ha alejado de b0 con varianza K²·T
    finales = np.array([sim.ruido_y_sesgo(n, f, 0.0, Kp, b0, rng)[1][-1] - b0 for _ in range(200)])
    T = (n - 1) / f
    assert finales.var() == pytest.approx(Kp ** 2 * T, rel=0.15)


def test_imu_con_ruido():
    t_s = sim.tiempos_ns(5.0, 200) / 1e9
    tr = sim.imu_ideal(lambda t: sim.pose_vio(t, TR), t_s, TR["paso_derivadas_s"])
    w_ideal, a_ideal = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"])
    w_m, a_m, b_g, b_a = sim.imu_con_ruido(tr, CFG["imu"], 200, np.random.default_rng(2))
    imu = CFG["imu"]
    assert (w_m - w_ideal - b_g).std() == pytest.approx(imu["ruido_gir"] * np.sqrt(200), rel=0.05)
    assert (a_m - a_ideal - b_a).std() == pytest.approx(imu["ruido_acc"] * np.sqrt(200), rel=0.05)
    assert b_g[0] == pytest.approx(imu["sesgo_inicial_gir"]) and b_a[0] == pytest.approx(imu["sesgo_inicial_acc"])