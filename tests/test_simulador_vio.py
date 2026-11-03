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