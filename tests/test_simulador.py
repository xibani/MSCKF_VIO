"""tests/test_simulador.py — Tests de vio/simulador.py."""

import numpy as np
import pytest

from vio import simulador as sim
from vio import so3
from vio.constantes import G


def trayectoria(f_hz=200, duracion_s=10.0, radio=1.0, omega=1.0):
    t_ns = sim.tiempos_ns(duracion_s, f_hz)
    return t_ns, sim.circulo(t_ns / 1e9, radio, omega, 0.0)


def test_tiempos():
    t = sim.tiempos_ns(1.0, 200)
    assert len(t) == 201 and t[0] == 0 and t[-1] == 1_000_000_000
    assert t.dtype == np.int64


def test_circulo_es_coherente():
    t_ns, tr = trayectoria()
    t_s = t_ns / 1e9
    # Radio constante y orientaciones válidas
    assert np.linalg.norm(tr["p_w"][:, :2], axis=1) == pytest.approx(np.ones(len(t_s)))
    assert all(so3.es_rotacion(R) for R in tr["R_w_i"])
    # La velocidad es la derivada de la posición (diferencias finitas centradas)
    dt = t_s[1] - t_s[0]
    v_numerica = (tr["p_w"][2:] - tr["p_w"][:-2]) / (2 * dt)
    assert v_numerica == pytest.approx(tr["v_w"][1:-1], abs=1e-4)
    # El eje x de la IMU (1ª columna de R) apunta en la dirección del movimiento
    x_imu = tr["R_w_i"][:, :, 0]
    direccion = tr["v_w"] / np.linalg.norm(tr["v_w"], axis=1, keepdims=True)
    assert x_imu == pytest.approx(direccion, abs=1e-12)


def test_imu_ideal_del_circulo():
    # En ejes de la IMU: hacia el centro (eje y) r·ω² = 1 m/s², y hacia arriba g
    _, tr = trayectoria()
    w_m, a_m = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"])
    assert a_m == pytest.approx(np.tile([0.0, 1.0, G], (len(a_m), 1)), abs=1e-12)
    assert w_m == pytest.approx(np.tile([0.0, 0.0, 1.0], (len(w_m), 1)))


def test_sesgos_se_suman():
    _, tr = trayectoria(duracion_s=1.0)
    b_g, b_a = np.array([0.01, 0.0, 0.0]), np.array([0.0, 0.0, 0.2])
    w0, a0 = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"])
    w1, a1 = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"], b_g=b_g, b_a=b_a)
    assert w1 - w0 == pytest.approx(np.tile(b_g, (len(w0), 1)))
    assert a1 - a0 == pytest.approx(np.tile(b_a, (len(a0), 1)))