"""tests/test_propagacion.py — Tests de vio/propagacion.py con datos sintéticos."""

import numpy as np
import pytest

from vio import simulador as sim
from vio import so3
from vio.config import cargar_config
from vio.constantes import G
from vio.estado import EstadoImu, estado_inicial
from vio.propagacion import propagar, propagar_secuencia

CFG = cargar_config("orientacion")["integracion"]


def reposo(R_w_i, duracion_s, f_hz=200, b_a=None):
    """Medidas de una IMU quieta con orientación R_w_i (opcionalmente con sesgo)."""
    t = sim.tiempos_ns(duracion_s, f_hz)
    a = R_w_i.T @ np.array([0.0, 0.0, G])
    if b_a is not None:
        a = a + b_a
    return t, np.zeros((len(t), 3)), np.tile(a, (len(t), 1))


def test_propagar_no_modifica_la_entrada():
    e = estado_inicial(np.eye(3))
    nuevo = propagar(e, np.array([0.1, 0.0, 0.0]), np.array([1.0, 0.0, G]), 0.01)
    assert np.array_equal(e.p_w, np.zeros(3)) and np.array_equal(e.R_w_i, np.eye(3))
    assert not np.array_equal(nuevo.p_w, e.p_w)


def test_reposo_plano_no_se_mueve():
    t, w, a = reposo(np.eye(3), 60.0)
    r = propagar_secuencia(estado_inicial(np.eye(3)), t, w, a, CFG)
    assert np.abs(r["p"]).max() < 1e-9
    assert np.abs(r["v"]).max() < 1e-9


def test_reposo_inclinado_no_se_mueve():
    R = so3.exp_so3([0.4, -0.7, 1.2])
    t, w, a = reposo(R, 60.0)
    r = propagar_secuencia(estado_inicial(R), t, w, a, CFG)
    assert np.abs(r["p"]).max() < 1e-6


def test_aceleracion_constante():
    # 1 m/s² en x durante 2 s, partiendo del reposo: p = ½·a·t² = 2 m, v = a·t = 2 m/s
    t = sim.tiempos_ns(2.0, 200)
    w = np.zeros((len(t), 3))
    a = np.tile([1.0, 0.0, G], (len(t), 1))       # fuerza específica: 1 m/s² + la que sostiene
    r = propagar_secuencia(estado_inicial(np.eye(3)), t, w, a, CFG)
    assert r["p"][-1] == pytest.approx([2.0, 0.0, 0.0], abs=1e-9)
    assert r["v"][-1] == pytest.approx([2.0, 0.0, 0.0], abs=1e-9)


def error_final_circulo(f_hz):
    t = sim.tiempos_ns(10.0, f_hz)
    tr = sim.circulo(t / 1e9, 1.0, 1.0, 0.0)
    w, a = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"])
    e0 = EstadoImu(tr["R_w_i"][0].copy(), tr["v_w"][0].copy(), tr["p_w"][0].copy(),
                   np.zeros(3), np.zeros(3))
    r = propagar_secuencia(e0, t, w, a, CFG)
    return np.linalg.norm(r["p"][-1] - tr["p_w"][-1])


def test_circulo_reproduce_la_trayectoria():
    assert error_final_circulo(200) < 1e-4          # < 0,1 mm tras 10 s y 1,6 vueltas (etapa 12)


def test_circulo_metodo_de_segundo_orden():
    # Doblar la frecuencia divide el error por 4: el error es proporcional a Δt² (etapa 12;
    # con la orientación del principio del intervalo se dividía por 2: primer orden)
    cociente = error_final_circulo(200) / error_final_circulo(400)
    assert cociente == pytest.approx(4.0, abs=0.2)


def test_sesgo_de_acelerometro_sin_corregir():
    # 0,05 m/s² hacia arriba sin corregir: error = ½·b·t² = 2,5 m en 10 s
    t, w, a = reposo(np.eye(3), 10.0, b_a=np.array([0.0, 0.0, 0.05]))
    r = propagar_secuencia(estado_inicial(np.eye(3)), t, w, a, CFG)
    assert r["p"][-1] == pytest.approx([0.0, 0.0, 2.5], abs=1e-6)


def test_inclinacion_de_un_grado_sin_detectar():
    # La placa está plana pero creemos que está inclinada 1°: la gravedad mal restada
    # deja g·sen(1°) ≈ 0,171 m/s² en horizontal -> ½·0,171·10² ≈ 8,56 m en 10 s
    t, w, a = reposo(np.eye(3), 10.0)
    R_creido = so3.exp_so3([np.radians(1.0), 0.0, 0.0])
    r = propagar_secuencia(estado_inicial(R_creido), t, w, a, CFG)
    esperado = 0.5 * G * np.sin(np.radians(1.0)) * 10.0 ** 2
    assert np.linalg.norm(r["p"][-1]) == pytest.approx(esperado, rel=1e-3)


def test_sesgo_de_giroscopio_sin_corregir_crece_como_t_al_cubo():
    # 0,1 °/s sobre x sin corregir: la inclinación creída crece como b·t, la gravedad mal
    # restada como g·b·t, la velocidad como g·b·t²/2 y la posición como g·b·t³/6
    b = np.radians(0.1)
    t = sim.tiempos_ns(10.0, 200)
    w = np.tile([b, 0.0, 0.0], (len(t), 1))
    a = np.tile([0.0, 0.0, G], (len(t), 1))
    r = propagar_secuencia(estado_inicial(np.eye(3)), t, w, a, CFG)
    esperado = G * b * 10.0 ** 3 / 6                 # ≈ 2,85 m
    assert np.linalg.norm(r["p"][-1]) == pytest.approx(esperado, rel=0.01)