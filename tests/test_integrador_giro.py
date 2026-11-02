"""tests/test_integrador_giro.py — Tests de vio/integrador_giro.py con datos sintéticos."""

import numpy as np
import pytest

from vio import so3
from vio.config import cargar_config
from vio.integrador_giro import IntegradorGiro, integrar_secuencia, paso_giro

CFG = cargar_config("orientacion")["integracion"]
F_HZ = 200


def tiempos(duracion_s, f_hz=F_HZ):
    n = int(duracion_s * f_hz) + 1
    return np.arange(n, dtype=np.int64) * (1_000_000_000 // f_hz)


def test_paso_sin_giro_no_cambia():
    R = so3.exp_so3([0.1, 0.2, 0.3])
    assert paso_giro(R, np.zeros(3), np.zeros(3), 0.005) == pytest.approx(R)


def test_velocidad_constante():
    # 90 °/s sobre z durante 1 s -> 90° sobre z
    t = tiempos(1.0)
    w = np.tile([0.0, 0.0, np.pi / 2], (len(t), 1))
    R, huecos = integrar_secuencia(np.eye(3), np.zeros(3), t, w, CFG)
    assert R[-1] == pytest.approx(so3.exp_so3([0.0, 0.0, np.pi / 2]), abs=1e-9)
    assert huecos == 0


def test_resta_el_sesgo():
    t = tiempos(1.0)
    sesgo = np.radians([0.5, -0.3, 0.2])
    w = np.tile([0.0, 0.0, np.pi / 2], (len(t), 1)) + sesgo
    R, _ = integrar_secuencia(np.eye(3), sesgo, t, w, CFG)
    assert R[-1] == pytest.approx(so3.exp_so3([0.0, 0.0, np.pi / 2]), abs=1e-9)


def test_sesgo_sin_restar_deriva():
    # Un sesgo de 1 °/s sin corregir se acumula: 60° en 60 s
    t = tiempos(60.0)
    w = np.tile(np.radians([0.0, 0.0, 1.0]), (len(t), 1))
    R, _ = integrar_secuencia(np.eye(3), np.zeros(3), t, w, CFG)
    assert np.degrees(so3.angulo_entre(np.eye(3), R[-1])) == pytest.approx(60.0, abs=1e-6)


def test_trapecio_exacto_con_aceleracion_angular_constante():
    # w(t) = alfa · t sobre x  ->  ángulo(t) = alfa · t² / 2 (exacto con la regla del trapecio)
    alfa = 2.0                                       # rad/s²
    t = tiempos(1.0)
    t_s = t / 1e9
    w = np.zeros((len(t), 3))
    w[:, 0] = alfa * t_s
    R, _ = integrar_secuencia(np.eye(3), np.zeros(3), t, w, CFG)
    assert R[-1] == pytest.approx(so3.exp_so3([alfa * 1.0 ** 2 / 2, 0.0, 0.0]), abs=1e-9)


def test_usa_los_dt_reales():
    # Muestras irregulares: el ángulo final solo depende del tiempo total
    rng = np.random.default_rng(2)
    dt_ns = rng.integers(4_000_000, 6_000_001, 400)  # entre 4 y 6 ms
    t = np.concatenate([[0], np.cumsum(dt_ns)])
    w = np.tile([0.0, 1.0, 0.0], (len(t), 1))        # 1 rad/s sobre y
    R, _ = integrar_secuencia(np.eye(3), np.zeros(3), t, w, CFG)
    angulo_esperado = t[-1] / 1e9 * 1.0
    assert R[-1] == pytest.approx(so3.exp_so3([0.0, angulo_esperado, 0.0]), abs=1e-9)


def test_cuenta_huecos():
    t = tiempos(1.0)
    t = np.delete(t, range(100, 120))                # 20 muestras seguidas perdidas = 105 ms
    w = np.zeros((len(t), 3))
    _, huecos = integrar_secuencia(np.eye(3), np.zeros(3), t, w, CFG)
    assert huecos == 1


def test_sigue_siendo_rotacion_tras_mucho_tiempo():
    # 10 minutos girando a la vez sobre los tres ejes
    rng = np.random.default_rng(3)
    integrador = IntegradorGiro(np.eye(3), np.zeros(3), 0, np.zeros(3), CFG)
    t_ns = 0
    for _ in range(600 * F_HZ):
        t_ns += 5_000_000
        R = integrador.actualizar(t_ns, rng.normal(0.0, 2.0, 3))
    assert so3.es_rotacion(R, tolerancia=1e-12)