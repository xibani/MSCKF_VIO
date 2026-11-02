"""tests/test_verificar.py — Tests de captura/verificar.py con marcas sintéticas."""

import numpy as np
import pytest

from captura.verificar import analizar_intervalos, analizar_latencia, variacion_pct

F_HZ = 200.0
PERIODO_NS = 5_000_000          # 5 ms
FACTOR_HUECO = 1.5


def marcas(n, sigma_ns=0.0, semilla=0):
    """n marcas a 200 Hz desde t = 1e12 ns, con ruido gaussiano opcional."""
    rng = np.random.default_rng(semilla)
    t = 1_000_000_000_000 + np.arange(n) * PERIODO_NS + rng.normal(0.0, sigma_ns, n)
    return np.round(t).astype(np.int64)


def test_marcas_perfectas():
    r = analizar_intervalos(marcas(2001), F_HZ, FACTOR_HUECO)
    assert r["frecuencia_hz"] == pytest.approx(200.0)
    assert r["duracion_s"] == pytest.approx(10.0)
    assert r["perdidas"] == 0 and r["huecos"] == 0 and r["no_crecientes"] == 0
    assert r["jitter_ms"] == pytest.approx(0.0, abs=1e-9)


def test_cuenta_perdidas():
    t = marcas(2001)
    t = np.delete(t, [100, 500, 501])        # 1 muestra suelta + 2 seguidas
    r = analizar_intervalos(t, F_HZ, FACTOR_HUECO)
    assert r["perdidas"] == 3
    assert r["huecos"] == 2
    assert r["perdidas_pct"] == pytest.approx(100 * 3 / 2001)


def test_jitter_es_sigma_por_raiz_de_dos():
    # Si cada marca tiene ruido σ, Δt = t2 - t1 tiene σ·√2
    sigma_ns = 100_000                       # 0.1 ms
    r = analizar_intervalos(marcas(20001, sigma_ns), F_HZ, FACTOR_HUECO)
    assert r["jitter_ms"] == pytest.approx(0.1 * np.sqrt(2), rel=0.03)


def test_detecta_marcas_no_crecientes():
    t = marcas(100)
    t[50] = t[49]                            # dos marcas iguales
    r = analizar_intervalos(t, F_HZ, FACTOR_HUECO)
    assert r["no_crecientes"] == 1


def test_latencia():
    t_sensor = marcas(100)
    t_llegada = t_sensor + 30_000_000                # todas llegan 30 ms después
    t_llegada[10] += 20_000_000                      # una llega 50 ms después
    r = analizar_latencia(t_sensor, t_llegada)
    assert r["min_ms"] == pytest.approx(30.0)
    assert r["max_ms"] == pytest.approx(50.0)
    assert r["media_ms"] == pytest.approx(30.2)


def test_variacion_pct():
    assert variacion_pct([8000, 8000, 8000]) == 0.0
    assert variacion_pct([7900, 8000, 8100]) == pytest.approx(2.5)