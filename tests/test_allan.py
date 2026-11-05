"""tests/test_allan.py — Etapa 7.3: varianza de Allan (vio/allan.py y herramientas/analizar_allan.py).

La idea de los tests: generar ruido con N y K CONOCIDOS y comprobar que el análisis los recupera.
"""

import copy

import numpy as np
import pytest

from captura import euroc
from vio import allan
from vio import simulador as sim
from vio.config import cargar_config

CFG = cargar_config("calibracion")["allan"]
F_HZ = 200.0
DT = 1.0 / F_HZ


def test_tamanos_cluster():
    m = allan.tamanos_cluster(100_000, 10, 0.1)
    assert m[0] == 1 and m[-1] == 10_000
    assert np.all(np.diff(m) > 0)                         # crecientes y sin repetir
    with pytest.raises(ValueError):
        allan.tamanos_cluster(10, 10, 0.1)


def test_ruido_blanco_da_pendiente_menos_un_medio():
    # Solo ruido blanco: σ(τ) = N/√τ en todo τ
    N = 1e-3
    x = np.random.default_rng(0).normal(0.0, N * np.sqrt(F_HZ), 200_000)
    tau, sigma = allan.desviacion_allan(x, DT, allan.tamanos_cluster(len(x), 10, 0.1))
    for tau_prueba in (0.1, 1.0, 10.0):
        i = np.argmin(np.abs(tau - tau_prueba))
        assert sigma[i] * np.sqrt(tau[i]) == pytest.approx(N, rel=0.06)
    assert allan.ruido_blanco(tau, sigma, 0.1, 1.0) == pytest.approx(N, rel=0.03)
    assert np.median(allan.pendiente_local(tau, sigma)) == pytest.approx(-0.5, abs=0.05)


def test_paseo_aleatorio_da_pendiente_mas_un_medio():
    # Solo paseo aleatorio de la señal (el sesgo que deriva): σ(τ) = K·√(τ/3)
    K = 1e-4
    x = np.cumsum(K * np.sqrt(DT) * np.random.default_rng(1).normal(0.0, 1.0, 200_000))
    tau, sigma = allan.desviacion_allan(x, DT, allan.tamanos_cluster(len(x), 10, 0.1))
    for tau_prueba in (1.0, 10.0):
        i = np.argmin(np.abs(tau - tau_prueba))
        assert sigma[i] / np.sqrt(tau[i] / 3.0) == pytest.approx(K, rel=0.25)


def test_recupera_el_ruido_del_simulador():
    # El mismo modelo de ruido que el simulador (etapa 11) con los valores de la hoja de datos:
    # el análisis tiene que devolver los N y K con los que se generó
    N, K = 8.7e-5, 2.0e-5
    blanco, sesgo = sim.ruido_y_sesgo(300_000, F_HZ, N, K, np.zeros(3), np.random.default_rng(2))
    for eje in range(3):
        r = allan.analizar_eje(blanco[:, eje] + sesgo[:, eje], DT, CFG)
        assert r["N"] == pytest.approx(N, rel=0.05)
        assert r["K_observado"] and r["K"] == pytest.approx(K, rel=0.5)     # K: pocos clusters, ±40 %
        assert r["tau_B"] < r["tau"][-1]                          # el mínimo queda dentro de la curva


def test_sin_paseo_visible_da_cota_superior():
    # Paseo tan pequeño que en esta duración no llega a verse: K es una cota (por encima)
    N, K = 1e-3, 1e-7
    blanco, sesgo = sim.ruido_y_sesgo(100_000, F_HZ, N, K, np.zeros(3), np.random.default_rng(3))
    r = allan.analizar_eje(blanco[:, 0] + sesgo[:, 0], DT, CFG)
    assert not r["K_observado"] and r["K"] > K


def test_guardar_y_leer(tmp_path):
    adoptado = {"ruido_gir": 1.5e-4, "ruido_acc": 2.5e-3, "paseo_sesgo_gir": 3e-5, "paseo_sesgo_acc": 6e-4}
    ruta = tmp_path / "imu_ruido.yaml"
    allan.guardar_ruido_imu(ruta, adoptado, {"gir": {}}, {"fecha": "hoy"})
    assert allan.cargar_ruido_imu(ruta) == pytest.approx(adoptado)


def test_analizar_grabacion_euroc(tmp_path):
    # De principio a fin: una grabación EuRoC como la de grabar_imu (reposo: g en z)
    from herramientas.analizar_allan import analizar_grabacion
    n = 120_000                                                   # 10 minutos a 200 Hz
    rng = np.random.default_rng(4)
    b_g, s_g = sim.ruido_y_sesgo(n, F_HZ, 8.7e-5, 2.0e-5, np.zeros(3), rng)
    b_a, s_a = sim.ruido_y_sesgo(n, F_HZ, 3.9e-3, 4.0e-4, np.zeros(3), rng)
    w = b_g + s_g
    a = b_a + s_a + np.array([0.0, 0.0, 9.80665])
    t_ns = np.arange(n, dtype=np.int64) * 5_000_000
    t_ns = np.delete(t_ns, [5000, 5001, 90000])                   # 3 muestras perdidas
    w, a = np.delete(w, [5000, 5001, 90000], axis=0), np.delete(a, [5000, 5001, 90000], axis=0)
    ruta_csv = euroc.ruta_imu(tmp_path)
    ruta_csv.parent.mkdir(parents=True)
    with open(ruta_csv, "w", encoding="utf-8") as f:
        f.write(euroc.CABECERA_IMU)
        for k in range(len(t_ns)):
            f.write(euroc.linea_imu(t_ns[k], w[k], a[k]))

    cfg = copy.deepcopy(CFG)
    cfg["descartar_inicio_s"] = 10.0
    adoptado = analizar_grabacion(tmp_path, cfg, tmp_path / "imu_ruido.yaml")
    assert adoptado["ruido_gir"] == pytest.approx(8.7e-5, rel=0.1)
    assert adoptado["ruido_acc"] == pytest.approx(3.9e-3, rel=0.1)
    assert (tmp_path / "allan.png").exists()
    leido = allan.cargar_ruido_imu(tmp_path / "imu_ruido.yaml")
    assert leido == pytest.approx(adoptado)