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
        assert not r["periodica"]                                 # un paseo solo sube: no es una joroba


def test_sin_paseo_visible_da_cota_superior():
    # Paseo tan pequeño que en esta duración no llega a verse: K es una cota (por encima)
    N, K = 1e-3, 1e-7
    blanco, sesgo = sim.ruido_y_sesgo(100_000, F_HZ, N, K, np.zeros(3), np.random.default_rng(3))
    r = allan.analizar_eje(blanco[:, 0] + sesgo[:, 0], DT, CFG)
    assert not r["K_observado"] and r["K"] > K


def test_perturbacion_periodica():
    # Ruido blanco + una oscilación del sesgo de 450 s (como la temperatura con el ventilador):
    # la curva sube, tiene su máximo en τ ≈ 0,371·T0 y vuelve a bajar. No es un paseo aleatorio
    f_hz, T0 = 10.0, 450.0
    t = np.arange(int(4 * 3600 * f_hz)) / f_hz
    x = np.random.default_rng(5).normal(0.0, 1e-3 * np.sqrt(f_hz), len(t)) + 1e-3 * np.sin(2 * np.pi * t / T0)
    r = allan.analizar_eje(x, 1.0 / f_hz, CFG)
    assert r["periodica"] and r["periodo_s"] == pytest.approx(T0, rel=0.05)
    assert not r["K_observado"]                                   # K no se puede medir: solo la cota
    assert r["N"] == pytest.approx(1e-3, rel=0.05)                # N sí: está a la izquierda de la joroba


def test_periodo_dominante():
    # El periodo sale del espectro, también cuando el ruido blanco tapa el lóbulo principal
    f_hz = 10.0
    t = np.arange(int(4 * 3600 * f_hz)) / f_hz
    ruido = np.random.default_rng(8).normal(0.0, 1e-3 * np.sqrt(f_hz), len(t))
    for T0 in (120.0, 450.0, 1800.0):
        x = ruido + 1e-3 * np.sin(2 * np.pi * t / T0) + 1e-7 * t       # con una deriva lenta encima
        assert allan.periodo_dominante(t, x, 1.0, 20.0) == pytest.approx(T0, rel=0.05)
    # Con el 58 % de las muestras perdidas (dos programas leyendo la IMU): con los instantes
    # reales el periodo sigue saliendo bien (contando muestras saldría 0,42 · T0)
    quedan = np.random.default_rng(9).random(len(t)) < 0.42
    x = ruido + 1e-3 * np.sin(2 * np.pi * t / 450.0)
    assert allan.periodo_dominante(t[quedan], x[quedan], 1.0, 20.0) == pytest.approx(450.0, rel=0.05)


def test_guardar_y_leer(tmp_path):
    adoptado = {"ruido_gir": 1.5e-4, "ruido_acc": 2.5e-3, "paseo_sesgo_gir": 3e-5, "paseo_sesgo_acc": 6e-4}
    ruta = tmp_path / "imu_ruido.yaml"
    allan.guardar_ruido_imu(ruta, adoptado, {"gir": {}}, {"fecha": "hoy"})
    assert allan.cargar_ruido_imu(ruta) == pytest.approx(adoptado)


def grabacion_sintetica(carpeta, n, quitar, semilla):
    """Grabación EuRoC como la de grabar_imu (reposo: g en z), sin las muestras de `quitar`."""
    rng = np.random.default_rng(semilla)
    b_g, s_g = sim.ruido_y_sesgo(n, F_HZ, 8.7e-5, 2.0e-5, np.zeros(3), rng)
    b_a, s_a = sim.ruido_y_sesgo(n, F_HZ, 3.9e-3, 4.0e-4, np.zeros(3), rng)
    w = b_g + s_g
    a = b_a + s_a + np.array([0.0, 0.0, 9.80665])
    t_ns = np.arange(n, dtype=np.int64) * 5_000_000
    t_ns, w, a = np.delete(t_ns, quitar), np.delete(w, quitar, axis=0), np.delete(a, quitar, axis=0)
    ruta_csv = euroc.ruta_imu(carpeta)
    ruta_csv.parent.mkdir(parents=True)
    with open(ruta_csv, "w", encoding="utf-8") as f:
        f.write(euroc.CABECERA_IMU)
        for k in range(len(t_ns)):
            f.write(euroc.linea_imu(t_ns[k], w[k], a[k]))


def cfg_corta():
    cfg = copy.deepcopy(CFG)
    cfg["descartar_inicio_s"] = 10.0
    return cfg


def test_analizar_grabacion_euroc(tmp_path):
    # De principio a fin, 10 minutos a 200 Hz con 3 muestras perdidas: válida
    from herramientas.analizar_allan import analizar_grabacion
    grabacion_sintetica(tmp_path, 120_000, [5000, 5001, 90000], 4)
    adoptado, motivos = analizar_grabacion(tmp_path, cfg_corta(), tmp_path / "imu_ruido.yaml", F_HZ, 1.5)
    assert motivos == []
    assert adoptado["ruido_gir"] == pytest.approx(8.7e-5, rel=0.1)
    assert adoptado["ruido_acc"] == pytest.approx(3.9e-3, rel=0.1)
    assert (tmp_path / "allan.png").exists()
    leido = allan.cargar_ruido_imu(tmp_path / "imu_ruido.yaml")
    assert leido == pytest.approx(adoptado)


def test_rechaza_grabacion_con_perdidas(tmp_path):
    # Como con dos programas leyendo la IMU: falta la mitad de las muestras, al azar
    from herramientas.analizar_allan import analizar_grabacion
    n = 120_000
    quitar = np.flatnonzero(np.random.default_rng(6).random(n) < 0.5)
    grabacion_sintetica(tmp_path, n, quitar, 7)
    ruta_yaml = tmp_path / "imu_ruido.yaml"
    adoptado, motivos = analizar_grabacion(tmp_path, cfg_corta(), ruta_yaml, F_HZ, 1.5)
    assert adoptado is None and not ruta_yaml.exists()            # no se guarda nada...
    assert (tmp_path / "allan.png").exists()                      # ...salvo la figura, para verlo
    assert len(motivos) == 1 and "50." in motivos[0]              # las pérdidas REALES, ~50 %
    # Con --forzar sí se guarda, y quedan apuntados los motivos
    adoptado, motivos = analizar_grabacion(tmp_path, cfg_corta(), ruta_yaml, F_HZ, 1.5, forzar=True)
    assert adoptado is not None and ruta_yaml.exists() and motivos
