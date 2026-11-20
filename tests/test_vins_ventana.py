"""tests/test_vins_ventana.py — Etapa 35: ventana deslizante (vins/ventana.py) en el simulador.

Una secuencia de 5 s (como el test del MSCKF de la etapa 12), con el mismo error inicial y los
mismos modelos de ruido para VINS y para el MSCKF (herramientas/vins_simulacion.py). La prueba
de consistencia de 60 s con varias secuencias es la herramienta:
    python -m herramientas.vins_simulacion --duracion 60 --nees 6
"""

import copy

import numpy as np
import pytest

from herramientas.vins_simulacion import (CLAVES_RUIDO, K_SIM, POSE, TAMANO, configuraciones, nees, resumen,
                                         una_secuencia)
from vio import simulador as sim
from vio import so3
from vio.config import cargar_config
from vio.estado import EstadoImu
from vins import factores as fa
from vins import marginalizacion as mg
from vins import ventana as vt

CFG_SIM = cargar_config("simulador")["vio"]


@pytest.fixture(scope="module")
def simulacion():
    cfg_sim = copy.deepcopy(CFG_SIM)
    cfg_sim["trayectoria"]["duracion_s"] = 5.0
    cfg_sim["mundo"]["puntos_por_superficie"] = 100          # menos puntos: test más rápido
    cfg_v, cfg_m = configuraciones(cfg_sim)
    r = una_secuencia(cfg_sim, cfg_v, cfg_m, np.random.default_rng(11))
    r["cfg_v"] = cfg_v
    return r


def test_ventana_tamano_y_marginalizaciones(simulacion):
    rec = simulacion["vins"]["recuentos"]
    N = simulacion["cfg_v"]["ventana"]["tamano"]
    assert max(x["frames"] for x in rec) == N                # tras deslizar quedan N (llega a N + 1)
    marg = [x["marginalizacion"] for x in rec]
    assert marg.count("antiguo") > 10 and marg.count("segundo") > 10     # las dos estrategias
    assert all(m is None for m in marg[:N])                  # hasta llenarse no se desliza
    assert np.mean([x["puntos_activos"] for x in rec[N:]]) > 20


def test_vins_mejor_o_igual_que_msckf(simulacion):
    """Criterio de salida (en corto): error de trayectoria <= el del MSCKF en la misma secuencia."""
    s = simulacion["s"]
    m_v, _ = resumen("vins", simulacion["vins"], s, 20)
    m_m, _ = resumen("msckf", simulacion["msckf"], s, 20)
    assert m_v["rms_pos"] <= m_m["rms_pos"]
    assert m_v["rms_pos"] < 0.10                             # menos de 10 cm en 4 m recorridos
    assert m_v["inclinacion_final"] < 0.2                    # la gravedad la hace observable


def test_vins_nees_razonable(simulacion):
    """Una secuencia no basta para medir la consistencia (eso lo hace la herramienta con
    varias), pero sí para detectar un estimador demasiado confiado (NEES >> 6) o demasiado
    prudente (NEES << 6). Con el rayo del ancla exacto (1 parámetro, como VINS) salía ~24."""
    v = simulacion["vins"]
    valor = nees(v["errores"], v["P"], POSE)[20:].mean()
    assert 2.0 < valor < 12.0


@pytest.fixture(scope="module")
def ventana_corta():
    """La ventana tras una secuencia de 2 s (arrancada en la verdad)."""
    cfg_sim = copy.deepcopy(CFG_SIM)
    cfg_sim["trayectoria"]["duracion_s"] = 2.0
    cfg_sim["mundo"]["puntos_por_superficie"] = 60
    cfg_v, _ = configuraciones(cfg_sim)
    s = sim.generar_secuencia(cfg_sim, K_SIM, TAMANO, np.random.default_rng(5))
    v = s["verdad"]
    e0 = EstadoImu(v["R_w_i"][0], v["v_w"][0], v["p_w"][0], v["b_g"][0], v["b_a"][0])
    ven = vt.VentanaDeslizante(e0, fa.sigmas_iniciales(cfg_v["inicial"]), s["imu"]["t_ns"][0], s["imu"]["w"][0],
                               s["imu"]["a"][0], s["R_c_i"], s["t_c_i"], K_SIM[0, 0], cfg_v,
                               {k: cfg_sim["imu"][k] for k in CLAVES_RUIDO})
    vt.ejecutar_secuencia(ven, s["imu"], s["obs"], s["camaras"]["t_ns"])
    return ven


def test_invariantes_de_los_puntos(ventana_corta):
    """Tras una secuencia: cada punto está anclado en un frame de la ventana, todas sus
    observaciones son de frames de la ventana y ninguna es anterior al ancla."""
    ven = ventana_corta
    ids_frames = {f.id for f in ven.frames}
    for p in ven.puntos.values():
        assert p.ancla in ids_frames
        assert set(p.obs) <= ids_frames
        assert min(p.obs) == p.ancla
        assert np.array_equal(p.xy_ancla, p.obs[p.ancla])
    assert len(ven.pres) == len(ven.frames) - 1
    assert isinstance(ven.prior, mg.FactorPriorMarginal)
    assert set(ven.prior_ids) <= ids_frames
    # Las preintegraciones encadenan los instantes de los frames
    for k, pre in enumerate(ven.pres):
        assert pre.dt_total == pytest.approx((ven.frames[k + 1].t_ns - ven.frames[k].t_ns) / 1e9, abs=1e-9)


def test_descarte_con_el_prior_tocando_el_segundo(ventana_corta):
    """El caso delicado: el 2.º más reciente está en el prior (porque observaba un punto
    anclado en un frame ya marginalizado) y hay que marginalizarlo del prior. Se comprueba que
    la secuencia lo ejercita; su corrección la cubren la consistencia y el error."""
    assert ventana_corta.descartes_en_prior > 0


def test_reanclar_conserva_el_punto_3d():
    cfg_v, _ = configuraciones(CFG_SIM)
    R_c_i, t_c_i = np.array(CFG_SIM["camara"]["R_i_c"]).T, np.zeros(3)
    e = EstadoImu(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    ven = vt.VentanaDeslizante(e, np.ones(15), 0, np.zeros(3), np.zeros(3), R_c_i, t_c_i, 643.0, cfg_v,
                               {"ruido_gir": 1e-4, "ruido_acc": 1e-3, "paseo_sesgo_gir": 1e-5, "paseo_sesgo_acc": 1e-4})
    viejo = vt.Frame(0, 0, EstadoImu(so3.exp_so3([0.1, 0.2, 0.3]), np.zeros(3), np.array([0.0, 0.0, 1.0]),
                                     np.zeros(3), np.zeros(3)))
    nuevo = vt.Frame(1, 1, EstadoImu(so3.exp_so3([0.15, 0.1, 0.35]), np.zeros(3), np.array([0.1, -0.05, 1.02]),
                                     np.zeros(3), np.zeros(3)))
    theta = np.array([0.05, -0.1, 0.4])
    p = vt.PuntoVentana(7, 0, theta[:2].copy(), theta.copy(), {0: theta[:2].copy(), 1: np.array([0.06, -0.11])})

    def mundo(theta, frame):
        R_c_w, t_c_w = ven.pose_camara(frame.estado)
        return R_c_w.T @ (np.array([theta[0], theta[1], 1.0]) / theta[2] - t_c_w)

    antes = mundo(p.theta, viejo)
    ven._reanclar(p, viejo, nuevo)
    assert p.ancla == 1 and np.array_equal(p.xy_ancla, p.obs[1])
    assert mundo(p.theta, nuevo) == pytest.approx(antes, abs=1e-12)