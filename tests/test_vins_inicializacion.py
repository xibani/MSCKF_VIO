"""tests/test_vins_inicializacion.py — Etapa 36: inicialización visual-inercial
(vins/inicializacion.py).

Piezas con datos exactos (cada una aislada) y los dos tests del plan en el simulador:
    - arranca en movimiento con error de escala < 5 % y de gravedad < 1°;
    - rechaza el movimiento insuficiente (reposo, giro puro y, además, el círculo suave del
      simulador, cuya escala no queda determinada en 1 s).
"Movimiento" es la trayectoria del simulador con un vaivén vertical como el de una placa
llevada a mano (±0,15 m a 4 rad/s, ~2,4 m/s² de pico): la inicialización necesita aceleración.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from vio import simulador as sim
from vio import so3
from vio.config import cargar_config
from vio.estado import EstadoImu
from vins import inicializacion as ini
from vins import ventana as vt

CFG_SIM = cargar_config("simulador")["vio"]
CFG_VINS = cargar_config("vins")
K = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])
CLAVES_RUIDO = ("ruido_gir", "ruido_acc", "paseo_sesgo_gir", "paseo_sesgo_acc")


# ---------------------------------------------------------------------
# Escenarios
# ---------------------------------------------------------------------
def secuencia(rng, pose=None, excitada=False, duracion_s=3.0, sin_ruido=False):
    cfg_sim = copy.deepcopy(CFG_SIM)
    cfg_sim["trayectoria"]["duracion_s"] = duracion_s
    if excitada:
        cfg_sim["trayectoria"]["oscilacion_z_m"] = 0.15
        cfg_sim["trayectoria"]["omega_z_rad_s"] = 4.0
    if sin_ruido:
        cfg_sim["camara"]["ruido_px"] = 1e-9
        for k in CLAVES_RUIDO:
            cfg_sim["imu"][k] *= 1e-9
    return sim.generar_secuencia(cfg_sim, K, (640, 480), rng, pose=pose), cfg_sim


def ventana_de_frames(s, cfg_sim, primera=20, cada=2, n=11):
    """n frames de la secuencia (una imagen de cada `cada`), con sus preintegraciones a b = 0."""
    o, cam, imu = s["obs"], s["camaras"], s["imu"]
    js = [primera + cada * q for q in range(n)]
    frames = [SimpleNamespace(t_ns=int(cam["t_ns"][j]),
                              obs={int(i): x for i, x in zip(o["id"][o["imagen"] == j], o["xy"][o["imagen"] == j])})
              for j in js]
    ruido = {k: max(cfg_sim["imu"][k], 1e-12) for k in CLAVES_RUIDO}
    pres = [ini.preintegrar_entre(imu["t_ns"], imu["w"], imu["a"], frames[q].t_ns, frames[q + 1].t_ns,
                                  np.zeros(3), np.zeros(3), ruido) for q in range(n - 1)]
    return frames, pres, [cam["indice_imu"][j] for j in js]


def cfg_test():
    cfg = copy.deepcopy(CFG_VINS)
    cfg["visual"]["ruido_px"] = CFG_SIM["camara"]["ruido_px"]
    return cfg


def errores(r, s, idx):
    """Error relativo de escala (longitud de la trayectoria) y de gravedad [°] (inclinación del
    primer frame)."""
    v = s["verdad"]
    p_est = np.array([e.p_w for e in r["estados"]])
    l_est = np.sum(np.linalg.norm(np.diff(p_est, axis=0), axis=1))
    l_ver = np.sum(np.linalg.norm(np.diff(v["p_w"][idx], axis=0), axis=1))
    z_est = r["estados"][0].R_w_i.T @ [0.0, 0.0, 1.0]
    z_ver = v["R_w_i"][idx[0]].T @ [0.0, 0.0, 1.0]
    return l_est / l_ver - 1.0, np.degrees(np.arccos(np.clip(z_est @ z_ver, -1.0, 1.0)))


# =====================================================================
# Piezas
# =====================================================================
def test_preintegrar_entre_igual_que_la_ventana():
    """Misma semántica que la ventana (anadir_imu + propagar_hasta), también con instantes
    que caen entre dos muestras."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 200) * 5_000_000 + 1_000
    w, a = rng.normal(0, 1, (200, 3)), rng.normal(0, 1, (200, 3)) + [0, 0, 9.8]
    t_ini, t_fin = int(t[20] + 2_345_678), int(t[150] + 1_234_567)
    pre = ini.preintegrar_entre(t, w, a, t_ini, t_fin, np.zeros(3), np.zeros(3), CFG_SIM["imu"])

    e0 = EstadoImu(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    ven = vt.VentanaDeslizante(e0, np.ones(15), t[0], w[0], a[0], np.eye(3), np.zeros(3), 643.0,
                               CFG_VINS, CFG_SIM["imu"])
    for k in range(1, 21):
        ven.anadir_imu(t[k], w[k], a[k])
    ven.propagar_hasta(t_ini)
    ven.pre = ven.pre.__class__(np.zeros(3), np.zeros(3), CFG_SIM["imu"])   # empezar en t_ini
    for k in range(21, 151):
        ven.anadir_imu(t[k], w[k], a[k])
    ven.propagar_hasta(t_fin)

    assert pre.dt_total == pytest.approx((t_fin - t_ini) / 1e9, abs=1e-12)
    assert pre.dt_total == pytest.approx(ven.pre.dt_total, abs=1e-12)
    assert pre.dv == pytest.approx(ven.pre.dv, abs=1e-12)
    assert pre.dp == pytest.approx(ven.pre.dp, abs=1e-12)


def test_traslacion_con_giro_en_escena_plana():
    """Puntos en un plano (una pared): la traslación con la rotación conocida sale exacta."""
    rng = np.random.default_rng(1)
    R_nl = so3.exp_so3([0.05, -0.2, 0.1])
    t_nl = np.array([0.3, -0.1, 0.05])
    obs_l, obs_n = {}, {}
    for i in range(40):
        P_l = np.array([rng.uniform(-1, 1), rng.uniform(-0.7, 0.7), 3.0])     # pared a 3 m
        P_n = R_nl @ P_l + t_nl
        obs_l[i], obs_n[i] = P_l[:2] / P_l[2], P_n[:2] / P_n[2]
    t, n_inl = ini.traslacion_con_giro(obs_l, obs_n, R_nl, 643.0, CFG_VINS["inicializacion"],
                                       np.random.default_rng(0))
    assert n_inl == 40
    assert t == pytest.approx(t_nl / np.linalg.norm(t_nl), abs=1e-9)


def test_sesgo_giroscopio_y_alineacion_exactos_con_poses_verdaderas():
    """Con las rotaciones y los centros de cámara VERDADEROS (a una escala conocida) y la IMU
    sin ruido (reintegradas con los sesgos verdaderos), el sesgo del giroscopio, la escala, la
    gravedad y las velocidades salen exactos: las ecuaciones de los pasos 2–4 no tienen errores."""
    s, cfg_sim = secuencia(np.random.default_rng(2), sin_ruido=True)
    frames, pres, idx = ventana_de_frames(s, cfg_sim)
    v = s["verdad"]
    p_i_c = -s["R_c_i"].T @ s["t_c_i"]
    R_b = [v["R_w_i"][k] for k in idx]
    escala = 0.37
    c_sfm = [(v["p_w"][k] + v["R_w_i"][k] @ p_i_c) / escala for k in idx]

    # Paso 2: sesgo del giroscopio desde las rotaciones verdaderas
    b_g = ini.estimar_sesgo_giroscopio(R_b, pres)
    assert b_g == pytest.approx(v["b_g"][idx[0]], abs=1e-6)

    # Pasos 3–4 con las preintegraciones en los sesgos verdaderos
    for pre in pres:
        pre.reintegrar(v["b_g"][idx[0]], v["b_a"][idx[0]])
    vel, g, s_est, rms = ini.alinear_lineal(R_b, c_sfm, pres, p_i_c)
    assert s_est == pytest.approx(escala, rel=1e-5)
    assert g == pytest.approx([0.0, 0.0, -9.80665], abs=1e-4)
    assert np.abs(vel - v["v_w"][idx]).max() < 1e-4
    vel, g, s_est, rms = ini.refinar_gravedad(R_b, c_sfm, pres, p_i_c, g + [0.1, -0.1, 0.3], 4)
    assert np.linalg.norm(g) == pytest.approx(9.80665, abs=1e-12)
    assert s_est == pytest.approx(escala, rel=1e-5)
    assert rms < 1e-6


# =====================================================================
# Los dos tests del plan
# =====================================================================
@pytest.mark.parametrize("semilla", [1, 2, 3])
def test_arranca_en_movimiento(semilla):
    s, cfg_sim = secuencia(np.random.default_rng(semilla), excitada=True, duracion_s=4.0)
    frames, pres, idx = ventana_de_frames(s, cfg_sim)
    r = ini.inicializar(frames, pres, s["R_c_i"], s["t_c_i"], K[0, 0], cfg_test())
    assert r["ok"], r["motivo"]
    e_escala, e_gravedad = errores(r, s, idx)
    assert abs(e_escala) < 0.05
    assert e_gravedad < 1.0
    # Y lo demás, razonable: sesgo del giroscopio, puntos y preintegraciones reintegradas
    assert np.degrees(np.linalg.norm(r["b_g"] - s["verdad"]["b_g"][idx[0]])) < 0.5     # °/s
    assert len(r["puntos"]) > 30
    assert r["incertidumbre_escala"] < CFG_VINS["inicializacion"]["incertidumbre_escala_max"]


def test_rechaza_el_reposo():
    R0, p0 = so3.exp_so3([0.1, -0.05, 0.3]), np.array([0.0, 0.0, 1.5])
    s, cfg_sim = secuencia(np.random.default_rng(4), pose=lambda t: (R0, p0))
    frames, pres, _ = ventana_de_frames(s, cfg_sim)
    r = ini.inicializar(frames, pres, s["R_c_i"], s["t_c_i"], K[0, 0], cfg_test())
    assert not r["ok"] and "poco excitada" in r["motivo"]


def test_rechaza_el_giro_puro():
    """Girando sobre sí misma (la cámara está a 3 cm de la IMU) la IMU sí ve aceleración (la
    gravedad cambia de dirección en sus ejes) pero no hay paralaje sin giro."""
    R0, p0 = so3.exp_so3([0.1, -0.05, 0.3]), np.array([0.0, 0.0, 1.5])
    s, cfg_sim = secuencia(np.random.default_rng(5), pose=lambda t: (
        so3.exp_so3([0.0, 0.0, 0.8 * t]) @ so3.exp_so3([0.2 * np.sin(2 * t), 0.0, 0.0]) @ R0, p0))
    frames, pres, _ = ventana_de_frames(s, cfg_sim)
    r = ini.inicializar(frames, pres, s["R_c_i"], s["t_c_i"], K[0, 0], cfg_test())
    assert not r["ok"] and "paralaje" in r["motivo"]
    assert r["excitacion"] > CFG_VINS["inicializacion"]["excitacion_min_m_s2"]   # no lo para la IMU


def test_rechaza_el_movimiento_suave():
    """El círculo suave del simulador (0,2 m/s²): SfM y alineación salen, pero con b_a
    desconocido la escala no queda determinada en 1 s (σ_L/L ~12 %): se rechaza en vez de
    arrancar con un 10 % de error de escala."""
    s, cfg_sim = secuencia(np.random.default_rng(1))
    frames, pres, idx = ventana_de_frames(s, cfg_sim)
    r = ini.inicializar(frames, pres, s["R_c_i"], s["t_c_i"], K[0, 0], cfg_test())
    assert not r["ok"] and "escala sin determinar" in r["motivo"]
    assert r["incertidumbre_escala"] > 0.05


# =====================================================================
# Etapa 37 — Sesgo del giroscopio en reposo
# =====================================================================
def test_sesgo_en_reposo():
    """Quieta: estima el sesgo verdadero. Girando a velocidad CONSTANTE (giroscopio estable, pero
    la gravedad cambia de dirección en los ejes de la IMU): no lo confunde con reposo."""
    cfg_e = CFG_VINS["estimador"]
    R0, p0 = so3.exp_so3([0.1, -0.05, 0.3]), np.array([0.0, 0.0, 1.5])
    s, _ = secuencia(np.random.default_rng(6), pose=lambda t: (R0, p0), duracion_s=2.0)
    det = ini.SesgoEnReposo(cfg_e)
    quieta = [det.actualizar(t, w, a) for t, w, a in zip(s["imu"]["t_ns"], s["imu"]["w"], s["imu"]["a"])]
    assert any(quieta)
    assert np.degrees(np.abs(det.b_g - s["verdad"]["b_g"][-1])).max() < 0.05     # °/s

    s, _ = secuencia(np.random.default_rng(7), duracion_s=2.0,
                     pose=lambda t: (so3.exp_so3([0.6 * t, 0.0, 0.0]) @ R0, p0))   # 34 °/s alrededor de x
    det = ini.SesgoEnReposo(cfg_e)
    assert not any(det.actualizar(t, w, a) for t, w, a in zip(s["imu"]["t_ns"], s["imu"]["w"], s["imu"]["a"]))