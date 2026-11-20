"""tests/test_vins_estimador.py — Etapa 37: VINS con la interfaz del MSCKF (vins/estimador.py) y
dentro del pipeline (vio/pipeline.py con estimador.tipo = vins).

    - la interfaz: msckf.ejecutar_secuencia (hecha para el MSCKF) funciona igual con VINS;
    - la inicialización visual-inercial en marcha: arranca quieta, se mueve, inicializa sola y
      sigue a la verdad;
    - los ZUPT como factores de velocidad nula: en reposo, sin ellos la velocidad deriva;
    - el pipeline con VINS sobre una grabación sintética (arranque en reposo), con recuentos.
"""

import copy

import numpy as np
import pytest

from herramientas.comparar_trayectoria import alinear
from herramientas.grabar_simulacion import guardar_grabacion_sintetica
from apps.pi_vio import eventos_grabacion
from vio import msckf as ms
from vio import simulador as sim
from vio import so3
from vio.config import cargar_config
from vio.estado import EstadoImu
from vio.pipeline import PipelineVio, leer_poses, linea_recuentos
from vins import estimador as es
from vins import factores as fa
from vins.estimador import VinsEstimador

CFG_SIM = cargar_config("simulador")["vio"]
CFG_VINS = cargar_config("vins")
K = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])
CLAVES_RUIDO = ("ruido_gir", "ruido_acc", "paseo_sesgo_gir", "paseo_sesgo_acc")


def cfg_vins_sim():
    cfg = copy.deepcopy(CFG_VINS)
    cfg["visual"]["ruido_px"] = CFG_SIM["camara"]["ruido_px"]
    return cfg


def ruido_sim(cfg_sim):
    return {k: cfg_sim["imu"][k] for k in CLAVES_RUIDO}


def verdad_en(s, k):
    v = s["verdad"]
    return EstadoImu(v["R_w_i"][k], v["v_w"][k], v["p_w"][k], v["b_g"][k], v["b_a"][k])


# =====================================================================
# Interfaz
# =====================================================================
def test_interfaz_como_el_msckf():
    """ejecutar_secuencia del MSCKF (anadir_imu, procesar_imagen, estado, P) funciona con VINS
    arrancado en reposo (con el estado verdadero) y da las mismas claves de recuentos."""
    cfg_sim = copy.deepcopy(CFG_SIM)
    cfg_sim["trayectoria"]["duracion_s"] = 2.0
    cfg_sim["mundo"]["puntos_por_superficie"] = 80
    s = sim.generar_secuencia(cfg_sim, K, (640, 480), np.random.default_rng(1))
    imu = s["imu"]
    vins = VinsEstimador(imu["t_ns"][0], imu["w"][0], imu["a"][0], s["R_c_i"], s["t_c_i"], K[0, 0],
                         cfg_vins_sim(), ruido_sim(cfg_sim), estado=verdad_en(s, 0),
                         sigmas0=fa.sigmas_iniciales(CFG_VINS["inicial"]))
    res = ms.ejecutar_secuencia(vins, imu, s["obs"], s["camaras"]["t_ns"])

    claves_msckf = set(es.recuentos_vacios())
    assert all(claves_msckf <= set(r) for r in res["recuentos"])
    assert res["P_imu"].shape == (len(s["camaras"]["t_ns"]), 15, 15)
    assert np.all(np.linalg.eigvalsh(res["P_imu"][-1]) > 0)
    assert vins.n_imagenes == len(s["camaras"]["t_ns"])
    errores = [np.linalg.norm(e.p_w - s["verdad"]["p_w"][k]) for k, e in zip(s["camaras"]["indice_imu"], res["estados"])]
    assert max(errores) < 0.10                                  # como el test del MSCKF (etapa 12)
    # Salidas para el mapa: algún frame ha salido con sus puntos
    salientes = [r["saliente"] for r in res["recuentos"] if r["saliente"] is not None]
    assert salientes and all({"imagen", "t_ns", "R_w_c", "p_w_c", "obs"} <= set(sal) for sal in salientes)


# =====================================================================
# Inicialización visual-inercial en marcha
# =====================================================================
def test_inicializa_solo_al_moverse_y_sigue_a_la_verdad():
    """1 s quieta, 1 s de arranque suave y después el movimiento con vaivén: mientras está
    quieta no inicializa (no hay pose), y después arranca sola y sigue a la verdad (alineando
    la primera pose: VINS define su mundo al inicializar)."""
    cfg_sim = copy.deepcopy(CFG_SIM)
    cfg_sim["trayectoria"]["duracion_s"] = 5.0
    cfg_sim["trayectoria"]["oscilacion_z_m"] = 0.15
    cfg_sim["trayectoria"]["omega_z_rad_s"] = 4.0
    cfg_sim["mundo"]["puntos_por_superficie"] = 100
    cfg_tr = cfg_sim["trayectoria"]
    s = sim.generar_secuencia(cfg_sim, K, (640, 480), np.random.default_rng(2),
                              pose=lambda t: sim.pose_con_reposo(t, cfg_tr, 1.0, 1.0))
    imu, t_cam = s["imu"], s["camaras"]["t_ns"]
    vins = VinsEstimador(imu["t_ns"][0], imu["w"][0], imu["a"][0], s["R_c_i"], s["t_c_i"], K[0, 0],
                         cfg_vins_sim(), ruido_sim(cfg_sim))
    t_est, p_est, R_est, motivos = [], [], [], []
    siguiente = 1
    for j, t in enumerate(t_cam):
        while siguiente < len(imu["t_ns"]) and imu["t_ns"][siguiente] <= t:
            vins.anadir_imu(imu["t_ns"][siguiente], imu["w"][siguiente], imu["a"][siguiente])
            siguiente += 1
        vins.propagar_hasta(t)
        en = s["obs"]["imagen"] == j
        r = vins.procesar_imagen(t, s["obs"]["id"][en], s["obs"]["xy"][en])
        if vins.inicializado:
            t_est.append(t)
            p_est.append(vins.estado.p_w)
            R_est.append(vins.estado.R_w_i)
        else:
            motivos.append(r["motivo_inicializacion"])

    t_inicio = (t_est[0] - t_cam[0]) / 1e9
    assert t_inicio > 1.0                                      # quieta no inicializa
    assert any("poco excitada" in m for m in motivos)
    assert t_inicio < 4.0                                      # moviéndose, sí
    t_v, p_v, R_v = s["verdad"]["t_ns"], s["verdad"]["p_w"], s["verdad"]["R_w_i"]
    p_ali, _ = alinear(np.array(t_est), np.array(p_est), np.array(R_est), t_v, p_v, R_v)
    error = np.linalg.norm(np.array(p_est) - p_ali, axis=1)
    assert np.sqrt(np.mean(error ** 2)) < 0.05
    recorrido = np.sum(np.linalg.norm(np.diff(np.array(p_est), axis=0), axis=1))
    recorrido_v = np.sum(np.linalg.norm(np.diff(p_ali, axis=0), axis=1))
    assert abs(recorrido / recorrido_v - 1.0) < 0.05


# =====================================================================
# ZUPT como factor
# =====================================================================
@pytest.mark.parametrize("con_zupt", [False, True])
def test_zupt_en_reposo(con_zupt):
    """3 s quieta, sin paralaje (no se triangula nada): solo la IMU. Sin ZUPT la velocidad
    deriva con el ruido; con el factor de velocidad nula en cada imagen se queda en ~0."""
    cfg_sim = copy.deepcopy(CFG_SIM)
    cfg_sim["trayectoria"]["duracion_s"] = 3.0
    cfg_sim["mundo"]["puntos_por_superficie"] = 60
    R0, p0 = so3.exp_so3([0.1, -0.05, 0.3]), np.array([0.0, 0.0, 1.5])
    s = sim.generar_secuencia(cfg_sim, K, (640, 480), np.random.default_rng(3), pose=lambda t: (R0, p0))
    imu, t_cam = s["imu"], s["camaras"]["t_ns"]
    # Arranque con un pequeño error de velocidad (como tras una inicialización en reposo)
    inicial = verdad_en(s, 0)
    inicial.v_w = inicial.v_w + [0.02, -0.01, 0.0]
    vins = VinsEstimador(imu["t_ns"][0], imu["w"][0], imu["a"][0], s["R_c_i"], s["t_c_i"], K[0, 0],
                         cfg_vins_sim(), ruido_sim(cfg_sim), estado=inicial,
                         sigmas0=fa.sigmas_iniciales(CFG_VINS["inicial"]))
    siguiente = 1
    for j, t in enumerate(t_cam):
        while siguiente < len(imu["t_ns"]) and imu["t_ns"][siguiente] <= t:
            vins.anadir_imu(imu["t_ns"][siguiente], imu["w"][siguiente], imu["a"][siguiente])
            siguiente += 1
        en = s["obs"]["imagen"] == j
        vins.procesar_imagen(t, s["obs"]["id"][en], s["obs"]["xy"][en])
        if con_zupt:
            vins.corregir_reposo(cargar_config("vio")["reposo"]["sigma_velocidad_m_s"])
    v_final = np.linalg.norm(vins.estado.v_w)
    deriva = np.linalg.norm(vins.estado.p_w - p0)
    if con_zupt:
        assert v_final < 0.005 and deriva < 0.01
    else:
        assert v_final > 0.01                                  # sin información, el error se queda


# =====================================================================
# Pipeline
# =====================================================================
@pytest.fixture(scope="module")
def grabacion_corta(tmp_path_factory):
    """Grabación sintética de 6 s (3 s quieta, como la del MSCKF de la etapa 13), con manchas."""
    cfg_sim = copy.deepcopy(CFG_SIM)
    cfg_sim["trayectoria"]["duracion_s"] = 6.0
    cfg_sim["mundo"]["puntos_por_superficie"] = 100
    ext = {"R_i_c": CFG_SIM["camara"]["R_i_c"], "p_i_c_m": CFG_SIM["camara"]["p_i_c_m"], "t_d_ns": 0}
    carpeta = tmp_path_factory.mktemp("vins37")
    s = guardar_grabacion_sintetica(carpeta, cfg_sim, K, (640, 480), ext, np.random.default_rng(4))
    return carpeta, s


def test_pipeline_con_vins(grabacion_corta):
    carpeta, s = grabacion_corta
    cfgs = {n: copy.deepcopy(cargar_config(n)) for n in ("vio", "frontend", "msckf", "triangulacion", "vins")}
    cfgs["vio"]["estimador"] = {"tipo": "vins", "inicializacion_vins": "reposo"}
    pipeline = PipelineVio(K, np.zeros(5), s["R_c_i"], s["t_c_i"], 0, cfgs)
    resultados, lineas = [], []
    for evento in eventos_grabacion(carpeta, 0):
        if evento[0] == "imu":
            pipeline.nueva_imu(*evento[1:])
            continue
        r = pipeline.nueva_imagen(*evento[1:])
        if r is not None:
            resultados.append(r)
            lineas.append(linea_recuentos(r, pipeline.filtro.P))      # como pi_vio
    assert isinstance(pipeline.filtro, VinsEstimador)
    assert len(resultados) > 60
    assert sum(r["msckf"]["quieto"] for r in resultados) > 10          # ZUPT en el reposo
    assert all(np.isfinite([float(x) for x in l.split(",")]).all() for l in lineas)
    t_est = np.array([r["t_ns"] for r in resultados])
    p_est = np.array([r["p_w"] for r in resultados])
    R_est = np.array([r["R_w_i"] for r in resultados])
    p_v, _ = alinear(t_est, p_est, R_est, *leer_poses(carpeta / "verdad.csv"))
    assert np.sqrt(np.mean(np.sum((p_est - p_v) ** 2, axis=1))) < 0.10


def test_pipeline_sigue_con_el_msckf_por_defecto():
    cfgs = {n: cargar_config(n) for n in ("vio", "frontend", "msckf", "triangulacion")}
    assert cfgs["vio"]["estimador"]["tipo"] == "msckf"
    pipeline = PipelineVio(K, np.zeros(5), np.eye(3), np.zeros(3), 0, cfgs)
    assert pipeline.tipo_estimador == "msckf"