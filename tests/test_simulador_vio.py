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


# =====================================================================
# Paso 11.2
# =====================================================================
def test_ruido_blanco_tiene_sigma_N_raiz_f():
    rng = np.random.default_rng(0)
    N, f = 3.9e-3, 200
    blanco, _ = sim.ruido_y_sesgo(100_000, f, N, 0.0, np.zeros(3), rng)
    assert blanco.std(axis=0) == pytest.approx(np.full(3, N * np.sqrt(f)), rel=0.02)


def test_paseo_aleatorio_del_sesgo():
    rng = np.random.default_rng(1)
    Kp, f, n = 4.0e-4, 200, 1001                     # 5 s de simulación
    b0 = np.array([0.05, -0.04, 0.08])
    # 1. Cada paso suma un ruido de sigma = K·√Δt
    _, sesgo = sim.ruido_y_sesgo(20_000, f, 0.0, Kp, b0, rng)
    assert sesgo[0] == pytest.approx(b0)
    assert np.diff(sesgo, axis=0).std() == pytest.approx(Kp * np.sqrt(1 / f), rel=0.02)
    # 2. Tras T segundos, el sesgo se ha alejado de b0 con varianza K²·T
    finales = np.array([sim.ruido_y_sesgo(n, f, 0.0, Kp, b0, rng)[1][-1] - b0 for _ in range(200)])
    T = (n - 1) / f
    assert finales.var() == pytest.approx(Kp ** 2 * T, rel=0.15)


def test_imu_con_ruido():
    t_s = sim.tiempos_ns(5.0, 200) / 1e9
    tr = sim.imu_ideal(lambda t: sim.pose_vio(t, TR), t_s, TR["paso_derivadas_s"])
    w_ideal, a_ideal = sim.medidas_imu(tr["R_w_i"], tr["a_w"], tr["w_i"])
    w_m, a_m, b_g, b_a = sim.imu_con_ruido(tr, CFG["imu"], 200, np.random.default_rng(2))
    imu = CFG["imu"]
    assert (w_m - w_ideal - b_g).std() == pytest.approx(imu["ruido_gir"] * np.sqrt(200), rel=0.05)
    assert (a_m - a_ideal - b_a).std() == pytest.approx(imu["ruido_acc"] * np.sqrt(200), rel=0.05)
    assert b_g[0] == pytest.approx(imu["sesgo_inicial_gir"]) and b_a[0] == pytest.approx(imu["sesgo_inicial_acc"])



# =====================================================================
# Paso 11.3
# =====================================================================
def test_puntos_en_las_paredes():
    lx, ly, lz = CFG["mundo"]["habitacion_m"]
    p = sim.puntos_habitacion((lx, ly, lz), 100, np.random.default_rng(3))
    assert p.shape == (600, 3)
    # Cada punto está sobre alguna de las 6 superficies...
    en_superficie = ((np.abs(np.abs(p[:, 0]) - lx / 2) < 1e-12) | (np.abs(np.abs(p[:, 1]) - ly / 2) < 1e-12)
                     | (np.abs(p[:, 2]) < 1e-12) | (np.abs(p[:, 2] - lz) < 1e-12))
    assert en_superficie.all()
    # ...y dentro de la habitación
    assert (np.abs(p[:, 0]) <= lx / 2).all() and (np.abs(p[:, 1]) <= ly / 2).all()
    assert (p[:, 2] >= 0).all() and (p[:, 2] <= lz).all()


def test_extrinseca_y_pose_camara():
    R_c_i, t_c_i = sim.extrinseca(CFG["camara"])
    assert so3.es_rotacion(R_c_i)
    R_w_i, p_w_i = sim.pose_vio(3.7, TR)
    R_c_w, t_c_w = sim.pose_camara(R_w_i, p_w_i, R_c_i, t_c_i)
    # El eje óptico (z de la cámara) apunta hacia el eje x de la IMU
    assert R_c_w.T[:, 2] == pytest.approx(R_w_i[:, 0], abs=1e-12)
    # El centro de la cámara está en p_w_i + R_w_i · p_i_c
    centro = -R_c_w.T @ t_c_w
    assert centro == pytest.approx(p_w_i + R_w_i @ np.array(CFG["camara"]["p_i_c_m"]), abs=1e-12)
    # Por los dos caminos (mundo -> IMU -> cámara y mundo -> cámara) sale lo mismo
    p_w = np.array([2.0, -1.0, 0.5])
    por_imu = R_c_i @ (R_w_i.T @ (p_w - p_w_i)) + t_c_i
    assert R_c_w @ p_w + t_c_w == pytest.approx(por_imu, abs=1e-12)


def test_observar_puntos():
    rng = np.random.default_rng(4)
    puntos = sim.puntos_habitacion(CFG["mundo"]["habitacion_m"], 300, rng)
    R_c_i, t_c_i = sim.extrinseca(CFG["camara"])
    R_c_w, t_c_w = sim.pose_camara(*sim.pose_vio(1.0, TR), R_c_i, t_c_i)
    # Sin ruido: dentro de la imagen, delante, y xy = proyección exacta
    ids, uv, xy = sim.observar_puntos(R_c_w, t_c_w, puntos, K, TAMANO, 0.2, 0.0, rng)
    assert len(ids) > 50
    p_c = puntos[ids] @ R_c_w.T + t_c_w
    assert (p_c[:, 2] > 0.2).all()
    assert (uv[:, 0] >= 0).all() and (uv[:, 0] < 640).all() and (uv[:, 1] >= 0).all() and (uv[:, 1] < 480).all()
    assert xy == pytest.approx(p_c[:, :2] / p_c[:, 2:3], abs=1e-12)
    # Los no vistos están detrás o fuera de la imagen
    otros = np.setdiff1d(np.arange(len(puntos)), ids)
    q = puntos[otros] @ R_c_w.T + t_c_w
    u = 643.0 * q[:, 0] / q[:, 2] + 320.0
    v = 643.0 * q[:, 1] / q[:, 2] + 240.0
    assert ((q[:, 2] <= 0.2) | (u < 0) | (u >= 640) | (v < 0) | (v >= 480)).all()
    # Con ruido: los mismos puntos, desplazados 1 px de desviación típica
    ids2, uv2, _ = sim.observar_puntos(R_c_w, t_c_w, puntos, K, TAMANO, 0.2, 1.0, rng)
    assert (ids2 == ids).all()
    assert (uv2 - uv).std() == pytest.approx(1.0, rel=0.15)



# =====================================================================
# Paso 11.4
# =====================================================================
@pytest.fixture(scope="module")
def secuencia_sin_ruido_visual():
    """Secuencia de 10 s sin ruido en la cámara (una sola vez para todos los tests)."""
    cfg = copy.deepcopy(CFG)
    cfg["trayectoria"]["duracion_s"] = 10.0
    cfg["camara"]["ruido_px"] = 0.0
    return sim.generar_secuencia(cfg, K, TAMANO, np.random.default_rng(5))


def test_generar_secuencia(secuencia_sin_ruido_visual):
    s = secuencia_sin_ruido_visual
    f_imu, f_cam = TR["frecuencia_imu_hz"], CFG["camara"]["frecuencia_hz"]
    assert len(s["imu"]["t_ns"]) == 10 * f_imu + 1 and s["imu"]["t_ns"].dtype == np.int64
    assert len(s["camaras"]["t_ns"]) == 10 * f_cam + 1
    # Cada imagen coincide con una muestra de IMU, cada f_imu / f_cam muestras
    assert (np.diff(s["camaras"]["indice_imu"]) == f_imu // f_cam).all()
    assert (s["imu"]["t_ns"][s["camaras"]["indice_imu"]] == s["camaras"]["t_ns"]).all()
    # Todas las observaciones son de instantes de cámara, con muchos puntos por imagen
    assert np.isin(s["obs"]["t_ns"], s["camaras"]["t_ns"]).all()
    assert np.bincount(s["obs"]["imagen"]).min() > 50


def test_triangular_con_poses_verdaderas_recupera_los_puntos(secuencia_sin_ruido_visual):
    # Cruza tres módulos: simulador (etapa 11), pistas (etapa 8) y triangulación (etapa 9)
    from vio import triangulacion as tri
    s = secuencia_sin_ruido_visual
    cfg_tri = cargar_config("triangulacion")
    pistas, punto_de_pista = sim.pistas_desde_observaciones(s["obs"])
    largas = [p for p in pistas if p.longitud() >= 5][::10]   # una de cada 10: basta y es más rápido
    fiables = 0
    for pista in largas:
        poses = [(s["camaras"]["R_c_w"][j], s["camaras"]["t_c_w"][j]) for j in pista.imagenes]
        r = tri.triangular(poses, pista.xy, cfg_tri, K[0, 0])
        if r["fiable"]:
            fiables += 1
            assert np.linalg.norm(r["p_w"] - s["puntos"][punto_de_pista[pista.id]]) < 1e-9
    assert fiables / len(largas) > 0.8


def test_pistas_desde_observaciones(secuencia_sin_ruido_visual):
    obs = secuencia_sin_ruido_visual["obs"]
    pistas, punto_de_pista = sim.pistas_desde_observaciones(obs)
    # No se pierde ni se repite ninguna observación
    assert sum(p.longitud() for p in pistas) == len(obs["id"])
    # Cada pista: imágenes consecutivas, ids de pista únicos
    for p in pistas:
        assert (np.diff(p.imagenes) == 1).all()
    assert len({p.id for p in pistas}) == len(pistas)
    # Lo guardado en la pista coincide con la observación original
    p = pistas[0]
    fila = np.flatnonzero((obs["id"] == punto_de_pista[p.id]) & (obs["imagen"] == p.imagenes[0]))[0]
    assert p.xy[0] == pytest.approx(tuple(obs["xy"][fila]))
    assert p.t_ns[0] == obs["t_ns"][fila]