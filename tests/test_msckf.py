"""tests/test_msckf.py — Tests del MSCKF (vio/msckf.py), con el simulador de la etapa 11."""

import copy

import numpy as np
import pytest
from scipy.stats import chi2

from vio import msckf as ms
from vio import simulador as sim
from vio import so3
from vio.config import cargar_config
from vio.estado import EstadoImu
from vio.propagacion import propagar

CFG = cargar_config("msckf")
CFG_SIM = cargar_config("simulador")["vio"]
K = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])


def estado_ejemplo():
    """Un estado cualquiera (no trivial) para los tests de jacobianos."""
    return EstadoImu(so3.exp_so3([0.3, -0.2, 1.0]), np.array([0.5, -0.2, 0.1]), np.array([1.0, 2.0, 0.5]),
                     np.array([0.01, -0.02, 0.005]), np.array([0.1, -0.05, 0.2]))


def covarianza_al_azar(n, semilla):
    """Una matriz simétrica y positiva cualquiera (n x n)."""
    A = np.random.default_rng(semilla).normal(0.0, 1.0, (n, n))
    return A @ A.T / n + 0.1 * np.eye(n)


# =====================================================================
# Paso 12.1
# =====================================================================
def test_inyectar_y_diferencia_son_inversas():
    e = estado_ejemplo()
    dx = np.random.default_rng(0).normal(0.0, 0.01, 15)
    assert ms.diferencia_imu(ms.inyectar_imu(e, dx), e) == pytest.approx(dx, abs=1e-12)


def test_matrices_error():
    e = estado_ejemplo()
    w_hat, a_hat = np.array([0.3, -0.5, 0.8]), np.array([1.0, 0.5, 9.5])
    F, G = ms.matrices_error(e.R_w_i, w_hat, a_hat)
    assert F.shape == (15, 15) and G.shape == (15, 12)
    assert F[ms.TH, ms.TH] == pytest.approx(-so3.hat(w_hat))
    assert F[ms.V, ms.TH] == pytest.approx(-e.R_w_i @ so3.hat(a_hat))
    assert F[ms.P, ms.V] == pytest.approx(np.eye(3))
    assert np.abs(F[ms.BG]).max() == 0 and np.abs(F[ms.BA]).max() == 0    # los sesgos no dependen de nada


def test_phi_igual_que_diferencias_finitas():
    # Perturbar el estado, propagar un paso y ver cómo cambia el error: columna a columna de Phi
    e = estado_ejemplo()
    w_m, a_m = np.array([0.3, -0.5, 0.8]), np.array([1.0, 0.5, 9.5])
    discrepancias = []
    for dt in (0.005, 0.0025):
        F, G = ms.matrices_error(e.R_w_i, w_m - e.b_g, a_m - e.b_a)
        Phi, _ = ms.transicion(F, G, np.zeros((12, 12)), dt)
        nominal = propagar(e, w_m, a_m, dt)
        eps = 1e-6
        Phi_num = np.zeros((15, 15))
        for k in range(15):
            d = np.zeros(15)
            d[k] = eps
            mas = propagar(ms.inyectar_imu(e, d), w_m, a_m, dt)
            menos = propagar(ms.inyectar_imu(e, -d), w_m, a_m, dt)
            Phi_num[:, k] = (ms.diferencia_imu(mas, nominal) - ms.diferencia_imu(menos, nominal)) / (2 * eps)
        discrepancias.append(np.abs(Phi - Phi_num).max())
        assert np.abs(Phi_num - np.eye(15)).max() > 100 * discrepancias[-1]   # el test no es trivial
    # Coinciden salvo términos en dt²: al dividir dt por 2, la discrepancia baja a la cuarta parte
    assert discrepancias[0] < 2e-4
    assert discrepancias[1] / discrepancias[0] == pytest.approx(0.25, abs=0.02)


def test_propagar_covarianza_con_clones():
    P = covarianza_al_azar(15 + 12, 1)                         # IMU + 2 clones
    e = estado_ejemplo()
    F, G = ms.matrices_error(e.R_w_i, np.array([0.3, -0.5, 0.8]), np.array([1.0, 0.5, 9.5]))
    Phi, Qd = ms.transicion(F, G, ms.ruido_continuo(CFG["imu"]), 0.005)
    P2 = ms.propagar_covarianza(P, Phi, Qd)
    assert np.abs(P2 - P2.T).max() < 1e-15 and np.linalg.eigvalsh(P2).min() > 0
    assert P2[15:, 15:] == pytest.approx(P[15:, 15:])         # los clones no cambian
    assert P2[:15, 15:] == pytest.approx(Phi @ P[:15, 15:])   # su relación con la IMU, sí
    assert np.trace(P2[:15, :15]) > np.trace(Phi @ P[:15, :15] @ Phi.T)   # Q_d añade incertidumbre


def test_nees_solo_imu():
    # Solo IMU: el error real de [δθ, δv, δp] debe tener el tamaño que dice P (NEES ≈ 9).
    # P0 muy pequeña: así el error lo produce el ruido de la IMU y se pone a prueba Q_d
    tr_cfg = CFG_SIM["trayectoria"]
    t_ns = sim.tiempos_ns(3.0, 200)
    tr = sim.imu_ideal(lambda t: sim.pose_vio(t, tr_cfg), t_ns / 1e9, tr_cfg["paso_derivadas_s"])
    P0 = ms.covarianza_inicial(CFG["inicial"]) * 1e-6
    Qc = ms.ruido_continuo(CFG["imu"])
    rng = np.random.default_rng(0)
    n_sim = 30
    suma = 0.0
    for _ in range(n_sim):
        w_m, a_m, b_g, b_a = sim.imu_con_ruido(tr, CFG_SIM["imu"], 200, rng)
        verdad_0 = EstadoImu(tr["R_w_i"][0], tr["v_w"][0], tr["p_w"][0], b_g[0], b_a[0])
        e = ms.inyectar_imu(verdad_0, -(np.linalg.cholesky(P0) @ rng.normal(0.0, 1.0, 15)))
        P = P0.copy()
        for k in range(1, len(t_ns)):
            dt = (t_ns[k] - t_ns[k - 1]) / 1e9
            w_medio, a_medio = 0.5 * (w_m[k - 1] + w_m[k]), 0.5 * (a_m[k - 1] + a_m[k])
            F, G = ms.matrices_error(e.R_w_i, w_medio - e.b_g, a_medio - e.b_a)
            Phi, Qd = ms.transicion(F, G, Qc, dt)
            P = ms.propagar_covarianza(P, Phi, Qd)
            e = propagar(e, w_medio, a_medio, dt)
        verdad = EstadoImu(tr["R_w_i"][-1], tr["v_w"][-1], tr["p_w"][-1], b_g[-1], b_a[-1])
        error = ms.diferencia_imu(verdad, e)[:9]
        suma += error @ np.linalg.solve(P[:9, :9], error)
    nees = suma / n_sim
    assert chi2.ppf(0.0005, 9 * n_sim) / n_sim < nees < chi2.ppf(0.9995, 9 * n_sim) / n_sim



# =====================================================================
# Paso 12.2
# =====================================================================
R_C_I, T_C_I = sim.extrinseca(CFG_SIM["camara"])
R_I_C, P_I_C = R_C_I.T, -R_C_I.T @ T_C_I


def test_jacobiano_clon_igual_que_diferencias_finitas():
    e = estado_ejemplo()
    J = ms.jacobiano_clon(e.R_w_i, R_I_C, P_I_C)
    R_c, p_c = ms.pose_camara_desde_imu(e.R_w_i, e.p_w, R_I_C, P_I_C)
    eps = 1e-6
    J_num = np.zeros((6, 15))
    for k in range(15):
        d = np.zeros(15)
        d[k] = eps
        columnas = []
        for signo in (1.0, -1.0):
            e2 = ms.inyectar_imu(e, signo * d)
            R2, p2 = ms.pose_camara_desde_imu(e2.R_w_i, e2.p_w, R_I_C, P_I_C)
            columnas.append(np.concatenate([so3.log_so3(R_c.T @ R2), p2 - p_c]))   # error del clon
        J_num[:, k] = (columnas[0] - columnas[1]) / (2 * eps)
    assert np.abs(J - J_num).max() < 1e-8


def test_clonar_covarianza():
    P = covarianza_al_azar(15, 2)
    e = estado_ejemplo()
    J = ms.jacobiano_clon(e.R_w_i, R_I_C, P_I_C)
    P2 = ms.clonar_covarianza(P, J)
    assert P2.shape == (21, 21) and np.abs(P2 - P2.T).max() < 1e-15
    assert P2[:15, :15] == pytest.approx(P)                  # la IMU no cambia
    assert P2[15:, 15:] == pytest.approx(J @ P @ J.T)        # el clon: J·P·Jᵀ
    assert P2[15:, :15] == pytest.approx(J @ P)              # y está relacionado con la IMU
    # Con la cámara en el mismo sitio y con los mismos ejes que la IMU, el clon es [δθ, δp]
    J_identidad = ms.jacobiano_clon(e.R_w_i, np.eye(3), np.zeros(3))
    P3 = ms.clonar_covarianza(P, J_identidad)
    indices = np.r_[0:3, 6:9]
    assert P3[15:, 15:] == pytest.approx(P[np.ix_(indices, indices)])
    # Al clonar no se gana información: P sigue sin autovalores negativos (el clon repite a la IMU)
    assert np.linalg.eigvalsh(P2).min() > -1e-12


def test_borrar_clon():
    P = covarianza_al_azar(15 + 18, 3)                      # IMU + 3 clones
    P2 = ms.borrar_clon(P, 0)                               # olvidar el más antiguo
    quedan = np.r_[0:15, 21:33]
    assert P2.shape == (27, 27)
    assert P2 == pytest.approx(P[np.ix_(quedan, quedan)])
    P3 = ms.borrar_clon(P, 1)                               # o uno intermedio
    assert P3 == pytest.approx(P[np.ix_(np.r_[0:21, 27:33], np.r_[0:21, 27:33])])



# =====================================================================
# Paso 12.3
# =====================================================================
def test_jacobianos_observacion_igual_que_diferencias_finitas():
    R_w_c, p_w_c = so3.exp_so3([0.1, 1.4, -0.2]), np.array([0.5, -0.3, 1.2])
    p_w = p_w_c + R_w_c @ np.array([0.4, -0.3, 3.0])        # un punto delante de la cámara
    h, H_clon, H_f = ms.jacobianos_observacion(R_w_c, p_w_c, p_w)
    eps = 1e-6
    H_clon_num, H_f_num = np.zeros((2, 6)), np.zeros((2, 3))
    for k in range(6):
        d = np.zeros(6)
        d[k] = eps
        h_mas = ms.jacobianos_observacion(*ms.inyectar_clon(R_w_c, p_w_c, d), p_w)[0]
        h_menos = ms.jacobianos_observacion(*ms.inyectar_clon(R_w_c, p_w_c, -d), p_w)[0]
        H_clon_num[:, k] = (h_mas - h_menos) / (2 * eps)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        H_f_num[:, k] = (ms.jacobianos_observacion(R_w_c, p_w_c, p_w + d)[0]
                         - ms.jacobianos_observacion(R_w_c, p_w_c, p_w - d)[0]) / (2 * eps)
    assert np.abs(H_clon - H_clon_num).max() < 1e-8
    assert np.abs(H_f - H_f_num).max() < 1e-8


def test_proyectar_nulo():
    rng = np.random.default_rng(4)
    n = 6                                                     # 6 observaciones: 12 filas
    H_f = rng.normal(0.0, 1.0, (2 * n, 3))
    H_x = rng.normal(0.0, 1.0, (2 * n, 40))
    r = rng.normal(0.0, 1.0, 2 * n)
    H0, r0 = ms.proyectar_nulo(H_x, H_f, r)
    assert H0.shape == (2 * n - 3, 40) and r0.shape == (2 * n - 3,)
    # A cumple Aᵀ·H_f = 0, y sus columnas son ortonormales (Aᵀ·A = I): el ruido sigue siendo sigma²·I
    Q, _ = np.linalg.qr(H_f, mode="complete")
    A = Q[:, 3:]
    assert np.abs(A.T @ H_f).max() < 1e-12
    assert A.T @ A == pytest.approx(np.eye(2 * n - 3), abs=1e-12)
    assert H0 == pytest.approx(A.T @ H_x) and r0 == pytest.approx(A.T @ r)


def test_correccion_kalman():
    P = covarianza_al_azar(27, 5)
    H = np.random.default_rng(6).normal(0.0, 1.0, (9, 27))
    r = np.random.default_rng(7).normal(0.0, 1.0, 9)
    dx, P2 = ms.correccion_kalman(P, H, r, 0.5)
    assert np.abs(P2 - P2.T).max() < 1e-12 and np.linalg.eigvalsh(P2).min() > 0
    assert np.trace(P2) < np.trace(P)                         # la medida reduce la incertidumbre
    # Igual que la forma simple con la K óptima (etapa 10)
    S = H @ P @ H.T + 0.5 * np.eye(9)
    K_opt = P @ H.T @ np.linalg.inv(S)
    assert dx == pytest.approx(K_opt @ r)
    assert P2 == pytest.approx((np.eye(27) - K_opt @ H) @ P, abs=1e-10)


def test_prueba_chi2():
    rng = np.random.default_rng(8)
    P = covarianza_al_azar(21, 9) * 1e-4
    H0 = rng.normal(0.0, 1.0, (9, 21))
    S = H0 @ P @ H0.T + 0.01 * np.eye(9)
    L = np.linalg.cholesky(S)
    # Residuos con la covarianza que espera el filtro: pasan el 95 % de las veces
    pasan = [ms.prueba_chi2(H0, L @ rng.normal(0.0, 1.0, 9), P, 0.01, 0.95)[0] for _ in range(2000)]
    assert np.mean(pasan) == pytest.approx(0.95, abs=0.015)
    # Un residuo con un error grosero no pasa
    malo = L @ rng.normal(0.0, 1.0, 9) + 2.0
    assert not ms.prueba_chi2(H0, malo, P, 0.01, 0.95)[0]



# =====================================================================
# Paso 12.4
# =====================================================================
def test_propagar_hasta_parte_el_paso():
    # Sin giro y con medidas constantes, partir un paso en dos da lo mismo que el paso entero
    # (con giro solo casi: cada medio paso usa la orientación de su propia mitad)
    e = estado_ejemplo()
    w_m, a_m = e.b_g.copy(), np.array([1.0, 0.5, 9.5])     # w_m - b_g = 0: sin giro
    P0 = covarianza_al_azar(15, 10) * 1e-3
    args = (np.eye(3), np.zeros(3), 643.0, CFG, cargar_config("triangulacion"))
    entero = ms.Msckf(e, P0, 0, w_m, a_m, *args)
    entero.anadir_imu(10_000_000, w_m, a_m)
    partido = ms.Msckf(e, P0, 0, w_m, a_m, *args)
    partido.propagar_hasta(4_000_000)                       # una imagen a los 4 ms
    partido.anadir_imu(10_000_000, w_m, a_m)
    assert partido.t_ns == entero.t_ns == 10_000_000
    assert ms.diferencia_imu(partido.estado, entero.estado) == pytest.approx(np.zeros(15), abs=1e-12)
    assert partido.P == pytest.approx(entero.P, rel=1e-4)      # Phi se corta en dt²: difiere en dt³


@pytest.fixture(scope="module")
def simulacion():
    """Una secuencia de 5 s, el MSCKF sobre ella y la IMU sola, con el mismo arranque."""
    from vio.propagacion import propagar_secuencia
    cfg_sim = copy.deepcopy(CFG_SIM)
    cfg_sim["trayectoria"]["duracion_s"] = 5.0
    cfg_sim["mundo"]["puntos_por_superficie"] = 150           # menos puntos: test más rápido
    rng = np.random.default_rng(11)
    s = sim.generar_secuencia(cfg_sim, K, (640, 480), rng)
    v = s["verdad"]

    # Arranque con un error inicial sorteado con P0 (como en la etapa 10)
    P0 = ms.covarianza_inicial(CFG["inicial"])
    verdad_0 = EstadoImu(v["R_w_i"][0], v["v_w"][0], v["p_w"][0], v["b_g"][0], v["b_a"][0])
    error_0 = np.linalg.cholesky(P0) @ rng.normal(0.0, 1.0, 15)
    inicial = ms.inyectar_imu(verdad_0, -error_0)

    filtro = ms.Msckf(inicial, P0, s["imu"]["t_ns"][0], s["imu"]["w"][0], s["imu"]["a"][0],
                      s["R_c_i"], s["t_c_i"], K[0, 0], CFG, cargar_config("triangulacion"))
    res = ms.ejecutar_secuencia(filtro, s["imu"], s["obs"], s["camaras"]["t_ns"])
    solo_imu = propagar_secuencia(inicial, s["imu"]["t_ns"], s["imu"]["w"], s["imu"]["a"],
                                  cargar_config("orientacion")["integracion"])

    # Error real en cada imagen
    errores = []
    for j, k in enumerate(s["camaras"]["indice_imu"]):
        verdad = EstadoImu(v["R_w_i"][k], v["v_w"][k], v["p_w"][k], v["b_g"][k], v["b_a"][k])
        errores.append(ms.diferencia_imu(verdad, res["estados"][j]))
    return {"s": s, "filtro": filtro, "res": res, "solo_imu": solo_imu,
            "errores": np.array(errores), "error_0": error_0}


def test_msckf_corrige_la_deriva_de_la_imu(simulacion):
    s, errores = simulacion["s"], simulacion["errores"]
    error_msckf = np.linalg.norm(errores[-1, ms.P])
    error_imu = np.linalg.norm(simulacion["solo_imu"]["p"][-1] - s["verdad"]["p_w"][-1])
    assert error_msckf < 0.1                                  # menos de 10 cm tras 5 s (~3,7 m recorridos)
    assert error_msckf < 0.05 * error_imu                     # la IMU sola se va a metros
    # El error ya no crece como t²: el de la segunda mitad no es mucho mayor que el de la primera
    mitad = len(errores) // 2
    assert np.linalg.norm(errores[mitad:, ms.P], axis=1).max() < 3 * np.linalg.norm(errores[:mitad, ms.P], axis=1).max()


def test_msckf_estima_los_sesgos(simulacion):
    errores, P_final = simulacion["errores"], simulacion["res"]["P_imu"][-1]
    # La incertidumbre del sesgo del giroscopio baja mucho respecto a P0, y el error real con ella
    sigma_0 = CFG["inicial"]["sigma_sesgo_gir_rad_s"]
    assert np.sqrt(np.diag(P_final)[ms.BG]).max() < sigma_0 / 5
    assert np.linalg.norm(errores[-1, ms.BG]) < np.linalg.norm(simulacion["error_0"][ms.BG]) / 5
    # Y los errores finales de los sesgos caben en sus ±3 sigma
    sigmas = np.sqrt(np.diag(P_final))
    assert (np.abs(errores[-1, 9:15]) < 3 * sigmas[9:15]).all()


def test_msckf_ventana_y_covarianza(simulacion):
    filtro, recuentos = simulacion["filtro"], simulacion["res"]["recuentos"]
    assert len(filtro.clones) == CFG["ventana"]["clones_max"] - 1
    assert filtro.P.shape == (15 + 6 * len(filtro.clones),) * 2
    # Simétrica y sin autovalores negativos (puede ser casi singular: cada clon nace como copia
    # exacta de la IMU, y esa relación tarda en deshacerse)
    assert np.abs(filtro.P - filtro.P.T).max() < 1e-12
    assert np.linalg.eigvalsh(filtro.P).min() > -1e-12 * np.abs(filtro.P).max()
    # Los clones son imágenes consecutivas, las más recientes
    imagenes = [c["imagen"] for c in filtro.clones]
    assert imagenes == list(range(filtro.n_imagenes - len(imagenes), filtro.n_imagenes))
    # Se usan casi todas las pistas listas, y pocas no pasan la prueba chi-cuadrado
    usadas = sum(r["usadas"] for r in recuentos)
    rechazadas = sum(r["chi2"] for r in recuentos)
    assert usadas > 300 and rechazadas < 0.1 * usadas


def test_msckf_nees_razonable(simulacion):
    # Una sola simulación no basta para medir la consistencia (eso lo hace la herramienta con
    # muchas), pero sí para detectar un filtro muy confiado: NEES de la pose >> 6
    errores, P_imu = simulacion["errores"], simulacion["res"]["P_imu"]
    pose = np.r_[0:3, 6:9]
    nees = [e[pose] @ np.linalg.solve(P[np.ix_(pose, pose)], e[pose]) for e, P in zip(errores, P_imu)]
    assert np.mean(nees[20:]) < 15