"""
vins/inicializacion.py — Etapa 36: inicialización visual-inercial (Qin et al., VINS-Mono,
T-RO 2018, sección V; Qin y Shen, "Robust initialization of monocular visual-inertial
estimation on aerial robots", IROS 2017).

Entrada: los frames de la ventana (instante y observaciones) y las preintegraciones entre
frames consecutivos. Salida: el estado de cada frame en el mundo (z hacia arriba, gravedad
(0, 0, −9,80665), rumbo del primer frame = 0, origen en el primer frame), los puntos (α, β, ρ)
anclados en su primera observación y el sesgo del giroscopio. Pasos:

    1. SfM solo visual, a escala arbitraria, en los ejes de la cámara de un frame l:
       frame l con paralaje suficiente respecto al más reciente → pose relativa (rotación del
       giroscopio + traslación lineal, o matriz esencial) → triangulación → PnP del resto →
       ajuste de haces solo visual (optimizador de la etapa 34 sin factores IMU).
    2. Sesgo del giroscopio:  min Σ |Log(ΔR̂_k,k+1(b_g)ᵀ·R_bk_bk+1^sfm)|², lineal con J^R_bg.
    3. Velocidades, gravedad y escala, lineal (X = [v_0 … v_n, g^c0, s]).
    4. Refinado de la gravedad con |g| = 9,80665 (2 grados en el plano tangente).
    5. Paso al mundo.
    6. Ajuste de haces visual-inercial de la ventana (lo que VINS hace justo después): corrige
       escala y gravedad con los pesos correctos y estima b_a.
Se rechaza (y se espera a la ventana siguiente) si la IMU está casi quieta, falta paralaje, falla
la pose relativa o algún PnP, el ajuste solo visual deja error alto, s <= 0, |g| sale lejos de 9,8,
el residuo de la alineación es alto o, tras el ajuste visual-inercial, la escala sigue sin
determinar (incertidumbre relativa de la longitud de la trayectoria por encima del umbral).
Criterio medido en el simulador con ventanas de 1 s: con el círculo suave (0,2 m/s² de
excitación) σ_L/L ≈ 12–13 % y errores de escala de −8 a −13 %; con un vaivén como el de una placa
a mano (1,9 m/s²) σ_L/L ≈ 0,7 % y errores <= 1,5 %.

Diferencias con VINS-Mono (decisiones documentadas):
- El frame l se elige con el paralaje medio QUITANDO EL GIRO (con el giroscopio preintegrado),
  no con el paralaje bruto: con un giro puro el bruto supera el umbral y la esencial da una
  traslación sin sentido.
- Pose relativa l–n: por defecto, rotación del GIROSCOPIO y traslación lineal (2 puntos +
  RANSAC), no la matriz esencial de 5 puntos. Con una escena casi plana (una pared) la esencial
  tiene varias soluciones compatibles con todos los puntos: en el simulador, con datos exactos,
  salía a 9,6° de la verdad. La esencial sigue disponible (pose_relativa: esencial), con la
  comprobación de que su rotación coincide con la del giroscopio. El ajuste de haces posterior
  refina las rotaciones solo con la imagen.
- Gauge del ajuste solo visual: pose del frame l fija (6 grados) y DISTANCIA l–n fija (1 grado,
  la escala). VINS fija la traslación entera del frame n, que quita además dos grados que sí
  se observan.
- Sesgo del acelerómetro: la alineación lineal lo deja en 0 (como VINS). En el simulador (b_a de
  0,05–0,08 m/s² y una trayectoria suave) eso solo ya da un 14 % de error de escala con las poses
  VERDADERAS. Estimarlo en la alineación lineal funciona con poses exactas pero no con las de la
  SfM (pesar con la covarianza de la IMU ignora el error de la SfM y la escala se hunde), así que
  lo corrige el paso 6, que es un ajuste conjunto con los pesos correctos.
- Velocidades en los ejes de c0 (no en los de cada frame, como VINS): las ecuaciones quedan
  algo más sencillas; la solución es la misma.
"""

import cv2
import numpy as np

from vio import so3
from vio import triangulacion as tri
from vio.constantes import G
from vio.estado import EstadoImu
from vins import factores as fa
from vins import optimizador as op
from vins.preintegracion import DIM, P, Preintegracion

E_Z = np.array([0.0, 0.0, 1.0])


# =====================================================================
# Paso 36.0 — Preintegración entre dos instantes cualesquiera
# =====================================================================
def preintegrar_entre(t_imu, w, a, t_ini_ns, t_fin_ns, b_g, b_a, ruido):
    """Preintegración de t_ini a t_fin con la MISMA semántica que el MSCKF y la ventana
    (anadir_imu + propagar_hasta): de t_ini a la primera muestra posterior, la media de la
    última muestra <= t_ini y esa; entre muestras, la media de cada dos; de la última muestra
    <= t_fin a t_fin, esa muestra sola.

    t_imu: (N,) int64 [ns]. w, a: (N, 3). Devuelve: Preintegracion.
    """
    pre = Preintegracion(b_g, b_a, ruido)
    k = int(np.searchsorted(t_imu, t_ini_ns, side="right")) - 1     # última muestra <= t_ini
    if k < 0:
        raise ValueError("No hay muestras de IMU antes del instante inicial")
    t_act, w_ant, a_ant = int(t_ini_ns), w[k], a[k]
    k += 1
    while k < len(t_imu) and t_imu[k] <= t_fin_ns:
        dt = (int(t_imu[k]) - t_act) / 1e9
        if dt > 0:
            pre.anadir(dt, 0.5 * (w_ant + w[k]), 0.5 * (a_ant + a[k]))
            t_act = int(t_imu[k])
        w_ant, a_ant = w[k], a[k]
        k += 1
    dt = (int(t_fin_ns) - t_act) / 1e9
    if dt > 0:
        pre.anadir(dt, w_ant, a_ant)
    return pre


# =====================================================================
# Paso 36.1 — SfM solo visual
# =====================================================================
def rotacion_relativa_giro(pres, desde, hasta):
    """R_b_desde_b_hasta a partir del giroscopio: producto de los ΔR de las preintegraciones."""
    R = np.eye(3)
    for k in range(desde, hasta):
        R = R @ pres[k].dR
    return R


def paralaje_sin_giro(obs_a, obs_b, R_cb_ca, focal_px):
    """Paralaje medio [px] entre dos frames quitando el giro: cada rayo de a se gira a la
    orientación de b y se mide lo que falta. Devuelve: (paralaje_px, ids comunes)."""
    comunes = [i for i in obs_a if i in obs_b]
    if not comunes:
        return 0.0, comunes
    d = []
    for i in comunes:
        rayo = R_cb_ca @ np.array([obs_a[i][0], obs_a[i][1], 1.0])
        d.append(focal_px * np.linalg.norm(obs_b[i] - rayo[:2] / rayo[2]))
    return float(np.mean(d)), comunes


def elegir_frame_l(frames, pres, R_c_i, focal_px, cfg):
    """El frame más antiguo con suficientes correspondencias y paralaje sin giro respecto al
    más reciente (VINS: >= 20 correspondencias y ~30 px). Devuelve: (l o None, paralaje_px)."""
    n = len(frames) - 1
    mejor = 0.0
    for l in range(n):
        R_bl_bn = rotacion_relativa_giro(pres, l, n)
        R_cn_cl = R_c_i @ R_bl_bn.T @ R_c_i.T
        paralaje, comunes = paralaje_sin_giro(frames[l].obs, frames[n].obs, R_cn_cl, focal_px)
        if len(comunes) >= cfg["correspondencias_min"]:
            mejor = max(mejor, paralaje)
            if paralaje >= cfg["paralaje_min_px"]:
                return l, paralaje
    return None, mejor


def pose_relativa_esencial(obs_l, obs_n, focal_px, cfg):
    """Matriz esencial con RANSAC (5 puntos) entre l y n y su descomposición.

    Devuelve: (R_cn_cl, t_cn_cl (|t| = 1), número de inliers) o None.
    Convención de OpenCV: x_n ∝ R·x_l + t.
    """
    comunes = [i for i in obs_l if i in obs_n]
    p_l = np.array([obs_l[i] for i in comunes], dtype=np.float64)
    p_n = np.array([obs_n[i] for i in comunes], dtype=np.float64)
    E, mascara = cv2.findEssentialMat(p_l, p_n, cameraMatrix=np.eye(3), method=cv2.RANSAC,
                                      prob=cfg["confianza_ransac"],
                                      threshold=cfg["umbral_ransac_px"] / focal_px)
    if E is None or E.shape != (3, 3):
        return None
    n_inliers, R, t, _ = cv2.recoverPose(E, p_l, p_n, cameraMatrix=np.eye(3), mask=mascara)
    return R, t.ravel(), int(n_inliers)


def triangular_puntos(frames, poses, puntos_3d, cfg):
    """Triangula los puntos aún sin 3D vistos en al menos dos frames con pose.
    poses: dict índice de frame -> (R_c_w, t_c_w). puntos_3d: dict id -> p_w (se amplía).
    Devuelve: número de puntos nuevos."""
    n_nuevos = 0
    ids = set()
    for k in poses:
        ids.update(frames[k].obs)
    for i in ids:
        if i in puntos_3d:
            continue
        ks = [k for k in sorted(poses) if i in frames[k].obs]
        if len(ks) < 2:
            continue
        ps = [poses[k] for k in ks]
        xy = np.array([frames[k].obs[i] for k in ks])
        if tri.angulo_paralaje(ps, xy) < cfg["paralaje_min_triangular_grados"]:
            continue
        p_w = tri.triangular_lineal(ps, xy)
        if p_w is None:
            continue
        if min(tri.transformar(R, t, p_w)[2] for R, t in ps) <= 0:
            continue
        puntos_3d[i] = p_w
        n_nuevos += 1
    return n_nuevos


def pose_por_pnp(frame, puntos_3d, focal_px, cfg):
    """Pose (R_c_w, t_c_w) de un frame con PnP + RANSAC sobre los puntos con 3D (cámara
    normalizada: K = I). Devuelve None si no hay bastantes puntos o falla."""
    ids = [i for i in frame.obs if i in puntos_3d]
    if len(ids) < cfg["puntos_pnp_min"]:
        return None
    objetos = np.array([puntos_3d[i] for i in ids], dtype=np.float64)
    imagen = np.array([frame.obs[i] for i in ids], dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(objetos, imagen, np.eye(3), None,
                                                 reprojectionError=cfg["umbral_ransac_px"] / focal_px,
                                                 confidence=cfg["confianza_ransac"],
                                                 flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < cfg["puntos_pnp_min"]:
        return None
    # Refinado con todos los inliers (Levenberg-Marquardt de OpenCV)
    idx = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(objetos[idx], imagen[idx], np.eye(3), None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def traslacion_con_giro(obs_l, obs_n, R_nl, focal_px, cfg, rng):
    """Traslación t_cn_cl (|t| = 1) conocida la rotación R_nl = R_cn_cl (del giroscopio).

    Restricción epipolar con E = [t]x·R:  x̄_nᵀ·[t]x·R·x̄_l = 0   <=>   ((R·x̄_l) × x̄_n)ᵀ·t = 0,
    lineal en t: con 2 puntos, t = a_1 × a_2 (a_k = (R·x̄_l,k) × x̄_n,k). RANSAC con la distancia
    de x_n a su recta epipolar; después, mínimos cuadrados con todos los inliers (vector singular
    de A) y el signo que deja más puntos delante de las dos cámaras.
    Sin la ambigüedad de la esencial con escenas planas (ver sfm_visual).
    Devuelve: (t, número de inliers) o None.
    """
    comunes = [i for i in obs_l if i in obs_n]
    if len(comunes) < 2:
        return None
    xl = np.array([[obs_l[i][0], obs_l[i][1], 1.0] for i in comunes])
    xn = np.array([[obs_n[i][0], obs_n[i][1], 1.0] for i in comunes])
    A = np.cross(xl @ R_nl.T, xn)                     # fila k: (R·x̄_l,k) × x̄_n,k
    umbral = cfg["umbral_ransac_px"] / focal_px

    def distancias(t):
        lineas = (so3.hat(t) @ R_nl @ xl.T).T         # recta epipolar de cada x_l en la imagen n
        return np.abs(np.sum(lineas * xn, axis=1)) / np.linalg.norm(lineas[:, :2], axis=1)

    # 1. RANSAC con muestras mínimas de 2
    mejores = np.zeros(len(comunes), dtype=bool)
    for _ in range(cfg["iteraciones_ransac"]):
        i, j = rng.choice(len(comunes), 2, replace=False)
        t = np.cross(A[i], A[j])
        if np.linalg.norm(t) < 1e-12:
            continue
        inliers = distancias(t / np.linalg.norm(t)) < umbral
        if inliers.sum() > mejores.sum():
            mejores = inliers
    if mejores.sum() < 2:
        return None
    # 2. Todos los inliers: t = vector singular de menor valor singular de A_inliers
    t = np.linalg.svd(A[mejores])[2][-1]
    # 3. Signo: profundidades positivas.  z_n·x̄_n = z_l·R·x̄_l + t  (mínimos cuadrados en z_l, z_n)
    delante = 0
    for k in np.flatnonzero(mejores):
        M = np.column_stack([R_nl @ xl[k], -xn[k]])
        z = np.linalg.lstsq(M, -t, rcond=None)[0]
        delante += 1 if (z[0] > 0 and z[1] > 0) else -1
    if delante < 0:
        t = -t
    return t, int(mejores.sum())


class FactorDistancia:
    """Fija la escala del SfM: r = |p_b − p_a| − d0 (con la interfaz de los priores)."""

    def __init__(self, a, b, d0, sigma):
        self.indices = [a, b]
        self.d0 = d0
        self.W = np.array([[1.0 / sigma ** 2]])

    def informacion(self):
        return self.W

    def evaluar_bloques(self, estados):
        d = estados[1].p_w - estados[0].p_w
        u = d / np.linalg.norm(d)
        J_a, J_b = np.zeros((1, DIM)), np.zeros((1, DIM))
        J_a[0, P] = -u
        J_b[0, P] = u
        return np.array([np.linalg.norm(d) - self.d0]), [J_a, J_b]


def ajuste_haces_visual(frames, poses, puntos_3d, l, n, focal_px, cfg_vins):
    """Ajuste de haces solo visual con el optimizador de la etapa 34.

    Truco: cada frame es un EstadoImu con la pose de la CÁMARA (R_w_i = R_c_wᵀ, p_w = centro)
    y extrínseca identidad; velocidad y sesgos no intervienen (un prior débil los deja quietos).
    Gauge: pose de l fija y distancia l–n fija.
    Devuelve: (poses ajustadas, dict id -> (ancla, xy_ancla, θ) en ejes de la cámara del ancla,
    error RMS de reproyección [px]).
    """
    cfg = cfg_vins["inicializacion"]
    ks = sorted(poses)
    mapa = {k: q for q, k in enumerate(ks)}
    estados = []
    for k in ks:
        R_c_w, t_c_w = poses[k]
        estados.append(EstadoImu(R_c_w.T, np.zeros(3), -R_c_w.T @ t_c_w, np.zeros(3), np.zeros(3)))

    # Puntos anclados en su primera observación (entre los frames con pose)
    puntos, visuales, ids = [], [], []
    for i, p_w in puntos_3d.items():
        kobs = [k for k in ks if i in frames[k].obs]
        h = kobs[0]
        R_c_w, t_c_w = poses[h]
        x, y, z = R_c_w @ p_w + t_c_w
        lp = len(puntos)
        puntos.append(op.Punto(mapa[h], frames[h].obs[i], np.array([x / z, y / z, 1.0 / z])))
        ids.append(i)
        for k in kobs[1:]:
            visuales.append(fa.FactorVisual(lp, mapa[k], frames[k].obs[i]))

    # Priores: gauge (l fijo; distancia l–n fija) y velocidad/sesgos quietos en todos
    grande, fijo = cfg["sigma_libre"], cfg["sigma_fijo"]
    priores = []
    for k in ks:
        s = np.full(15, grande)
        s[3:6] = s[9:15] = 1.0                        # v, b_g, b_a: sin información, a 0
        if k == l:
            s[0:3] = s[6:9] = fijo                    # orientación y posición de l
        priores.append(fa.FactorPriorEstado(mapa[k], estados[mapa[k]], s))
    d0 = np.linalg.norm(estados[mapa[n]].p_w - estados[mapa[l]].p_w)
    priores.append(FactorDistancia(mapa[l], mapa[n], d0, fijo))

    prob = op.ProblemaLote(estados, puntos, [], visuales, None, np.eye(3), np.zeros(3), priores=priores)
    opt = op.OptimizadorLote(cfg_vins, focal_px)
    opt.optimizar(prob, iteraciones_max=cfg["iteraciones_ba"])

    # Error RMS de reproyección (visuales + anclas)
    validas, usados = opt.observaciones_validas(prob)
    errores = [r for _, r, *_ in validas]
    errores += [f.evaluar(prob.thetas[f.punto])[0] for f in prob.factores_ancla if f.punto in usados]
    rms_px = focal_px * float(np.sqrt(np.mean([e @ e for e in errores]))) if errores else np.inf

    poses_aj = {k: (prob.estados[q].R_w_i.T, -prob.estados[q].R_w_i.T @ prob.estados[q].p_w)
                for k, q in mapa.items()}
    puntos_aj = {i: (ks[prob.puntos[q].ancla], prob.puntos[q].xy_ancla, prob.thetas[q].copy())
                 for q, i in enumerate(ids) if q in usados}
    return poses_aj, puntos_aj, rms_px


def sfm_visual(frames, pres, R_c_i, focal_px, cfg_vins):
    """Estructura y poses de cámara a escala arbitraria en los ejes de la cámara l.

    Devuelve: dict con ok, motivo, l, paralaje_px, poses (índice -> (R_c_w, t_c_w), w = c_l),
    puntos (id -> (ancla, xy_ancla, θ)), rms_px.
    """
    cfg = cfg_vins["inicializacion"]
    n = len(frames) - 1
    res = {"ok": False, "motivo": "", "l": None, "paralaje_px": 0.0}

    # 1. Frame l
    l, paralaje = elegir_frame_l(frames, pres, R_c_i, focal_px, cfg)
    res["paralaje_px"] = paralaje
    if l is None:
        res["motivo"] = "paralaje insuficiente"
        return res
    res["l"] = l

    # 2. Pose relativa de n respecto a l
    R_nl_giro = R_c_i @ rotacion_relativa_giro(pres, l, n).T @ R_c_i.T
    if cfg["pose_relativa"] == "giro":
        # Rotación del giroscopio y traslación lineal (por defecto: ver la nota de abajo)
        rel = traslacion_con_giro(frames[l].obs, frames[n].obs, R_nl_giro, focal_px, cfg,
                                  np.random.default_rng(cfg["semilla_ransac"]))
        if rel is None or rel[1] < cfg["inliers_min"]:
            res["motivo"] = "traslación l–n con pocos inliers"
            return res
        R_nl, t_nl = R_nl_giro, rel[0]
    else:
        # Matriz esencial (5 puntos), como VINS. Con puntos casi coplanares (una pared) tiene
        # varias soluciones compatibles: en el simulador, sin ruido y con los 35 puntos como
        # inliers, daba una rotación a 9,6° de la verdad (el giroscopio, a 0,4°)
        rel = pose_relativa_esencial(frames[l].obs, frames[n].obs, focal_px, cfg)
        if rel is None or rel[2] < cfg["inliers_min"]:
            res["motivo"] = "matriz esencial con pocos inliers"
            return res
        R_nl, t_nl, _ = rel
        if np.degrees(so3.angulo_entre(R_nl, R_nl_giro)) > cfg["diferencia_giro_max_grados"]:
            res["motivo"] = "la esencial no coincide con el giroscopio"
            return res

    # 3. Poses de l y n, triangulación, PnP del resto (en pasadas, hasta que no avance)
    poses = {l: (np.eye(3), np.zeros(3)), n: (R_nl, t_nl)}
    puntos_3d = {}
    triangular_puntos(frames, poses, puntos_3d, cfg)
    avanza = True
    while avanza and len(poses) < len(frames):
        avanza = False
        for k in sorted(set(range(len(frames))) - set(poses), key=lambda k: min(abs(k - l), abs(k - n))):
            pose = pose_por_pnp(frames[k], puntos_3d, focal_px, cfg)
            if pose is not None:
                poses[k] = pose
                avanza = True
        triangular_puntos(frames, poses, puntos_3d, cfg)
    if len(poses) < len(frames):
        res["motivo"] = f"PnP fallido en {len(frames) - len(poses)} frames"
        return res

    # 4. Ajuste de haces solo visual
    poses, puntos, rms = ajuste_haces_visual(frames, poses, puntos_3d, l, n, focal_px, cfg_vins)
    res.update({"poses": poses, "puntos": puntos, "rms_px": rms})
    if rms > cfg["error_ba_max_px"]:
        res["motivo"] = f"error de reproyección alto tras el ajuste ({rms:.2f} px)"
        return res
    res["ok"] = True
    return res


# =====================================================================
# Paso 36.2 — Sesgo del giroscopio
# =====================================================================
def estimar_sesgo_giroscopio(R_c0_b, pres):
    """Un paso de Gauss-Newton de  min Σ_k |Log(ΔR̂_kᵀ·R_bk_bk+1)|²  en δb_g:
        ΔR(b_lin + δb) = ΔR̂·Exp(J^R_bg·δb)
        r_k(δb) = Log(Exp(−J·δb)·ΔR̂ᵀ·R_sfm) ≈ r_k − J·δb       (r_k pequeño)
        δb = (Σ JᵀJ)⁻¹·Σ Jᵀ·r_k
    R_c0_b: lista de R_c0_bk (orientación de cada frame en ejes de c0, de la SfM).
    Devuelve: b_g = b_lin + δb (todas las preintegraciones deben tener el mismo b_lin).
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for k, pre in enumerate(pres):
        R_sfm = R_c0_b[k].T @ R_c0_b[k + 1]
        r = so3.log_so3(pre.dR.T @ R_sfm)
        A += pre.J_R_bg.T @ pre.J_R_bg
        b += pre.J_R_bg.T @ r
    return pres[0].b_g_lin + np.linalg.solve(A, b)


# =====================================================================
# Paso 36.3 — Velocidades, gravedad y escala (lineal)
# =====================================================================
def ecuaciones_alineacion(R_c0_b, c_sfm, pres, p_i_c, g0=None, base=None):
    """Sistema lineal A·X = b de la alineación visual-inercial.

    Modelo (en ejes de c0; p_k posición métrica de la IMU, c_k centro de la cámara de la SfM):
        p_k = s·c_k − R_k·p_i_c
        R_iᵀ·(p_j − p_i − v_i·Δt − ½·g·Δt²) = Δp̂_ij
        R_iᵀ·(v_j − v_i − g·Δt) = Δv̂_ij
    Sustituyendo p_k, para cada par (i, j = i + 1):
        R_iᵀ(c_j − c_i)·s − R_iᵀΔt·v_i − ½Δt²·R_iᵀ·g = Δp̂ + R_iᵀR_j·p_i_c − p_i_c
        −R_iᵀ·v_i + R_iᵀ·v_j − Δt·R_iᵀ·g                = Δv̂
    Incógnitas X = [v_0 … v_n (3 cada una), g (3), s] (paso 3), o, con g0 y base (paso 4),
    g = g0 + base·w y X = [v_0 … v_n, w (2), s].
    Devuelve: (A, b).
    """
    n_f = len(R_c0_b)
    refinado = g0 is not None
    n_g = 2 if refinado else 3
    n_x = 3 * n_f + n_g + 1
    filas_A, filas_b = [], []
    for i, pre in enumerate(pres):
        j = i + 1
        dt = pre.dt_total
        Ri_T = R_c0_b[i].T
        Ap = np.zeros((3, n_x))
        Av = np.zeros((3, n_x))
        Ap[:, 3 * i:3 * i + 3] = -Ri_T * dt
        Av[:, 3 * i:3 * i + 3] = -Ri_T
        Av[:, 3 * j:3 * j + 3] = Ri_T
        Cg_p = -0.5 * dt ** 2 * Ri_T                 # coeficiente de g en la ecuación de p
        Cg_v = -dt * Ri_T                            # y en la de v
        bp = pre.dp + Ri_T @ R_c0_b[j] @ p_i_c - p_i_c
        bv = pre.dv.copy()
        cg = slice(3 * n_f, 3 * n_f + n_g)
        if refinado:
            Ap[:, cg] = Cg_p @ base
            Av[:, cg] = Cg_v @ base
            bp = bp - Cg_p @ g0
            bv = bv - Cg_v @ g0
        else:
            Ap[:, cg] = Cg_p
            Av[:, cg] = Cg_v
        Ap[:, -1] = Ri_T @ (c_sfm[j] - c_sfm[i])
        filas_A += [Ap, Av]
        filas_b += [bp, bv]
    return np.vstack(filas_A), np.concatenate(filas_b)


def alinear_lineal(R_c0_b, c_sfm, pres, p_i_c):
    """Paso 3: mínimos cuadrados. Devuelve: (v (n, 3), g (3,), s, residuo RMS)."""
    A, b = ecuaciones_alineacion(R_c0_b, c_sfm, pres, p_i_c)
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    n_f = len(R_c0_b)
    rms = float(np.sqrt(np.mean((A @ X - b) ** 2)))
    return X[:3 * n_f].reshape(n_f, 3), X[3 * n_f:3 * n_f + 3], float(X[-1]), rms


def base_tangente(g):
    """Dos vectores unitarios perpendiculares a g y entre sí (plano tangente)."""
    u = g / np.linalg.norm(g)
    auxiliar = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    b1 = np.cross(u, auxiliar)
    b1 /= np.linalg.norm(b1)
    return np.column_stack([b1, np.cross(u, b1)])


def refinar_gravedad(R_c0_b, c_sfm, pres, p_i_c, g, iteraciones):
    """Paso 4: g = |G|·ĝ + w1·b1 + w2·b2, resolviendo (v, w, s) e iterando (b_a = 0, como VINS;
    el sesgo del acelerómetro lo estima después el ajuste visual-inercial del paso 6).
    Devuelve: (v, g (con |g| = G), s, residuo RMS)."""
    n_f = len(R_c0_b)
    for _ in range(iteraciones):
        g0 = G * g / np.linalg.norm(g)
        base = base_tangente(g0)
        A, b = ecuaciones_alineacion(R_c0_b, c_sfm, pres, p_i_c, g0, base)
        X = np.linalg.lstsq(A, b, rcond=None)[0]
        g = g0 + base @ X[3 * n_f:3 * n_f + 2]
    g = G * g / np.linalg.norm(g)
    rms = float(np.sqrt(np.mean((A @ X - b) ** 2)))
    return X[:3 * n_f].reshape(n_f, 3), g, float(X[-1]), rms


# =====================================================================
# Paso 36.4 — Excitación de la IMU
# =====================================================================
def excitacion_imu(pres):
    """Desviación típica [m/s²] de Δv̂/Δt entre las preintegraciones (VINS: < 0,25 = poca)."""
    a = np.array([pre.dv / pre.dt_total for pre in pres])
    return float(np.sqrt(np.sum((a - a.mean(axis=0)) ** 2) / (len(a) - 1)))


# =====================================================================
# Paso 36.5 — Ajuste de haces visual-inercial de la ventana inicializada
# =====================================================================
def longitud_trayectoria(estados):
    """Suma de las distancias entre frames consecutivos [m]."""
    return float(sum(np.linalg.norm(b.p_w - a.p_w) for a, b in zip(estados[:-1], estados[1:])))


def incertidumbre_escala(prob, opt):
    """σ_L / L: incertidumbre relativa de la longitud de la trayectoria L = Σ|p_k+1 − p_k|.

    Covarianza de los estados: P = H⁻¹ (con los puntos eliminados por Schur), y propagación
    lineal: σ_L² = ∇Lᵀ·P·∇L, con ∂L/∂δp_k = u_k−1 − u_k (u: dirección de cada tramo).
    Mide si la escala está determinada: con poca aceleración (o b_a confundido con la gravedad),
    el ajuste converge igual pero L puede variar mucho sin que el coste lo note.
    """
    from vins import marginalizacion as mg          # aquí: evita un import circular
    lin = opt.linealizar(prob)
    H, _, _ = mg.marginalizar(lin, [], prob.n_estados, 1e-12, todos=True)
    P = np.linalg.inv(H)
    grad = np.zeros(DIM * prob.n_estados)
    for k in range(prob.n_estados - 1):
        d = prob.estados[k + 1].p_w - prob.estados[k].p_w
        u = d / np.linalg.norm(d)
        grad[DIM * (k + 1) + 6:DIM * (k + 1) + 9] += u
        grad[DIM * k + 6:DIM * k + 9] -= u
    return float(np.sqrt(grad @ P @ grad) / longitud_trayectoria(prob.estados))


def ajuste_visual_inercial(frames, estados, puntos, pres, R_c_i, t_c_i, focal_px, cfg_vins):
    """El primer ajuste conjunto tras la alineación lineal (lo mismo que hace VINS justo después
    de inicializar): factores IMU + visuales + priores, con el optimizador de la etapa 34.

    Gauge: rumbo y posición del frame 0 (FactorPriorGauge, como en el lote de la etapa 34).
    Sesgos: prior débil en el frame 0 con las σ de P0 (sección "inicial"); los paseos aleatorios
    de los factores IMU los llevan al resto. Lo demás (inclinación, velocidades, escala), libre.
    Devuelve: (estados, puntos, informe del optimizador).
    """
    cfg = cfg_vins["inicializacion"]
    puntos_lista, visuales, ids = [], [], []
    for i, (h, xy, theta) in puntos.items():
        q = len(puntos_lista)
        puntos_lista.append(op.Punto(h, xy, theta))
        ids.append(i)
        for k, fr in enumerate(frames):
            if k != h and i in fr.obs:
                visuales.append(fa.FactorVisual(q, k, fr.obs[i]))
    imu = [fa.FactorImu(k, k + 1, pre) for k, pre in enumerate(pres)]
    gauge = fa.FactorPriorGauge(0, estados[0].R_w_i, estados[0].p_w, cfg_vins["gauge"]["sigma_rumbo_rad"],
                                cfg_vins["gauge"]["sigma_posicion_m"])
    sigmas = np.full(15, cfg["sigma_libre"])
    sigmas[9:12] = cfg_vins["inicial"]["sigma_sesgo_gir_rad_s"]
    sigmas[12:15] = cfg_vins["inicial"]["sigma_sesgo_acc_m_s2"]
    sesgos = fa.FactorPriorEstado(0, estados[0], sigmas)
    prob = op.ProblemaLote(estados, puntos_lista, imu, visuales, gauge, R_c_i.T, -R_c_i.T @ t_c_i,
                           priores=[sesgos])
    opt = op.OptimizadorLote(cfg_vins, focal_px)
    informe = opt.optimizar(prob, iteraciones_max=cfg["iteraciones_ajuste_vi"])
    informe["incertidumbre_escala"] = incertidumbre_escala(prob, opt)
    puntos_aj = {i: (puntos_lista[q].ancla, puntos_lista[q].xy_ancla, prob.thetas[q].copy())
                 for q, i in enumerate(ids)}
    return prob.estados, puntos_aj, informe


# =====================================================================
# Paso 36.6 — Todo junto y paso al mundo
# =====================================================================
def inicializar(frames, pres, R_c_i, t_c_i, focal_px, cfg_vins):
    """Inicialización visual-inercial de una ventana.

    frames: lista de objetos con .t_ns y .obs (id -> xy (2,) normalizadas).
    pres: Preintegracion de cada frame al siguiente (todas con el mismo sesgo de linealización;
          normalmente 0). No se modifican: se devuelven reintegradas con el b_g estimado.
    R_c_i, t_c_i: extrínseca T_c_i. focal_px: para pasar los umbrales de px a normalizadas.
    Devuelve: dict con ok, motivo, y si ok: estados (lista de EstadoImu en el mundo), puntos
    (id -> (índice del ancla, xy_ancla, θ)), pres (reintegradas), escala, g_c0, b_g, y los
    diagnósticos (paralaje_px, rms_px, excitacion, g_lineal, residuo).
    """
    cfg = cfg_vins["inicializacion"]
    R_i_c = R_c_i.T
    p_i_c = -R_c_i.T @ t_c_i
    res = {"ok": False, "motivo": ""}

    # 0. ¿Se ha movido lo bastante la IMU? (sin excitación, escala y gravedad no se separan)
    res["excitacion"] = excitacion_imu(pres)
    if res["excitacion"] < cfg["excitacion_min_m_s2"]:
        res["motivo"] = f"IMU poco excitada ({res['excitacion']:.3f} m/s²)"
        return res

    # 1. SfM solo visual (w = c_l = c0)
    sfm = sfm_visual(frames, pres, R_c_i, focal_px, cfg_vins)
    res.update({k: sfm.get(k) for k in ("l", "paralaje_px", "rms_px")})
    if not sfm["ok"]:
        res["motivo"] = sfm["motivo"]
        return res
    n_f = len(frames)
    R_c0_c = [sfm["poses"][k][0].T for k in range(n_f)]               # cámara k -> c0
    c_sfm = [-sfm["poses"][k][0].T @ sfm["poses"][k][1] for k in range(n_f)]   # centros
    R_c0_b = [R @ R_c_i for R in R_c0_c]                              # IMU k -> c0

    # 2. Sesgo del giroscopio (dos pasadas: reintegrar y repetir)
    pres = [pre.copia() for pre in pres]
    b_g = pres[0].b_g_lin.copy()
    for _ in range(cfg["iteraciones_sesgo_giro"]):
        b_g = estimar_sesgo_giroscopio(R_c0_b, pres)
        for pre in pres:
            pre.reintegrar(b_g, pre.b_a_lin)
    res["b_g"] = b_g

    # 3. Alineación lineal
    v, g, s, rms_lin = alinear_lineal(R_c0_b, c_sfm, pres, p_i_c)
    res["g_lineal"] = float(np.linalg.norm(g))
    if abs(res["g_lineal"] - G) > cfg["tolerancia_gravedad_m_s2"]:
        res["motivo"] = f"|g| lejos de 9,8 ({res['g_lineal']:.2f} m/s²)"
        return res

    # 4. Refinado de la gravedad
    v, g, s, rms = refinar_gravedad(R_c0_b, c_sfm, pres, p_i_c, g, cfg["iteraciones_gravedad"])
    res.update({"escala": s, "g_c0": g, "residuo": rms})
    if s <= 0:
        res["motivo"] = f"escala no positiva ({s:.3f})"
        return res
    if rms > cfg["residuo_max"]:
        res["motivo"] = f"residuo de la alineación alto ({rms:.4f})"
        return res

    # 5. Al mundo: g -> (0, 0, −G), rumbo del frame 0 = 0, origen en el frame 0
    u = g / np.linalg.norm(g)
    eje = np.cross(u, -E_Z)
    angulo = np.arctan2(np.linalg.norm(eje), u @ -E_Z)
    R_al = so3.exp_so3(eje / np.linalg.norm(eje) * angulo) if np.linalg.norm(eje) > 1e-12 else np.eye(3)
    rumbo = so3.angulos_zyx(R_al @ R_c0_b[0])[2]
    R_w_c0 = so3.exp_so3([0.0, 0.0, -rumbo]) @ R_al
    p = [R_w_c0 @ (s * c_sfm[k] - R_c0_b[k] @ p_i_c) for k in range(n_f)]
    estados = [EstadoImu(R_w_c0 @ R_c0_b[k], R_w_c0 @ v[k], p[k] - p[0], b_g.copy(), np.zeros(3))
               for k in range(n_f)]
    # Puntos: la escala multiplica las profundidades -> ρ / s; el rayo (α, β) no cambia
    puntos = {i: (h, xy, np.array([th[0], th[1], th[2] / s])) for i, (h, xy, th) in sfm["puntos"].items()}

    # 6. Ajuste de haces visual-inercial de la ventana (escala, gravedad y b_a, con sus pesos)
    if cfg["ajuste_visual_inercial"]:
        longitud_antes = longitud_trayectoria(estados)
        estados, puntos, informe = ajuste_visual_inercial(frames, estados, puntos, pres, R_c_i, t_c_i,
                                                          focal_px, cfg_vins)
        res["correccion_escala"] = longitud_trayectoria(estados) / longitud_antes
        res["chi2_gdl"] = informe["chi2"] / max(informe["gdl"], 1)
        res["incertidumbre_escala"] = informe["incertidumbre_escala"]
        res.update({"estados": estados, "puntos": puntos})       # para diagnosticar aunque se rechace
        if informe["incertidumbre_escala"] > cfg["incertidumbre_escala_max"]:
            res["motivo"] = (f"escala sin determinar (σ_L/L = {100 * informe['incertidumbre_escala']:.1f} %): "
                             "falta aceleración")
            return res
        if not informe["convergido"]:
            res["motivo"] = "el ajuste visual-inercial no converge"
            return res
    res.update({"ok": True, "estados": estados, "puntos": puntos, "pres": pres,
                "b_a": estados[0].b_a.copy(), "b_g": estados[0].b_g.copy()})
    return res