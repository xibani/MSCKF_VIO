"""
vio/msckf.py — MSCKF: Multi-State Constraint Kalman Filter (Mourikis y Roumeliotis, 2007).

Estado NOMINAL: la IMU (R_w_i, v_w, p_w, b_g, b_a; un EstadoImu de la etapa 6) y M copias
("clones") de la pose de la cámara (R_w_c, p_w_c) en las últimas imágenes.
Estado de ERROR (lo que estima el filtro de Kalman): un vector pequeño alrededor del nominal,
    [δθ, δv, δp, δb_g, δb_a | δθ_c1, δp_c1 | δθ_c2, δp_c2 | ...]       15 + 6·M números
con la orientación en ejes locales:  R_verdadera = R_estimada · exp_so3(δθ).
Los puntos 3D NO están en el estado: se triangulan y se eliminan de las ecuaciones.

Referencia de las ecuaciones de error: J. Solà, "Quaternion kinematics for the error-state
Kalman filter" (2017), con error local de orientación.
"""

import numpy as np
from scipy.stats import chi2

from vio import so3
from vio import triangulacion as tri
from vio.constantes import GRAVEDAD_W
from vio.propagacion import propagar

DIM_IMU = 15     # tamaño del error de la IMU
DIM_CLON = 6     # tamaño del error de cada clon: [δθ_c, δp_c]
# Posición de cada parte dentro del error de la IMU
TH = slice(0, 3)     # δθ    orientación
V = slice(3, 6)      # δv    velocidad
P = slice(6, 9)      # δp    posición
BG = slice(9, 12)    # δb_g  sesgo del giroscopio
BA = slice(12, 15)   # δb_a  sesgo del acelerómetro


# =====================================================================
# Paso 12.1 — Estado de error y propagación de la covarianza
# =====================================================================
def matrices_error(R_w_i, w_hat, a_hat):
    """F (15x15) y G (15x12) de la evolución del error: d(δx)/dt = F·δx + G·n.

    w_hat = w_m - b_g [rad/s], a_hat = a_m - b_a [m/s²] (en ejes de la IMU).
    n = [n_g, n_a, n_bg, n_ba]: ruidos blancos del giroscopio, del acelerómetro y de los
    paseos aleatorios de los dos sesgos.
        d(δθ)/dt  = -[w_hat]x·δθ - δb_g - n_g
        d(δv)/dt  = -R·[a_hat]x·δθ - R·δb_a - R·n_a
        d(δp)/dt  = δv
        d(δb_g)/dt = n_bg,   d(δb_a)/dt = n_ba
    """
    I3 = np.eye(3)
    F = np.zeros((DIM_IMU, DIM_IMU))
    F[TH, TH] = -so3.hat(w_hat)
    F[TH, BG] = -I3
    F[V, TH] = -R_w_i @ so3.hat(a_hat)
    F[V, BA] = -R_w_i
    F[P, V] = I3

    G = np.zeros((DIM_IMU, 12))
    G[TH, 0:3] = -I3
    G[V, 3:6] = -R_w_i
    G[BG, 6:9] = I3
    G[BA, 9:12] = I3
    return F, G


def ruido_continuo(cfg_imu):
    """Q_c (12x12): densidades de los cuatro ruidos al cuadrado, cada una en sus 3 ejes."""
    diagonal = ([cfg_imu["ruido_gir"] ** 2] * 3 + [cfg_imu["ruido_acc"] ** 2] * 3
                + [cfg_imu["paseo_sesgo_gir"] ** 2] * 3 + [cfg_imu["paseo_sesgo_acc"] ** 2] * 3)
    return np.diag(diagonal)


def transicion(F, G, Qc, dt):
    """Matriz de transición y ruido de un paso de duración dt [s].

        Phi ≈ I + F·dt + ½·(F·dt)²          Q_d ≈ G·Q_c·Gᵀ·dt
    """
    Fdt = F * dt
    Phi = np.eye(DIM_IMU) + Fdt + 0.5 * Fdt @ Fdt
    Qd = G @ Qc @ G.T * dt
    return Phi, Qd


def propagar_covarianza(P_total, Phi, Qd):
    """P <- Phi·P·Phiᵀ + Q_d para la IMU; los clones no cambian, pero sí su relación con ella.

    P_total: (15 + 6·M, 15 + 6·M). Con P = [[P_II, P_IC], [P_CI, P_CC]]:
        P_II <- Phi·P_II·Phiᵀ + Q_d      P_IC <- Phi·P_IC      P_CC sin cambios
    Devuelve una matriz nueva.
    """
    P_total = P_total.copy()
    P_total[:DIM_IMU, :DIM_IMU] = Phi @ P_total[:DIM_IMU, :DIM_IMU] @ Phi.T + Qd
    if len(P_total) > DIM_IMU:
        P_total[:DIM_IMU, DIM_IMU:] = Phi @ P_total[:DIM_IMU, DIM_IMU:]
        P_total[DIM_IMU:, :DIM_IMU] = P_total[:DIM_IMU, DIM_IMU:].T
    return 0.5 * (P_total + P_total.T)            # simétrica también con redondeo


def inyectar_imu(estado, dx):
    """Aplica un error dx (15,) al estado nominal. Devuelve un EstadoImu nuevo."""
    nuevo = estado.copia()
    nuevo.R_w_i = estado.R_w_i @ so3.exp_so3(dx[TH])
    nuevo.v_w = estado.v_w + dx[V]
    nuevo.p_w = estado.p_w + dx[P]
    nuevo.b_g = estado.b_g + dx[BG]
    nuevo.b_a = estado.b_a + dx[BA]
    return nuevo


def diferencia_imu(verdad, estimado):
    """Error real (15,) entre dos EstadoImu: lo que habría que inyectar al estimado para
    obtener la verdad. Es la operación inversa de inyectar_imu."""
    dx = np.zeros(DIM_IMU)
    dx[TH] = so3.log_so3(estimado.R_w_i.T @ verdad.R_w_i)
    dx[V] = verdad.v_w - estimado.v_w
    dx[P] = verdad.p_w - estimado.p_w
    dx[BG] = verdad.b_g - estimado.b_g
    dx[BA] = verdad.b_a - estimado.b_a
    return dx


def covarianza_inicial(cfg_ini):
    """P0 (15x15) diagonal a partir de las desviaciones típicas del YAML."""
    sigmas = ([cfg_ini["sigma_orientacion_rad"]] * 3 + [cfg_ini["sigma_velocidad_m_s"]] * 3
              + [cfg_ini["sigma_posicion_m"]] * 3 + [cfg_ini["sigma_sesgo_gir_rad_s"]] * 3
              + [cfg_ini["sigma_sesgo_acc_m_s2"]] * 3)
    return np.diag(np.array(sigmas) ** 2)


# =====================================================================
# Paso 12.2 — Clones de la pose de la cámara
# =====================================================================
def pose_camara_desde_imu(R_w_i, p_w_i, R_i_c, p_i_c):
    """Pose de la cámara en el mundo: R_w_c = R_w_i·R_i_c,  p_w_c = p_w_i + R_w_i·p_i_c."""
    return R_w_i @ R_i_c, p_w_i + R_w_i @ p_i_c


def jacobiano_clon(R_w_i, R_i_c, p_i_c):
    """J (6x15): cómo cambia el error del clon [δθ_c, δp_c] con el error de la IMU.

    Con R_verdadera = R_w_i·exp(δθ):
        R_w_c = R_w_i·exp(δθ)·R_i_c = R_w_i·R_i_c·exp(R_i_cᵀ·δθ)   =>  δθ_c = R_i_cᵀ·δθ
        p_w_c = p_w_i + δp + R_w_i·exp(δθ)·p_i_c                    =>  δp_c ≈ δp - R_w_i·[p_i_c]x·δθ
    """
    J = np.zeros((DIM_CLON, DIM_IMU))
    J[0:3, TH] = R_i_c.T
    J[3:6, TH] = -R_w_i @ so3.hat(p_i_c)
    J[3:6, P] = np.eye(3)
    return J


def clonar_covarianza(P_total, J):
    """Añade el error de un clon nuevo al final de P.

    El clon es una función del error actual de la IMU: δx_clon = J_total·δx, con
    J_total = [J | 0 ...]. Por la regla de propagación de covarianzas:
        P <- [I; J_total] · P · [I; J_total]ᵀ
    """
    n = len(P_total)
    J_total = np.zeros((DIM_CLON, n))
    J_total[:, :DIM_IMU] = J
    JP = J_total @ P_total
    nueva = np.zeros((n + DIM_CLON, n + DIM_CLON))
    nueva[:n, :n] = P_total
    nueva[n:, :n] = JP
    nueva[:n, n:] = JP.T
    nueva[n:, n:] = JP @ J_total.T
    return nueva


def borrar_clon(P_total, posicion):
    """Olvida el clon número 'posicion' (0 = el más antiguo): borra sus 6 filas y columnas."""
    inicio = DIM_IMU + DIM_CLON * posicion
    quedan = [k for k in range(len(P_total)) if not inicio <= k < inicio + DIM_CLON]
    return P_total[np.ix_(quedan, quedan)]


def inyectar_clon(R_w_c, p_w_c, dx_clon):
    """Aplica el error [δθ_c, δp_c] (6,) a un clon. Devuelve (R_w_c, p_w_c) nuevos."""
    return R_w_c @ so3.exp_so3(dx_clon[0:3]), p_w_c + dx_clon[3:6]


# =====================================================================
# Paso 12.3 — Corrección con las pistas
# =====================================================================
def jacobianos_observacion(R_w_c, p_w_c, p_w):
    """Predicción de una observación y sus jacobianos.

    p_c = R_w_cᵀ·(p_w - p_w_c),  h = (x/z, y/z).  Con R_w_c verdadera = R_w_c·exp(δθ_c):
        p_c ≈ p_c + [p_c]x·δθ_c - R_w_cᵀ·δp_c + R_w_cᵀ·δp_f
    Devuelve: h (2,), H_clon (2x6) respecto a [δθ_c, δp_c], H_f (2x3) respecto al punto.
    """
    p_c = R_w_c.T @ (p_w - p_w_c)
    x, y, z = p_c
    h = np.array([x / z, y / z])
    J_proy = np.array([[1.0 / z, 0.0, -x / z ** 2],
                       [0.0, 1.0 / z, -y / z ** 2]])
    H_clon = J_proy @ np.hstack([so3.hat(p_c), -R_w_c.T])
    H_f = J_proy @ R_w_c.T
    return h, H_clon, H_f


def residuo_pista(poses_clones, posiciones, xy, p_w, dim):
    """Residuos y jacobianos de todas las observaciones de una pista.

    poses_clones: lista de (R_w_c, p_w_c) de los clones en que se vio, en orden.
    posiciones: posición de cada uno de esos clones en la ventana (0 = el más antiguo).
    xy: (n, 2) observaciones. p_w: (3,) punto triangulado. dim: tamaño del estado de error.
    Devuelve: r (2n,), H_x (2n x dim), H_f (2n x 3).
    Etapa 13: las n observaciones a la vez (mismas fórmulas que jacobianos_observacion).
    """
    n = len(xy)
    R_w_c = np.array([pose[0] for pose in poses_clones])          # (n, 3, 3)
    p_w_c = np.array([pose[1] for pose in poses_clones])          # (n, 3)
    R_c_w = np.transpose(R_w_c, (0, 2, 1))                        # R_w_cᵀ de cada clon
    # 1. El punto en cada cámara y su proyección
    p_c = np.einsum("kij,kj->ki", R_c_w, p_w - p_w_c)              # (n, 3)
    x, y, z = p_c[:, 0], p_c[:, 1], p_c[:, 2]
    r = (np.asarray(xy) - p_c[:, :2] / p_c[:, 2:3]).ravel()        # x0, y0, x1, y1...
    # 2. Jacobiano de la proyección, (n, 2, 3)
    J_proy = np.zeros((n, 2, 3))
    J_proy[:, 0, 0] = 1.0 / z
    J_proy[:, 0, 2] = -x / z ** 2
    J_proy[:, 1, 1] = 1.0 / z
    J_proy[:, 1, 2] = -y / z ** 2
    # 3. [p_c]x de cada observación, (n, 3, 3)
    hat_pc = np.zeros((n, 3, 3))
    hat_pc[:, 0, 1], hat_pc[:, 0, 2] = -z, y
    hat_pc[:, 1, 0], hat_pc[:, 1, 2] = z, -x
    hat_pc[:, 2, 0], hat_pc[:, 2, 1] = -y, x
    # 4. H_clon = J_proy·[ [p_c]x | -R_w_cᵀ ],  H_f = J_proy·R_w_cᵀ
    H_clon = np.concatenate([J_proy @ hat_pc, -(J_proy @ R_c_w)], axis=2)   # (n, 2, 6)
    H_f = (J_proy @ R_c_w).reshape(2 * n, 3)
    H_x = np.zeros((2 * n, dim))
    for k, posicion in enumerate(posiciones):
        columna = DIM_IMU + DIM_CLON * posicion
        H_x[2 * k:2 * k + 2, columna:columna + DIM_CLON] = H_clon[k]
    return r, H_x, H_f


def proyectar_nulo(H_x, H_f, r):
    """Elimina el error del punto de las ecuaciones.

    r ≈ H_x·δx + H_f·δp_f + ruido. Con A (2n x (2n-3)) cuyas columnas son ortonormales y
    perpendiculares a las de H_f (Aᵀ·H_f = 0), multiplicando por Aᵀ:
        Aᵀ·r ≈ Aᵀ·H_x·δx + Aᵀ·ruido          (el punto ha desaparecido)
    A son las últimas 2n - 3 columnas de Q en la descomposición QR completa de H_f.
    Devuelve: (H_0 = Aᵀ·H_x, r_0 = Aᵀ·r).
    """
    Q, _ = np.linalg.qr(H_f, mode="complete")
    A = Q[:, 3:]
    return A.T @ H_x, A.T @ r


_UMBRALES_CHI2 = {}     # (confianza, grados de libertad) -> umbral: se calcula una sola vez


def umbral_chi2(confianza, grados):
    """chi2.ppf(confianza, grados), recordando los ya calculados (etapa 13: scipy es lento)."""
    clave = (confianza, grados)
    if clave not in _UMBRALES_CHI2:
        _UMBRALES_CHI2[clave] = chi2.ppf(confianza, grados)
    return _UMBRALES_CHI2[clave]


def prueba_chi2(H0, r0, P_total, sigma2, confianza):
    """¿Encaja la pista con lo que el filtro espera? Devuelve (bool, d, umbral).

    Si todo es correcto, r_0 es gaussiano con covarianza S = H_0·P·H_0ᵀ + sigma²·I, así que
    d = r_0ᵀ·S⁻¹·r_0 sigue una chi-cuadrado con len(r_0) grados de libertad (etapa 10).
    """
    S = H0 @ P_total @ H0.T + sigma2 * np.eye(len(r0))
    d = r0 @ np.linalg.solve(S, r0)
    umbral = umbral_chi2(confianza, len(r0))
    return d <= umbral, d, umbral


def comprimir_qr(H, r):
    """Reduce las filas de la corrección sin perder información (etapa 13).

    Con más filas (m) que columnas (n), H = Q·R_qr con Q (m x n) de columnas ortonormales y
    R_qr (n x n) triangular. Multiplicar por Qᵀ no cambia la información ni el ruido
    (Qᵀ·σ²I·Q = σ²I), y deja n ecuaciones en lugar de m:  Qᵀ·r = R_qr·δx + ruido.
    Devuelve: (H_comprimida, r_comprimido); sin cambios si m <= n.
    """
    if len(r) <= H.shape[1]:
        return H, r
    Q, R_qr = np.linalg.qr(H)                      # modo "reduced": Q (m x n)
    return R_qr, Q.T @ r


def correccion_kalman(P_total, H, r, sigma2):
    """Corrección de Kalman con todas las filas apiladas (forma de Joseph, como en la etapa 10).

    Devuelve: (dx, P) con dx el error estimado (que se inyectará) y P corregida.
    """
    S = H @ P_total @ H.T + sigma2 * np.eye(len(r))
    K = np.linalg.solve(S, H @ P_total).T          # K = P·Hᵀ·S⁻¹ (S es simétrica)
    dx = K @ r
    I_KH = np.eye(len(P_total)) - K @ H
    P_nueva = I_KH @ P_total @ I_KH.T + sigma2 * K @ K.T
    return dx, 0.5 * (P_nueva + P_nueva.T)

# =====================================================================
# Paso 12.4 — El filtro completo
# =====================================================================
class Msckf:
    """MSCKF: se alimenta con muestras de IMU (anadir_imu) e imágenes (procesar_imagen)."""

    def __init__(self, estado, P0, t_ns, w_m, a_m, R_c_i, t_c_i, focal_px, cfg, cfg_tri,
                 g_w=GRAVEDAD_W):
        """estado: EstadoImu inicial en t_ns, con su covarianza P0 (15x15).
        w_m, a_m: primera muestra de la IMU (en t_ns).
        R_c_i, t_c_i: extrínseca T_c_i. focal_px: para pasar el ruido de px a normalizadas.
        cfg: dict de config/msckf.yaml. cfg_tri: dict de config/triangulacion.yaml.
        """
        self.estado = estado.copia()
        self.P = P0.copy()
        self.t_ns = int(t_ns)
        self.w_ant = np.array(w_m, dtype=float)
        self.a_ant = np.array(a_m, dtype=float)
        self.R_i_c = R_c_i.T
        self.p_i_c = -R_c_i.T @ t_c_i                 # posición de la cámara en ejes de la IMU
        self.focal = focal_px
        self.cfg = cfg
        self.cfg_tri = cfg_tri
        self.g_w = np.array(g_w, dtype=float)
        self.Qc = ruido_continuo(cfg["imu"])
        self.sigma2 = (cfg["correccion"]["ruido_px"] / focal_px) ** 2
        self.M = cfg["ventana"]["clones_max"]
        self.clones = []          # lista de dicts {imagen, t_ns, R_w_c, p_w_c}, del más antiguo al más nuevo
        self.trozos = {}          # id del punto -> {"imagenes": [...], "xy": [...]}
        self.n_imagenes = 0

    # ---------------- Propagación ----------------
    def _paso(self, w_m, a_m, dt):
        """Un paso de propagación: primero la covarianza (con el estado al principio del paso)."""
        F, G = matrices_error(self.estado.R_w_i, w_m - self.estado.b_g, a_m - self.estado.b_a)
        Phi, Qd = transicion(F, G, self.Qc, dt)
        self.P = propagar_covarianza(self.P, Phi, Qd)
        self.estado = propagar(self.estado, w_m, a_m, dt, self.g_w)

    def anadir_imu(self, t_ns, w_m, a_m):
        """Una muestra nueva de la IMU: propaga desde el instante actual hasta t_ns,
        con la media de la muestra anterior y esta (regla del trapecio, como en la etapa 6)."""
        w_m = np.asarray(w_m, dtype=float)
        a_m = np.asarray(a_m, dtype=float)
        dt = (int(t_ns) - self.t_ns) / 1e9
        if dt > 0:
            self._paso(0.5 * (self.w_ant + w_m), 0.5 * (self.a_ant + a_m), dt)
            self.t_ns = int(t_ns)
        self.w_ant, self.a_ant = w_m, a_m

    def propagar_hasta(self, t_ns):
        """Propaga hasta un instante entre dos muestras (el de una imagen), con la última
        muestra recibida. La siguiente muestra continuará desde ese instante."""
        dt = (int(t_ns) - self.t_ns) / 1e9
        if dt > 0:
            self._paso(self.w_ant, self.a_ant, dt)
            self.t_ns = int(t_ns)

    # ---------------- Imágenes ----------------
    def _clonar(self, indice):
        """Añade un clon con la pose actual de la cámara."""
        R_w_c, p_w_c = pose_camara_desde_imu(self.estado.R_w_i, self.estado.p_w, self.R_i_c, self.p_i_c)
        J = jacobiano_clon(self.estado.R_w_i, self.R_i_c, self.p_i_c)
        self.P = clonar_covarianza(self.P, J)
        self.clones.append({"imagen": indice, "t_ns": self.t_ns, "R_w_c": R_w_c, "p_w_c": p_w_c})

    def _inyectar(self, dx):
        """Aplica el error estimado a la IMU y a todos los clones."""
        self.estado = inyectar_imu(self.estado, dx[:DIM_IMU])
        self.estado.R_w_i = so3.reortonormalizar(self.estado.R_w_i)
        for k, clon in enumerate(self.clones):
            inicio = DIM_IMU + DIM_CLON * k
            clon["R_w_c"], clon["p_w_c"] = inyectar_clon(clon["R_w_c"], clon["p_w_c"],
                                                          dx[inicio:inicio + DIM_CLON])

    def _corregir(self, trozos, est):
        """Corrección con las pistas listas. Rellena los recuentos de est."""
        cfg_cor = self.cfg["correccion"]
        primera = self.clones[0]["imagen"]
        filas_H, filas_r = [], []
        for trozo in trozos:
            n = len(trozo["imagenes"])
            if n < cfg_cor["observaciones_min"]:
                est["cortas"] += 1
                continue
            # 1. Triangular con las poses de los clones (etapa 9)
            posiciones = [imagen - primera for imagen in trozo["imagenes"]]
            poses_clones = [(self.clones[k]["R_w_c"], self.clones[k]["p_w_c"]) for k in posiciones]
            poses_c_w = [(R.T, -R.T @ p) for R, p in poses_clones]
            xy = np.array(trozo["xy"])
            res = tri.triangular(poses_c_w, xy, self.cfg_tri, self.focal)
            if not res["fiable"]:
                est["sin_triangular"] += 1
                continue
            # 2. Residuos y jacobianos; 3. eliminar el punto
            r, H_x, H_f = residuo_pista(poses_clones, posiciones, xy, res["p_w"], len(self.P))
            H0, r0 = proyectar_nulo(H_x, H_f, r)
            # 4. Prueba chi-cuadrado
            pasa, _, _ = prueba_chi2(H0, r0, self.P, self.sigma2, cfg_cor["confianza_chi2"])
            if not pasa:
                est["chi2"] += 1
                continue
            filas_H.append(H0)
            filas_r.append(r0)
            est["usadas"] += 1

        # 5. Una sola corrección con todas las pistas aceptadas
        if filas_H:
            H = np.vstack(filas_H)
            r = np.concatenate(filas_r)
            est["filas"] = len(r)
            H, r = comprimir_qr(H, r)                  # etapa 13: menos filas, mismo resultado
            dx, self.P = correccion_kalman(self.P, H, r, self.sigma2)
            self._inyectar(dx)

    def procesar_imagen(self, t_ns, ids, xy):
        """Una imagen nueva con las observaciones de los puntos seguidos.

        ids: (V,) identificador de cada punto (el mismo en todas las imágenes en que se ve).
        xy: (V, 2) coordenadas normalizadas sin distorsión.
        Devuelve: dict con los recuentos de esta imagen.
        """
        indice = self.n_imagenes
        est = {"usadas": 0, "cortas": 0, "sin_triangular": 0, "chi2": 0, "filas": 0}

        # 1. Llevar el estado al instante de la imagen y clonar la pose de la cámara
        self.propagar_hasta(t_ns)
        self._clonar(indice)

        # 2. Añadir las observaciones a sus pistas
        vistos = set()
        for k in range(len(ids)):
            trozo = self.trozos.setdefault(int(ids[k]), {"imagenes": [], "xy": []})
            trozo["imagenes"].append(indice)
            trozo["xy"].append(xy[k])
            vistos.add(int(ids[k]))

        # 3. Listas para corregir: las perdidas y las que ya ocupan toda la ventana
        listos = []
        for id_punto in list(self.trozos):
            if id_punto not in vistos or len(self.trozos[id_punto]["imagenes"]) >= self.M:
                listos.append(self.trozos.pop(id_punto))
        self._corregir(listos, est)

        # 4. Ventana llena: olvidar el clon más antiguo (ninguna pista activa lo usa ya)
        if len(self.clones) >= self.M:
            self.P = borrar_clon(self.P, 0)
            self.clones.pop(0)

        self.n_imagenes += 1
        est["clones"] = len(self.clones)
        return est


def ejecutar_secuencia(filtro, imu, obs, t_camaras):
    """Pasa el filtro por una secuencia completa (sin conexión a los sensores).

    imu: dict t_ns, w, a (la primera muestra ya se usó al crear el filtro).
    obs: dict t_ns, imagen, id, xy (una fila por observación, como el simulador).
    t_camaras: (C,) instantes de las imágenes [ns].
    Devuelve: dict con, en cada imagen, el estado estimado, P de la IMU y los recuentos.
    """
    estados, P_imu, recuentos = [], [], []
    siguiente = 1                                   # próxima muestra de IMU por usar
    for j, t_cam in enumerate(t_camaras):
        # 1. Todas las muestras de IMU hasta el instante de la imagen
        while siguiente < len(imu["t_ns"]) and imu["t_ns"][siguiente] <= t_cam:
            filtro.anadir_imu(imu["t_ns"][siguiente], imu["w"][siguiente], imu["a"][siguiente])
            siguiente += 1
        # 2. La imagen
        en_imagen = obs["imagen"] == j
        recuentos.append(filtro.procesar_imagen(t_cam, obs["id"][en_imagen], obs["xy"][en_imagen]))
        estados.append(filtro.estado.copia())
        P_imu.append(filtro.P[:DIM_IMU, :DIM_IMU].copy())
    return {"estados": estados, "P_imu": np.array(P_imu), "recuentos": recuentos}