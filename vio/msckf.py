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
    """
    n = len(xy)
    r = np.zeros(2 * n)
    H_x = np.zeros((2 * n, dim))
    H_f = np.zeros((2 * n, 3))
    for k, ((R_w_c, p_w_c), posicion) in enumerate(zip(poses_clones, posiciones)):
        h, H_clon, H_punto = jacobianos_observacion(R_w_c, p_w_c, p_w)
        r[2 * k:2 * k + 2] = xy[k] - h
        columna = DIM_IMU + DIM_CLON * posicion
        H_x[2 * k:2 * k + 2, columna:columna + DIM_CLON] = H_clon
        H_f[2 * k:2 * k + 2] = H_punto
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


def prueba_chi2(H0, r0, P_total, sigma2, confianza):
    """¿Encaja la pista con lo que el filtro espera? Devuelve (bool, d, umbral).

    Si todo es correcto, r_0 es gaussiano con covarianza S = H_0·P·H_0ᵀ + sigma²·I, así que
    d = r_0ᵀ·S⁻¹·r_0 sigue una chi-cuadrado con len(r_0) grados de libertad (etapa 10).
    """
    S = H0 @ P_total @ H0.T + sigma2 * np.eye(len(r0))
    d = r0 @ np.linalg.solve(S, r0)
    umbral = chi2.ppf(confianza, len(r0))
    return d <= umbral, d, umbral


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