"""
vins/preintegracion.py — Etapa 33: preintegración de la IMU entre dos keyframes i y j.

Referencias: Forster, Carlone, Dellaert y Scaramuzza, "On-Manifold Preintegration for Real-Time
Visual-Inertial Odometry", T-RO 2017 (Ec. 26–27 la medida, Ec. 59–63 la covarianza, apéndice B
los jacobianos de sesgo, Ec. 44 la corrección de primer orden); VINS-Mono (Qin et al., T-RO
2018), sección IV-B, que usa el punto medio.

Qué se resume: todas las muestras entre i y j se condensan en (ΔR, Δv, Δp), una medida relativa
expresada en los ejes de la IMU en i que NO depende del estado en i:
    R_j = R_i·ΔR
    v_j = v_i + g_w·Δt_ij + R_i·Δv
    p_j = p_i + v_i·Δt_ij + ½·g_w·Δt_ij² + R_i·Δp
(con R = R_w_i, v = v_w, p = p_w y el modelo del acelerómetro a_m = R_w_iᵀ(a_w − g_w) + b_a).

Discretización: LA MISMA que vio/propagacion.propagar, para que preintegrar y aplicar al estado
en i dé exactamente lo mismo que propagar muestra a muestra. En cada intervalo δt, con
ω̂ = w_m − b̂_g y â = a_m − b̂_a (w_m, a_m: medias de dos muestras consecutivas, como hace
anadir_imu del MSCKF):
    E_h = Exp(ω̂·δt/2)        rotación de medio intervalo
    E   = Exp(ω̂·δt)          rotación del intervalo completo
    R_h = ΔR_ik·E_h          orientación (relativa) a mitad del intervalo
    Δp ← Δp + Δv·δt + ½·R_h·â·δt²
    Δv ← Δv + R_h·â·δt
    ΔR ← ΔR_ik·E
El plan escribe la versión de Euler (R_h → ΔR_ik); aquí se usa R_h porque es lo que hace
propagar (etapa 12: con la orientación del principio el método era de primer orden).

Error de la medida (convención del proyecto, error local por la derecha):
    ΔR_verdadera = ΔR̂·Exp(δθ),   Δv_verdadera = Δv̂ + δv,   Δp_verdadera = Δp̂ + δp
Orden del error: [δθ, δv, δp, δb_g, δb_a] (15), el mismo que el MSCKF.
"""

import numpy as np

from vio import so3
from vio.config import cargar_config
from vio.constantes import GRAVEDAD_W
from vio.estado import EstadoImu

DIM = 15  # tamaño del error preintegrado con los sesgos
# Posición de cada parte dentro del error (igual que en vio/msckf.py)
TH = slice(0, 3)  # δθ    orientación relativa
V = slice(3, 6)  # δv    velocidad relativa
P = slice(6, 9)  # δp    posición relativa
BG = slice(9, 12)  # δb_g  sesgo del giroscopio
BA = slice(12, 15)  # δb_a  sesgo del acelerómetro

# Densidades de ruido que necesita la preintegración (mismas claves que msckf.yaml e imu_ruido.yaml)
CLAVES_RUIDO = ("ruido_gir", "ruido_acc", "paseo_sesgo_gir", "paseo_sesgo_acc")

# Por debajo de este ángulo [rad] los jacobianos de SO(3) usan el desarrollo en serie
UMBRAL_ANGULO_PEQUENO = 1e-6


# =====================================================================
# Paso 33.1 — Jacobiano derecho de SO(3) y su inversa
# =====================================================================
def jacobiano_derecho(phi):
    """J_r(φ) (3x3): Exp(φ + δφ) ≈ Exp(φ)·Exp(J_r(φ)·δφ)   (Forster, Ec. 8).

    J_r(φ) = I − (1 − cos θ)/θ²·[φ]x + (θ − sen θ)/θ³·[φ]x²,   θ = |φ|
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)
    Phi = so3.hat(phi)
    if theta < UMBRAL_ANGULO_PEQUENO:
        # Serie: (1 − cos θ)/θ² ≈ ½ − θ²/24,  (θ − sen θ)/θ³ ≈ 1/6 − θ²/120
        return np.eye(3) - (0.5 - theta ** 2 / 24.0) * Phi + (1.0 / 6.0 - theta ** 2 / 120.0) * Phi @ Phi
    a = (1.0 - np.cos(theta)) / theta ** 2
    b = (theta - np.sin(theta)) / theta ** 3
    return np.eye(3) - a * Phi + b * Phi @ Phi


def jacobiano_derecho_inv(phi):
    """J_r⁻¹(φ) (3x3): Log(Exp(φ)·Exp(δφ)) ≈ φ + J_r⁻¹(φ)·δφ   (Forster, Ec. 9).

    J_r⁻¹(φ) = I + ½·[φ]x + (1/θ² − (1 + cos θ)/(2·θ·sen θ))·[φ]x²
    Se usará en los jacobianos del residuo r_R de la etapa 34.
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)
    Phi = so3.hat(phi)
    if theta < UMBRAL_ANGULO_PEQUENO:
        # Serie: 1/θ² − (1 + cos θ)/(2θ sen θ) ≈ 1/12 + θ²/720
        return np.eye(3) + 0.5 * Phi + (1.0 / 12.0 + theta ** 2 / 720.0) * Phi @ Phi
    c = 1.0 / theta ** 2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) + 0.5 * Phi + c * Phi @ Phi


# =====================================================================
# Paso 33.2 — Ruido de la IMU para la preintegración
# =====================================================================
def ruido_desde_config(cfg_pre):
    """Densidades de ruido (continuas) para la preintegración, ya infladas.

    cfg_pre: sección "preintegracion" de config/vins.yaml.
    Lee config/<fichero_ruido>.yaml (Allan, etapa 7.3) y multiplica las cuatro densidades
    por factor_inflado (las σ; las varianzas quedan multiplicadas por factor²).
    Devuelve: dict con ruido_gir [rad/s/√Hz], ruido_acc [m/s²/√Hz],
              paseo_sesgo_gir [rad/s²/√Hz], paseo_sesgo_acc [m/s³/√Hz].
    """
    medido = cargar_config(cfg_pre["fichero_ruido"])
    factor = float(cfg_pre["factor_inflado"])
    return {clave: factor * float(medido[clave]) for clave in CLAVES_RUIDO}


# =====================================================================
# Paso 33.3 — Matrices de un paso: transición del error y entrada de ruido
# =====================================================================
def matrices_paso(dR, w_hat, a_hat, dt):
    """Linealización de UN paso de la preintegración (la discretización de punto medio).

    dR: (3, 3) ΔR_ik al principio del paso. w_hat, a_hat: (3,) medidas sin sesgo.
    Devuelve: (A, B_g, B_a, E) con
        A   (9x9): δ_{k+1} = A·δ_k + B_g·δω + B_a·δa,   δ = [δθ, δv, δp]
        B_g (9x3): efecto de una perturbación δω de la velocidad angular sin sesgo
        B_a (9x3): efecto de una perturbación δa de la aceleración sin sesgo
        E   (3x3): Exp(ω̂·δt), para avanzar ΔR

    Derivación (perturbando ΔR_ik·Exp(δθ), ω̂ + δω, â + δa):
        δθ_{k+1} = Eᵀ·δθ + J_r(ω̂δt)·δt·δω
        δθ_h     = E_hᵀ·δθ + J_r(ω̂δt/2)·(δt/2)·δω              (a mitad del paso)
        R_h·Exp(δθ_h)·(â + δa) ≈ R_h·â − R_h·[â]x·δθ_h + R_h·δa
        δv_{k+1} = δv + δt·(−R_h[â]x·δθ_h + R_h·δa)
        δp_{k+1} = δp + δt·δv + ½·δt²·(−R_h[â]x·δθ_h + R_h·δa)
    Son las del plan (Forster, Ec. 59) con R_h y E_h en lugar de ΔR_ik e I: la derivada
    EXACTA del esquema que se integra, así los jacobianos de sesgo coinciden con las
    diferencias finitas hasta el redondeo.
    """
    I3 = np.eye(3)
    E_h = so3.exp_so3(w_hat * 0.5 * dt)
    E = so3.exp_so3(w_hat * dt)
    R_h = dR @ E_h
    Jr = jacobiano_derecho(w_hat * dt)
    Jr_h = jacobiano_derecho(w_hat * 0.5 * dt)
    Ra = R_h @ so3.hat(a_hat)                 # R_h·[â]x

    # Transición del error [δθ, δv, δp]
    A = np.eye(9)
    A[TH, TH] = E.T
    A[V, TH] = -Ra @ E_h.T * dt
    A[P, TH] = -0.5 * Ra @ E_h.T * dt ** 2
    A[P, V] = I3 * dt

    # Entrada de una perturbación de la velocidad angular sin sesgo
    B_g = np.zeros((9, 3))
    B_g[TH] = Jr * dt
    B_g[V] = -Ra @ Jr_h * (0.5 * dt) * dt
    B_g[P] = -0.5 * Ra @ Jr_h * (0.5 * dt) * dt ** 2

    # Entrada de una perturbación de la aceleración sin sesgo
    B_a = np.zeros((9, 3))
    B_a[V] = R_h * dt
    B_a[P] = 0.5 * R_h * dt ** 2
    return A, B_g, B_a, E


# =====================================================================
# Paso 33.4 — La preintegración
# =====================================================================
class Preintegracion:
    """Medida preintegrada entre dos keyframes, con su covarianza y sus jacobianos de sesgo.

    Uso:
        pre = Preintegracion(b_g, b_a, ruido)
        for cada intervalo entre i y j:  pre.anadir(dt_s, w_medio, a_medio)
        pre.dR, pre.dv, pre.dp, pre.dt_total, pre.cov, pre.J_R_bg, ...

    Las muestras se guardan para poder reintegrar si el sesgo estimado cambia mucho
    (y, en la etapa 35, para fundir dos preintegraciones).
    """

    def __init__(self, b_g, b_a, ruido, con_covarianza=True):
        """b_g, b_a: (3,) sesgos con los que se integra (los del keyframe i) = punto de
        linealización de la corrección de primer orden.
        ruido: dict de densidades continuas (ver ruido_desde_config).
        con_covarianza: False solo para tests de Monte Carlo (ahorra tiempo).
        """
        self.b_g_lin = np.array(b_g, dtype=float)
        self.b_a_lin = np.array(b_a, dtype=float)
        # Solo las cuatro densidades (el dict puede traer más claves, p. ej. el del simulador)
        self.ruido = {clave: float(ruido[clave]) for clave in CLAVES_RUIDO}
        self.con_covarianza = con_covarianza
        self.medidas = []   # lista de (dt_s, w_m (3,), a_m (3,)) tal como llegaron
        self._reiniciar()

    def _reiniciar(self):
        """Medida vacía: ΔR = I, Δv = Δp = 0, covarianza y jacobianos nulos."""
        self.dt_total = 0.0
        self.dR = np.eye(3)
        self.dv = np.zeros(3)
        self.dp = np.zeros(3)
        self.cov = np.zeros((DIM, DIM))
        # Jacobianos de sesgo (apéndice B de Forster): cómo cambia cada medida con el sesgo
        self.J_R_bg = np.zeros((3, 3))   # ∂(δθ)/∂b_g
        self.J_v_bg = np.zeros((3, 3))   # ∂Δv/∂b_g
        self.J_v_ba = np.zeros((3, 3))   # ∂Δv/∂b_a
        self.J_p_bg = np.zeros((3, 3))   # ∂Δp/∂b_g
        self.J_p_ba = np.zeros((3, 3))   # ∂Δp/∂b_a

    # ---------------- Integración ----------------
    def anadir(self, dt_s, w_m, a_m):
        """Un intervalo de duración dt_s [s] con las medidas representativas w_m [rad/s] y
        a_m [m/s²] (la media de sus dos muestras, como en Msckf.anadir_imu).
        Los intervalos con dt_s <= 0 se ignoran.
        """
        if dt_s <= 0:
            return
        w_m = np.array(w_m, dtype=float)
        a_m = np.array(a_m, dtype=float)
        self.medidas.append((float(dt_s), w_m, a_m))
        self._paso(float(dt_s), w_m, a_m)

    def _paso(self, dt, w_m, a_m):
        """Integra un intervalo con los sesgos de linealización."""
        w_hat = w_m - self.b_g_lin
        a_hat = a_m - self.b_a_lin
        A, B_g, B_a, E = matrices_paso(self.dR, w_hat, a_hat, dt)

        # 1. Jacobianos de sesgo: un cambio δb_g del sesgo es δω = −δb_g (y δa = −δb_a), así
        #    que J_{k+1} = A·J_k − B. Se escriben por bloques, como en el plan:
        #    ∂ΔR/∂b_g ← Eᵀ·∂ΔR/∂b_g − J_r·δt  (bloque TH de B_g), etc.
        J_bg = np.vstack([self.J_R_bg, self.J_v_bg, self.J_p_bg])     # (9, 3)
        J_ba = np.vstack([np.zeros((3, 3)), self.J_v_ba, self.J_p_ba])  # (9, 3): ΔR no depende de b_a
        J_bg = A @ J_bg - B_g
        J_ba = A @ J_ba - B_a
        self.J_R_bg, self.J_v_bg, self.J_p_bg = J_bg[TH], J_bg[V], J_bg[P]
        self.J_v_ba, self.J_p_ba = J_ba[V], J_ba[P]

        # 2. Covarianza 15x15: Σ ← F·Σ·Fᵀ + G·Q_d·Gᵀ
        #    F = [[A, −B_g, −B_a], [0, I, 0], [0, 0, I]]  (el error del sesgo entra como −δb)
        #    Ruido blanco en discreto: σ_d² = σ_c²/δt (cada media de muestras), y paseo de los
        #    sesgos: σ_c²·δt en sus filas.
        if self.con_covarianza:
            F = np.eye(DIM)
            F[0:9, 0:9] = A
            F[0:9, BG] = -B_g
            F[0:9, BA] = -B_a
            G = np.zeros((DIM, 12))
            G[0:9, 0:3] = B_g
            G[0:9, 3:6] = B_a
            G[BG, 6:9] = np.eye(3)
            G[BA, 9:12] = np.eye(3)
            Qd = np.diag([self.ruido["ruido_gir"] ** 2 / dt] * 3
                         + [self.ruido["ruido_acc"] ** 2 / dt] * 3
                         + [self.ruido["paseo_sesgo_gir"] ** 2 * dt] * 3
                         + [self.ruido["paseo_sesgo_acc"] ** 2 * dt] * 3)
            self.cov = F @ self.cov @ F.T + G @ Qd @ G.T
            self.cov = 0.5 * (self.cov + self.cov.T)   # simétrica también con redondeo

        # 3. Medida (orden: Δp usa Δv antiguo; Δv y Δp usan ΔR antiguo)
        R_h = self.dR @ so3.exp_so3(w_hat * 0.5 * dt)
        self.dp = self.dp + self.dv * dt + 0.5 * R_h @ a_hat * dt ** 2
        self.dv = self.dv + R_h @ a_hat * dt
        self.dR = self.dR @ E
        self.dt_total += dt

    def reintegrar(self, b_g, b_a):
        """Vuelve a integrar todas las muestras guardadas con sesgos nuevos (que pasan a ser
        el punto de linealización)."""
        self.b_g_lin = np.array(b_g, dtype=float)
        self.b_a_lin = np.array(b_a, dtype=float)
        self._reiniciar()
        for dt, w_m, a_m in self.medidas:
            self._paso(dt, w_m, a_m)

    # ---------------- Cambio de sesgo ----------------
    def corregida(self, b_g, b_a):
        """Medida para otros sesgos, sin reintegrar (primer orden; Forster, Ec. 44):
            ΔR(b) ≈ ΔR̂·Exp(J_R_bg·δb_g)
            Δv(b) ≈ Δv̂ + J_v_bg·δb_g + J_v_ba·δb_a
            Δp(b) ≈ Δp̂ + J_p_bg·δb_g + J_p_ba·δb_a
        con δb = b − b_lin. Devuelve: (dR (3, 3), dv (3,), dp (3,)).
        """
        dbg = np.asarray(b_g, dtype=float) - self.b_g_lin
        dba = np.asarray(b_a, dtype=float) - self.b_a_lin
        dR = self.dR @ so3.exp_so3(self.J_R_bg @ dbg)
        dv = self.dv + self.J_v_bg @ dbg + self.J_v_ba @ dba
        dp = self.dp + self.J_p_bg @ dbg + self.J_p_ba @ dba
        return dR, dv, dp

    def actualizar_sesgo(self, b_g, b_a, umbral_bg, umbral_ba):
        """Reintegra si el sesgo se ha alejado del de linealización más que los umbrales
        (|δb_g| > umbral_bg o |δb_a| > umbral_ba). Si no, se seguirá usando corregida().
        Devuelve: True si ha reintegrado.
        """
        dbg = np.linalg.norm(np.asarray(b_g, dtype=float) - self.b_g_lin)
        dba = np.linalg.norm(np.asarray(b_a, dtype=float) - self.b_a_lin)
        if dbg > umbral_bg or dba > umbral_ba:
            self.reintegrar(b_g, b_a)
            return True
        return False

    # ---------------- Uso de la medida ----------------
    def predecir(self, estado_i, g_w=GRAVEDAD_W):
        """Estado en j a partir del estado en i (con los sesgos de estado_i, corrigiendo la
        medida en primer orden si difieren de los de linealización). Los sesgos no cambian.
            R_j = R_i·ΔR,  v_j = v_i + g·Δt + R_i·Δv,  p_j = p_i + v_i·Δt + ½·g·Δt² + R_i·Δp
        Devuelve: EstadoImu nuevo.
        """
        g_w = np.asarray(g_w, dtype=float)
        dR, dv, dp = self.corregida(estado_i.b_g, estado_i.b_a)
        T = self.dt_total
        R_i = estado_i.R_w_i
        return EstadoImu(
            R_w_i=R_i @ dR,
            v_w=estado_i.v_w + g_w * T + R_i @ dv,
            p_w=estado_i.p_w + estado_i.v_w * T + 0.5 * g_w * T ** 2 + R_i @ dp,
            b_g=estado_i.b_g.copy(),
            b_a=estado_i.b_a.copy(),
        )

    # ---------------- Etapa 35: fusión de dos preintegraciones ----------------
    def copia(self):
        """Copia independiente (medidas, covarianza y jacobianos)."""
        nueva = Preintegracion(self.b_g_lin, self.b_a_lin, self.ruido, self.con_covarianza)
        nueva.medidas = [(dt, w.copy(), a.copy()) for dt, w, a in self.medidas]
        nueva.dt_total = self.dt_total
        nueva.dR, nueva.dv, nueva.dp = self.dR.copy(), self.dv.copy(), self.dp.copy()
        nueva.cov = self.cov.copy()
        for nombre in ("J_R_bg", "J_v_bg", "J_v_ba", "J_p_bg", "J_p_ba"):
            setattr(nueva, nombre, getattr(self, nombre).copy())
        return nueva

    def fusionada(self, otra):
        """Preintegración de i a k a partir de esta (i → j) y la siguiente (j → k), SIN volver
        a integrar las muestras: se componen las medidas, la covarianza y los jacobianos.

        Se usa al descartar el segundo frame más reciente (etapa 35): así no se pierde nada
        de la información de la IMU. Las dos deben estar linealizadas en el mismo sesgo; si
        `otra` lo está en otro, se reintegra antes (solo ella) con el de esta.

        Medidas (concatenar los dos tramos):
            ΔR_ik = ΔR_ij·ΔR_jk
            Δv_ik = Δv_ij + ΔR_ij·Δv_jk
            Δp_ik = Δp_ij + Δv_ij·Δt_jk + ΔR_ij·Δp_jk
        Error: x_ik = M·x_ij + N·y_jk, con x_ij el error 15 de i→j (su parte de sesgo es el
        paseo acumulado hasta j) e y_jk el error propio de j→k. Perturbando lo de arriba:
            δθ_ik = ΔR_jkᵀ·δθ_ij + δθ_jk + J^R_g,jk·δb_g,ij
            δv_ik = δv_ij − ΔR_ij·[Δv_jk]x·δθ_ij + ΔR_ij·(δv_jk + J^v_jk·δb_ij)
            δp_ik = δp_ij + Δt_jk·δv_ij − ΔR_ij·[Δp_jk]x·δθ_ij + ΔR_ij·(δp_jk + J^p_jk·δb_ij)
            δb_ik = δb_ij + δb_jk
        (el sesgo de j a k arranca con el error acumulado de i a j, y entra por los jacobianos
        de sesgo del segundo tramo).
        Covarianza: Σ_ik = M·Σ_ij·Mᵀ + N·Σ_jk·Nᵀ.  Jacobianos de sesgo (el mismo sesgo en los dos
        tramos): J_ik = M[0:9, 0:9]·J_ij + M[0:9, 9:15].
        Devuelve: una Preintegracion nueva (no modifica ninguna de las dos).
        """
        # 1. El segundo tramo, linealizado en el mismo sesgo que el primero
        segunda = otra.copia()
        if not (np.array_equal(segunda.b_g_lin, self.b_g_lin) and np.array_equal(segunda.b_a_lin, self.b_a_lin)):
            segunda.reintegrar(self.b_g_lin, self.b_a_lin)

        dR_ij, dv_ij, dp_ij = self.dR, self.dv, self.dp
        dR_jk, dv_jk, dp_jk, T_jk = segunda.dR, segunda.dv, segunda.dp, segunda.dt_total
        I3 = np.eye(3)

        # 2. Matrices de composición del error
        M = np.eye(DIM)
        M[TH, TH] = dR_jk.T
        M[TH, BG] = segunda.J_R_bg
        M[V, TH] = -dR_ij @ so3.hat(dv_jk)
        M[V, BG] = dR_ij @ segunda.J_v_bg
        M[V, BA] = dR_ij @ segunda.J_v_ba
        M[P, TH] = -dR_ij @ so3.hat(dp_jk)
        M[P, V] = I3 * T_jk
        M[P, BG] = dR_ij @ segunda.J_p_bg
        M[P, BA] = dR_ij @ segunda.J_p_ba
        N = np.eye(DIM)
        N[V, V] = dR_ij
        N[P, P] = dR_ij

        # 3. Resultado
        nueva = self.copia()
        nueva.medidas = self.medidas + segunda.medidas
        nueva.dt_total = self.dt_total + T_jk
        nueva.dR = dR_ij @ dR_jk
        nueva.dv = dv_ij + dR_ij @ dv_jk
        nueva.dp = dp_ij + dv_ij * T_jk + dR_ij @ dp_jk
        nueva.cov = M @ self.cov @ M.T + N @ segunda.cov @ N.T
        nueva.cov = 0.5 * (nueva.cov + nueva.cov.T)
        J_ik = M[0:9, 0:9] @ self.jacobiano_sesgos() + M[0:9, 9:15]
        nueva.J_R_bg = J_ik[TH, 0:3]
        nueva.J_v_bg, nueva.J_v_ba = J_ik[V, 0:3], J_ik[V, 3:6]
        nueva.J_p_bg, nueva.J_p_ba = J_ik[P, 0:3], J_ik[P, 3:6]
        return nueva

    def jacobiano_sesgos(self):
        """Los cinco jacobianos juntos (9x6): filas [δθ, δv, δp], columnas [b_g, b_a]."""
        J = np.zeros((9, 6))
        J[TH, 0:3] = self.J_R_bg
        J[V, 0:3], J[V, 3:6] = self.J_v_bg, self.J_v_ba
        J[P, 0:3], J[P, 3:6] = self.J_p_bg, self.J_p_ba
        return J