"""
vins/factores.py — Etapa 34: residuos y jacobianos de los factores de VINS.

Factores:
    FactorImu        entre dos keyframes consecutivos, con la preintegración de la etapa 33
    FactorVisual     un punto anclado en el keyframe h (rayo + profundidad inversa) visto en j
    FactorAncla      la observación de ese punto en su propio ancla h (etapa 35)
    FactorPriorGauge rumbo y posición de un keyframe: fija los 4 grados sin observar
    FactorPriorEstado (etapa 35) el estado completo de un keyframe con sus σ (arranque)

Interfaz común de los priores (etapa 35; el optimizador los trata a todos igual, también al
de marginalización de vins/marginalizacion.py):
    indices                     keyframes del problema que toca (lista)
    evaluar_bloques(estados)    (r, [J_k]) con un jacobiano (n_r, 15) por keyframe de indices
    informacion()               W (n_r, n_r)

Error de cada keyframe (como el MSCKF): [δθ, δv, δp, δb_g, δb_a], 15 números, con
    R ← R·Exp(δθ),   v ← v + δv,   p ← p + δp,   b ← b + δb.
Todos los jacobianos son respecto a ESE error (en la variedad), y los tests los comprueban
con diferencias finitas aplicando vio.msckf.inyectar_imu.

Referencias: Forster et al., T-RO 2017 (residuos Ec. 45 y jacobianos, apéndice C);
Qin et al., VINS-Mono, T-RO 2018 (Ec. 16 el factor IMU, Ec. 18 el visual con profundidad
inversa anclada en la primera observación).
"""

import numpy as np

from vio import so3
from vio.msckf import diferencia_imu
from vio.constantes import GRAVEDAD_W
from vins.preintegracion import BA, BG, DIM, P, TH, V, jacobiano_derecho, jacobiano_derecho_inv

E_Z = np.array([0.0, 0.0, 1.0])


# =====================================================================
# Paso 34.1 — Pérdidas robustas
# =====================================================================
def coste_robusto(s, tipo, c):
    """ρ(s), con s = rᵀ·W·r (error blanqueado al cuadrado) y c el umbral en unidades de σ.

    ninguna: ρ = s
    huber:   ρ = s si s <= c²;  2·c·√s − c² si no      (cuadrática cerca, lineal lejos)
    cauchy:  ρ = c²·log(1 + s/c²)                       (VINS-Mono usa Cauchy con c = 1)
    """
    if tipo == "ninguna":
        return s
    if tipo == "huber":
        return s if s <= c ** 2 else 2.0 * c * np.sqrt(s) - c ** 2
    if tipo == "cauchy":
        return c ** 2 * np.log1p(s / c ** 2)
    raise ValueError(f"Pérdida desconocida: {tipo}")


def peso_robusto(s, tipo, c):
    """ρ'(s): el peso con el que entra el factor en Gauss-Newton (mínimos cuadrados
    reponderados): H += ρ'·JᵀWJ,  b += ρ'·JᵀWr. Derivada de coste_robusto respecto a s."""
    if tipo == "ninguna":
        return 1.0
    if tipo == "huber":
        return 1.0 if s <= c ** 2 else c / np.sqrt(s)
    if tipo == "cauchy":
        return 1.0 / (1.0 + s / c ** 2)
    raise ValueError(f"Pérdida desconocida: {tipo}")


# =====================================================================
# Paso 34.2 — Factor IMU
# =====================================================================
class FactorImu:
    """Restricción entre los keyframes i y j a partir de la preintegración pre (i → j).

    Residuo (15), en el mismo orden que la covarianza preintegrada:
        r_R  = Log( ΔR̃ᵀ·R_iᵀ·R_j )
        r_v  = R_iᵀ·(v_j − v_i − g·Δt) − Δṽ
        r_p  = R_iᵀ·(p_j − p_i − v_i·Δt − ½·g·Δt²) − Δp̃
        r_bg = b_g,j − b_g,i,   r_ba = b_a,j − b_a,i
    donde (ΔR̃, Δṽ, Δp̃) = pre.corregida(b_g,i, b_a,i): la medida corregida en primer orden
    al sesgo ACTUAL del keyframe i. Peso: la inversa de pre.cov.
    """

    def __init__(self, i, j, pre, g_w=GRAVEDAD_W):
        """i, j: índices de los keyframes en la ventana. pre: Preintegracion de i a j."""
        self.i = i
        self.j = j
        self.pre = pre
        self.g_w = np.asarray(g_w, dtype=float)

    def informacion(self):
        """W = Σ⁻¹ (15x15). Se recalcula cada vez: la covarianza cambia si se reintegra."""
        W = np.linalg.inv(self.pre.cov)
        return 0.5 * (W + W.T)

    def evaluar(self, e_i, e_j):
        """Residuo y jacobianos en los estados e_i, e_j (EstadoImu).

        Devuelve: (r (15,), J_i (15, 15), J_j (15, 15)), con J_x = ∂r/∂(error de x).
        """
        pre = self.pre
        T = pre.dt_total
        g = self.g_w
        R_i, R_j = e_i.R_w_i, e_j.R_w_i

        # 1. Medida corregida al sesgo actual de i (Forster, Ec. 44)
        dbg = e_i.b_g - pre.b_g_lin
        dR, dv, dp = pre.corregida(e_i.b_g, e_i.b_a)

        # 2. Residuo
        dif_v = e_j.v_w - e_i.v_w - g * T                        # en el mundo
        dif_p = e_j.p_w - e_i.p_w - e_i.v_w * T - 0.5 * g * T ** 2
        r = np.zeros(DIM)
        r[TH] = so3.log_so3(dR.T @ R_i.T @ R_j)
        r[V] = R_i.T @ dif_v - dv
        r[P] = R_i.T @ dif_p - dp
        r[BG] = e_j.b_g - e_i.b_g
        r[BA] = e_j.b_a - e_i.b_a

        # 3. Jacobianos (perturbando cada estado por la derecha):
        #    r_R: Log(Exp(r)·Exp(a)) ≈ r + J_r⁻¹(r)·a, y Exp(−x)·Exp(r) = Exp(r)·Exp(−Exp(r)ᵀx)
        #    r_v, r_p: R_iᵀ → (I − [δθ_i]x)·R_iᵀ   =>   ∂(R_iᵀ·u)/∂δθ_i = [R_iᵀ·u]x
        Jr_inv = jacobiano_derecho_inv(r[TH])
        Exp_r_T = so3.exp_so3(r[TH]).T
        I3 = np.eye(3)

        J_i = np.zeros((DIM, DIM))
        J_i[TH, TH] = -Jr_inv @ R_j.T @ R_i
        # Cambio de b_g,i: ΔR̃ = ΔR̂·Exp(J_R_bg·δb) → ΔR̃·Exp(J_r(J_R_bg·δb)·J_R_bg·d)
        J_i[TH, BG] = -Jr_inv @ Exp_r_T @ jacobiano_derecho(pre.J_R_bg @ dbg) @ pre.J_R_bg
        J_i[V, TH] = so3.hat(R_i.T @ dif_v)
        J_i[V, V] = -R_i.T
        J_i[V, BG] = -pre.J_v_bg
        J_i[V, BA] = -pre.J_v_ba
        J_i[P, TH] = so3.hat(R_i.T @ dif_p)
        J_i[P, V] = -R_i.T * T
        J_i[P, P] = -R_i.T
        J_i[P, BG] = -pre.J_p_bg
        J_i[P, BA] = -pre.J_p_ba
        J_i[BG, BG] = -I3
        J_i[BA, BA] = -I3

        J_j = np.zeros((DIM, DIM))
        J_j[TH, TH] = Jr_inv
        J_j[V, V] = R_i.T
        J_j[P, P] = R_i.T
        J_j[BG, BG] = I3
        J_j[BA, BA] = I3
        return r, J_i, J_j


# =====================================================================
# Paso 34.3 — Factor visual con el punto anclado (rayo + profundidad inversa)
# =====================================================================
# Etapa 35 (cambio de la decisión 5 del plan): cada punto tiene 3 parámetros θ = (α, β, ρ)
# anclados en el keyframe h de su primera observación,  P_ch = [α, β, 1]/ρ.  VINS-Mono usa solo
# ρ y toma el rayo (α, β) = observación en h como EXACTO; con 1 px de ruido eso hace al
# estimador demasiado confiado (NEES de la pose ~24 en vez de 6 en el simulador) y menos
# preciso. Con (α, β) como parámetros, la observación del ancla es un residuo más (FactorAncla).
DIM_PUNTO = 3


def punto_en_camara_j(e_h, e_j, theta, R_i_c, p_i_c):
    """Cadena de transformaciones del punto (Qin et al., Ec. 18, con el rayo como parámetro):
        P_ch = [α, β, 1]/ρ                  en la cámara h
        P_ih = R_i_c·P_ch + p_i_c           en la IMU h
        P_w  = R_w_ih·P_ih + p_w_ih         en el mundo
        P_ij = R_w_ijᵀ·(P_w − p_w_ij)       en la IMU j
        P_cj = R_i_cᵀ·(P_ij − p_i_c)        en la cámara j
    theta: (3,) = (α, β, ρ). Devuelve: dict con todos los intermedios.
    """
    alfa, beta, rho = theta
    rayo = np.array([alfa, beta, 1.0])
    P_ch = rayo / rho
    P_ih = R_i_c @ P_ch + p_i_c
    P_w = e_h.R_w_i @ P_ih + e_h.p_w
    P_ij = e_j.R_w_i.T @ (P_w - e_j.p_w)
    P_cj = R_i_c.T @ (P_ij - p_i_c)
    return {"rayo": rayo, "P_ch": P_ch, "P_ih": P_ih, "P_w": P_w, "P_ij": P_ij, "P_cj": P_cj}


class FactorVisual:
    """Observación xy_j (plano normalizado) en el keyframe j != h de un punto anclado en h.

    Residuo (2): r = [x/z, y/z](P_cj) − xy_j,  con σ = ruido_px / focal_px.
    Residuo en el plano normalizado (decisión 6 del plan): la OV5647 es pinhole sin gran
    angular, no hace falta la tangente de la esfera unidad que usa VINS.
    """

    def __init__(self, punto, j, xy_j):
        """punto: índice del punto en el problema. j: keyframe que lo observa. xy_j: (2,)."""
        self.punto = punto
        self.j = j
        self.xy_j = np.asarray(xy_j, dtype=float)

    def evaluar(self, e_h, e_j, theta, R_i_c, p_i_c):
        """Residuo y jacobianos.

        e_h, e_j: EstadoImu del ancla y del que observa (h != j). theta: (3,) del punto.
        Devuelve: (r (2,), J_h (2, 15), J_j (2, 15), J_punto (2, 3), z_j [m])
        z_j: profundidad en la cámara j; si es <= 0 el residuo no tiene sentido (detrás).
        """
        c = punto_en_camara_j(e_h, e_j, theta, R_i_c, p_i_c)
        x, y, z = c["P_cj"]

        # 1. Residuo
        r = np.array([x / z, y / z]) - self.xy_j

        # 2. Proyección: ∂[x/z, y/z]/∂P_cj
        J_proy = np.array([[1.0 / z, 0.0, -x / z ** 2],
                           [0.0, 1.0 / z, -y / z ** 2]])
        A = J_proy @ R_i_c.T                   # ∂r/∂P_ij  (2x3)

        # 3. Respecto al keyframe j:  P_ij = R_jᵀ·(P_w − p_j)
        #    δθ_j: R_jᵀ → (I − [δθ]x)·R_jᵀ  =>  ∂P_ij/∂δθ_j = [P_ij]x;   ∂P_ij/∂δp_j = −R_jᵀ
        J_j = np.zeros((2, DIM))
        J_j[:, TH] = A @ so3.hat(c["P_ij"])
        J_j[:, P] = A @ (-e_j.R_w_i.T)

        # 4. Respecto al ancla h:  P_w = R_h·Exp(δθ)·P_ih + p_h + δp
        #    ∂P_w/∂δθ_h = −R_h·[P_ih]x;   ∂P_w/∂δp_h = I;   y ∂P_ij/∂P_w = R_jᵀ
        B = A @ e_j.R_w_i.T                    # ∂r/∂P_w  (2x3)
        J_h = np.zeros((2, DIM))
        J_h[:, TH] = B @ (-e_h.R_w_i @ so3.hat(c["P_ih"]))
        J_h[:, P] = B

        # 5. Respecto al punto θ = (α, β, ρ):  ∂P_ch/∂θ = [[1/ρ, 0, −α/ρ²], [0, 1/ρ, −β/ρ²], [0, 0, −1/ρ²]]
        #    y ∂P_w/∂P_ch = R_h·R_i_c
        alfa, beta, rho = theta
        dPch = np.array([[1.0 / rho, 0.0, -alfa / rho ** 2],
                         [0.0, 1.0 / rho, -beta / rho ** 2],
                         [0.0, 0.0, -1.0 / rho ** 2]])
        J_punto = B @ e_h.R_w_i @ R_i_c @ dPch
        return r, J_h, J_j, J_punto, z


class FactorAncla:
    """La observación del punto en su propio keyframe ancla: r = (α, β) − xy_h (2).

    Solo depende del punto (en la cámara h, P_ch ∝ [α, β, 1] se proyecta en (α, β)): no toca
    ninguna pose. Mismo σ que el resto de observaciones.
    """

    J = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])

    def __init__(self, punto, xy_h):
        self.punto = punto
        self.xy_h = np.asarray(xy_h, dtype=float)

    def evaluar(self, theta):
        """Devuelve: (r (2,), J_punto (2, 3))."""
        return np.asarray(theta[:2], dtype=float) - self.xy_h, self.J


# =====================================================================
# Paso 34.4 — Prior de gauge (rumbo y posición de un keyframe)
# =====================================================================
class FactorPriorGauge:
    """Fija los 4 grados de libertad que VIO no observa (x, y, z y rumbo) en un keyframe.

    Residuo (4):  r = [ e_zᵀ·Log(R·R0ᵀ),  p − p0 ]
    Log(R·R0ᵀ) es la rotación de R0 a R en ejes del MUNDO; su componente z es el cambio de
    rumbo, y un cambio de gauge R → R_z(ψ)·R da exactamente Log = ψ·e_z. El alabeo y el
    cabeceo (que la gravedad hace observables) no se tocan.
    Jacobiano: R·Exp(δθ)·R0ᵀ = Exp(φ)·Exp(R0·δθ)  =>  ∂Log/∂δθ = J_r⁻¹(φ)·R0
    """

    def __init__(self, k, R0, p0, sigma_rumbo, sigma_posicion):
        """k: keyframe al que se aplica. R0, p0: valores de referencia. σ: desviaciones."""
        self.k = k
        self.R0 = np.array(R0, dtype=float)
        self.p0 = np.array(p0, dtype=float)
        self.W = np.diag([1.0 / sigma_rumbo ** 2] + [1.0 / sigma_posicion ** 2] * 3)

    @property
    def indices(self):
        return [self.k]

    def informacion(self):
        return self.W

    def evaluar_bloques(self, estados):
        """Interfaz común de los priores: estados = [estado del keyframe k]."""
        r, J = self.evaluar(estados[0])
        return r, [J]

    def evaluar(self, e):
        """Devuelve: (r (4,), J (4, 15))."""
        phi = so3.log_so3(e.R_w_i @ self.R0.T)
        r = np.concatenate([[phi[2]], e.p_w - self.p0])
        J = np.zeros((4, DIM))
        J[0, TH] = E_Z @ jacobiano_derecho_inv(phi) @ self.R0
        J[1:4, P] = np.eye(3)
        return r, J


# =====================================================================
# Paso 35.1 — Prior sobre el estado completo de un keyframe (arranque de la ventana)
# =====================================================================
class FactorPriorEstado:
    """Lo que se sabe del primer keyframe al arrancar: x0 con incertidumbre diagonal.

    Residuo (15):  r = x ⊟ x0 = [Log(R0ᵀ·R), v − v0, p − p0, b_g − b_g0, b_a − b_a0]
    W = diag(1/σ²). Con las σ de P0 del MSCKF fija también el gauge (rumbo y posición), pero
    con la incertidumbre real del arranque: el error inicial de rumbo no se da por exacto.
    Jacobiano: ∂Log(R0ᵀ·R·Exp(δθ))/∂δθ = J_r⁻¹(Log(R0ᵀ·R)); el resto, identidad.
    """

    def __init__(self, k, x0, sigmas):
        """k: keyframe. x0: EstadoImu de referencia. sigmas: (15,) en el orden del error."""
        self.k = k
        self.x0 = x0.copia()
        self.W = np.diag(1.0 / np.asarray(sigmas, dtype=float) ** 2)

    @property
    def indices(self):
        return [self.k]

    def informacion(self):
        return self.W

    def evaluar_bloques(self, estados):
        e = estados[0]
        r = diferencia_imu(e, self.x0)
        J = np.eye(DIM)
        J[TH, TH] = jacobiano_derecho_inv(r[TH])
        return r, [J]


def sigmas_iniciales(cfg_ini):
    """(15,) desviaciones típicas del arranque a partir de la sección "inicial" del YAML
    (las mismas claves que config/msckf.yaml)."""
    return np.array([cfg_ini["sigma_orientacion_rad"]] * 3 + [cfg_ini["sigma_velocidad_m_s"]] * 3
                    + [cfg_ini["sigma_posicion_m"]] * 3 + [cfg_ini["sigma_sesgo_gir_rad_s"]] * 3
                    + [cfg_ini["sigma_sesgo_acc_m_s2"]] * 3)


# =====================================================================
# Paso 37.1 — Velocidad nula (ZUPT de las etapas 13 y 32 como factor)
# =====================================================================
class FactorVelocidadNula:
    """"La IMU está quieta" (o casi, si solo gira) en un keyframe: r = v (3), W = I/σ².

    El pipeline lo pide con corregir_reposo(σ) tras procesar la imagen; la ventana lo guarda con
    el frame y lo usa en todas las optimizaciones mientras el frame siga en ella (al marginalizar
    el frame pasa al prior; al descartarlo se tira, como sus observaciones).
    """

    def __init__(self, k, sigma):
        self.indices = [k]
        self.W = np.eye(3) / sigma ** 2

    def informacion(self):
        return self.W

    def evaluar_bloques(self, estados):
        J = np.zeros((3, DIM))
        J[:, V] = np.eye(3)
        return estados[0].v_w.copy(), [J]


# =====================================================================
# Paso 37.2 — Prior de arranque tras la inicialización visual-inercial
# =====================================================================
class FactorPriorArranque:
    """Lo que se sabe del primer frame tras la inicialización de la etapa 36:
        - rumbo y posición: los define él (el mundo se fija ahí) → σ del gauge, muy pequeñas;
        - sesgos: un prior débil (las σ de P0);
        - inclinación y velocidad: nada (las dan los datos de la ventana).
    Residuo (10): [e_zᵀ·Log(R·R0ᵀ), p − p0, b_g − b_g0, b_a − b_a0].
    """

    def __init__(self, k, x0, sigma_rumbo, sigma_posicion, sigma_bg, sigma_ba):
        self.indices = [k]
        self.x0 = x0.copia()
        self.W = np.diag([1.0 / sigma_rumbo ** 2] + [1.0 / sigma_posicion ** 2] * 3
                         + [1.0 / sigma_bg ** 2] * 3 + [1.0 / sigma_ba ** 2] * 3)

    def informacion(self):
        return self.W

    def evaluar_bloques(self, estados):
        e = estados[0]
        phi = so3.log_so3(e.R_w_i @ self.x0.R_w_i.T)
        r = np.concatenate([[phi[2]], e.p_w - self.x0.p_w, e.b_g - self.x0.b_g, e.b_a - self.x0.b_a])
        J = np.zeros((10, DIM))
        J[0, TH] = E_Z @ jacobiano_derecho_inv(phi) @ self.x0.R_w_i
        J[1:4, P] = np.eye(3)
        J[4:7, BG] = np.eye(3)
        J[7:10, BA] = np.eye(3)
        return r, [J]