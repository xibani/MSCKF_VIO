"""
vins/marginalizacion.py — Etapa 35: marginalización por Schur y prior en forma de residuo.

Al sacar de la ventana unas variables x_m (un keyframe y los puntos anclados en él), los
factores que las tocan se linealizan en la estimación actual y se resume su información sobre
las variables que se quedan, x_r:
    [ H_mm  H_mr ] [δx_m]     [b_m]
    [ H_rm  H_rr ] [δx_r] = − [b_r]
    H* = H_rr − H_rm·H_mm⁻¹·H_mr,      b* = b_r − H_rm·H_mm⁻¹·b_m
(Leutenegger et al., OKVIS, IJRR 2015, sección IV; Qin et al., VINS-Mono, sección VI-C).

Para que el prior entre en el optimizador como un factor más, se escribe como residuo:
    H* = V·S·Vᵀ  (solo autovalores > ε·máx)
    J_p = S^½·Vᵀ,      r_p0 = S^-½·Vᵀ·b*
    r_p(x) = r_p0 + J_p·(x ⊟ x_lin)
Así ½·|r_p|² = cte + b*ᵀ·δx + ½·δxᵀ·H*·δx alrededor de x_lin (J_pᵀJ_p = H*, J_pᵀr_p0 = b*).
Convención de signo: aquí b = Σ JᵀWr (el gradiente), como en vins/optimizador.py.

J_p y r_p0 NO se recalculan después (primeras estimaciones, como VINS): x_lin se guarda y
el prior sigue siendo lineal en x ⊟ x_lin. Diferencia con VINS: en el jacobiano del residuo
se incluye J_r⁻¹ de la parte de rotación (la derivada exacta de Log(R_linᵀ·R·Exp(δθ))); VINS
la aproxima por la identidad.
"""

import numpy as np

from vio.msckf import diferencia_imu
from vins.factores import DIM_PUNTO
from vins.preintegracion import DIM, TH, jacobiano_derecho_inv


def bloque(k):
    """Filas/columnas del keyframe k en el vector de estados."""
    return slice(DIM * k, DIM * (k + 1))


# =====================================================================
# Paso 35.2 — Prior de marginalización como factor
# =====================================================================
class FactorPriorMarginal:
    """r(x) = r0 + J·(x ⊟ x_lin) sobre varios keyframes. Residuo ya blanqueado: W = I."""

    def __init__(self, indices, x_lin, J, r0):
        """indices: keyframes que toca (en el problema donde se usa; la ventana los vuelve a
        asignar en cada imagen). x_lin: lista de EstadoImu (punto de linealización).
        J: (n_r, 15·len(indices)). r0: (n_r,)."""
        self.indices = list(indices)
        self.x_lin = [e.copia() for e in x_lin]
        self.J = np.asarray(J, dtype=float)
        self.r0 = np.asarray(r0, dtype=float)

    def informacion(self):
        return np.eye(len(self.r0))

    def evaluar_bloques(self, estados):
        """estados: los de self.indices, en ese orden. Devuelve (r, [J_k])."""
        # 1. x ⊟ x_lin, keyframe a keyframe (lo que hay que inyectar a x_lin para obtener x)
        dx = np.concatenate([diferencia_imu(e, e_lin) for e, e_lin in zip(estados, self.x_lin)])
        r = self.r0 + self.J @ dx
        # 2. Jacobiano respecto al error de x: ∂(x ⊟ x_lin)/∂δx = diag(J_r⁻¹(δθ_lin), I)
        bloques = []
        for k in range(len(estados)):
            D = np.eye(DIM)
            D[TH, TH] = jacobiano_derecho_inv(dx[bloque(k)][TH])
            bloques.append(self.J[:, bloque(k)] @ D)
        return r, bloques


# =====================================================================
# Paso 35.3 — Complemento de Schur
# =====================================================================
def inversa_simetrica(A, eps_relativo):
    """Pseudoinversa de una matriz simétrica por autovalores (los < ε·máx cuentan como 0)."""
    s, V = np.linalg.eigh(0.5 * (A + A.T))
    umbral = eps_relativo * max(s.max(), 0.0)
    s_inv = np.zeros_like(s)
    s_inv[s > umbral] = 1.0 / s[s > umbral]
    return (V * s_inv) @ V.T


def marginalizar(lin, indices_m, n_estados, eps_relativo, todos=False):
    """Schur de unas ecuaciones normales: elimina TODOS los puntos que aparecen en lin y los
    keyframes indices_m.
    todos: si True, se quedan todos los keyframes que no están en indices_m, aunque los
    factores no los toquen (la ventana lo usa con indices_m = [] para la covarianza).

    lin: dict de OptimizadorLote.linealizar con SOLO los factores que se marginalizan.
    Devuelve: (H* (15·R, 15·R), b* (15·R,), indices_r) con indices_r los keyframes que se
    quedan y tocan esos factores (en orden creciente).
    """
    # 1. Puntos (H_ll diagonal por bloques 3x3): todos los que tienen información se van
    H = lin["H_xx"].copy()
    b = lin["b_x"].copy()
    for l in range(len(lin["H_ll"])):
        if lin["H_ll"][l, 2, 2] <= 0:                    # punto sin observaciones válidas
            continue
        cols = slice(DIM_PUNTO * l, DIM_PUNTO * (l + 1))
        H_xl_inv = lin["H_xl"][:, cols] @ np.linalg.inv(lin["H_ll"][l])
        H -= H_xl_inv @ lin["H_xl"][:, cols].T
        b -= H_xl_inv @ lin["b_l"][cols]

    # 2. Keyframes que se van (m) y los que se quedan y están tocados por estos factores (r)
    indices_r = [k for k in range(n_estados)
                 if k not in indices_m and (todos or np.any(H[bloque(k), :] != 0.0))]
    filas_m = (np.concatenate([np.arange(DIM * k, DIM * (k + 1)) for k in indices_m])
               if indices_m else np.zeros(0, dtype=int))
    filas_r = (np.concatenate([np.arange(DIM * k, DIM * (k + 1)) for k in indices_r])
               if indices_r else np.zeros(0, dtype=int))

    # 3. H* = H_rr − H_rm·H_mm⁻¹·H_mr,   b* = b_r − H_rm·H_mm⁻¹·b_m
    H_mm_inv = (inversa_simetrica(H[np.ix_(filas_m, filas_m)], eps_relativo)
                if len(filas_m) else np.zeros((0, 0)))
    H_rm = H[np.ix_(filas_r, filas_m)]
    H_estrella = H[np.ix_(filas_r, filas_r)] - H_rm @ H_mm_inv @ H_rm.T
    b_estrella = b[filas_r] - H_rm @ H_mm_inv @ b[filas_m]
    return 0.5 * (H_estrella + H_estrella.T), b_estrella, indices_r


def prior_desde_schur(H_estrella, b_estrella, indices_r, estados, eps_relativo):
    """FactorPriorMarginal a partir de (H*, b*), linealizado en los estados actuales.

    estados: lista de EstadoImu del problema (se guardan los de indices_r como x_lin).
    """
    s, V = np.linalg.eigh(H_estrella)
    validos = s > eps_relativo * max(s.max(), 0.0)
    s, V = s[validos], V[:, validos]
    J = np.sqrt(s)[:, np.newaxis] * V.T                 # S^½·Vᵀ
    r0 = (V.T @ b_estrella) / np.sqrt(s)                # S^-½·Vᵀ·b*
    return FactorPriorMarginal(indices_r, [estados[k] for k in indices_r], J, r0)