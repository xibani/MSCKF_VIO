"""
vins/optimizador.py — Etapa 34 (puntos de 3 parámetros desde la 35): optimización en lote.

Problema:  min  ½·Σ rᵀ·W·r   (IMU y priores)   +   ½·Σ ρ(rᵀ·W·r)   (visuales, pérdida robusta)
sobre los estados de N keyframes (15·N, en la variedad) y M puntos θ = (α, β, ρ) (3·M).

Cada iteración de Levenberg-Marquardt:
    1. Linealizar: H = Σ ρ'·JᵀWJ,  b = Σ ρ'·JᵀWr, por bloques
           [ H_xx  H_xl ] [δx]     [b_x]
           [ H_lx  H_ll ] [δθ] = − [b_l]       H_ll DIAGONAL POR BLOQUES 3x3 (uno por punto)
    2. Amortiguar: H ← H + μ·diag(H)
    3. Schur sobre los puntos (la inversa de H_ll es la de cada bloque 3x3):
           H_xx* = H_xx − H_xl·H_ll⁻¹·H_lx,     b_x* = b_x − H_xl·H_ll⁻¹·b_l
           H_xx*·δx = −b_x*;   δθ_l = H_ll,l⁻¹·(−b_l − H_lx·δx)_l
    4. Aplicar en la variedad (inyectar_imu) y aceptar el paso solo si el coste baja.

Puntos (cambio de la decisión 5 en la etapa 35): VINS usa 1 parámetro por punto (ρ) y H_ll
diagonal, con el rayo del ancla tomado como exacto. Aquí son 3 (α, β, ρ) y la observación del
ancla es un residuo (FactorAncla); el coste es invertir M bloques 3x3 en lugar de M escalares.

Gauge (decisión documentada en la etapa 34): los 4 grados sin observar (x, y, z, rumbo) los fija
un prior (FactorPriorGauge en el lote; en la ventana, el del primer frame y luego el de
marginalización), no μ.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from vio.msckf import inyectar_imu
from vins import factores as fa
from vins.factores import DIM_PUNTO
from vins.preintegracion import DIM


@dataclass
class Punto:
    """Un punto anclado en el keyframe h de su primera observación.

    ancla: índice del keyframe h.  xy_ancla: (2,) observación en h (plano normalizado): es
    la medida del FactorAncla.  theta: (3,) = (α, β, ρ), con P_ch = [α, β, 1]/ρ.
    """
    ancla: int
    xy_ancla: np.ndarray
    theta: np.ndarray


class ProblemaLote:
    """Todo lo que se optimiza y los factores que lo relacionan."""

    def __init__(self, estados, puntos, factores_imu, factores_visuales, prior_gauge, R_i_c, p_i_c,
                 priores=()):
        """estados: lista de EstadoImu (N keyframes). puntos: lista de Punto (M).
        factores_imu: lista de FactorImu. factores_visuales: lista de FactorVisual (las
        observaciones fuera del ancla; las del ancla se crean aquí, una por punto).
        prior_gauge: FactorPriorGauge o None. R_i_c, p_i_c: extrínseca (fija en las etapas 33–37).
        priores (etapa 35): más priores con la interfaz común (FactorPriorEstado, el de
        marginalización...). Todos, también el de gauge, quedan en self.priores.
        """
        self.estados = [e.copia() for e in estados]
        self.puntos = puntos
        self.thetas = np.array([pt.theta for pt in puntos], dtype=float).reshape(-1, DIM_PUNTO)
        self.factores_imu = factores_imu
        self.factores_visuales = factores_visuales
        self.factores_ancla = [fa.FactorAncla(l, pt.xy_ancla) for l, pt in enumerate(puntos)]
        self.prior_gauge = prior_gauge
        self.priores = ([prior_gauge] if prior_gauge is not None else []) + list(priores)
        self.R_i_c = np.asarray(R_i_c, dtype=float)
        self.p_i_c = np.asarray(p_i_c, dtype=float)

    @property
    def n_estados(self):
        return len(self.estados)

    @property
    def n_puntos(self):
        return len(self.puntos)


class OptimizadorLote:
    """Gauss-Newton / Levenberg-Marquardt con complemento de Schur sobre los puntos."""

    def __init__(self, cfg_vins, focal_px):
        """cfg_vins: config/vins.yaml completo. focal_px: focal de la cámara [px]."""
        self.cfg_opt = cfg_vins["optimizador"]
        cfg_vis = cfg_vins["visual"]
        cfg_pre = cfg_vins["preintegracion"]
        sigma = cfg_vis["ruido_px"] / focal_px               # σ en el plano normalizado
        self.W_visual = np.eye(2) / sigma ** 2
        self.perdida = cfg_vis["perdida"]
        self.umbral_perdida = cfg_vis["umbral_perdida"]
        self.z_min = cfg_vis["profundidad_min_m"]
        self.rho_min = 1.0 / cfg_vis["profundidad_max_m"]
        self.rho_max = 1.0 / cfg_vis["profundidad_min_m"]
        self.umbral_bg = cfg_pre["umbral_reintegrar_bg_rad_s"]
        self.umbral_ba = cfg_pre["umbral_reintegrar_ba_m_s2"]

    # =================================================================
    # Paso 34.5 — Qué observaciones valen y coste
    # =================================================================
    def observaciones_validas(self, prob, estados=None, thetas=None):
        """Evalúa cada FactorVisual una vez. Devuelve: (lista de (factor, r, J_h, J_j, J_punto)
        de las válidas, conjunto de puntos con alguna válida). Válida: delante de la cámara j
        (z > profundidad_min). El ancla de un punto solo cuenta si el punto tiene alguna válida
        (si no, su profundidad no está determinada)."""
        estados = prob.estados if estados is None else estados
        thetas = prob.thetas if thetas is None else thetas
        validas, puntos = [], set()
        for f in prob.factores_visuales:
            pt = prob.puntos[f.punto]
            r, J_h, J_j, J_p, z = f.evaluar(estados[pt.ancla], estados[f.j], thetas[f.punto],
                                            prob.R_i_c, prob.p_i_c)
            if z > self.z_min:
                validas.append((f, r, J_h, J_j, J_p))
                puntos.add(f.punto)
        return validas, puntos

    def evaluar_coste(self, prob, estados=None, thetas=None):
        """Coste ½·Σ(...) en (estados, thetas), por defecto los del problema.

        Devuelve: dict con coste (con pérdida robusta), chi2 (sin pérdida: Σ rᵀWr),
        n_residuos (contando solo las observaciones válidas), n_visuales_validos y
        n_puntos_usados.
        """
        estados = prob.estados if estados is None else estados
        thetas = prob.thetas if thetas is None else thetas
        coste, chi2, n_res = 0.0, 0.0, 0

        for f in prob.factores_imu:
            r, _, _ = f.evaluar(estados[f.i], estados[f.j])
            s = r @ f.informacion() @ r
            coste += 0.5 * s
            chi2 += s
            n_res += DIM

        validas, puntos = self.observaciones_validas(prob, estados, thetas)
        residuos = [r for _, r, *_ in validas]
        residuos += [fa_.evaluar(thetas[fa_.punto])[0] for fa_ in prob.factores_ancla if fa_.punto in puntos]
        for r in residuos:
            s = r @ self.W_visual @ r
            coste += 0.5 * fa.coste_robusto(s, self.perdida, self.umbral_perdida)
            chi2 += s
            n_res += 2

        for pr in prob.priores:
            r, _ = pr.evaluar_bloques([estados[k] for k in pr.indices])
            s = r @ pr.informacion() @ r
            coste += 0.5 * s
            chi2 += s
            n_res += len(r)
        return {"coste": coste, "chi2": chi2, "n_residuos": n_res, "n_visuales_validos": len(validas),
                "n_puntos_usados": len(puntos)}

    # =================================================================
    # Paso 34.6 — Linealización: ecuaciones normales por bloques
    # =================================================================
    def linealizar(self, prob):
        """H y b en el estado actual del problema.

        Devuelve: dict con H_xx (15N, 15N), H_xl (15N, 3M), H_ll (M, 3, 3) (los bloques
        diagonales), b_x (15N,), b_l (3M,).
        """
        nx = DIM * prob.n_estados
        M = prob.n_puntos
        H_xx = np.zeros((nx, nx))
        H_xl = np.zeros((nx, DIM_PUNTO * M))
        H_ll = np.zeros((M, DIM_PUNTO, DIM_PUNTO))
        b_x = np.zeros(nx)
        b_l = np.zeros(DIM_PUNTO * M)

        def bloque(k):
            return slice(DIM * k, DIM * (k + 1))

        def bloque_punto(l):
            return slice(DIM_PUNTO * l, DIM_PUNTO * (l + 1))

        # 1. Factores IMU: tocan a dos keyframes
        for f in prob.factores_imu:
            r, J_i, J_j = f.evaluar(prob.estados[f.i], prob.estados[f.j])
            W = f.informacion()
            Ji, Jj = bloque(f.i), bloque(f.j)
            H_xx[Ji, Ji] += J_i.T @ W @ J_i
            H_xx[Ji, Jj] += J_i.T @ W @ J_j
            H_xx[Jj, Ji] += J_j.T @ W @ J_i
            H_xx[Jj, Jj] += J_j.T @ W @ J_j
            b_x[Ji] += J_i.T @ W @ r
            b_x[Jj] += J_j.T @ W @ r

        # 2. Factores visuales: tocan al ancla h, al keyframe j y al punto l
        validas, puntos_usados = self.observaciones_validas(prob)
        for f, r, J_h, J_j, J_p in validas:
            h, j, l = prob.puntos[f.punto].ancla, f.j, f.punto
            s = r @ self.W_visual @ r
            W = fa.peso_robusto(s, self.perdida, self.umbral_perdida) * self.W_visual
            Jh, Jj, Jl = bloque(h), bloque(j), bloque_punto(l)
            H_xx[Jh, Jh] += J_h.T @ W @ J_h
            H_xx[Jh, Jj] += J_h.T @ W @ J_j
            H_xx[Jj, Jh] += J_j.T @ W @ J_h
            H_xx[Jj, Jj] += J_j.T @ W @ J_j
            H_xl[Jh, Jl] += J_h.T @ W @ J_p
            H_xl[Jj, Jl] += J_j.T @ W @ J_p
            H_ll[l] += J_p.T @ W @ J_p
            b_x[Jh] += J_h.T @ W @ r
            b_x[Jj] += J_j.T @ W @ r
            b_l[Jl] += J_p.T @ W @ r

        # 3. Anclas: solo el punto (de los puntos con alguna observación válida)
        for f in prob.factores_ancla:
            if f.punto not in puntos_usados:
                continue
            r, J_p = f.evaluar(prob.thetas[f.punto])
            s = r @ self.W_visual @ r
            W = fa.peso_robusto(s, self.perdida, self.umbral_perdida) * self.W_visual
            H_ll[f.punto] += J_p.T @ W @ J_p
            b_l[bloque_punto(f.punto)] += J_p.T @ W @ r

        # 4. Priores (gauge, estado inicial, marginalización): pueden tocar varios keyframes
        for pr in prob.priores:
            r, Js = pr.evaluar_bloques([prob.estados[k] for k in pr.indices])
            W = pr.informacion()
            for k_a, J_a in zip(pr.indices, Js):
                b_x[bloque(k_a)] += J_a.T @ W @ r
                for k_b, J_b in zip(pr.indices, Js):
                    H_xx[bloque(k_a), bloque(k_b)] += J_a.T @ W @ J_b

        return {"H_xx": H_xx, "H_xl": H_xl, "H_ll": H_ll, "b_x": b_x, "b_l": b_l}

    # =================================================================
    # Paso 34.7 — Resolver: con Schur y (para comprobar) con H completa
    # =================================================================
    @staticmethod
    def amortiguar(lin, mu):
        """H + μ·diag(H) en H_xx y en cada bloque de H_ll (que sigue siendo diagonal por bloques)."""
        H_xx = lin["H_xx"] + mu * np.diag(np.diag(lin["H_xx"]))
        H_ll = lin["H_ll"].copy()
        for l in range(len(H_ll)):
            H_ll[l] += mu * np.diag(np.diag(H_ll[l]))
        return H_xx, H_ll

    @staticmethod
    def inversas_puntos(H_ll):
        """Inversa de cada bloque 3x3. Los puntos sin información (bloque nulo) quedan a 0:
        no se mueven. Devuelve: (M, 3, 3)."""
        inv = np.zeros_like(H_ll)
        for l in range(len(H_ll)):
            if H_ll[l, 2, 2] > 0:                       # tiene alguna observación válida
                inv[l] = np.linalg.inv(H_ll[l])
        return inv

    def resolver_schur(self, lin, mu):
        """Paso (δx, δθ) eliminando los puntos con el complemento de Schur.

        Lanza np.linalg.LinAlgError si H_xx* no es definida positiva.
        Devuelve: (dx (15N,), dl (M, 3)).
        """
        H_xx, H_ll = self.amortiguar(lin, mu)
        H_xl, b_x, b_l = lin["H_xl"], lin["b_x"], lin["b_l"]
        M = len(H_ll)
        H_ll_inv = self.inversas_puntos(H_ll)

        # 1. Sistema reducido, punto a punto: H_xx* = H_xx − Σ_l H_xl,l·H_ll,l⁻¹·H_lx,l
        H_red = H_xx.copy()
        b_red = b_x.copy()
        for l in range(M):
            cols = slice(DIM_PUNTO * l, DIM_PUNTO * (l + 1))
            H_xl_inv = H_xl[:, cols] @ H_ll_inv[l]               # (15N, 3)
            H_red -= H_xl_inv @ H_xl[:, cols].T
            b_red -= H_xl_inv @ b_l[cols]
        H_red = 0.5 * (H_red + H_red.T)

        # 2. Resolver por Cholesky (simétrica y definida positiva si todo va bien)
        factor = scipy.linalg.cho_factor(H_red)
        dx = scipy.linalg.cho_solve(factor, -b_red)

        # 3. Sustitución hacia atrás: δθ_l = H_ll,l⁻¹·(−b_l − H_lx,l·δx)
        dl = np.zeros((M, DIM_PUNTO))
        for l in range(M):
            cols = slice(DIM_PUNTO * l, DIM_PUNTO * (l + 1))
            dl[l] = H_ll_inv[l] @ (-b_l[cols] - H_xl[:, cols].T @ dx)
        return dx, dl

    def resolver_completo(self, lin, mu):
        """El mismo paso resolviendo H entera (densa). Solo para el test de Schur."""
        H_xx, H_ll = self.amortiguar(lin, mu)
        nx, M = len(lin["b_x"]), len(H_ll)
        con_info = [l for l in range(M) if H_ll[l, 2, 2] > 0]
        cols = np.concatenate([np.arange(DIM_PUNTO * l, DIM_PUNTO * (l + 1)) for l in con_info])
        n = nx + len(cols)
        H = np.zeros((n, n))
        H[:nx, :nx] = H_xx
        H[:nx, nx:] = lin["H_xl"][:, cols]
        H[nx:, :nx] = lin["H_xl"][:, cols].T
        for q, l in enumerate(con_info):
            H[nx + 3 * q:nx + 3 * q + 3, nx + 3 * q:nx + 3 * q + 3] = H_ll[l]
        b = np.concatenate([lin["b_x"], lin["b_l"][cols]])
        delta = np.linalg.solve(H, -b)
        dl = np.zeros((M, DIM_PUNTO))
        dl[con_info] = delta[nx:].reshape(-1, DIM_PUNTO)
        return delta[:nx], dl

    # =================================================================
    # Paso 34.8 — Aplicar el paso en la variedad
    # =================================================================
    def aplicar(self, prob, dx, dl):
        """Estados y puntos con el paso aplicado (no modifica el problema).

        Estados: R ← R·Exp(δθ), el resto suma (inyectar_imu del MSCKF).
        Puntos: θ ← θ + δθ, con ρ acotada a [1/profundidad_max, 1/profundidad_min]
        (ρ > 0 siempre: un punto no puede pasar detrás del ancla).
        Devuelve: (estados, thetas).
        """
        estados = [inyectar_imu(e, dx[DIM * k:DIM * (k + 1)]) for k, e in enumerate(prob.estados)]
        thetas = prob.thetas + dl
        thetas[:, 2] = np.clip(thetas[:, 2], self.rho_min, self.rho_max)
        return estados, thetas

    # =================================================================
    # Paso 34.9 — Bucle de Levenberg-Marquardt
    # =================================================================
    def reintegrar_si_hace_falta(self, prob):
        """Reintegra las preintegraciones cuyo sesgo de i se ha alejado del de linealización
        más que los umbrales (etapa 33). Devuelve: número de reintegraciones."""
        n = 0
        for f in prob.factores_imu:
            e_i = prob.estados[f.i]
            if f.pre.actualizar_sesgo(e_i.b_g, e_i.b_a, self.umbral_bg, self.umbral_ba):
                n += 1
        return n

    def optimizar(self, prob, iteraciones_max=None):
        """Optimiza el problema (lo modifica). iteraciones_max: si se da, sustituye a la del
        YAML (la ventana deslizante hace pocas por imagen). Devuelve un informe (dict):
        iteraciones (pasos aceptados), coste_inicial, coste_final, chi2, gdl, historial
        (coste tras cada paso aceptado), reintegraciones, convergido (bool), motivo (str).
        """
        cfg = self.cfg_opt
        mu = cfg["mu_inicial"]
        actual = self.evaluar_coste(prob)
        informe = {"coste_inicial": actual["coste"], "historial": [actual["coste"]],
                   "iteraciones": 0, "reintegraciones": 0, "convergido": False,
                   "motivo": "iteraciones_max"}

        if iteraciones_max is None:
            iteraciones_max = cfg["iteraciones_max"]
        for _ in range(iteraciones_max):
            # 1. Si el sesgo se ha movido mucho, reintegrar (cambia ligeramente el coste)
            n_re = self.reintegrar_si_hace_falta(prob)
            if n_re:
                informe["reintegraciones"] += n_re
                actual = self.evaluar_coste(prob)

            # 2. Linealizar en el estado actual
            lin = self.linealizar(prob)

            # 3. Buscar un μ que haga bajar el coste
            aceptado = False
            for _ in range(cfg["intentos_max"]):
                try:
                    dx, dl = self.resolver_schur(lin, mu)
                except np.linalg.LinAlgError:
                    mu *= cfg["factor_subir"]
                    continue
                estados, thetas = self.aplicar(prob, dx, dl)
                nuevo = self.evaluar_coste(prob, estados, thetas)
                if nuevo["coste"] < actual["coste"]:
                    aceptado = True
                    break
                mu *= cfg["factor_subir"]
            if not aceptado:
                informe["motivo"] = "sin_paso_que_baje_el_coste"
                informe["convergido"] = True       # en un mínimo (o μ demasiado grande)
                break

            # 4. Aceptar
            bajada_rel = (actual["coste"] - nuevo["coste"]) / actual["coste"]
            prob.estados, prob.thetas = estados, thetas
            actual = nuevo
            mu = max(mu / cfg["factor_bajar"], cfg["mu_min"])
            informe["iteraciones"] += 1
            informe["historial"].append(actual["coste"])

            # 5. ¿Convergido? Paso pequeño o coste que casi no baja
            if np.abs(dx).max() < cfg["tolerancia_paso"] or bajada_rel < cfg["tolerancia_coste_rel"]:
                informe["convergido"] = True
                informe["motivo"] = "paso_o_bajada_pequenos"
                break

        # 6. Resumen final: χ² sin pérdida robusta y grados de libertad
        informe["coste_final"] = actual["coste"]
        informe["chi2"] = actual["chi2"]
        informe["gdl"] = actual["n_residuos"] - (DIM * prob.n_estados + DIM_PUNTO * actual["n_puntos_usados"])
        informe["mu_final"] = mu
        return informe