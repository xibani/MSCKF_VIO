"""
vins/ventana.py — Etapa 35: ventana deslizante de VINS-Mono con las dos marginalizaciones.

La ventana guarda los N + 1 frames más recientes (estado de 15 cada uno), las
preintegraciones entre frames consecutivos, los puntos (profundidad inversa anclada en su
primera observación DENTRO de la ventana) y un prior (al principio, el del primer frame; luego,
el de marginalización). Cada imagen:
    1. Cierra la preintegración y añade un frame con el estado predicho por la IMU.
    2. Añade las observaciones; los puntos nuevos se triangulan cuando tienen paralaje.
    3. Decide si el SEGUNDO más reciente es keyframe (paralaje sin giro respecto al tercero, o
       pocas pistas comunes), como VINS-Mono.
    4. Optimiza la ventana (Levenberg-Marquardt, etapa 34) con los factores IMU, visuales y el prior.
    5. Si la ventana está llena (N + 1 frames):
       - el 2.º más reciente es keyframe → se MARGINALIZA el más antiguo: sus factores (IMU con el
         siguiente, visuales de los puntos anclados en él y el prior) se resumen por Schur en un
         prior nuevo. Sus puntos se marginalizan con él y salen de la ventana; si la pista sigue,
         renace como punto nuevo anclado en la imagen siguiente.
       - si no → se DESCARTA el 2.º más reciente: sus observaciones se tiran, su preintegración se
         funde con la siguiente y, si el prior lo tocaba, se marginaliza solo del prior.

Diferencia con VINS-Mono (decisión documentada): VINS, al marginalizar el más antiguo, pasa la
profundidad de sus puntos al siguiente frame y los sigue usando con las observaciones que ya
estaban en el prior: la misma información cuenta dos veces. Aquí se marginalizan con el frame
(sin reutilizarlas), que es lo coherente con la prueba de consistencia (NEES).

Interfaz como la del MSCKF: anadir_imu(t_ns, w, a), propagar_hasta(t_ns) y
procesar_imagen(t_ns, ids, xy).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from vio import triangulacion as tri
from vio.constantes import GRAVEDAD_W
from vins import factores as fa
from vins import marginalizacion as mg
from vins import optimizador as op
from vins.preintegracion import DIM, Preintegracion


@dataclass
class Frame:
    """Un frame de la ventana. obs: id del punto -> xy (2,) normalizadas."""
    id: int
    t_ns: int
    estado: object
    obs: dict = field(default_factory=dict)


@dataclass
class PuntoVentana:
    """Un punto de la ventana. ancla: id del frame de su primera observación en la ventana.
    xy_ancla: la observación en el ancla (medida del FactorAncla).
    theta: (3,) = (α, β, ρ) en la cámara del ancla, o None si aún no se ha triangulado.
    obs: id de frame -> xy (2,), INCLUIDA la del ancla."""
    id: int
    ancla: int
    xy_ancla: np.ndarray
    theta: object
    obs: dict = field(default_factory=dict)


class VentanaDeslizante:
    """Estimador de ventana deslizante (VINS-Mono sin inicialización ni bucles)."""

    def __init__(self, estado, sigmas0, t_ns, w_m, a_m, R_c_i, t_c_i, focal_px, cfg_vins, ruido_imu,
                 g_w=GRAVEDAD_W):
        """estado: EstadoImu en t_ns (de la inicialización; en la etapa 35, la verdad).
        sigmas0: (15,) incertidumbre de ese estado (prior del primer frame).
        w_m, a_m: muestra de la IMU en t_ns. R_c_i, t_c_i: extrínseca T_c_i (como el MSCKF).
        cfg_vins: config/vins.yaml. ruido_imu: densidades para la preintegración.
        """
        self._configurar(R_c_i, t_c_i, focal_px, cfg_vins, ruido_imu, g_w)
        self.sigmas0 = np.asarray(sigmas0, dtype=float)

        # IMU: misma semántica que Msckf.anadir_imu / propagar_hasta
        self.t_ns = int(t_ns)
        self.w_ant = np.array(w_m, dtype=float)
        self.a_ant = np.array(a_m, dtype=float)
        self.estado_ref = estado.copia()          # estado en t_ns, antes del primer frame
        self.pre = Preintegracion(estado.b_g, estado.b_a, ruido_imu)

    def _configurar(self, R_c_i, t_c_i, focal_px, cfg_vins, ruido_imu, g_w):
        """Lo común a los dos constructores: configuración y ventana vacía."""
        self.cfg = cfg_vins
        self.cfg_v = cfg_vins["ventana"]
        self.N = self.cfg_v["tamano"]
        self.opt = op.OptimizadorLote(cfg_vins, focal_px)
        self.eps = cfg_vins["marginalizacion"]["eps_autovalor_relativo"]
        self.focal = focal_px
        self.ruido = ruido_imu
        self.g_w = np.asarray(g_w, dtype=float)
        self.R_c_i = np.asarray(R_c_i, dtype=float)
        self.t_c_i = np.asarray(t_c_i, dtype=float)
        self.R_i_c = self.R_c_i.T
        self.p_i_c = -self.R_c_i.T @ self.t_c_i

        self.frames = []          # del más antiguo al más nuevo
        self.pres = []            # pres[k]: de frames[k] a frames[k + 1]
        self.puntos = {}          # id del punto -> PuntoVentana
        self.prior = None         # FactorPriorEstado o FactorPriorMarginal
        self.prior_ids = []       # ids de los frames que toca el prior
        self.siguiente_id = 0
        self.descartes_en_prior = 0   # veces que el 2.º más reciente estaba en el prior (para los tests)
        self.zupt = {}            # etapa 37: id de frame -> σ de su factor de velocidad nula

    @classmethod
    def desde_inicializacion(cls, res, frames_ini, ids_frames, t_ns, w_m, a_m, R_c_i, t_c_i, focal_px,
                             cfg_vins, ruido_imu, g_w=GRAVEDAD_W):
        """Etapa 37: la ventana que deja una inicialización visual-inercial aceptada (etapa 36).

        res: lo que devuelve inicializacion.inicializar (estados, puntos, pres).
        frames_ini: los frames de la inicialización (.t_ns, .obs). ids_frames: el id de cada uno
        (el número de imagen, como en el pipeline). t_ns, w_m, a_m: instante del último frame y la
        última muestra de la IMU recibida (la IMU sigue desde ahí).
        Prior: FactorPriorArranque en el primer frame (rumbo y posición los define él; sesgos con
        las σ de P0). Los puntos sin 3D de la inicialización entran como pendientes.
        """
        v = cls.__new__(cls)
        v._configurar(R_c_i, t_c_i, focal_px, cfg_vins, ruido_imu, g_w)
        v.t_ns = int(t_ns)
        v.w_ant = np.array(w_m, dtype=float)
        v.a_ant = np.array(a_m, dtype=float)
        v.estado_ref = res["estados"][0].copia()
        v.sigmas0 = fa.sigmas_iniciales(cfg_vins["inicial"])

        v.frames = [Frame(int(i), int(f.t_ns), e.copia(), dict(f.obs))
                    for i, f, e in zip(ids_frames, frames_ini, res["estados"])]
        v.pres = [pre.copia() for pre in res["pres"]]
        v.pre = Preintegracion(v.frames[-1].estado.b_g, v.frames[-1].estado.b_a, ruido_imu)
        # Puntos: los de la inicialización (con θ) y los demás vistos, pendientes de triangular
        for k, f in enumerate(v.frames):
            for id_p, xy in f.obs.items():
                if id_p in v.puntos:
                    v.puntos[id_p].obs[f.id] = xy
                elif id_p in res["puntos"] and res["puntos"][id_p][0] == k:
                    h, xy_h, theta = res["puntos"][id_p]
                    v.puntos[id_p] = PuntoVentana(id_p, f.id, xy_h, theta.copy(), {f.id: xy})
                elif id_p not in res["puntos"]:
                    v.puntos[id_p] = PuntoVentana(id_p, f.id, xy, None, {f.id: xy})
        cfg_ini = cfg_vins["inicial"]
        v.prior = fa.FactorPriorArranque(0, v.frames[0].estado, cfg_vins["gauge"]["sigma_rumbo_rad"],
                                         cfg_vins["gauge"]["sigma_posicion_m"],
                                         cfg_ini["sigma_sesgo_gir_rad_s"], cfg_ini["sigma_sesgo_acc_m_s2"])
        v.prior_ids = [v.frames[0].id]
        v.siguiente_id = v.frames[-1].id + 1
        return v

    def corregir_reposo(self, sigma_v):
        """Etapa 37: ZUPT como factor de velocidad nula en el frame más reciente (se usa desde la
        siguiente optimización y mientras el frame siga en la ventana)."""
        if self.frames:
            self.zupt[self.frames[-1].id] = float(sigma_v)

    @property
    def estado(self):
        """Estado del frame más reciente (el estimado de la última imagen)."""
        return self.frames[-1].estado if self.frames else self.estado_ref

    # =================================================================
    # IMU
    # =================================================================
    def anadir_imu(self, t_ns, w_m, a_m):
        """Muestra nueva: el intervalo desde la anterior, con la media de las dos."""
        w_m = np.asarray(w_m, dtype=float)
        a_m = np.asarray(a_m, dtype=float)
        dt = (int(t_ns) - self.t_ns) / 1e9
        if dt > 0:
            self.pre.anadir(dt, 0.5 * (self.w_ant + w_m), 0.5 * (self.a_ant + a_m))
            self.t_ns = int(t_ns)
        self.w_ant, self.a_ant = w_m, a_m

    def propagar_hasta(self, t_ns):
        """Hasta un instante entre dos muestras (el de una imagen), con la última muestra."""
        dt = (int(t_ns) - self.t_ns) / 1e9
        if dt > 0:
            self.pre.anadir(dt, self.w_ant, self.a_ant)
            self.t_ns = int(t_ns)

    # =================================================================
    # Geometría
    # =================================================================
    def pose_camara(self, estado):
        """(R_c_w, t_c_w) de la cámara a partir del estado de la IMU (como simulador.pose_camara)."""
        R_c_w = self.R_c_i @ estado.R_w_i.T
        return R_c_w, self.t_c_i - R_c_w @ estado.p_w

    def _frame(self, id_frame):
        for f in self.frames:
            if f.id == id_frame:
                return f
        raise KeyError(id_frame)

    # =================================================================
    # Paso 35.4 — Imagen nueva
    # =================================================================
    def procesar_imagen(self, t_ns, ids, xy):
        """ids: (V,) identificadores de los puntos; xy: (V, 2) normalizadas sin distorsión.
        Devuelve: dict con los recuentos de esta imagen."""
        t0 = time.perf_counter()

        # 1. Cerrar la preintegración y crear el frame con el estado predicho
        self.propagar_hasta(t_ns)
        if self.frames:
            estado = self.pre.predecir(self.frames[-1].estado, self.g_w)
            self.pres.append(self.pre)
        else:
            estado = self.pre.predecir(self.estado_ref, self.g_w)
        frame = Frame(self.siguiente_id, int(t_ns), estado)
        self.siguiente_id += 1
        self.frames.append(frame)
        self.pre = Preintegracion(estado.b_g, estado.b_a, self.ruido)
        if self.prior is None:
            self.prior = fa.FactorPriorEstado(0, estado, self.sigmas0)
            self.prior_ids = [frame.id]

        # 2. Observaciones
        for id_p, xy_p in zip(ids, xy):
            id_p = int(id_p)
            xy_p = np.array(xy_p, dtype=float)
            frame.obs[id_p] = xy_p
            if id_p in self.puntos:
                self.puntos[id_p].obs[frame.id] = xy_p
            else:
                self.puntos[id_p] = PuntoVentana(id_p, frame.id, xy_p, None, {frame.id: xy_p})

        # 3. ¿El segundo más reciente es keyframe? (se decide antes de optimizar, como VINS)
        es_kf, paralaje_px, comunes = self._segundo_es_keyframe()

        # 4. Triangular los puntos nuevos y optimizar
        tri_cuentas = self._triangular_pendientes()
        informe = {"iteraciones": 0, "coste_final": 0.0}
        filas = 0
        activos = 0
        if len(self.frames) >= 2:
            prob, ids_puntos = self._problema()
            activos = prob.n_puntos
            filas = 2 * (len(prob.factores_visuales) + prob.n_puntos)
            informe = self.opt.optimizar(prob, iteraciones_max=self.cfg_v["iteraciones_max"])
            self._guardar(prob, ids_puntos)

        # 5. Etapa 37: observaciones atípicas (error de reproyección alto tras optimizar)
        rechazadas, errores_px = self._rechazar_atipicos()

        # 6. Ventana llena: marginalizar el más antiguo o descartar el segundo más reciente
        marginalizacion, saliente, puntos_mapa = None, None, []
        if len(self.frames) > self.N:
            if es_kf:
                saliente, puntos_mapa = self._marginalizar_antiguo()
                marginalizacion = "antiguo"
            else:
                self._descartar_segundo()
                marginalizacion = "segundo"

        return {"frames": len(self.frames), "puntos_activos": activos,
                "pendientes": sum(1 for p in self.puntos.values() if p.theta is None),
                "triangulados": tri_cuentas["triangulados"], "tri": tri_cuentas,
                "keyframe": es_kf, "paralaje_px": paralaje_px,
                "pistas_comunes": comunes, "marginalizacion": marginalizacion,
                "iteraciones": informe["iteraciones"], "coste": informe["coste_final"],
                "filas": filas, "rechazadas": rechazadas, "errores_px": errores_px,
                "saliente": saliente, "puntos_mapa": puntos_mapa,
                "tiempo_ms": 1000.0 * (time.perf_counter() - t0)}

    # =================================================================
    # Paso 35.5 — Selección de keyframe
    # =================================================================
    def _segundo_es_keyframe(self):
        """Paralaje medio [px] entre el 3.º y el 2.º frames más recientes, quitando el giro:
        cada rayo del 3.º se gira a la orientación del 2.º (con las orientaciones estimadas, que
        en el más nuevo vienen del giroscopio) y se mide lo que queda: la parte debida a la
        traslación. Devuelve: (es_keyframe, paralaje_px, pistas_comunes)."""
        if len(self.frames) < 3:
            return True, 0.0, 0
        a, b = self.frames[-3], self.frames[-2]
        # Etapa 37: un frame con ZUPT se queda (se marginalizará, y su "v = 0" pasará al prior).
        # Si se descartara, su factor se tiraría con él: en reposo (sin paralaje, todos se
        # descartaban) VINS derivaba 2 cm en 3 s y el MSCKF, con un ZUPT por imagen, 0,01 cm
        if b.id in self.zupt:
            return True, 0.0, 0
        comunes = [i for i in a.obs if i in b.obs]
        if len(comunes) < self.cfg_v["pistas_min_keyframe"]:
            return True, 0.0, len(comunes)
        # R_cb_ca = R_c_i · R_w_ibᵀ · R_w_ia · R_i_c   (rayo de la cámara a a la cámara b)
        R_ba = self.R_c_i @ b.estado.R_w_i.T @ a.estado.R_w_i @ self.R_i_c
        paralajes = []
        for i in comunes:
            rayo = R_ba @ np.array([a.obs[i][0], a.obs[i][1], 1.0])
            prediccion = rayo[:2] / rayo[2]
            paralajes.append(self.focal * np.linalg.norm(b.obs[i] - prediccion))
        media = float(np.mean(paralajes))
        return media >= self.cfg_v["paralaje_keyframe_px"], media, len(comunes)

    # =================================================================
    # Paso 35.6 — Triangulación de los puntos nuevos
    # =================================================================
    def _triangular_pendientes(self):
        """DLT (etapa 9) con las poses actuales de todos los frames que ven el punto; se acepta
        si el ángulo entre rayos llega al mínimo y queda delante de todas las cámaras.
        Devuelve: dict de recuentos (triangulados, cortas = una sola observación, sin_paralaje,
        detras = detrás de alguna cámara o demasiado lejos)."""
        z_min = self.cfg["visual"]["profundidad_min_m"]
        z_max = self.cfg["visual"]["profundidad_max_m"]
        cuentas = {"triangulados": 0, "cortas": 0, "sin_paralaje": 0, "detras": 0}
        for p in self.puntos.values():
            if p.theta is not None:
                continue
            if len(p.obs) < 2:
                cuentas["cortas"] += 1
                continue
            ids_f = list(p.obs)
            poses = [self.pose_camara(self._frame(i).estado) for i in ids_f]
            xy = np.array([p.obs[i] for i in ids_f])
            if tri.angulo_paralaje(poses, xy) < self.cfg_v["paralaje_min_triangular_grados"]:
                cuentas["sin_paralaje"] += 1
                continue
            p_w = tri.triangular_lineal(poses, xy)
            if p_w is None:
                cuentas["sin_paralaje"] += 1
                continue
            profundidades = [tri.transformar(R, t, p_w)[2] for R, t in poses]
            z_ancla = profundidades[ids_f.index(p.ancla)]
            if min(profundidades) < z_min or z_ancla > z_max:
                cuentas["detras"] += 1
                continue
            # Rayo inicial = la observación en el ancla; ρ de la triangulación
            p.theta = np.array([p.xy_ancla[0], p.xy_ancla[1], 1.0 / z_ancla])
            cuentas["triangulados"] += 1
        return cuentas

    # =================================================================
    # Paso 37.3 — Observaciones atípicas
    # =================================================================
    def _rechazar_atipicos(self):
        """Tras optimizar: quita las observaciones con error de reproyección > rechazo_px (o
        detrás de la cámara). Si la atípica es la del ancla, el punto entero (su rayo es esa
        observación). VINS hace algo parecido (outlierRejection, con el error medio del punto).
        Devuelve: (número de observaciones quitadas, sus errores [px])."""
        umbral = self.cfg_v["rechazo_px"]
        z_min = self.cfg["visual"]["profundidad_min_m"]
        errores, quitadas = [], 0
        for id_p in list(self.puntos):
            p = self.puntos[id_p]
            if p.theta is None or len(p.obs) < 2:
                continue
            error_ancla = self.focal * np.linalg.norm(p.theta[:2] - p.xy_ancla)
            if error_ancla > umbral:
                errores.append(float(error_ancla))
                quitadas += 1
                del self.puntos[id_p]
                continue
            e_h = self._frame(p.ancla).estado
            for id_f in [i for i in p.obs if i != p.ancla]:
                c = fa.punto_en_camara_j(e_h, self._frame(id_f).estado, p.theta, self.R_i_c, self.p_i_c)
                z = c["P_cj"][2]
                error = np.inf if z <= z_min else self.focal * np.linalg.norm(c["P_cj"][:2] / z - p.obs[id_f])
                if error > umbral:
                    errores.append(float(error) if np.isfinite(error) else -1.0)
                    quitadas += 1
                    del p.obs[id_f]
        return quitadas, errores

    # =================================================================
    # Paso 35.7 — Problema de la ventana
    # =================================================================
    def _vincular_prior(self, mapa):
        """Pone en el prior los índices actuales de sus frames (cambian al deslizar)."""
        if isinstance(self.prior, fa.FactorPriorEstado):
            self.prior.k = mapa[self.prior_ids[0]]
        else:
            self.prior.indices = [mapa[i] for i in self.prior_ids]
        return self.prior

    def _problema(self):
        """ProblemaLote con toda la ventana. Puntos: los triangulados con al menos una
        observación además del ancla (la del ancla la añade ProblemaLote como FactorAncla).
        Devuelve: (problema, ids de los puntos en su orden)."""
        mapa = {f.id: k for k, f in enumerate(self.frames)}
        imu = [fa.FactorImu(k, k + 1, pre, self.g_w) for k, pre in enumerate(self.pres)]
        puntos, visuales, ids_puntos = [], [], []
        for p in self.puntos.values():
            if p.theta is None or len(p.obs) < 2:
                continue
            l = len(puntos)
            puntos.append(op.Punto(mapa[p.ancla], p.xy_ancla, p.theta))
            ids_puntos.append(p.id)
            for id_f, xy_f in p.obs.items():
                if id_f != p.ancla:
                    visuales.append(fa.FactorVisual(l, mapa[id_f], xy_f))
        estados = [f.estado for f in self.frames]
        priores = [self._vincular_prior(mapa)]
        priores += [fa.FactorVelocidadNula(mapa[i], sigma) for i, sigma in self.zupt.items()]
        prob = op.ProblemaLote(estados, puntos, imu, visuales, None, self.R_i_c, self.p_i_c,
                               priores=priores)
        return prob, ids_puntos

    def _guardar(self, prob, ids_puntos):
        """Copia el resultado de la optimización a los frames y a los puntos."""
        for f, e in zip(self.frames, prob.estados):
            f.estado = e
        for id_p, theta in zip(ids_puntos, prob.thetas):
            self.puntos[id_p].theta = theta.copy()

    # =================================================================
    # Paso 35.8 — Marginalizar el frame más antiguo
    # =================================================================
    def _marginalizar_antiguo(self):
        prob, ids_puntos = self._problema()
        f0 = self.frames[0]

        # 1. Factores que tocan al frame 0: el IMU con el siguiente, los visuales de los puntos
        #    anclados en él y el prior (que siempre lo toca: es el más antiguo)
        puntos_m = {l for l, id_p in enumerate(ids_puntos) if self.puntos[id_p].ancla == f0.id}
        #    (de los priores, solo los que tocan al frame 0: el principal y su ZUPT, si tiene)
        sub = op.ProblemaLote(prob.estados, prob.puntos, [prob.factores_imu[0]],
                              [f for f in prob.factores_visuales if f.punto in puntos_m], None,
                              self.R_i_c, self.p_i_c, priores=[pr for pr in prob.priores if 0 in pr.indices])
        sub.thetas = prob.thetas

        # 2. Schur y prior nuevo, linealizado en la estimación actual
        H_e, b_e, idx_r = mg.marginalizar(self.opt.linealizar(sub), [0], prob.n_estados, self.eps)
        self.prior = mg.prior_desde_schur(H_e, b_e, idx_r, prob.estados, self.eps)
        self.prior_ids = [self.frames[k].id for k in idx_r]

        # 3. Etapa 37: lo que sale, para el mapa (como el clon saliente del MSCKF)
        saliente, puntos_mapa = self._salida_para_mapa(f0)

        # 4. Sacar el frame, su preintegración, su ZUPT y sus puntos (también los no triangulados)
        for id_p in [i for i, p in self.puntos.items() if p.ancla == f0.id]:
            del self.puntos[id_p]
        for p in self.puntos.values():          # por seguridad: nadie más lo observa
            p.obs.pop(f0.id, None)
        self.zupt.pop(f0.id, None)
        self.frames.pop(0)
        self.pres.pop(0)
        return saliente, puntos_mapa

    def _salida_para_mapa(self, f0):
        """El frame que sale y sus puntos, con el formato del MSCKF (etapa 18):
            saliente: {imagen, t_ns, R_w_c, p_w_c, obs: [(id, xy, z, paralaje [°])]}
            puntos_mapa: [(id, p_w, paralaje [°])]
        Los puntos son los anclados en él (los únicos que observa) con 3D; z = 1/ρ."""
        R_w_c = f0.estado.R_w_i @ self.R_i_c
        p_w_c = f0.estado.p_w + f0.estado.R_w_i @ self.p_i_c
        obs, puntos_mapa = [], []
        for p in self.puntos.values():
            if p.ancla != f0.id or p.theta is None or len(p.obs) < 2:
                continue
            alfa, beta, rho = p.theta
            p_w = R_w_c @ (np.array([alfa, beta, 1.0]) / rho) + p_w_c
            ids_f = list(p.obs)
            poses = [self.pose_camara(self._frame(i).estado) for i in ids_f]
            paralaje = tri.angulo_paralaje(poses, np.array([p.obs[i] for i in ids_f]))
            obs.append((p.id, p.xy_ancla.copy(), 1.0 / rho, paralaje))
            puntos_mapa.append((p.id, p_w, paralaje))
        saliente = {"imagen": f0.id, "t_ns": f0.t_ns, "R_w_c": R_w_c, "p_w_c": p_w_c, "obs": obs}
        return saliente, puntos_mapa

    # =================================================================
    # Paso 35.9 — Descartar el segundo frame más reciente
    # =================================================================
    def _descartar_segundo(self):
        k = len(self.frames) - 2
        fd, siguiente = self.frames[k], self.frames[k + 1]

        # 1. IMU: fundir de frames[k-1] a frames[k] con de frames[k] a frames[k+1]
        self.pres[k - 1] = self.pres[k - 1].fusionada(self.pres[k])
        del self.pres[k]

        # 2. Si el prior lo toca, marginalizarlo SOLO del prior (sus observaciones se tiran)
        if fd.id in self.prior_ids:
            self.descartes_en_prior += 1
            mapa = {f.id: q for q, f in enumerate(self.frames)}
            estados = [f.estado for f in self.frames]
            sub = op.ProblemaLote(estados, [], [], [], None, self.R_i_c, self.p_i_c,
                                  priores=[self._vincular_prior(mapa)])
            H_e, b_e, idx_r = mg.marginalizar(self.opt.linealizar(sub), [k], len(estados), self.eps)
            self.prior = mg.prior_desde_schur(H_e, b_e, idx_r, estados, self.eps)
            self.prior_ids = [self.frames[q].id for q in idx_r]

        # 3. Puntos: quitar la observación; los anclados en él pasan al frame siguiente
        for id_p in list(self.puntos):
            p = self.puntos[id_p]
            if fd.id not in p.obs:
                continue
            if p.ancla != fd.id:
                del p.obs[fd.id]
                continue
            if siguiente.id not in p.obs:
                del self.puntos[id_p]              # solo lo veía este frame
                continue
            self._reanclar(p, fd, siguiente)
            del p.obs[fd.id]
        self.zupt.pop(fd.id, None)
        self.frames.pop(k)

    def _reanclar(self, p, viejo, nuevo):
        """Cambia el ancla del punto p de `viejo` a `nuevo` conservando su posición 3D:
        P_w con el ancla vieja, a la cámara nueva, θ' = (x/z, y/z, 1/z). La medida del ancla
        pasa a ser la observación en el frame nuevo (que deja de ser un FactorVisual: se sigue
        usando una sola vez)."""
        if p.theta is not None:
            alfa, beta, rho = p.theta
            R_c_w, t_c_w = self.pose_camara(viejo.estado)
            p_w = R_c_w.T @ (np.array([alfa, beta, 1.0]) / rho - t_c_w)
            R_n, t_n = self.pose_camara(nuevo.estado)
            x, y, z = tri.transformar(R_n, t_n, p_w)
            p.theta = np.array([x / z, y / z, 1.0 / z]) if z > self.cfg["visual"]["profundidad_min_m"] else None
        p.ancla = nuevo.id
        p.xy_ancla = p.obs[nuevo.id]

    # =================================================================
    # Paso 35.10 — Covarianza (para el NEES)
    # =================================================================
    def covarianza_ultimo(self):
        """Covarianza (15x15) del error del frame más reciente: inversa de H en el estado
        actual, con los puntos eliminados por Schur (incluye el prior)."""
        prob, _ = self._problema()
        lin = self.opt.linealizar(prob)
        H, _, _ = mg.marginalizar(lin, [], prob.n_estados, self.eps, todos=True)
        P = np.linalg.inv(H)
        k = len(self.frames) - 1
        return P[DIM * k:DIM * (k + 1), DIM * k:DIM * (k + 1)]


def ejecutar_secuencia(ventana, imu, obs, t_camaras, con_covarianza=False):
    """Pasa la ventana por una secuencia completa (como msckf.ejecutar_secuencia).

    imu: dict t_ns, w, a (la primera muestra ya se usó al crear la ventana).
    obs: dict t_ns, imagen, id, xy. t_camaras: (C,) instantes de las imágenes [ns].
    Devuelve: dict con el estado del frame más reciente tras cada imagen, su covarianza (si
    con_covarianza) y los recuentos.
    """
    estados, P, recuentos = [], [], []
    siguiente = 1
    for j, t_cam in enumerate(t_camaras):
        while siguiente < len(imu["t_ns"]) and imu["t_ns"][siguiente] <= t_cam:
            ventana.anadir_imu(imu["t_ns"][siguiente], imu["w"][siguiente], imu["a"][siguiente])
            siguiente += 1
        en_imagen = obs["imagen"] == j
        recuentos.append(ventana.procesar_imagen(t_cam, obs["id"][en_imagen], obs["xy"][en_imagen]))
        estados.append(ventana.estado.copia())
        if con_covarianza:
            P.append(ventana.covarianza_ultimo())
    return {"estados": estados, "P": np.array(P), "recuentos": recuentos}