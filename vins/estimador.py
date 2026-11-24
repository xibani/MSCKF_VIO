"""
vins/estimador.py — Etapa 37: VINS con la misma interfaz que el MSCKF (vio/msckf.Msckf), para
enchufarlo en vio/pipeline.PipelineVio sin cambiar el front-end, el mapa ni el visor.

Interfaz (la que usa el pipeline):
    anadir_imu(t_ns, w, a)       una muestra de la IMU
    propagar_hasta(t_ns)         hasta el instante de una imagen
    procesar_imagen(t_ns, ids, xy) -> dict de recuentos (las claves del MSCKF y algunas más)
    corregir_reposo(sigma_v)     ZUPT (etapas 13 y 32), aquí un factor de velocidad nula
    t_ns, estado, P, n_imagenes  instante, estado propagado, covarianza (15x15) y nº de imágenes
    inicializado                 False mientras busca una ventana que inicialice (modo visual-inercial)

Dos maneras de arrancar:
    - "reposo" (etapa 5.1): el pipeline ya sabe el estado (quieto, inclinación por la gravedad,
      sesgo del giroscopio) y crea el estimador con él: la ventana empieza en la primera imagen.
    - "visual_inercial" (etapa 36): el estimador se crea con la primera muestra de la IMU y va
      guardando frames (uno cada intervalo_inicializacion_s) y muestras; con frames_inicializacion
      frames intenta inicializar (cada reintento_s como mucho) y, cuando una ventana se acepta,
      crea la ventana deslizante a partir de ella.
"""

import time
from types import SimpleNamespace

import numpy as np

from vio import so3
from vio.constantes import GRAVEDAD_W
from vio.estado import EstadoImu
from vins import inicializacion as ini
from vins import ventana as vt
from vins.preintegracion import DIM


def recuentos_vacios():
    """Las claves que el pipeline, pi_vio y vio_recuentos.csv esperan del MSCKF, a cero."""
    return {"usadas": 0, "cortas": 0, "sin_triangular": 0, "chi2": 0, "sin_paralaje": 0,
            "reproyeccion": 0, "detras": 0, "errores_px": [], "filas": 0,
            "puntos_mapa": [], "saliente": None}


class VinsEstimador:
    """VINS (ventana deslizante de la etapa 35 + inicialización de la 36) para el pipeline."""

    def __init__(self, t_ns, w_m, a_m, R_c_i, t_c_i, focal_px, cfg_vins, ruido_imu,
                 estado=None, sigmas0=None, g_w=GRAVEDAD_W):
        """t_ns, w_m, a_m: primera muestra de la IMU. R_c_i, t_c_i: extrínseca T_c_i.
        estado, sigmas0: con ellos, arranque "reposo" (ventana desde ya); sin ellos,
        arranque "visual_inercial" (la ventana se crea al inicializar).
        """
        self.R_c_i = np.asarray(R_c_i, dtype=float)
        self.t_c_i = np.asarray(t_c_i, dtype=float)
        self.focal = focal_px
        self.cfg = cfg_vins
        self.cfg_e = cfg_vins["estimador"]
        self.ruido = ruido_imu
        self.g_w = np.asarray(g_w, dtype=float)
        self.n_imagenes = 0
        self._P = None                    # covarianza del último frame (se calcula al pedirla)
        self.ultimo_intento = None        # resultado del último intento de inicializar
        self.intentos = 0

        if estado is not None:
            self.ventana = vt.VentanaDeslizante(estado, sigmas0, t_ns, w_m, a_m, self.R_c_i, self.t_c_i,
                                                focal_px, cfg_vins, ruido_imu, g_w)
        else:
            self.ventana = None
            # Hasta inicializar: todas las muestras de la IMU (se recortan las viejas) y los
            # frames candidatos; la orientación, solo con el giroscopio (para el front-end)
            self.imu = [(int(t_ns), np.array(w_m, dtype=float), np.array(a_m, dtype=float))]
            self.candidatos = []
            self._t = int(t_ns)
            self._R_giro = np.eye(3)
            self._w_ant = np.array(w_m, dtype=float)
            # Etapa 37: sesgo del giroscopio en los reposos (se usa al preintegrar para inicializar)
            self.reposo = ini.SesgoEnReposo(self.cfg_e)
            self.reposo.actualizar(t_ns, w_m, a_m)

    # =================================================================
    # Estado (lo que lee el pipeline)
    # =================================================================
    @property
    def inicializado(self):
        return self.ventana is not None

    @property
    def t_ns(self):
        return self.ventana.t_ns if self.inicializado else self._t

    @property
    def estado(self):
        """Estado en el instante actual: el del último frame llevado hasta t_ns con la IMU
        recibida desde entonces. Antes de inicializar, solo la orientación del giroscopio."""
        if not self.inicializado:
            return EstadoImu(self._R_giro.copy(), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        v = self.ventana
        return v.pre.predecir(v.frames[-1].estado, self.g_w) if v.frames else v.estado_ref

    @property
    def P(self):
        """Covarianza (15x15) del error del último frame. Se calcula al pedirla (una
        linealización de la ventana) y se guarda hasta la siguiente imagen."""
        if self._P is None:
            if self.inicializado and len(self.ventana.frames) >= 2 and self.cfg_e["covarianza_en_recuentos"]:
                self._P = self.ventana.covarianza_ultimo()
            else:
                self._P = np.zeros((DIM, DIM))
        return self._P

    @property
    def sesgo_giro(self):
        """Sesgo del giroscopio que se usa antes de inicializar: el del último reposo, o 0."""
        if self.inicializado or self.reposo.b_g is None:
            return np.zeros(3)
        return self.reposo.b_g

    # =================================================================
    # IMU
    # =================================================================
    def anadir_imu(self, t_ns, w_m, a_m):
        if self.inicializado:
            self.ventana.anadir_imu(t_ns, w_m, a_m)
            return
        w_m = np.asarray(w_m, dtype=float)
        self.reposo.actualizar(t_ns, w_m, a_m)
        dt = (int(t_ns) - self._t) / 1e9
        if dt > 0:
            self._R_giro = self._R_giro @ so3.exp_so3((0.5 * (self._w_ant + w_m) - self.sesgo_giro) * dt)
            self._t = int(t_ns)
        self._w_ant = w_m
        self.imu.append((int(t_ns), w_m, np.asarray(a_m, dtype=float)))

    def propagar_hasta(self, t_ns):
        if self.inicializado:
            self.ventana.propagar_hasta(t_ns)
            return
        dt = (int(t_ns) - self._t) / 1e9
        if dt > 0:
            self._R_giro = self._R_giro @ so3.exp_so3((self._w_ant - self.sesgo_giro) * dt)
            self._t = int(t_ns)

    # =================================================================
    # Imagen
    # =================================================================
    def procesar_imagen(self, t_ns, ids, xy):
        """Devuelve: dict con las claves del MSCKF (usadas, cortas, sin_triangular, chi2,
        sin_paralaje, reproyeccion, detras, errores_px, filas, puntos_mapa, saliente) y las de
        VINS (inicializado, iteraciones, coste, keyframe, marginalizacion, tiempo_ms...)."""
        t0 = time.perf_counter()
        self._P = None
        indice = self.n_imagenes
        self.n_imagenes += 1
        est = recuentos_vacios()
        if not self.inicializado:
            est.update(self._intentar_inicializar(t_ns, ids, xy, indice))
            est["tiempo_ms"] = 1000.0 * (time.perf_counter() - t0)
            return est

        r = self.ventana.procesar_imagen(t_ns, ids, xy)
        tri = r["tri"]
        est.update({
            "usadas": r["puntos_activos"], "cortas": tri["cortas"],
            "sin_triangular": tri["sin_paralaje"] + tri["detras"], "sin_paralaje": tri["sin_paralaje"],
            "detras": tri["detras"], "reproyeccion": r["rechazadas"], "errores_px": r["errores_px"],
            "filas": r["filas"], "puntos_mapa": r["puntos_mapa"], "saliente": r["saliente"],
            "inicializado": True, "iteraciones": r["iteraciones"], "coste": r["coste"],
            "keyframe": r["keyframe"], "marginalizacion": r["marginalizacion"], "frames": r["frames"],
        })
        est["tiempo_ms"] = 1000.0 * (time.perf_counter() - t0)
        return est

    def corregir_reposo(self, sigma_v):
        """ZUPT: factor de velocidad nula en el frame más reciente (antes de inicializar, nada)."""
        if self.inicializado:
            self.ventana.corregir_reposo(sigma_v)
            self._P = None

    # =================================================================
    # Paso 37.4 — Inicialización visual-inercial en marcha
    # =================================================================
    def _intentar_inicializar(self, t_ns, ids, xy, indice):
        """Guarda la imagen como frame candidato si toca y, con la ventana de candidatos llena,
        intenta inicializar. Devuelve: dict con inicializado y el motivo del último intento."""
        cfg = self.cfg_e
        nuevo = SimpleNamespace(t_ns=int(t_ns), obs={int(i): np.array(p, dtype=float) for i, p in zip(ids, xy)},
                                id=indice)
        if self.candidatos and (t_ns - self.candidatos[-1].t_ns) / 1e9 < cfg["intervalo_inicializacion_s"]:
            return {"inicializado": False, "motivo_inicializacion": "esperando"}
        self.candidatos.append(nuevo)
        self.candidatos = self.candidatos[-cfg["frames_inicializacion"]:]
        # Muestras de la IMU: solo hacen falta desde la anterior al primer candidato
        t_primero = self.candidatos[0].t_ns
        while len(self.imu) > 1 and self.imu[1][0] <= t_primero:
            self.imu.pop(0)
        if len(self.candidatos) < cfg["frames_inicializacion"]:
            return {"inicializado": False, "motivo_inicializacion": "acumulando frames"}
        if self.ultimo_intento is not None and (t_ns - self.ultimo_intento) / 1e9 < cfg["reintento_s"]:
            return {"inicializado": False, "motivo_inicializacion": "esperando reintento"}

        # Intento
        self.ultimo_intento = int(t_ns)
        self.intentos += 1
        t_imu = np.array([m[0] for m in self.imu], dtype=np.int64)
        w = np.array([m[1] for m in self.imu])
        a = np.array([m[2] for m in self.imu])
        frames = self.candidatos
        # Preintegración con el sesgo del giroscopio del último reposo (0 si no ha habido ninguno)
        pres = [ini.preintegrar_entre(t_imu, w, a, frames[q].t_ns, frames[q + 1].t_ns,
                                      self.sesgo_giro, np.zeros(3), self.ruido) for q in range(len(frames) - 1)]
        res = ini.inicializar(frames, pres, self.R_c_i, self.t_c_i, self.focal, self.cfg)
        if not res["ok"]:
            return {"inicializado": False, "motivo_inicializacion": res["motivo"]}

        # Aceptada: ventana deslizante desde la inicialización (la IMU sigue desde t_ns)
        self.ventana = vt.VentanaDeslizante.desde_inicializacion(
            res, frames, [f.id for f in frames], t_ns, self.imu[-1][1], self.imu[-1][2],
            self.R_c_i, self.t_c_i, self.focal, self.cfg, self.ruido, self.g_w)
        self.imu, self.candidatos = [], []
        return {"inicializado": True, "motivo_inicializacion": "aceptada",
                "escala_incertidumbre": res["incertidumbre_escala"]}