"""
vio/pipeline.py — El VIO completo: inicialización en reposo + front-end + MSCKF.

Recibe las muestras de la IMU y las imágenes una a una, de cualquier fuente (una grabación
o los sensores en vivo): el código del filtro es el mismo en los dos casos.
    1. Mientras no hay filtro: se acumulan muestras de IMU en reposo (etapa 5.1).
    2. Con t_reposo_s acumulados: sesgo del giroscopio e inclinación -> se crea el MSCKF.
    3. Cada imagen: IMU hasta su instante, seguimiento de puntos (etapa 8), MSCKF (etapa 12).
    4. Etapa 18: puntos para el mapa disperso y, cuando un clon sale de la ventana del
       filtro, quizá un keyframe de mapeo (vio/mapeo.py).
"""

import time

import numpy as np

from vio import calibracion_imu as ci
from vio import inicializacion as ini
from vio import mapeo
from vio import modelo_camara as mc
from vio import msckf as ms
from vio.estado import EstadoImu
from vio.frontend import FrontEnd

CABECERA_POSES = "#t_ns,p_x,p_y,p_z,r00,r01,r02,r10,r11,r12,r20,r21,r22\n"


def linea_pose(t_ns, R_w_i, p_w):
    """Una fila de un CSV de poses: tiempo, posición y la matriz de orientación fila a fila."""
    numeros = list(np.asarray(p_w, dtype=float)) + list(
        np.asarray(R_w_i, dtype=float).ravel()
    )
    return f"{int(t_ns)}," + ",".join(f"{x:.9f}" for x in numeros) + "\n"


def leer_poses(ruta):
    """Lee un CSV de poses. Devuelve: (t_ns (N,) int64, p (N, 3), R (N, 3, 3))."""
    tiempos, filas = [], []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            if linea.startswith("#") or not linea.strip():
                continue
            campos = linea.strip().split(",")
            tiempos.append(int(campos[0]))
            filas.append([float(c) for c in campos[1:13]])
    datos = np.array(filas)
    return (
        np.array(tiempos, dtype=np.int64),
        datos[:, 0:3],
        datos[:, 3:12].reshape(-1, 3, 3),
    )


class PipelineVio:
    """Inicialización + front-end + MSCKF, alimentado muestra a muestra e imagen a imagen."""

    def __init__(self, K, dist, R_c_i, t_c_i, t_d_ns, cfgs, calib_acc=None, frontend=None):
        """K, dist: intrínsecos. R_c_i, t_c_i: extrínseca T_c_i. t_d_ns: desfase de Kalibr.
        cfgs: dict con los dicts de config "vio", "frontend", "msckf" y "triangulacion".
        calib_acc: (sesgo (3,), escala (3,)) del acelerómetro (etapa 7.2), o None para
            usar la medida tal cual.
        frontend: el front-end ya creado (etapa 22: FrontEndLIghtGlue), o None para el de
            siempre, FAST + Lucas-Kanade (etapa 8).
        """
        self.K, self.dist = K, dist
        self.R_c_i, self.t_c_i = R_c_i, t_c_i
        self.t_d_ns = int(t_d_ns)
        self.cfgs = cfgs
        if frontend is None:
            self.frontend = FrontEnd(K, dist, cfgs["frontend"])
        self.frontend = frontend
        self.filtro = None
        self.reposo = []  # muestras (t, w, a) mientras se inicializa
        self.pendientes = []  # muestras recibidas que aún no ha usado el filtro
        self.problemas = []  # por qué falló el último intento de inicializar
        self.intentos = 0
        self.uv_anterior = {}  # id -> posición [px] de cada punto en la imagen anterior
        self.calib_acc = calib_acc
        self.descartadas = 0  # imágenes anteriores al estado del filtro (relojes mal)
        # Etapa 18: la imagen de cada clon mientras esté en la ventana (número de imagen -> gris),
        # para poder enviarla si al salir se elige como keyframe de mapeo
        self.grises = {}
        self.selector = mapeo.SelectorKeyframes(cfgs["vio"]["mapeo"])
        self.n_keyframes = 0

    # ---------------- Inicialización ----------------
    def _inicializar(self):
        """Intenta crear el filtro con las muestras en reposo acumuladas."""
        cfg_ini = self.cfgs["vio"]["inicializacion"]
        t = np.array([m[0] for m in self.reposo], dtype=np.int64)
        w = np.array([m[1] for m in self.reposo])
        a = np.array([m[2] for m in self.reposo])
        reposo = ini.estimar_reposo(w, a)
        self.intentos += 1
        self.problemas = ini.comprobar_reposo(reposo, cfg_ini)
        if self.problemas:
            self.reposo = []  # se movió: volver a empezar a acumular
            return
        # Estado inicial (etapa 5.1): inclinación de la gravedad, rumbo 0, en el origen y
        # quieto; sesgo del giroscopio = su media en reposo; el del acelerómetro, 0
        R0 = ini.rotacion_desde_gravedad(reposo["a_media"])
        estado = EstadoImu(R0, np.zeros(3), np.zeros(3), reposo["b_g"], np.zeros(3))
        P0 = ms.covarianza_inicial(self.cfgs["msckf"]["inicial"])
        self.filtro = ms.Msckf(
            estado,
            P0,
            int(t[-1]),
            w[-1],
            a[-1],
            self.R_c_i,
            self.t_c_i,
            self.K[0, 0],
            self.cfgs["msckf"],
            self.cfgs["triangulacion"],
        )
        self.reposo = []

    # ---------------- Reposo ----------------
    def esta_quieto(self):
        """¿Están quietos los puntos seguidos? Mediana de su movimiento desde la imagen anterior.

        Se usa la cámara y no la IMU: con la IMU, un movimiento a velocidad constante no se
        distingue del reposo; los puntos, en cambio, solo se quedan quietos si la cámara lo está.
        """
        cfg = self.cfgs["vio"]["reposo"]
        desplazamientos = [
            np.linalg.norm(uv - self.uv_anterior[i])
            for i, uv in zip(self.frontend.ids, self.frontend.puntos)
            if i in self.uv_anterior
        ]
        if len(desplazamientos) < cfg["puntos_min"]:
            return False
        return float(np.median(desplazamientos)) < cfg["desplazamiento_max_px"]

    # ---------------- Mapa (etapa 18) ----------------
    def _keyframe(self, saliente):
        """El clon que acaba de salir de la ventana: KeyframeMapeo o None.

        Su imagen ya no hace falta en ningún caso, así que se borra (y las anteriores, si
        quedara alguna).
        """
        gris = self.grises.pop(saliente["imagen"], None)
        for indice in [k for k in self.grises if k < saliente["imagen"]]:
            del self.grises[indice]
        if gris is None:
            return None
        kf = mapeo.keyframe_desde_clon(
            saliente, gris, self.selector, self.cfgs["vio"]["mapeo"]
        )
        if kf is not None:
            self.n_keyframes += 1
        return kf

    # ---------------- Datos ----------------
    def nueva_imu(self, t_ns, w_m, a_m):
        """Una muestra de la IMU (t_ns [ns], w_m [rad/s], a_m [m/s²])."""
        # Etapa 7.2: a_medida = escala·a + sesgo  ->  a = (a_medida - sesgo) / escala.
        # Se corrige aquí, antes de todo, para que la inicialización y el filtro vean lo mismo
        a_m = np.asarray(a_m, dtype=float)
        if self.calib_acc is not None:
            a_m = ci.corregir_acelerometro(a_m, self.calib_acc[0], self.calib_acc[1])
        if self.filtro is None:
            self.reposo.append(
                (int(t_ns), np.asarray(w_m, dtype=float), np.asarray(a_m, dtype=float))
            )
            duracion_s = (self.reposo[-1][0] - self.reposo[0][0]) / 1e9
            if duracion_s >= self.cfgs["vio"]["inicializacion"]["t_reposo_s"]:
                self._inicializar()
        else:
            self.pendientes.append((int(t_ns), w_m, a_m))

    def nueva_imagen(self, t_camara_ns, gris):
        """Una imagen (t_camara_ns: marca del sensor [ns]).

        Devuelve: None si el filtro aún no está listo, o un dict con t_ns, R_w_i, p_w, v_w,
        los recuentos del front-end y del MSCKF, los tiempos de cada parte [ms] y, para el
        mapa (etapa 17): puntos_mapa (ids (N,), p_w (N, 3)) y keyframe (KeyframeMapeo o None).
        """
        if self.filtro is None:
            return None
        t_ns = int(t_camara_ns) + self.t_d_ns  # convención de Kalibr: t_cam + t_d
        if t_ns <= self.filtro.t_ns:
            self.descartadas += 1  # imagen anterior al estado actual
            return None  # imagen anterior al estado actual

        # 1. IMU hasta el instante de la imagen (las posteriores esperan a la siguiente)
        t0 = time.perf_counter_ns()
        while self.pendientes and self.pendientes[0][0] <= t_ns:
            t_k, w_k, a_k = self.pendientes.pop(0)
            self.filtro.anadir_imu(t_k, w_k, a_k)
        t1 = time.perf_counter_ns()

        # 2. Seguimiento de puntos (etapa 8): ids y posiciones de los puntos vistos ahora
        res_fe = self.frontend.procesar(gris, t_ns)
        ids = self.frontend.ids
        xy = (
            mc.normalizar(self.frontend.puntos, self.K, self.dist)
            if len(ids)
            else np.zeros((0, 2))
        )
        t2 = time.perf_counter_ns()

        # 3. MSCKF (etapa 12): propagar hasta la imagen, clonar, corregir
        self.grises[self.filtro.n_imagenes] = gris  # etapa 18
        est = self.filtro.procesar_imagen(t_ns, ids, xy)

        # 4. Si los puntos no se mueven, además, velocidad nula (etapa 13)
        est["quieto"] = self.esta_quieto()
        if est["quieto"]:
            self.filtro.corregir_reposo(
                self.cfgs["vio"]["reposo"]["sigma_velocidad_m_s"]
            )
        self.uv_anterior = {
            int(i): uv.copy() for i, uv in zip(ids, self.frontend.puntos)
        }
        # 5. Etapa 18: lo que el filtro deja para el mapa
        puntos_mapa = mapeo.filtrar_puntos(
            est.pop("puntos_mapa"), self.cfgs["vio"]["mapeo"]
        )
        saliente = est.pop("saliente")
        keyframe = self._keyframe(saliente) if saliente is not None else None
        t3 = time.perf_counter_ns()

        e = self.filtro.estado
        return {
            "t_ns": t_ns,
            "R_w_i": e.R_w_i.copy(),
            "p_w": e.p_w.copy(),
            "v_w": e.v_w.copy(),
            "frontend": res_fe["estadisticas"],
            "msckf": est,
            "puntos_mapa": puntos_mapa,
            "keyframe": keyframe,
            "tiempos_ms": {
                "imu": (t1 - t0) / 1e6,
                "frontend": (t2 - t1) / 1e6,
                "msckf": (t3 - t2) / 1e6,
                "total": (t3 - t0) / 1e6,
            },
        }
