"""
vio/emparejador.py — Etapa 21: SuperPoint + LightGlue sobre DOS imágenes.

Hasta ahora un punto se SIGUE: FAST lo detecta en una imagen y Lucas-Kanade busca, en la
siguiente, una ventanita igual cerca de donde estaba. Funciona si entre las dos imágenes ha
cambiado poco. Aquí se hace otra cosa, EMPAREJAR:

    1. SuperPoint mira cada imagen POR SEPARADO y devuelve
         - puntos: posiciones (u, v) llamativas (esquinas, cruces);
         - descriptores: de cada punto, una lista de 256 números que resume cómo es la
           imagen a su alrededor. Dos vistas del mismo sitio dan listas parecidas aunque
           haya cambiado el ángulo, la distancia o la luz.
    2. LightGlue recibe los puntos y descriptores de las DOS imágenes y decide qué punto de
       la primera es el mismo que cuál de la segunda. No compara cada punto por su cuenta:
       mira todos a la vez, y por eso distingue dos esquinas parecidas por sus vecinas.
       De cada pareja da una confianza de 0 a 1; un punto puede quedarse sin pareja.

Las dos son redes neuronales ya entrenadas (no se entrenan aquí): se descargan sus pesos la
primera vez. Necesitan PyTorch, así que este módulo es para el PC, no para la Pi. Los
imports de torch y lightglue van DENTRO de la clase para que el resto del proyecto (y los
tests de la Pi) puedan importar este archivo sin tenerlos instalados.

Lo demás del archivo no usa redes: comprobar las parejas contra la verdad del simulador
(etapa 20) y dibujarlas.
"""

import time

import cv2
import numpy as np

from vio import escena
from vio import modelo_camara as mc


# =====================================================================
# Paso 21.2 — Las dos redes, detrás de una clase sencilla
# =====================================================================
def elegir_dispositivo(pedido, hay_cuda):
    """Dónde se ejecutan las redes: "cuda" (tarjeta gráfica NVIDIA) o "cpu" (procesador).

    pedido: "auto", "cpu" o "cuda" (config/lightglue.yaml). hay_cuda: bool, si PyTorch ve
    una tarjeta NVIDIA. Con "auto" se usa la tarjeta si la hay.
    """
    if pedido == "auto":
        return "cuda" if hay_cuda else "cpu"
    if pedido == "cuda" and not hay_cuda:
        raise RuntimeError("config/lightglue.yaml pide cuda, pero PyTorch no ve ninguna tarjeta NVIDIA")
    if pedido not in ("cpu", "cuda"):
        raise ValueError(f"dispositivo desconocido: {pedido}")
    return pedido


class EmparejadorLightGlue:
    """SuperPoint + LightGlue con entradas y salidas en NumPy.

    Uso:
        emparejador = EmparejadorLightGlue(cargar_config("lightglue"))
        a = emparejador.extraer(gris_a)             # puntos y descriptores de cada imagen
        b = emparejador.extraer(gris_b)
        indices, confianza, tiempo_ms = emparejador.emparejar(a, b)
        uv_a = a["puntos"][indices[:, 0]]           # la pareja k: uv_a[k] <-> uv_b[k]
        uv_b = b["puntos"][indices[:, 1]]
    """

    def __init__(self, cfg, extractor=None, emparejador=None):
        """cfg: config/lightglue.yaml.

        extractor, emparejador: solo para los tests, que pasan dos objetos falsos con la
        misma forma de trabajar que las redes. Normalmente se dejan en None y se crean aquí
        (la primera vez se descargan los pesos: unos 50 MB, hace falta internet).
        """
        import torch                                   # aquí dentro: solo en el PC

        self.torch = torch
        self.cfg = cfg
        self.dispositivo = elegir_dispositivo(cfg["dispositivo"], torch.cuda.is_available())
        if extractor is None or emparejador is None:
            from lightglue import LightGlue, SuperPoint

            cfg_sp, cfg_lg = cfg["superpoint"], cfg["lightglue"]
            extractor = SuperPoint(max_num_keypoints=cfg_sp["max_puntos"],
                                   detection_threshold=cfg_sp["umbral_deteccion"],
                                   nms_radius=cfg_sp["radio_nms_px"])
            emparejador = LightGlue(features="superpoint",
                                    filter_threshold=cfg_lg["umbral_confianza"],
                                    depth_confidence=cfg_lg["parada_temprana"],
                                    width_confidence=cfg_lg["poda_puntos"])
            # eval(): modo "usar la red", no "entrenarla". to(): llevarla a la tarjeta o a la CPU
            extractor = extractor.eval().to(self.dispositivo)
            emparejador = emparejador.eval().to(self.dispositivo)
        self.extractor = extractor
        self.emparejador = emparejador

    def extraer(self, gris):
        """SuperPoint sobre una imagen.

        gris: array (alto, ancho) uint8.
        Devuelve: dict con
            puntos        (N, 2) float64, (u, v) en píxeles de la imagen original
            puntuaciones  (N,) lo "llamativo" que es cada punto (de 0 a 1)
            tiempo_ms     lo que ha tardado
            _red          lo mismo en el formato de la red, para pasárselo a emparejar()
        """
        if gris.ndim != 2 or gris.dtype != np.uint8:
            raise ValueError("extraer() espera una imagen en gris de 8 bits (alto, ancho)")
        t0 = time.perf_counter()
        torch = self.torch
        # La red quiere un tensor (el array de PyTorch) de forma (1 canal, alto, ancho) con
        # valores de 0 a 1, y en el mismo dispositivo que ella
        tensor = torch.from_numpy(gris.astype(np.float32) / 255.0)[None].to(self.dispositivo)
        # no_grad(): no guardar lo necesario para entrenar (más rápido y menos memoria).
        # resize=None: sin cambiar el tamaño (por defecto la ampliaría a 1024 px de lado)
        with torch.no_grad():
            red = self.extractor.extract(tensor, resize=None)
        # La red trabaja por lotes: todo lleva delante una dimensión de tamaño 1 (este lote
        # tiene una sola imagen). [0] la quita; cpu().numpy() lo trae de vuelta a NumPy
        puntos = red["keypoints"][0].cpu().numpy().astype(np.float64)
        puntuaciones = red["keypoint_scores"][0].cpu().numpy().astype(np.float64)
        return {"puntos": puntos, "puntuaciones": puntuaciones, "_red": red,
                "tiempo_ms": 1e3 * (time.perf_counter() - t0)}

    def emparejar(self, a, b):
        """LightGlue entre dos extracciones (lo que devuelve extraer()).

        Devuelve: (indices (K, 2) int, confianza (K,) de 0 a 1, tiempo_ms).
        La fila k de indices dice: el punto indices[k, 0] de `a` es el indices[k, 1] de `b`.
        """
        if len(a["puntos"]) == 0 or len(b["puntos"]) == 0:
            return np.zeros((0, 2), dtype=int), np.zeros(0), 0.0
        t0 = time.perf_counter()
        with self.torch.no_grad():
            salida = self.emparejador({"image0": a["_red"], "image1": b["_red"]})
        indices = salida["matches"][0].cpu().numpy().astype(int).reshape(-1, 2)
        confianza = salida["scores"][0].cpu().numpy().astype(np.float64).reshape(-1)
        return indices, confianza, 1e3 * (time.perf_counter() - t0)


# =====================================================================
# Paso 21.3 — ¿Son correctas las parejas? (con la verdad del simulador)
# =====================================================================
def posicion_verdadera_en_b(uv_a, profundidad_a, pose_a, pose_b, K, dist):
    """Dónde está DE VERDAD, en la imagen B, cada punto de la imagen A.

    Solo se puede calcular en el simulador, que conoce la profundidad de cada píxel:
        píxel de A + su profundidad -> punto 3D en la cámara A -> en el mundo -> en la
        cámara B -> píxel de B.
    uv_a: (N, 2) [px]. profundidad_a: (alto, ancho) [m], de escena.renderizar_habitacion.
    pose_a, pose_b: (R_c_w, t_c_w) de cada cámara. dist: distorsión, o None.
    Devuelve: (uv_b (N, 2), visible (N,) bool): visible = delante de B y dentro de su imagen.
    """
    alto, ancho = profundidad_a.shape
    R_a, t_a = pose_a
    R_b, t_b = pose_b
    # 1. Píxel -> coordenadas normalizadas (x, y) y profundidad z en ese píxel (interpolada)
    if dist is None:
        xy = np.column_stack([(uv_a[:, 0] - K[0, 2]) / K[0, 0], (uv_a[:, 1] - K[1, 2]) / K[1, 1]])
    else:
        xy = mc.normalizar(uv_a, K, dist)
    z = escena.muestrear_bilineal(profundidad_a, uv_a[:, 0], uv_a[:, 1])
    # 2. Cámara A -> mundo -> cámara B
    p_c_a = z[:, None] * np.column_stack([xy, np.ones(len(xy))])
    p_w = (p_c_a - t_a) @ R_a                                  # R_aᵀ·(p_c - t_a)
    p_c_b = p_w @ R_b.T + t_b
    # 3. Cámara B -> píxel (solo los que están delante: no se puede dividir por z <= 0)
    delante = p_c_b[:, 2] > 0.05
    uv_b = np.full((len(uv_a), 2), np.nan)
    if np.any(delante):
        ceros = np.zeros(5) if dist is None else dist
        uv_b[delante] = mc.proyectar(p_c_b[delante], K, ceros)
    visible = delante.copy()
    visible[delante] = ((uv_b[delante, 0] >= 0) & (uv_b[delante, 0] <= ancho - 1)
                        & (uv_b[delante, 1] >= 0) & (uv_b[delante, 1] <= alto - 1))
    return uv_b, visible


def evaluar_parejas(uv_b, uv_b_verdad, visible, umbral_px):
    """Compara dónde dice el emparejador que está cada punto con dónde está de verdad.

    uv_b: (K, 2) lo encontrado. uv_b_verdad, visible: de posicion_verdadera_en_b.
    Una pareja es correcta si el punto se ve en B y el error es menor que umbral_px.
    Devuelve: dict con error_px (K,) (infinito si el punto no se ve en B), correctas (K,) bool,
    n, n_correctas, fraccion_correctas y error_mediano_px (de las correctas).
    """
    error = np.full(len(uv_b), np.inf)
    error[visible] = np.linalg.norm(uv_b[visible] - uv_b_verdad[visible], axis=1)
    correctas = error < umbral_px
    n = len(uv_b)
    return {
        "error_px": error,
        "correctas": correctas,
        "n": n,
        "n_correctas": int(correctas.sum()),
        "fraccion_correctas": float(correctas.mean()) if n else 0.0,
        "error_mediano_px": float(np.median(error[correctas])) if correctas.any() else float("nan"),
    }


# =====================================================================
# Paso 21.4 — Dibujar las parejas
# =====================================================================
VERDE = (0, 200, 0)          # colores en el orden de OpenCV: azul, verde, rojo
ROJO = (0, 0, 255)
AMARILLO = (0, 220, 220)


def dibujar_parejas(gris_a, gris_b, uv_a, uv_b, correctas=None, max_lineas=300):
    """Las dos imágenes juntas, A a la izquierda y B a la derecha, y una línea por pareja.

    uv_a, uv_b: (K, 2) [px], la pareja k une uv_a[k] con uv_b[k].
    correctas: (K,) bool o None. Con la verdad: verde las correctas y rojo las falsas; sin
    ella (fotos reales), todas en amarillo.
    max_lineas: con más parejas, se dibujan solo max_lineas repartidas (si no, no se ve nada).
    Devuelve: imagen en color (alto, 2·ancho, 3) uint8.
    """
    alto, ancho = gris_a.shape
    if gris_b.shape != gris_a.shape:
        raise ValueError("las dos imágenes deben tener el mismo tamaño")
    lienzo = cv2.cvtColor(np.hstack([gris_a, gris_b]), cv2.COLOR_GRAY2BGR)
    k_total = len(uv_a)
    paso = max(int(np.ceil(k_total / max_lineas)), 1)
    for k in range(0, k_total, paso):
        if correctas is None:
            color = AMARILLO
        else:
            color = VERDE if correctas[k] else ROJO
        p_a = (int(round(uv_a[k, 0])), int(round(uv_a[k, 1])))
        p_b = (int(round(uv_b[k, 0])) + ancho, int(round(uv_b[k, 1])))     # B, desplazada
        cv2.line(lienzo, p_a, p_b, color, 1, cv2.LINE_AA)
        cv2.circle(lienzo, p_a, 3, color, -1, cv2.LINE_AA)
        cv2.circle(lienzo, p_b, 3, color, -1, cv2.LINE_AA)
    return lienzo
