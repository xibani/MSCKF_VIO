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
