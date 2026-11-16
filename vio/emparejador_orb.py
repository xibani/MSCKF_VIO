"""
vio/emparejador_orb.py — Etapa 28: un emparejador clásico, sin redes.

Hace lo mismo que EmparejadorLightGlue (etapa 21) y se usa igual, pero con ORB, un método de
2011 que trae OpenCV y no necesita PyTorch:
    - Detecta esquinas con FAST (el mismo detector del front-end de la etapa 8).
    - De cada una guarda un descriptor de 256 BITS: 256 comparaciones de brillo entre
      parejas de píxeles de alrededor ("¿este es más claro que aquel?"), orientadas según
      el giro de la esquina.
    - Dos puntos se parecen si sus descriptores difieren en pocos bits (distancia de Hamming).
Aguanta mucho menos que SuperPoint + LightGlue: falla en cuanto el punto de vista cambia de
verdad. Está aquí para dos cosas: probar el cierre de bucles en la Pi y en los tests, y
tener con qué comparar. Para datos reales, el emparejador es LightGlue.
"""

import time

import cv2
import numpy as np


class EmparejadorOrb:
    """ORB + fuerza bruta, con la misma forma de uso que EmparejadorLightGlue."""

    dispositivo = "cpu (ORB, sin redes)"

    def __init__(self, cfg_orb):
        """cfg_orb: sección "orb" de config/bucles.yaml."""
        self.cfg = cfg_orb
        self.orb = cv2.ORB_create(nfeatures=cfg_orb["max_puntos"], nlevels=cfg_orb["niveles"],
                                  fastThreshold=cfg_orb["umbral_fast"])
        self.fuerza_bruta = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extraer(self, gris):
        """Puntos y descriptores de una imagen.

        gris: (alto, ancho) uint8. Devuelve: dict con puntos (N, 2) [px], puntuaciones (N,),
        tiempo_ms y _red (aquí, los descriptores: (N, 32) uint8, 256 bits por punto).
        """
        if gris.ndim != 2 or gris.dtype != np.uint8:
            raise ValueError("extraer() espera una imagen en gris de 8 bits (alto, ancho)")
        t0 = time.perf_counter()
        esquinas, descriptores = self.orb.detectAndCompute(gris, None)
        puntos = np.array([e.pt for e in esquinas], dtype=np.float64).reshape(-1, 2)
        puntuaciones = np.array([e.response for e in esquinas], dtype=np.float64)
        return {"puntos": puntos, "puntuaciones": puntuaciones, "_red": descriptores,
                "tiempo_ms": 1e3 * (time.perf_counter() - t0)}

    def parejas_en_un_sentido(self, descriptores_a, descriptores_b):
        """Para cada punto de A, su mejor pareja en B, si es claramente mejor que la segunda.

        "Fuerza bruta": se compara cada descriptor de A con todos los de B. Se piden los dos
        más parecidos; si el primero no gana con claridad al segundo (prueba de la razón),
        el punto es ambiguo (hay otro casi igual) y se queda sin pareja.
        Devuelve: dict fila de A -> fila de B.
        """
        parejas = {}
        for vecinos in self.fuerza_bruta.knnMatch(descriptores_a, descriptores_b, k=2):
            if len(vecinos) == 2 and vecinos[0].distance < self.cfg["razon"] * vecinos[1].distance:
                parejas[vecinos[0].queryIdx] = vecinos[0].trainIdx
        return parejas

    def emparejar(self, a, b):
        """Parejas entre dos extracciones. Devuelve: (indices (K, 2) int, confianza (K,), tiempo_ms).

        Una pareja vale si lo es en los dos sentidos: el mejor de B para el punto de A, y
        el mejor de A para ese punto de B. La confianza es siempre 1: ORB no da una, y las
        dudosas ya se han quitado aquí.
        """
        if len(a["puntos"]) < 2 or len(b["puntos"]) < 2:
            return np.zeros((0, 2), dtype=int), np.zeros(0), 0.0
        t0 = time.perf_counter()
        de_a_a_b = self.parejas_en_un_sentido(a["_red"], b["_red"])
        de_b_a_a = self.parejas_en_un_sentido(b["_red"], a["_red"])
        indices = [(i, j) for i, j in de_a_a_b.items() if de_b_a_a.get(j) == i]
        indices = np.array(indices, dtype=int).reshape(-1, 2)
        return indices, np.ones(len(indices)), 1e3 * (time.perf_counter() - t0)


