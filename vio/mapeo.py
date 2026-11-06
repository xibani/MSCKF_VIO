"""
vio/mapeo.py — Etapa 18: lo que el VIO entrega para la reconstrucción 3D en el PC.

El MSCKF no guarda mapa: cada punto se triangula, se usa en una corrección y se olvida.
Para reconstruir, el filtro deja dos cosas por el camino (vio/msckf.py, etapa 17):
    - puntos_mapa: cada punto aceptado en una corrección, en el mundo -> mapa disperso.
    - el clon que sale de la ventana: su pose ya no va a cambiar más, y lleva la
      profundidad de los puntos aceptados que se vieron en él.
Este módulo decide qué clones salientes se convierten en KEYFRAMES DE MAPEO (imágenes
elegidas cada cierto movimiento, con su pose y sus profundidades) y los empaqueta.

Criterio de keyframe (como SVO, §VI: "a keyframe is selected if the Euclidean distance of
the new frame relative to all keyframes exceeds 12 % of the average scene depth"), con dos
cambios:
    - se compara solo con el ÚLTIMO keyframe (en streaming no hay mapa de keyframes);
    - se usa la MEDIANA de la profundidad (un punto lejano mal triangulado no la mueve);
y además un giro mínimo: girando sin desplazarse no hay paralaje, pero se ve otra parte de
la habitación, que Depth Anything sí puede densificar.
"""

from dataclasses import dataclass

import numpy as np

from vio import so3


@dataclass
class KeyframeMapeo:
    """Un keyframe de mapeo: la imagen, su pose definitiva y la profundidad de sus puntos."""

    t_ns: int  # instante de la imagen en el reloj del filtro (t_cámara + t_d) [ns]
    indice: int  # número de imagen dentro del filtro
    R_w_c: (
        np.ndarray
    )  # (3, 3) orientación de la cámara en el mundo (z del mundo hacia arriba)
    p_w_c: np.ndarray  # (3,) posición de la cámara en el mundo [m]
    ids: np.ndarray  # (N,) int64, identificador de cada punto (el del front-end)
    xy: np.ndarray  # (N, 2) coordenadas normalizadas SIN distorsión
    z: np.ndarray  # (N,) profundidad de cada punto en esta cámara [m]
    gris: np.ndarray  # (alto, ancho) uint8: la imagen ORIGINAL, con su distorsión


def filtrar_observaciones(obs, cfg):
    """Observaciones de un clon saliente que valen para el mapa.

    obs: lista de (id, xy (2,), z [m], paralaje [°]) que dejó el MSCKF.
    cfg: sección "mapeo" de config/vio.yaml.
    Se quedan las de paralaje suficiente y profundidad dentro de [profundidad_min_m,
    profundidad_max_m]. El MSCKF acepta pistas desde 1° de paralaje, porque para corregir la
    pose le basta con que encajen; para el mapa se pide más (error de profundidad ≈ z·σ/θ).
    Devuelve: (ids (N,) int64, xy (N, 2), z (N,)).
    """
    ids, xy, z = [], [], []
    for id_punto, xy_k, z_k, paralaje in obs:
        if paralaje < cfg["paralaje_min_grados"]:
            continue
        if not cfg["profundidad_min_m"] <= z_k <= cfg["profundidad_max_m"]:
            continue
        ids.append(int(id_punto))
        xy.append(xy_k)
        z.append(z_k)
    return (
        np.array(ids, dtype=np.int64),
        np.array(xy, dtype=float).reshape(-1, 2),
        np.array(z, dtype=float),
    )


def filtrar_puntos(puntos_mapa, cfg):
    """Puntos aceptados en una imagen que valen para el mapa disperso.

    puntos_mapa: lista de (id, p_w (3,), paralaje [°]) que dejó el MSCKF.
    Devuelve: (ids (N,) int64, p_w (N, 3)).
    """
    elegidos = [
        (int(i), p)
        for i, p, paralaje in puntos_mapa
        if paralaje >= cfg["paralaje_min_grados"]
    ]
    ids = np.array([i for i, _ in elegidos], dtype=np.int64)
    p_w = np.array([p for _, p in elegidos], dtype=float).reshape(-1, 3)
    return ids, p_w


def angulo_entre(R_a, R_b):
    """Ángulo de la rotación que lleva de R_a a R_b [grados]."""
    return float(np.degrees(np.linalg.norm(so3.log_so3(R_a.T @ R_b))))


class SelectorKeyframes:
    """Decide qué clones salientes son keyframes de mapeo."""

    def __init__(self, cfg):
        """cfg: sección "mapeo" de config/vio.yaml."""
        self.cfg = cfg
        self.ultimo = None  # (R_w_c, p_w_c) del último keyframe aceptado

    def decidir(self, R_w_c, p_w_c, z):
        """¿Es keyframe una imagen con esta pose y estas profundidades? Devuelve (bool, motivo).

        Si lo es, pasa a ser la referencia de los siguientes.
        """
        if len(z) < self.cfg["puntos_min"]:
            return False, "pocos puntos con profundidad"
        if self.ultimo is not None:
            R_ult, p_ult = self.ultimo
            distancia = float(np.linalg.norm(p_w_c - p_ult))
            umbral = self.cfg["fraccion_profundidad"] * float(np.median(z))
            giro = angulo_entre(R_ult, R_w_c)
            if distancia < umbral and giro < self.cfg["giro_min_grados"]:
                return False, "poco movimiento desde el último keyframe"
        self.ultimo = (R_w_c.copy(), p_w_c.copy())
        return True, "ok"


def keyframe_desde_clon(saliente, gris, selector, cfg):
    """Convierte un clon saliente del MSCKF en un KeyframeMapeo, o None si no lo es.

    saliente: dict del MSCKF con imagen, t_ns, R_w_c, p_w_c y obs.
    gris: la imagen de ese clon (la guardó el pipeline mientras el clon estaba en la ventana).
    """
    ids, xy, z = filtrar_observaciones(saliente["obs"], cfg)
    es_keyframe, _ = selector.decidir(saliente["R_w_c"], saliente["p_w_c"], z)
    if not es_keyframe:
        return None
    return KeyframeMapeo(
        t_ns=int(saliente["t_ns"]),
        indice=int(saliente["imagen"]),
        R_w_c=saliente["R_w_c"].copy(),
        p_w_c=saliente["p_w_c"].copy(),
        ids=ids,
        xy=xy,
        z=z,
        gris=gris,
    )
