"""
vio/bucles.py — Etapa 28: detectar que se ha vuelto a un sitio.

El VIO encadena movimientos pequeños, y sus errores se suman: después de un rato, la pose
tiene deriva. Si la cámara vuelve a un sitio donde ya estuvo, hay una segunda forma de saber
dónde está: comparar lo que ve ahora con lo que vio entonces. Esa medida no arrastra la
deriva del camino intermedio. Aquí se buscan esas parejas de keyframes ("bucles") y se mide
la pose de uno respecto al otro:
    1. Candidatos: qué parejas de keyframes merece la pena comparar.
    2. Emparejar sus imágenes (SuperPoint + LightGlue, etapa 21; u ORB, sin redes).
    3. PnP: con la profundidad de los puntos de una imagen (etapa 25) y dónde se ven en la
       otra, la pose de la segunda cámara respecto a la primera.
    4. Comprobación: la misma cuenta al revés tiene que dar la pose inversa.
La etapa 29 usa estas medidas para corregir las poses de todos los keyframes.

Una "vista" es lo mismo que en la etapa 26: dict con kf, z (alto, ancho) [m] y gris.
Pose relativa "de j en i": (R_i_j, p_i_j), con p_i = R_i_j·p_j + p_i_j. Es dónde está y
cómo está girada la cámara j vista desde la cámara i.
"""

import cv2
import numpy as np

from vio import medidas_mapa as mm
from vio import modelo_camara as mc
from vio import so3


# =====================================================================
# Paso 28.1 — Poses relativas
# =====================================================================
def pose_relativa(R_w_i, p_w_i, R_w_j, p_w_j):
    """La pose de la cámara j vista desde la cámara i, a partir de sus poses en el mundo.

    Un punto en ejes de j pasa al mundo con (R_w_j, p_w_j) y del mundo a i con la inversa de
    (R_w_i, p_w_i):  p_i = R_w_iᵀ·(R_w_j·p_j + p_w_j - p_w_i).
    Devuelve: (R_i_j (3, 3), p_i_j (3,)).
    """
    return R_w_i.T @ R_w_j, R_w_i.T @ (p_w_j - p_w_i)


def invertir_pose(R, p):
    """La pose inversa: si (R, p) es la de j en i, devuelve la de i en j."""
    return R.T, -R.T @ p


def diferencia_de_poses(R_1, p_1, R_2, p_2):
    """Cuánto se diferencian dos poses: (distancia entre posiciones [m], ángulo entre
    orientaciones [grados])."""
    return float(np.linalg.norm(p_1 - p_2)), float(np.degrees(so3.angulo_entre(R_1, R_2)))


# =====================================================================
# Paso 28.2 — Candidatos
# =====================================================================
def candidatos(vistas, cfg_candidatos):
    """Parejas de vistas (i, j), con i < j, que merece la pena comparar.

    Siempre: separadas al menos separacion_min keyframes (las vecinas se parecen, pero entre
    ellas no hay deriva que corregir).
    Con por_pose: además, que según el VIO las dos cámaras estén a menos de distancia_max_m y
    miren en direcciones a menos de angulo_max_grados (hacia dónde mira una cámara es la
    tercera columna de R_w_c: su eje z en el mundo). De cada j se quedan, como mucho, las
    max_por_keyframe más cercanas.
    Devuelve: lista de (i, j).
    """
    parejas = []
    for j in range(len(vistas)):
        posibles = []
        for i in range(j - cfg_candidatos["separacion_min"]):
            if not cfg_candidatos["por_pose"]:
                posibles.append((0.0, i))
                continue
            kf_i, kf_j = vistas[i]["kf"], vistas[j]["kf"]
            distancia = float(np.linalg.norm(kf_i["p_w_c"] - kf_j["p_w_c"]))
            coseno = float(np.clip(kf_i["R_w_c"][:, 2] @ kf_j["R_w_c"][:, 2], -1.0, 1.0))
            angulo = np.degrees(np.arccos(coseno))
            if distancia < cfg_candidatos["distancia_max_m"] and angulo < cfg_candidatos["angulo_max_grados"]:
                posibles.append((distancia, i))
        if cfg_candidatos["por_pose"]:
            posibles = sorted(posibles)[:cfg_candidatos["max_por_keyframe"]]
        parejas += [(i, j) for _, i in posibles]
    return parejas


# =====================================================================
# Paso 28.3 — PnP: la pose de una cámara a partir de puntos 3D de la otra
# =====================================================================
def puntos_3d_de_pixeles(uv, z_mapa, K, dist, cfg_pnp):
    """Los puntos 3D, en ejes de su cámara, de unos píxeles de una vista con profundidad.

    Como en la etapa 26: píxel -> rayo (x, y, 1) -> p_c = z·(x, y, 1), con la z leída de la
    imagen de profundidad (interpolada).
    uv: (N, 2) [px]. Devuelve: (p_c (N, 3), validos (N,) bool): no es válido un píxel sin
    profundidad o con ella fuera de [profundidad_min_m, profundidad_max_m].
    """
    z = mm.muestrear(z_mapa, uv)
    with np.errstate(invalid="ignore"):
        validos = (z >= cfg_pnp["profundidad_min_m"]) & (z <= cfg_pnp["profundidad_max_m"])
    p_c = np.zeros((len(uv), 3))
    if validos.any():
        xy = mc.normalizar(uv[validos], K, dist)
        p_c[validos] = z[validos, None] * np.column_stack([xy, np.ones(len(xy))])
    return p_c, validos


def pose_por_pnp(p_a, uv_b, K, dist, cfg_pnp):
    """La pose de la cámara A vista desde la B, con puntos 3D de A y sus píxeles en B.

    Es el problema PnP ("perspectiva con n puntos"): se conocen N puntos del espacio (aquí,
    en ejes de A) y en qué píxel se ve cada uno desde otra cámara; se busca dónde está esa
    cámara. Con RANSAC, porque entre las parejas hay falsas: se resuelve con 4 puntos al
    azar, se cuenta cuántos de los demás se proyectan cerca de su píxel, y al final se
    reajusta con todos los que encajan.
    Los píxeles de B se pasan antes a coordenadas normalizadas (sin intrínsecos ni
    distorsión), así que a OpenCV se le da una cámara "ideal": K = identidad.
    p_a: (N, 3) [m]. uv_b: (N, 2) [px]. Devuelve: None si no hay solución, o dict con
    R (3, 3) y p (3,) —la pose de A en B: p_b = R·p_a + p— y encajan (N,) bool.
    """
    if len(p_a) < 4:
        return None
    xy_b = mc.normalizar(uv_b, K, dist)
    cv2.setRNGSeed(cfg_pnp["semilla"])                   # el mismo azar cada vez: mismo resultado
    hay, giro, traslacion, filas = cv2.solvePnPRansac(
        p_a.reshape(-1, 1, 3), xy_b.reshape(-1, 1, 2), np.eye(3), None,
        iterationsCount=cfg_pnp["iteraciones"], reprojectionError=cfg_pnp["umbral_px"] / K[0, 0],
        confidence=cfg_pnp["confianza"], flags=cv2.SOLVEPNP_P3P)
    if not hay or filas is None or len(filas) < 4:
        return None
    filas = filas.ravel()
    # Reajuste con todos los puntos que encajan, partiendo de la solución del RANSAC
    _, giro, traslacion = cv2.solvePnP(p_a[filas].reshape(-1, 1, 3), xy_b[filas].reshape(-1, 1, 2), np.eye(3), None,
                                       rvec=giro, tvec=traslacion, useExtrinsicGuess=True,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
    encajan = np.zeros(len(p_a), dtype=bool)
    encajan[filas] = True
    return {"R": cv2.Rodrigues(giro)[0], "p": traslacion.ravel(), "encajan": encajan}


def pose_entre_vistas(uv_a, uv_b, vista_a, cfg_pnp):
    """PnP con los puntos de A que tienen profundidad. uv_a, uv_b: (K, 2), las parejas.

    Devuelve: (resultado de pose_por_pnp o None, con_profundidad (K,) bool). En el resultado,
    "encajan" se refiere a las K parejas (falso en las que no tenían profundidad), y la
    pose solo se da por buena si encajan encajan_min puntos y al menos fraccion_min de los
    que tenían profundidad.
    """
    kf = vista_a["kf"]
    p_a, validos = puntos_3d_de_pixeles(uv_a, vista_a["z"], kf["K"], kf["dist"], cfg_pnp)
    if validos.sum() < 4:
        return None, validos
    resultado = pose_por_pnp(p_a[validos], uv_b[validos], kf["K"], kf["dist"], cfg_pnp)
    if resultado is None:
        return None, validos
    encajan = np.zeros(len(uv_a), dtype=bool)
    encajan[np.flatnonzero(validos)[resultado["encajan"]]] = True
    resultado["encajan"] = encajan
    if encajan.sum() < cfg_pnp["encajan_min"] or encajan.sum() < cfg_pnp["fraccion_min"] * validos.sum():
        return None, validos
    return resultado, validos


# =====================================================================
# Paso 28.4 — Un bucle: emparejar, PnP en los dos sentidos y comprobar
# =====================================================================
def comprobar_bucle(vista_i, vista_j, extraccion_i, extraccion_j, emparejador, cfg):
    """¿Ven las vistas i y j el mismo sitio? Si es así, la pose de j vista desde i.

    extraccion_i, extraccion_j: lo que devuelve emparejador.extraer() para cada imagen.
    Devuelve: (bucle, motivo). bucle es None si no lo hay (y motivo dice por qué: "pocas
    parejas", "sin pose" o "no coinciden los dos sentidos"), o un dict con
        R, p              la pose de j en i: p_i = R·p_j + p
        parejas           número de parejas con confianza suficiente
        encajan           cuántas encajan con la pose (en el sentido que se ha usado)
        desacuerdo_m, desacuerdo_grados    diferencia entre los dos sentidos
        uv_i, uv_j        (M, 2) las parejas que encajan, para dibujarlas
    """
    indices, confianza, _ = emparejador.emparejar(extraccion_i, extraccion_j)
    indices = indices[confianza >= cfg["parejas"]["umbral_confianza"]]
    if len(indices) < cfg["parejas"]["minimas"]:
        return None, "pocas parejas"
    uv_i = extraccion_i["puntos"][indices[:, 0]]
    uv_j = extraccion_j["puntos"][indices[:, 1]]
    # Sentido 1: puntos 3D de j vistos en i -> la pose de j en i, directamente
    desde_j, _ = pose_entre_vistas(uv_j, uv_i, vista_j, cfg["pnp"])
    # Sentido 2: puntos 3D de i vistos en j -> la pose de i en j; invertida, la de j en i
    desde_i, _ = pose_entre_vistas(uv_i, uv_j, vista_i, cfg["pnp"])
    if desde_j is None or desde_i is None:
        return None, "sin pose"
    R_1, p_1 = desde_j["R"], desde_j["p"]
    R_2, p_2 = invertir_pose(desde_i["R"], desde_i["p"])
    desacuerdo_m, desacuerdo_grados = diferencia_de_poses(R_1, p_1, R_2, p_2)
    if desacuerdo_m > cfg["acuerdo"]["distancia_max_m"] or desacuerdo_grados > cfg["acuerdo"]["angulo_max_grados"]:
        return None, "no coinciden los dos sentidos"
    # Se queda la del sentido con más puntos
    if desde_j["encajan"].sum() >= desde_i["encajan"].sum():
        R, p, encajan = R_1, p_1, desde_j["encajan"]
    else:
        R, p, encajan = R_2, p_2, desde_i["encajan"]
    return {"R": R, "p": p, "parejas": len(indices), "encajan": int(encajan.sum()),
            "desacuerdo_m": desacuerdo_m, "desacuerdo_grados": desacuerdo_grados,
            "uv_i": uv_i[encajan], "uv_j": uv_j[encajan]}, "ok"


# =====================================================================
# Paso 28.5 — Todos los bucles de una sesión
# =====================================================================
def buscar_bucles(vistas, emparejador, cfg):
    """Todos los bucles entre las vistas.

    Cada imagen se pasa una sola vez por emparejador.extraer(); después se comprueba cada
    pareja candidata. A cada bucle se le añade lo que dice el VIO de esa misma pareja.
    Devuelve: (bucles, recuentos). bucles: lista de dict de comprobar_bucle, con además
        i, j                        posiciones en la lista `vistas` (i < j)
        indice_i, indice_j          el número de imagen de cada keyframe
        vio_m, vio_grados           diferencia entre la pose medida y la que dan las poses
                                    del VIO: la deriva acumulada entre i y j que ve el bucle
    recuentos: dict con candidatos, bucles y cuántos candidatos se han quedado por cada motivo.
    """
    parejas = candidatos(vistas, cfg["candidatos"])
    recuentos = {"candidatos": len(parejas), "bucles": 0, "pocas parejas": 0, "sin pose": 0,
                 "no coinciden los dos sentidos": 0}
    usadas = sorted({k for pareja in parejas for k in pareja})
    extracciones = {k: emparejador.extraer(vistas[k]["gris"]) for k in usadas}
    bucles = []
    for i, j in parejas:
        bucle, motivo = comprobar_bucle(vistas[i], vistas[j], extracciones[i], extracciones[j], emparejador, cfg)
        if bucle is None:
            recuentos[motivo] += 1
            continue
        kf_i, kf_j = vistas[i]["kf"], vistas[j]["kf"]
        R_vio, p_vio = pose_relativa(kf_i["R_w_c"], kf_i["p_w_c"], kf_j["R_w_c"], kf_j["p_w_c"])
        vio_m, vio_grados = diferencia_de_poses(bucle["R"], bucle["p"], R_vio, p_vio)
        bucle.update(i=i, j=j, indice_i=kf_i["indice"], indice_j=kf_j["indice"], vio_m=vio_m, vio_grados=vio_grados)
        bucles.append(bucle)
        recuentos["bucles"] += 1
    return bucles, recuentos


# =====================================================================
# Paso 28.6 — Comparar con la verdad (solo en el simulador)
# =====================================================================
def poses_verdaderas_de_camara(t_ns, t_ver, p_ver, R_ver, R_i_c, p_i_c):
    """La pose verdadera de la CÁMARA en unos instantes, a partir de la verdad del simulador.

    verdad.csv guarda la pose de la IMU (R_w_i, p_w_i). La cámara está fija a la IMU, girada
    R_i_c y desplazada p_i_c (la extrínseca):  R_w_c = R_w_i·R_i_c,  p_w_c = p_w_i + R_w_i·p_i_c.
    t_ns: (N,) instantes (los de los keyframes, en el reloj del filtro). t_ver, p_ver, R_ver:
    de leer_poses(verdad.csv). Devuelve: lista de N parejas (R_w_c, p_w_c), en el mundo del simulador.
    """
    poses = []
    for t in t_ns:
        k = min(int(np.searchsorted(t_ver, t)), len(t_ver) - 1)
        poses.append((R_ver[k] @ R_i_c, p_ver[k] + R_ver[k] @ p_i_c))
    return poses


def errores_de_bucles(bucles, poses_verdad):
    """El error de cada bucle frente a la pose relativa verdadera.

    poses_verdad: lista de (R_w_c, p_w_c) verdaderas, una por vista, en el mismo orden. La
    pose relativa no depende de qué mundo se use: se puede comparar sin alinear nada.
    Devuelve: lista de (error [m], error [grados]), una pareja por bucle.
    """
    errores = []
    for bucle in bucles:
        R_i, p_i = poses_verdad[bucle["i"]]
        R_j, p_j = poses_verdad[bucle["j"]]
        R_ver, p_ver = pose_relativa(R_i, p_i, R_j, p_j)
        errores.append(diferencia_de_poses(bucle["R"], bucle["p"], R_ver, p_ver))
    return errores
