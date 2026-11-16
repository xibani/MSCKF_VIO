"""
vio/densificar.py — Etapa 26: de la profundidad de cada keyframe a una sola nube densa.

La etapa 25 deja, por cada keyframe de mapeo, una imagen de profundidad en metros. Aquí:
    1. Cada píxel con profundidad se convierte en un punto 3D del mundo (con la pose del
       keyframe, que viene del VIO).
    2. Se quitan los píxeles dudosos de cada imagen: demasiado cerca o lejos, y los que
       están en un salto de profundidad.
    3. Cada punto se comprueba desde los demás keyframes: se queda si alguno lo ve a la
       misma profundidad.
    4. Los puntos de todos los keyframes se juntan en una rejilla de cubos (vóxeles): un
       punto por cubo ocupado.
En el simulador, además, se mide a qué distancia queda la nube de la habitación de verdad.

Una "vista" es un keyframe con su profundidad: dict con kf (el de protocolo.leer_keyframe),
z (alto, ancho) [m, NaN = sin dato] y gris (alto, ancho) uint8.
"""

import cv2
import numpy as np

from vio import escena
from vio import medidas_mapa as mm
from vio import modelo_camara as mc


# =====================================================================
# Paso 26.1 — De un píxel con profundidad a un punto del mundo
# =====================================================================
def rejilla_de_pixeles(tamano, paso):
    """Uno de cada `paso` píxeles, en las dos direcciones.

    tamano: (ancho, alto). Devuelve: (filas (N,) int, columnas (N,) int).
    """
    ancho, alto = tamano
    columnas, filas = np.meshgrid(np.arange(0, ancho, paso), np.arange(0, alto, paso))
    return filas.ravel(), columnas.ravel()


def rayos_de_pixeles(filas, columnas, K, dist):
    """El rayo de cada píxel, en ejes de la cámara: (x, y, 1), con (x, y) las coordenadas
    normalizadas sin distorsión del píxel (etapa 2). Devuelve: (N, 3)."""
    if len(filas) == 0:                                         # OpenCV no admite una lista vacía
        return np.zeros((0, 3))
    uv = np.column_stack([columnas, filas]).astype(float)       # u = columna, v = fila
    xy = mc.normalizar(uv, K, dist)
    return np.column_stack([xy, np.ones(len(xy))])


def puntos_en_mundo(z, rayos, R_w_c, p_w_c):
    """Puntos 3D del mundo a partir de la profundidad de cada rayo.

    En la cámara, el punto está en p_c = z·(x, y, 1): se avanza por el rayo hasta que la
    tercera coordenada vale z. En el mundo: p_w = R_w_c·p_c + p_w_c.
    z: (N,) [m]. rayos: (N, 3). Devuelve: (N, 3).
    """
    p_c = z[:, None] * rayos
    return p_c @ R_w_c.T + p_w_c


# =====================================================================
# Paso 26.2 — Los píxeles que no se usan
# =====================================================================
def mascara_de_bordes(z, salto_relativo, margen_px):
    """Qué píxeles NO están en un salto de profundidad.

    En el canto de un objeto la profundidad salta de golpe (del objeto a lo que tiene
    detrás). La red no da un salto limpio: lo reparte en unos cuantos píxeles, y esos
    píxeles, llevados al mundo, son puntos que flotan en el aire entre las dos superficies.
    Hay salto entre dos píxeles vecinos si sus profundidades difieren, en proporción, más
    que salto_relativo. Un píxel sin dato (NaN) cuenta como salto con sus vecinos.
    z: (alto, ancho) [m]. Devuelve: (alto, ancho) bool, True = se puede usar.
    """
    salto = np.zeros(z.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        # Cada píxel con su vecino de la derecha
        diferencia = np.abs(z[:, 1:] - z[:, :-1]) / np.minimum(z[:, 1:], z[:, :-1])
        hay = ~(diferencia < salto_relativo)             # con NaN la comparación da False: salto
        salto[:, 1:] |= hay
        salto[:, :-1] |= hay
        # Cada píxel con su vecino de abajo
        diferencia = np.abs(z[1:, :] - z[:-1, :]) / np.minimum(z[1:, :], z[:-1, :])
        hay = ~(diferencia < salto_relativo)
        salto[1:, :] |= hay
        salto[:-1, :] |= hay
    if margen_px > 0:
        # Ensanchar la zona de salto: cada píxel marcado marca a los que tiene alrededor
        lado = 2 * margen_px + 1
        salto = cv2.dilate(salto.astype(np.uint8), np.ones((lado, lado), np.uint8)) > 0
    return ~salto


def nube_de_vista(vista, cfg):
    """Los puntos del mundo que da una vista, ya sin los píxeles dudosos.

    vista: dict con kf, z y gris. cfg: config/densificar.yaml.
    Devuelve: dict con p_w (N, 3), gris (N,) uint8 (el brillo del píxel de cada punto) y los
    recuentos candidatos (píxeles de la rejilla), sin_dato (sin profundidad o fuera de
    [profundidad_min_m, profundidad_max_m]) y en_borde.
    """
    kf, z_mapa = vista["kf"], vista["z"]
    filas, columnas = rejilla_de_pixeles(kf["tamano"], cfg["paso_px"])
    z = z_mapa[filas, columnas]
    with np.errstate(invalid="ignore"):
        con_dato = (z >= cfg["profundidad_min_m"]) & (z <= cfg["profundidad_max_m"])
    lisos = mascara_de_bordes(z_mapa, cfg["bordes"]["salto_relativo"], cfg["bordes"]["margen_px"])
    usar = con_dato & lisos[filas, columnas]
    rayos = rayos_de_pixeles(filas[usar], columnas[usar], kf["K"], kf["dist"])
    return {"p_w": puntos_en_mundo(z[usar], rayos, kf["R_w_c"], kf["p_w_c"]),
            "gris": vista["gris"][filas[usar], columnas[usar]],
            "candidatos": len(z), "sin_dato": int((~con_dato).sum()),
            "en_borde": int((con_dato & ~lisos[filas, columnas]).sum())}


# =====================================================================
# Paso 26.3 — Cada punto, comprobado desde los demás keyframes
# =====================================================================
def campo_de_vision(K, dist, tamano):
    """Hasta dónde ve la cámara, en coordenadas normalizadas sin distorsión.

    Se normalizan los píxeles del marco de la imagen y se toman los extremos. Hace falta
    porque la fórmula de la distorsión solo vale dentro de la imagen: aplicada a un punto
    muy apartado del eje puede devolver un píxel de dentro, y parecería que la cámara lo ve.
    Devuelve: (x_min, x_max, y_min, y_max).
    """
    ancho, alto = tamano
    u = np.arange(ancho, dtype=float)
    v = np.arange(alto, dtype=float)
    marco = np.vstack([np.column_stack([u, np.zeros(ancho)]), np.column_stack([u, np.full(ancho, alto - 1.0)]),
                       np.column_stack([np.zeros(alto), v]), np.column_stack([np.full(alto, ancho - 1.0), v])])
    xy = mc.normalizar(marco, K, dist)
    return xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()


def comprobar_desde_otra(p_w, otra, umbral_relativo):
    """Mira unos puntos del mundo desde otra vista y los compara con su profundidad.

    Cada punto se lleva a los ejes de la otra cámara: p_c = R_w_cᵀ·(p_w - p_w_c). Su tercera
    coordenada es la profundidad a la que esa cámara debería verlo. Se proyecta a un píxel
    y se lee qué profundidad tiene la otra vista ahí:
        - parecida (dentro del umbral): la otra vista CONFIRMA el punto;
        - bastante mayor: la otra vista ve más lejos por ese mismo rayo, es decir, ve a
          través del sitio donde debería estar el punto: lo CONTRADICE;
        - bastante menor: hay algo delante que lo tapa. No dice nada del punto.
    Si el punto queda detrás de la cámara, fuera de la imagen o en un píxel sin dato,
    tampoco dice nada.
    p_w: (N, 3). otra: vista (dict con kf y z). Devuelve: (confirma (N,) bool, contradice (N,) bool).
    """
    kf = otra["kf"]
    p_c = (p_w - kf["p_w_c"]) @ kf["R_w_c"]              # cada fila: R_w_cᵀ·(p_w - p_w_c)
    x_min, x_max, y_min, y_max = campo_de_vision(kf["K"], kf["dist"], kf["tamano"])
    z_esperada = p_c[:, 2]
    delante = z_esperada > 1e-6
    xy = np.zeros((len(p_w), 2))
    xy[delante] = p_c[delante, :2] / z_esperada[delante, None]
    a_la_vista = delante & (xy[:, 0] > x_min) & (xy[:, 0] < x_max) & (xy[:, 1] > y_min) & (xy[:, 1] < y_max)

    confirma = np.zeros(len(p_w), dtype=bool)
    contradice = np.zeros(len(p_w), dtype=bool)
    if a_la_vista.any():
        uv = mc.proyectar(p_c[a_la_vista], kf["K"], kf["dist"])
        z_vista = mm.muestrear(otra["z"], uv)            # NaN fuera de la imagen o sin dato
        with np.errstate(invalid="ignore"):
            razon = z_vista / z_esperada[a_la_vista]
            confirma[a_la_vista] = np.abs(razon - 1.0) < umbral_relativo
            contradice[a_la_vista] = razon > 1.0 + umbral_relativo
    return confirma, contradice


def puntos_coherentes(nubes, vistas, cfg_coherencia):
    """Qué puntos de cada vista se quedan después de mirarlos desde todas las demás.

    Un punto se queda si lo confirman al menos vistas_min vistas y son más las que lo
    confirman que las que lo contradicen.
    nubes: lista de dict de nube_de_vista, una por vista. Devuelve: lista de (N,) bool.
    """
    se_quedan = []
    for i, nube in enumerate(nubes):
        confirmaciones = np.zeros(len(nube["p_w"]), dtype=int)
        contradicciones = np.zeros(len(nube["p_w"]), dtype=int)
        for j, otra in enumerate(vistas):
            if j == i:
                continue
            confirma, contradice = comprobar_desde_otra(nube["p_w"], otra, cfg_coherencia["umbral_relativo"])
            confirmaciones += confirma
            contradicciones += contradice
        se_quedan.append((confirmaciones >= cfg_coherencia["vistas_min"]) & (confirmaciones > contradicciones))
    return se_quedan


# =====================================================================
# Paso 26.4 — Un punto por cada cubo ocupado
# =====================================================================
def fundir_en_voxeles(p_w, gris, lado_m, puntos_min):
    """Junta los puntos que caen en el mismo cubo (vóxel) en uno solo: su media.

    El espacio se parte en cubos de lado lado_m. El cubo de un punto es la parte entera de
    sus coordenadas divididas por el lado: tres enteros. Los puntos con los mismos tres
    enteros están en el mismo cubo.
    p_w: (N, 3). gris: (N,). Devuelve: (p (M, 3), gris (M,) uint8, cuentas (M,) int): la
    posición media, el brillo medio y el número de puntos de cada cubo con puntos_min o más.
    """
    if len(p_w) == 0:
        return np.zeros((0, 3)), np.zeros(0, np.uint8), np.zeros(0, int)
    cubos = np.floor(p_w / lado_m).astype(np.int64)
    # np.unique agrupa las filas iguales: `de_que_cubo` dice a qué grupo pertenece cada punto
    _, de_que_cubo, cuentas = np.unique(cubos, axis=0, return_inverse=True, return_counts=True)
    de_que_cubo = de_que_cubo.ravel()
    n_cubos = len(cuentas)
    # np.bincount con pesos suma, para cada cubo, el valor de todos sus puntos
    p = np.column_stack([np.bincount(de_que_cubo, weights=p_w[:, eje], minlength=n_cubos) for eje in range(3)])
    p = p / cuentas[:, None]
    brillo = np.bincount(de_que_cubo, weights=gris.astype(float), minlength=n_cubos) / cuentas
    bastantes = cuentas >= puntos_min
    return p[bastantes], np.round(brillo[bastantes]).astype(np.uint8), cuentas[bastantes]


def densificar(vistas, cfg):
    """Toda la etapa: de las vistas a la nube densa.

    vistas: lista de dict con kf, z y gris. cfg: config/densificar.yaml.
    Devuelve: dict con p_w (M, 3), gris (M,) uint8, cuentas (M,) y por_vista: una lista de
    dict (indice, candidatos, sin_dato, en_borde, sin_confirmar, usados) con qué ha pasado
    con los píxeles de cada vista.
    """
    nubes = [nube_de_vista(vista, cfg) for vista in vistas]
    se_quedan = puntos_coherentes(nubes, vistas, cfg["coherencia"])
    por_vista = []
    for vista, nube, queda in zip(vistas, nubes, se_quedan):
        por_vista.append({"indice": vista["kf"]["indice"], "candidatos": nube["candidatos"],
                          "sin_dato": nube["sin_dato"], "en_borde": nube["en_borde"],
                          "sin_confirmar": int((~queda).sum()), "usados": int(queda.sum())})
    if vistas:
        todos = np.vstack([nube["p_w"][queda] for nube, queda in zip(nubes, se_quedan)])
        brillos = np.concatenate([nube["gris"][queda] for nube, queda in zip(nubes, se_quedan)])
    else:
        todos, brillos = np.zeros((0, 3)), np.zeros(0, np.uint8)
    p_w, gris, cuentas = fundir_en_voxeles(todos, brillos, cfg["voxel"]["lado_m"], cfg["voxel"]["puntos_min"])
    return {"p_w": p_w, "gris": gris, "cuentas": cuentas, "por_vista": por_vista}


# =====================================================================
# Paso 26.5 — Comparar con la habitación de verdad (solo en el simulador)
# =====================================================================
def transformacion_a_la_verdad(t_est, p_est, R_est, t_ver, p_ver, R_ver):
    """Cómo pasar del mundo del VIO al mundo del simulador.

    El filtro define su mundo al arrancar: el origen donde está y rumbo cero. El simulador
    tiene el suyo (el centro del suelo de la habitación). Se pasa de uno a otro haciendo
    coincidir la primera pose estimada con la verdadera de ese instante:
        R = R_ver(t0)·R_est(t0)ᵀ        t = p_ver(t0) - R·p_est(t0)
    t_est, p_est, R_est: la trayectoria estimada (vio_trayectoria.csv). t_ver, p_ver, R_ver:
    la verdadera (verdad.csv). Devuelve: (R (3, 3), t (3,)), con p_verdad = R·p_vio + t.
    """
    i = min(int(np.searchsorted(t_ver, t_est[0])), len(t_ver) - 1)
    R = R_ver[i] @ R_est[0].T
    return R, p_ver[i] - R @ p_est[0]


def distancia_a_la_habitacion(p, dimensiones):
    """Distancia de cada punto a la superficie más cercana de la habitación, y cuál es.

    La habitación es la de vio/escena.py: x en [-lx/2, lx/2], y en [-ly/2, ly/2], z en [0, lz],
    con las caras numeradas igual (0 y 1: x; 2 y 3: y; 4: suelo; 5: techo).
    Un punto de dentro está a (x - x_min) de la cara 0, a (x_max - x) de la 1, etc.: vale la
    menor. Un punto de fuera se lleva primero al punto más cercano de la habitación
    (recortando sus coordenadas), y su distancia es lo que se ha movido.
    p: (N, 3) en el mundo del simulador. Devuelve: (distancia (N,) [m], cara (N,) de 0 a 5).
    """
    lx, ly, lz = dimensiones
    minimo = np.array([-lx / 2, -ly / 2, 0.0])
    maximo = np.array([lx / 2, ly / 2, lz])
    recortado = np.clip(p, minimo, maximo)
    por_fuera = np.linalg.norm(p - recortado, axis=1)    # 0 para los puntos de dentro
    a_cada_cara = np.column_stack([recortado[:, 0] - minimo[0], maximo[0] - recortado[:, 0],
                                   recortado[:, 1] - minimo[1], maximo[1] - recortado[:, 1],
                                   recortado[:, 2] - minimo[2], maximo[2] - recortado[:, 2]])
    cara = np.argmin(a_cada_cara, axis=1)
    return a_cada_cara.min(axis=1) + por_fuera, cara


def evaluar_nube(p, dimensiones, cfg_evaluacion):
    """Cuánto se parece la nube a la habitación de verdad.

    p: (N, 3) puntos de la nube, YA en el mundo del simulador.
    Devuelve: dict con
        n               puntos de la nube
        error_mediano   mediana de la distancia de cada punto a su superficie [m]
        error_p90       el 90 % de los puntos están a menos de esto [m]
        cerca           fracción de puntos a menos de cerca_m de una superficie
        cubierto        fracción de la superficie de la habitación (en celdas de celda_m) que
                        tiene algún punto de la nube cerca
        cubierto_por_cara   lo mismo, cara a cara (6 números)
    """
    if len(p) == 0:
        return {"n": 0, "error_mediano": float("nan"), "error_p90": float("nan"), "cerca": float("nan"),
                "cubierto": 0.0, "cubierto_por_cara": [0.0] * escena.N_CARAS}
    distancia, cara = distancia_a_la_habitacion(p, dimensiones)
    es_bueno = distancia < cfg_evaluacion["cerca_m"]
    celda = cfg_evaluacion["celda_m"]
    ocupadas, totales = [], []
    for n in range(escena.N_CARAS):
        ancho_m, alto_m = escena.tamano_de_cara(n, dimensiones)
        columnas, filas = int(np.ceil(ancho_m / celda)), int(np.ceil(alto_m / celda))
        en_cara = es_bueno & (cara == n)
        a, b = escena.coordenadas_en_cara(p[en_cara], n, dimensiones)
        c = np.clip(np.floor(a / celda).astype(int), 0, columnas - 1)
        f = np.clip(np.floor(b / celda).astype(int), 0, filas - 1)
        rejilla = np.zeros((filas, columnas), dtype=bool)
        rejilla[f, c] = True
        ocupadas.append(int(rejilla.sum()))
        totales.append(filas * columnas)
    return {"n": len(p), "error_mediano": float(np.median(distancia)),
            "error_p90": float(np.percentile(distancia, 90)), "cerca": float(es_bueno.mean()),
            "cubierto": sum(ocupadas) / sum(totales),
            "cubierto_por_cara": [o / t for o, t in zip(ocupadas, totales)]}
