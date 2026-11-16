"""
vio/planos.py — Etapa 27: los planos que hay en la nube densa.

Una habitación está hecha sobre todo de superficies planas: paredes, suelo, techo, mesas.
En la nube densa de la etapa 26 cada una es una lámina de puntos con algo de grosor (el
error de la profundidad). Aquí se busca, una a una, cada lámina, y se describe con cuatro
números: un plano es el conjunto de puntos p que cumplen
        n · p = d
donde n es un vector de largo 1 perpendicular al plano (su NORMAL) y d es la distancia del
plano al origen, medida a lo largo de n.

    1. Un plano por tres puntos, y el plano que mejor pasa entre muchos.
    2. RANSAC: el plano que más puntos de la nube tiene cerca.
    3. Se quitan sus puntos y se busca el siguiente, hasta que no quede ninguno grande.
    4. Los planos que son la misma superficie se juntan.
    5. De cada plano: de qué tipo es (suelo, techo, pared...), qué superficie ocupa.
    6. Las medidas de la habitación: separación entre paredes opuestas, altura, ángulos.
En el simulador, además, cada plano se compara con la cara de verdad de la habitación.

Mundo: el del VIO, con z hacia arriba (la gravedad es -z).
"""

import numpy as np

ARRIBA = np.array([0.0, 0.0, 1.0])


# =====================================================================
# Paso 27.1 — Un plano
# =====================================================================
def plano_por_tres_puntos(p0, p1, p2):
    """El plano que pasa por tres puntos. Devuelve: (n (3,), d) o None si están en línea.

    Los vectores p1 - p0 y p2 - p0 están dentro del plano; su producto vectorial es
    perpendicular a los dos, es decir, al plano. d sale de que p0 cumple n·p0 = d.
    """
    n = np.cross(p1 - p0, p2 - p0)
    largo = np.linalg.norm(n)
    if largo < 1e-12:
        return None
    n = n / largo
    return n, float(n @ p0)


def distancia_al_plano(p, n, d):
    """Distancia CON SIGNO de cada punto al plano: positiva del lado hacia el que apunta n.
    p: (N, 3). Devuelve: (N,) [m]."""
    return p @ n - d


def ajustar_plano(p):
    """El plano que mejor pasa entre muchos puntos (mínimos cuadrados).

    Pasa por el centro de los puntos (su media). Y su normal es la dirección en la que los
    puntos, vistos desde ese centro, menos se extienden: una lámina se extiende mucho a lo
    largo y a lo ancho, y casi nada a lo grueso. Esas tres direcciones las da la
    descomposición en valores singulares (SVD) de los puntos centrados: las filas de Vt,
    ordenadas de más a menos extensión. La normal es la última.
    p: (N, 3), N >= 3. Devuelve: (n (3,), d, ejes (2, 3)): los ejes son las otras dos
    direcciones, dentro del plano.
    """
    centro = p.mean(axis=0)
    _, _, Vt = np.linalg.svd(p - centro, full_matrices=False)
    n = Vt[2]
    return n, float(n @ centro), Vt[:2]


# =====================================================================
# Paso 27.2 — RANSAC: el plano con más puntos cerca
# =====================================================================
def mejor_plano(p, cfg_ransac, rng):
    """El plano que más puntos de p tiene a menos de umbral_m.

    Se cogen tres puntos al azar, se saca su plano y se cuentan los puntos que están cerca.
    Se repite `iteraciones` veces y se queda el más votado. Después se reajusta con todos
    sus puntos (mínimos cuadrados) y se vuelven a contar, `refinados` veces: el plano de
    tres puntos concretos es solo una primera aproximación.
    p: (N, 3). Devuelve: (n, d, encajan (N,) bool), o None si no hay tres puntos.
    """
    if len(p) < 3:
        return None
    mejor, mejor_votos = None, -1
    for _ in range(cfg_ransac["iteraciones"]):
        i, j, k = rng.choice(len(p), 3, replace=False)
        plano = plano_por_tres_puntos(p[i], p[j], p[k])
        if plano is None:
            continue
        votos = int((np.abs(distancia_al_plano(p, *plano)) < cfg_ransac["umbral_m"]).sum())
        if votos > mejor_votos:
            mejor, mejor_votos = plano, votos
    if mejor is None:
        return None
    n, d = mejor
    encajan = np.abs(distancia_al_plano(p, n, d)) < cfg_ransac["umbral_m"]
    for _ in range(cfg_ransac["refinados"]):
        if encajan.sum() < 3:
            break
        n, d, _ = ajustar_plano(p[encajan])
        encajan = np.abs(distancia_al_plano(p, n, d)) < cfg_ransac["umbral_m"]
    return n, d, encajan


# =====================================================================
# Paso 27.3 — Un plano tras otro
# =====================================================================
def extraer_planos(p, cfg_ransac, rng):
    """Todos los planos grandes de la nube.

    Se busca el mejor plano, se apartan sus puntos y se repite con los que quedan, hasta
    que el mejor plano que queda tiene menos de puntos_min puntos o se llega a planos_max.
    p: (N, 3). Devuelve: lista de arrays de índices (los puntos de p de cada plano), en el
    orden en que han salido: los grandes suelen salir antes, pero no siempre.
    """
    quedan = np.arange(len(p))                           # índices de los puntos aún sin plano
    planos = []
    while len(planos) < cfg_ransac["planos_max"] and len(quedan) >= cfg_ransac["puntos_min"]:
        resultado = mejor_plano(p[quedan], cfg_ransac, rng)
        if resultado is None or resultado[2].sum() < cfg_ransac["puntos_min"]:
            break
        encajan = resultado[2]
        planos.append(quedan[encajan])
        quedan = quedan[~encajan]
    return planos


# =====================================================================
# Paso 27.4 — Juntar los planos que son la misma superficie
# =====================================================================
def orientar_normal(n, d, centro, punto_dentro):
    """Devuelve (n, d) con la normal apuntando hacia punto_dentro (dónde han estado las
    cámaras): n y -n describen el mismo plano, y así todas las normales miran hacia dentro
    de la habitación. centro: un punto del plano."""
    if n @ (punto_dentro - centro) < 0:
        return -n, -d
    return n, d


def son_la_misma_superficie(p_a, p_b, punto_dentro, cfg_fusion):
    """¿Dos grupos de puntos son la misma superficie plana?

    Sí, si sus normales (las dos mirando hacia dentro) forman menos de angulo_grados y el
    centro de cada grupo está a menos de distancia_m del plano del otro.
    Pasa cuando la lámina de puntos es más gruesa que el umbral del RANSAC: el primer plano
    se lleva el centro de la lámina y lo que sobra a los lados sale como planos aparte.
    """
    centro_a, centro_b = p_a.mean(axis=0), p_b.mean(axis=0)
    n_a, d_a, _ = ajustar_plano(p_a)
    n_b, d_b, _ = ajustar_plano(p_b)
    n_a, d_a = orientar_normal(n_a, d_a, centro_a, punto_dentro)
    n_b, d_b = orientar_normal(n_b, d_b, centro_b, punto_dentro)
    angulo = np.degrees(np.arccos(np.clip(n_a @ n_b, -1.0, 1.0)))
    separacion = max(abs(n_a @ centro_b - d_a), abs(n_b @ centro_a - d_b))
    return bool(angulo < cfg_fusion["angulo_grados"] and separacion < cfg_fusion["distancia_m"])


def fundir_planos(p, grupos, punto_dentro, cfg_fusion):
    """Junta los grupos que son la misma superficie, hasta que no quede ninguna pareja.

    grupos: lista de arrays de índices (de extraer_planos). Devuelve: otra lista, de mayor a menor.
    """
    grupos = [g for g in grupos]
    hubo_cambio = True
    while hubo_cambio:
        hubo_cambio = False
        for i in range(len(grupos)):
            for j in range(i + 1, len(grupos)):
                if son_la_misma_superficie(p[grupos[i]], p[grupos[j]], punto_dentro, cfg_fusion):
                    grupos[i] = np.concatenate([grupos[i], grupos[j]])
                    del grupos[j]
                    hubo_cambio = True
                    break
            if hubo_cambio:
                break
    return sorted(grupos, key=len, reverse=True)


# =====================================================================
# Paso 27.5 — Describir cada plano
# =====================================================================
def tipo_de_plano(n, tolerancia_grados):
    """"horizontal_arriba" (la normal mira hacia arriba: un suelo, una mesa),
    "horizontal_abajo" (mira hacia abajo: un techo), "pared" (vertical) o "inclinado".

    n: normal de largo 1, mirando hacia dentro de la habitación. El ángulo de la normal con
    la vertical es arccos(n_z): cerca de 0° u 180° el plano es horizontal; cerca de 90°, vertical.
    """
    angulo = np.degrees(np.arccos(np.clip(n @ ARRIBA, -1.0, 1.0)))
    if angulo < tolerancia_grados:
        return "horizontal_arriba"
    if angulo > 180.0 - tolerancia_grados:
        return "horizontal_abajo"
    if abs(angulo - 90.0) < tolerancia_grados:
        return "pared"
    return "inclinado"


def describir_plano(p_plano, punto_dentro, cfg):
    """Todo lo que se dice de un plano a partir de sus puntos.

    p_plano: (N, 3). cfg: config/planos.yaml.
    Devuelve: dict con
        n, d        el plano n·p = d, con n mirando hacia dentro
        tipo        de tipo_de_plano (describir_planos lo cambia luego a "suelo" o "techo")
        centro      (3,) la media de sus puntos
        ejes        (2, 3) dos direcciones dentro del plano, perpendiculares. En una pared,
                    la primera es horizontal y la segunda va hacia arriba
        extremos    (a_min, a_max, b_min, b_max) [m]: hasta dónde llegan los puntos a lo largo
                    de cada eje, medidos desde el centro (entre los percentiles 1 y 99)
        superficie  [m²] ocupada por los puntos (celdas de celda_m con algún punto)
        grosor      [m] desviación típica de la distancia de los puntos al plano
        puntos      N
    """
    centro = p_plano.mean(axis=0)
    n, d, ejes = ajustar_plano(p_plano)
    n, d = orientar_normal(n, d, centro, punto_dentro)
    tipo = tipo_de_plano(n, cfg["tipos"]["tolerancia_grados"])
    if tipo == "pared":
        horizontal = np.cross(ARRIBA, n)                 # perpendicular a la vertical y a la normal
        horizontal = horizontal / np.linalg.norm(horizontal)
        ejes = np.array([horizontal, np.cross(n, horizontal)])
    a = (p_plano - centro) @ ejes[0]
    b = (p_plano - centro) @ ejes[1]
    celda = cfg["medidas"]["celda_m"]
    celdas = np.unique(np.column_stack([np.floor(a / celda), np.floor(b / celda)]), axis=0)
    return {"n": n, "d": d, "tipo": tipo, "centro": centro, "ejes": ejes,
            "extremos": (float(np.percentile(a, 1)), float(np.percentile(a, 99)),
                         float(np.percentile(b, 1)), float(np.percentile(b, 99))),
            "superficie": len(celdas) * celda ** 2,
            "grosor": float(distancia_al_plano(p_plano, n, d).std()), "puntos": len(p_plano)}


def describir_planos(p, grupos, punto_dentro, cfg):
    """La descripción de cada plano, con el suelo y el techo señalados.

    De los horizontales que miran hacia arriba, el más bajo es el "suelo" (los demás se
    quedan en "horizontal_arriba": mesas, estantes). De los que miran hacia abajo, el más
    alto es el "techo". Devuelve: lista de dict de describir_plano, con "indices" añadido.
    """
    planos = []
    for indices in grupos:
        plano = describir_plano(p[indices], punto_dentro, cfg)
        plano["indices"] = indices
        planos.append(plano)
    hacia_arriba = [q for q in planos if q["tipo"] == "horizontal_arriba"]
    if hacia_arriba:
        min(hacia_arriba, key=lambda q: q["centro"][2])["tipo"] = "suelo"
    hacia_abajo = [q for q in planos if q["tipo"] == "horizontal_abajo"]
    if hacia_abajo:
        max(hacia_abajo, key=lambda q: q["centro"][2])["tipo"] = "techo"
    return planos


def buscar_planos(p, punto_dentro, cfg, rng):
    """Toda la etapa: de la nube a la lista de planos descritos.

    p: (N, 3) la nube densa. punto_dentro: (3,) un punto del interior de la habitación (la
    media de las posiciones de las cámaras). Devuelve: lista de dict (ver describir_plano).
    """
    grupos = extraer_planos(p, cfg["ransac"], rng)
    grupos = fundir_planos(p, grupos, punto_dentro, cfg["fusion"])
    return describir_planos(p, grupos, punto_dentro, cfg)


# =====================================================================
# Paso 27.6 — Las medidas de la habitación
# =====================================================================
def angulo_entre_normales(n_a, n_b):
    """Ángulo entre dos normales [grados], de 0 a 180."""
    return float(np.degrees(np.arccos(np.clip(n_a @ n_b, -1.0, 1.0))))


def separacion_entre_planos(plano_a, plano_b):
    """Distancia entre dos planos paralelos [m]: la media de la distancia del centro de cada
    uno al plano del otro (si no son paralelos del todo, las dos no coinciden)."""
    a_b = abs(plano_a["n"] @ plano_b["centro"] - plano_a["d"])
    b_a = abs(plano_b["n"] @ plano_a["centro"] - plano_b["d"])
    return float((a_b + b_a) / 2)


def medidas_de_habitacion(planos, tolerancia_grados):
    """Lo que se puede medir de la habitación con los planos encontrados.

    Devuelve: dict con
        altura      [m] del suelo al techo, o None si falta alguno
        opuestas    lista de (i, j, separación [m]): parejas de paredes enfrentadas (sus
                    normales, que miran hacia dentro, apuntan una contra otra)
        esquinas    lista de (i, j, ángulo [°]): parejas de paredes que no son paralelas, con
                    el ángulo que forman (en una habitación rectangular, 90°)
    i y j son posiciones en la lista `planos`.
    """
    suelo = [q for q in planos if q["tipo"] == "suelo"]
    techo = [q for q in planos if q["tipo"] == "techo"]
    altura = separacion_entre_planos(suelo[0], techo[0]) if suelo and techo else None
    paredes = [i for i, q in enumerate(planos) if q["tipo"] == "pared"]
    opuestas, esquinas = [], []
    for posicion, i in enumerate(paredes):
        for j in paredes[posicion + 1:]:
            angulo = angulo_entre_normales(planos[i]["n"], planos[j]["n"])
            if angulo > 180.0 - tolerancia_grados:
                opuestas.append((i, j, separacion_entre_planos(planos[i], planos[j])))
            elif angulo > tolerancia_grados:
                esquinas.append((i, j, angulo))
    return {"altura": altura, "opuestas": opuestas, "esquinas": esquinas}


# =====================================================================
# Paso 27.7 — Comparar con la habitación de verdad (solo en el simulador)
# =====================================================================
def caras_de_la_habitacion(dimensiones):
    """Las 6 caras de la habitación de vio/escena.py como planos n·p = d, con la normal
    hacia dentro. Devuelve: lista de (n (3,), d), en el orden de las caras (0 a 5)."""
    lx, ly, lz = dimensiones
    return [(np.array([1.0, 0.0, 0.0]), -lx / 2), (np.array([-1.0, 0.0, 0.0]), -lx / 2),
            (np.array([0.0, 1.0, 0.0]), -ly / 2), (np.array([0.0, -1.0, 0.0]), -ly / 2),
            (np.array([0.0, 0.0, 1.0]), 0.0), (np.array([0.0, 0.0, -1.0]), -lz)]


def comparar_con_la_verdad(planos, R, t, dimensiones):
    """Cada plano encontrado, frente a la cara de la habitación a la que más se parece.

    planos: en el mundo del VIO. R, t: de densificar.transformacion_a_la_verdad
    (p_verdad = R·p_vio + t). Un plano n·p = d pasa a ser (R·n)·p' = d + (R·n)·t.
    La cara que le corresponde es la de normal más parecida.
    Devuelve: lista de dict, uno por plano, con cara (0 a 5), error_angulo [°] entre las dos
    normales y error_m [m]: distancia del centro del plano a la cara.
    """
    caras = caras_de_la_habitacion(dimensiones)
    comparaciones = []
    for plano in planos:
        n = R @ plano["n"]
        centro = R @ plano["centro"] + t
        angulos = [angulo_entre_normales(n, n_cara) for n_cara, _ in caras]
        cara = int(np.argmin(angulos))
        n_cara, d_cara = caras[cara]
        comparaciones.append({"cara": cara, "error_angulo": angulos[cara],
                              "error_m": float(abs(n_cara @ centro - d_cara))})
    return comparaciones
