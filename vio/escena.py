"""
vio/escena.py — Etapa 20: una habitación con textura, vista por la cámara.

Hasta ahora el simulador dibujaba cada punto como una mancha sobre fondo negro (etapa 13).
A FAST y a Lucas-Kanade les basta, pero una red como SuperPoint está entrenada con fotos:
necesita una imagen "de verdad", con superficies que tengan dibujo. Aquí se pinta un dibujo
al azar en las 4 paredes, el suelo y el techo, y se calcula qué ve la cámara desde una pose.

Cómo se dibuja una imagen (trazado de rayos, del píxel al mundo):
    1. De cada píxel sale un rayo: la recta de todos los puntos 3D que caen en ese píxel.
    2. Se busca dónde choca el rayo con la habitación (qué cara y a qué distancia).
    3. El brillo del píxel es el del dibujo de esa cara en el punto del choque.
Es el camino inverso de la proyección de la etapa 2 (del mundo al píxel).

La habitación es la misma del simulador: x en [-lx/2, lx/2], y en [-ly/2, ly/2], z en [0, lz].
Caras, numeradas:  0: x = -lx/2    1: x = +lx/2    2: y = -ly/2    3: y = +ly/2
                   4: z = 0 (suelo)                5: z = lz (techo)
"""

import cv2
import numpy as np

from vio import modelo_camara as mc

N_CARAS = 6


# =====================================================================
# Paso 20.1 — El dibujo de cada cara
# =====================================================================
def tamano_de_cara(cara, dimensiones):
    """(ancho, alto) en metros de una cara. Sus dos ejes (a, b) son:

    caras 0 y 1 (x constante):  a = y + ly/2,  b = z
    caras 2 y 3 (y constante):  a = x + lx/2,  b = z
    caras 4 y 5 (z constante):  a = x + lx/2,  b = y + ly/2
    Así a y b van siempre de 0 al tamaño de la cara.
    """
    lx, ly, lz = dimensiones
    if cara in (0, 1):
        return ly, lz
    if cara in (2, 3):
        return lx, lz
    return lx, ly


def textura_aleatoria(ancho_m, alto_m, cfg, rng):
    """Un dibujo al azar para una superficie de ancho_m x alto_m.

    cfg: sección "textura" de config/simulador.yaml.
    Tiene dos ingredientes:
      - Manchas suaves de varios tamaños: como una pared con sombras y desconchones. Dan
        variación de brillo en todas partes, pero pocas esquinas.
      - Rectángulos de gris uniforme (como cuadros o carteles) de tamaños al azar: cada uno
        aporta 4 esquinas nítidas, y al solaparse, muchas más. Son lo que detectan FAST y
        SuperPoint.
    Al final se suaviza un poco: un borde perfectamente afilado no existe en una foto real,
    y al verlo de lejos produciría parpadeos (aliasing).
    Devuelve: array (alto_px, ancho_px) float32, con valores entre 0 y 255.
    """
    px_m = cfg["px_por_m"]
    ancho, alto = int(round(ancho_m * px_m)), int(round(alto_m * px_m))

    # 1. Manchas suaves: una cuadrícula de valores al azar, ampliada hasta el tamaño final.
    #    Cada tamaño de celda da manchas de ese tamaño; se suman con pesos iguales.
    dibujo = np.zeros((alto, ancho), dtype=np.float32)
    for celda_m in cfg["manchas_m"]:
        columnas = max(int(np.ceil(ancho_m / celda_m)), 2)
        filas = max(int(np.ceil(alto_m / celda_m)), 2)
        valores = rng.uniform(-1.0, 1.0, (filas, columnas)).astype(np.float32)
        dibujo += cv2.resize(valores, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    dibujo = cfg["gris_medio"] + cfg["contraste_manchas"] * dibujo / len(cfg["manchas_m"])

    # 2. Rectángulos: posición, tamaño y gris al azar
    n_rectangulos = int(round(cfg["rectangulos_por_m2"] * ancho_m * alto_m))
    lado_min, lado_max = cfg["lado_rectangulo_m"]
    for _ in range(n_rectangulos):
        w = int(rng.uniform(lado_min, lado_max) * px_m)
        h = int(rng.uniform(lado_min, lado_max) * px_m)
        c0 = int(rng.uniform(0, ancho - 1))
        f0 = int(rng.uniform(0, alto - 1))
        dibujo[f0:f0 + h, c0:c0 + w] = rng.uniform(cfg["gris_min"], cfg["gris_max"])

    # 3. Suavizado (sigma en píxeles del dibujo)
    dibujo = cv2.GaussianBlur(dibujo, (0, 0), cfg["suavizado_px"])
    return np.clip(dibujo, 0.0, 255.0)


def crear_habitacion(dimensiones, cfg, rng):
    """La habitación con un dibujo distinto en cada cara.

    Devuelve: dict con dimensiones (lx, ly, lz), px_por_m y texturas (lista de 6 arrays).
    """
    texturas = []
    for cara in range(N_CARAS):
        ancho_m, alto_m = tamano_de_cara(cara, dimensiones)
        texturas.append(textura_aleatoria(ancho_m, alto_m, cfg, rng))
    return {"dimensiones": tuple(dimensiones), "px_por_m": cfg["px_por_m"], "texturas": texturas}


# =====================================================================
# Paso 20.2 — Del píxel al mundo: rayos y choque con la habitación
# =====================================================================
def rayos_camara(K, dist, tamano):
    """La dirección del rayo de cada píxel, en ejes de la cámara.

    El píxel (u, v) corresponde a las coordenadas normalizadas (x, y) (etapa 2: quitar los
    intrínsecos y la distorsión). Todos los puntos p_c = z·(x, y, 1), con z > 0, caen en ese
    píxel: esa es la recta. Su dirección es (x, y, 1), y con ella el parámetro del rayo es
    directamente la profundidad z.
    Solo depende de la cámara, no de la pose: se calcula una vez y vale para todas las imágenes.
    tamano: (ancho, alto). dist: (k1, k2, p1, p2, k3) o None (sin distorsión).
    Devuelve: array (alto, ancho, 3).
    """
    ancho, alto = tamano
    u, v = np.meshgrid(np.arange(ancho, dtype=float), np.arange(alto, dtype=float))
    uv = np.column_stack([u.ravel(), v.ravel()])
    if dist is None:
        xy = np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1]])
    else:
        xy = mc.normalizar(uv, K, dist)
    rayos = np.column_stack([xy, np.ones(len(xy))])
    return rayos.reshape(alto, ancho, 3)


def chocar_con_habitacion(origen, direcciones, dimensiones):
    """Dónde choca cada rayo con las caras de la habitación, visto desde dentro.

    origen: (3,) punto del que salen todos los rayos, DENTRO de la habitación.
    direcciones: (N, 3). Los puntos de un rayo son origen + s·dirección, con s >= 0.

    En cada eje el rayo solo puede salir por una de las dos caras: la del lado hacia el que
    avanza. Por ejemplo, si la dirección tiene x > 0, por la cara x = +lx/2, y eso ocurre en
        origen_x + s·dirección_x = lx/2   ->   s = (lx/2 - origen_x) / dirección_x
    Se calcula s para los tres ejes y vale el MENOR: la primera cara que se alcanza.
    Devuelve: (s (N,), cara (N,) de 0 a 5).
    """
    lx, ly, lz = dimensiones
    minimo = np.array([-lx / 2, -ly / 2, 0.0])
    maximo = np.array([lx / 2, ly / 2, lz])
    if np.any(origen <= minimo) or np.any(origen >= maximo):
        raise ValueError(f"la cámara está fuera de la habitación: {origen}")

    s_ejes = np.full(direcciones.shape, np.inf)
    for eje in range(3):
        d = direcciones[:, eje]
        hacia_max = d > 0
        hacia_min = d < 0
        s_ejes[hacia_max, eje] = (maximo[eje] - origen[eje]) / d[hacia_max]
        s_ejes[hacia_min, eje] = (minimo[eje] - origen[eje]) / d[hacia_min]
        # d == 0: el rayo es paralelo a esas dos caras y no las toca (se queda en infinito)
    eje_choque = np.argmin(s_ejes, axis=1)
    s = s_ejes[np.arange(len(s_ejes)), eje_choque]
    hacia_max = direcciones[np.arange(len(s_ejes)), eje_choque] > 0
    cara = 2 * eje_choque + hacia_max.astype(int)        # 2·eje, +1 si es la cara del máximo
    return s, cara


def coordenadas_en_cara(p_w, cara, dimensiones):
    """Las dos coordenadas (a, b) [m] de un punto dentro de su cara (ver tamano_de_cara).

    p_w: (N, 3) puntos sobre la cara número `cara` (un entero). Devuelve: (a (N,), b (N,)).
    """
    lx, ly, _ = dimensiones
    if cara in (0, 1):
        return p_w[:, 1] + ly / 2, p_w[:, 2]
    if cara in (2, 3):
        return p_w[:, 0] + lx / 2, p_w[:, 2]
    return p_w[:, 0] + lx / 2, p_w[:, 1] + ly / 2


def muestrear_bilineal(textura, columna, fila):
    """Valor de `textura` en posiciones con decimales, por interpolación bilineal.

    Una posición (columna, fila) cae entre 4 píxeles del dibujo. Su valor es la media de los
    4, cada uno con un peso mayor cuanto más cerca está: si la parte decimal es (dc, df),
        valor = (1-dc)(1-df)·arriba_izq + dc(1-df)·arriba_der + (1-dc)df·abajo_izq + dc·df·abajo_der
    Sin esto (cogiendo el píxel más cercano) el dibujo avanzaría "a saltos" al moverse la
    cámara, y Lucas-Kanade no podría medir desplazamientos de décimas de píxel.
    columna, fila: (N,). Devuelve: (N,).
    """
    alto, ancho = textura.shape
    columna = np.clip(columna, 0.0, ancho - 1.001)        # dentro del dibujo (el borde, repetido)
    fila = np.clip(fila, 0.0, alto - 1.001)
    c0 = np.floor(columna).astype(int)
    f0 = np.floor(fila).astype(int)
    dc = columna - c0
    df = fila - f0
    return ((1 - dc) * (1 - df) * textura[f0, c0] + dc * (1 - df) * textura[f0, c0 + 1]
            + (1 - dc) * df * textura[f0 + 1, c0] + dc * df * textura[f0 + 1, c0 + 1])


# =====================================================================
# Paso 20.3 — La imagen desde una pose
# =====================================================================
def renderizar_habitacion(habitacion, R_c_w, t_c_w, rayos):
    """Lo que ve la cámara desde una pose.

    habitacion: de crear_habitacion. R_c_w, t_c_w: pose (mundo -> cámara, p_c = R_c_w·p_w + t_c_w).
    rayos: de rayos_camara.
    Devuelve: (imagen (alto, ancho) float de 0 a 255, profundidad (alto, ancho) [m]):
    la profundidad es la z, en ejes de la cámara, del punto que se ve en cada píxel. No la
    tendría una cámara real: es la "verdad" con la que comparar más adelante.
    """
    alto, ancho, _ = rayos.shape
    dimensiones = habitacion["dimensiones"]
    px_m = habitacion["px_por_m"]

    # 1. Los rayos, en el mundo. De p_c = R_c_w·p_w + t_c_w se despeja p_w = R_c_wᵀ·(p_c - t_c_w):
    #    el centro de la cámara (p_c = 0) está en -R_c_wᵀ·t_c_w y las direcciones giran con R_c_wᵀ
    origen = -R_c_w.T @ t_c_w
    direcciones = rayos.reshape(-1, 3) @ R_c_w                # cada fila: R_c_wᵀ · d_c

    # 2. Dónde choca cada rayo
    s, cara = chocar_con_habitacion(origen, direcciones, dimensiones)
    p_w = origen + s[:, None] * direcciones

    # 3. El brillo del dibujo en ese punto, cara por cara
    imagen = np.zeros(alto * ancho)
    for n in range(N_CARAS):
        en_cara = cara == n
        if not np.any(en_cara):
            continue
        a, b = coordenadas_en_cara(p_w[en_cara], n, dimensiones)
        imagen[en_cara] = muestrear_bilineal(habitacion["texturas"][n], a * px_m, b * px_m)

    # 4. Como la dirección es (x, y, 1), el parámetro s del rayo es la profundidad z
    return imagen.reshape(alto, ancho), s.reshape(alto, ancho)


def acabar_imagen(imagen, cfg, rng):
    """De la imagen ideal a la que daría un sensor: un poco desenfocada, con ruido y en 8 bits.

    cfg: sección "textura". El desenfoque imita a la lente (ningún objetivo enfoca un punto en
    un solo píxel) y el ruido, al sensor: sin ellos las imágenes serían demasiado fáciles.
    Devuelve: array (alto, ancho) uint8.
    """
    if cfg["desenfoque_px"] > 0:
        imagen = cv2.GaussianBlur(imagen, (0, 0), cfg["desenfoque_px"])
    if cfg["ruido_gris"] > 0:
        imagen = imagen + rng.normal(0.0, cfg["ruido_gris"], imagen.shape)
    return np.clip(np.round(imagen), 0, 255).astype(np.uint8)