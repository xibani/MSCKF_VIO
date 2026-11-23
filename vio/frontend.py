"""
vio/frontend.py — Detección y seguimiento de puntos de imagen en imagen.

Por cada imagen nueva:
    1. Seguir los puntos de la imagen anterior (Lucas-Kanade piramidal).
       Etapa 31: si el pipeline da el giro de la cámara medido por el giroscopio, cada punto
       se busca a partir de donde lo habría llevado ese giro, no desde donde estaba.
    2. Descartar los seguimientos dudosos (ida y vuelta, bordes).
    3. Descartar los que no encajan con un movimiento rígido de la cámara (RANSAC).
    4. Las pistas de los puntos perdidos se dan por terminadas y se entregan.
    5. Detectar puntos nuevos (FAST) en las zonas con pocos puntos.
"""

from dataclasses import dataclass, field

import os

import cv2
import numpy as np

from vio import modelo_camara as mc


# =====================================================================
# Paso 8.1 — Detección repartida por la imagen
# =====================================================================
def celda_de(u, v, ancho, alto, celdas_x, celdas_y):
    """Celda (fila, columna) de la rejilla en la que cae el píxel (u, v)."""
    columna = min(int(u / ancho * celdas_x), celdas_x - 1)
    fila = min(int(v / alto * celdas_y), celdas_y - 1)
    return fila, columna


def detectar(gris, puntos_existentes, cfg_det, margen_px):
    """Busca esquinas nuevas, repartidas y lejos de los puntos que ya se siguen.

    gris: imagen (alto, ancho) uint8.
    puntos_existentes: array (N, 2) [px] de los puntos que ya se siguen.
    cfg_det: sección "deteccion" de config/frontend.yaml.
    margen_px: franja junto al borde donde no se detecta (la misma que descarta el
        seguimiento: un punto ahí se perdería en la imagen siguiente).
    Devuelve: array (M, 2) float32 [px] con los puntos nuevos.
    """
    alto, ancho = gris.shape
    cx, cy = cfg_det["celdas_x"], cfg_det["celdas_y"]
    d_min = cfg_det["distancia_min_px"]

    # 1. Máscara: no buscar junto al borde ni alrededor de los puntos que ya existen
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    mascara[margen_px:alto - margen_px, margen_px:ancho - margen_px] = 255
    for u, v in puntos_existentes:
        cv2.circle(mascara, (int(round(u)), int(round(v))), d_min, 0, -1)   # -1 = círculo relleno

    # 2. Detector FAST
    fast = cv2.FastFeatureDetector_create(threshold=cfg_det["fast_umbral"], nonmaxSuppression=True)
    candidatos = fast.detect(gris, mascara)

    # 3. Cuántos puntos tiene ya cada celda
    ocupacion = np.zeros((cy, cx), dtype=int)
    for u, v in puntos_existentes:
        ocupacion[celda_de(u, v, ancho, alto, cx, cy)] += 1

    # 4. Elegir de mejor a peor respuesta, respetando el máximo por celda y el total
    candidatos = sorted(candidatos, key=lambda k: k.response, reverse=True)
    hueco = cfg_det["max_puntos"] - len(puntos_existentes)
    nuevos = []
    for k in candidatos:
        if len(nuevos) >= hueco:
            break
        u, v = k.pt
        celda = celda_de(u, v, ancho, alto, cx, cy)
        if ocupacion[celda] >= cfg_det["max_por_celda"]:
            continue
        # Lejos también de los nuevos ya elegidos (la máscara solo tiene los antiguos)
        if any((u - un) ** 2 + (v - vn) ** 2 < d_min ** 2 for un, vn in nuevos):
            continue
        nuevos.append((u, v))
        ocupacion[celda] += 1

    return np.array(nuevos, dtype=np.float32).reshape(-1, 2)


# =====================================================================
# Etapa 31 — Predicción de la posición de cada punto con el giro del giroscopio
# =====================================================================
def predecir_con_giro(puntos, R_act_ant, K, dist):
    """Dónde estaría cada punto en la imagen actual si la cámara solo hubiera girado.

    Un punto de la imagen es una DIRECCIÓN desde la cámara: d = (x, y, 1), con (x, y) sus
    coordenadas normalizadas sin distorsión. Si la cámara gira R_act_ant (d_act = R_act_ant·d_ant,
    vio/pipeline.giro_camara) y no se desplaza, esa misma dirección del mundo se ve en
    d_act, y su píxel es la proyección de d_act con K y la distorsión. Esto vale para cualquier
    profundidad: un giro puro mueve igual los puntos cercanos y los lejanos.
    Lo que no explica el giro (la traslación, que sí depende de la profundidad, y el error del
    giroscopio) lo corrige después Lucas-Kanade, partiendo de esta predicción.

    puntos: (N, 2) [px] en la imagen anterior. R_act_ant: (3, 3). K, dist: intrínsecos.
    Devuelve: (prediccion (N, 2) float32 [px], delante (N,) bool). delante = False si con ese
    giro la dirección queda detrás de la cámara (no se puede proyectar); su predicción se deja
    en la posición de partida.
    """
    n = len(puntos)
    prediccion = np.asarray(puntos, dtype=np.float32).reshape(-1, 2).copy()
    delante = np.zeros(n, dtype=bool)
    if n == 0:
        return prediccion, delante
    # 1. Píxeles -> direcciones (x, y, 1) en la cámara anterior
    xy = mc.normalizar(puntos, K, dist)
    d_ant = np.column_stack([xy, np.ones(n)])
    # 2. La misma dirección vista desde la cámara actual
    d_act = d_ant @ R_act_ant.T                       # cada fila: R_act_ant · d_ant
    delante = d_act[:, 2] > 0
    if not delante.any():
        return prediccion, delante
    # 3. Direcciones -> píxeles, con la distorsión (cámara en el origen y sin girar: ya está en sus ejes)
    proyectados, _ = cv2.projectPoints(d_act[delante].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                       K, np.asarray(dist, dtype=np.float64))
    prediccion[delante] = proyectados.reshape(-1, 2)
    return prediccion, delante


def sanear_prediccion(prediccion, puntos, forma, R_act_ant):
    """Predicción por giro utilizable por Lucas-Kanade: finita y dentro de la imagen (con un margen
    de media imagen). Las que no lo son vuelven a la posición de partida (como sin giro), con un
    aviso. Con un giro grande entre dos imágenes (salto del filtro, hueco de frames), una dirección
    casi perpendicular al eje óptico se proyecta, con la distorsión, a coordenadas enormes.
    """
    alto, ancho = forma
    p = np.asarray(prediccion, dtype=np.float32).reshape(-1, 2).copy()
    buenas = (np.all(np.isfinite(p), axis=1)
              & (p[:, 0] > -0.5 * ancho) & (p[:, 0] < 1.5 * ancho)
              & (p[:, 1] > -0.5 * alto) & (p[:, 1] < 1.5 * alto))
    if not buenas.all():
        angulo = np.degrees(np.arccos(np.clip((np.trace(R_act_ant) - 1.0) / 2.0, -1.0, 1.0)))
        print(f"AVISO frontend: {int((~buenas).sum())} de {len(p)} predicciones por giro inservibles "
              f"(giro entre imágenes {angulo:.1f}°): esas parten de su posición anterior", flush=True)
        p[~buenas] = np.asarray(puntos, dtype=np.float32).reshape(-1, 2)[~buenas]
    return p


def volcar_entradas_lk(gris_a, gris_b, p_a, prediccion, R):
    """Diagnóstico: si la variable de entorno VIO_VOLCAR_LK tiene una ruta, guarda ahí (y la
    sobrescribe en cada llamada) todo lo que recibe calcOpticalFlowPyrLK. Si el programa se queda
    colgado dentro, el archivo contiene las entradas exactas de la llamada que se atascó."""
    ruta = os.environ.get("VIO_VOLCAR_LK")
    if ruta:
        np.savez_compressed(ruta, gris_a=gris_a, gris_b=gris_b, p_a=np.asarray(p_a),
                            prediccion=np.asarray(prediccion), R=R)


# =====================================================================
# Paso 8.2 — Seguimiento con Lucas-Kanade y comprobación de ida y vuelta
# =====================================================================
def seguir(gris_ant, gris_act, puntos, cfg_seg, giro=None):
    """Sigue los puntos de la imagen anterior a la actual.

    puntos: array (N, 2) [px] en la imagen anterior.
    giro: None, o (R_act_ant, K, dist) para empezar a buscar donde lo lleva el giro (etapa 31).
        La vuelta parte, igual, de donde llevaría el giro inverso (R_act_antᵀ) a lo encontrado:
        así la comprobación ida-vuelta sigue siendo independiente (no se le da la respuesta).
    Devuelve: (puntos_act (N, 2), validos (N,) bool, error_ida_vuelta (N,) [px]).
    """
    n = len(puntos)
    if n == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, bool), np.zeros(0)

    parametros = dict(
        winSize=(cfg_seg["ventana_px"], cfg_seg["ventana_px"]),
        maxLevel=cfg_seg["niveles_piramide"],
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                  cfg_seg["iteraciones"], cfg_seg["epsilon"]))
    p0 = puntos.reshape(-1, 1, 2).astype(np.float32)

    # 1. Ida: de la imagen anterior a la actual
    if giro is None:
        p1, estado_ida, _ = cv2.calcOpticalFlowPyrLK(gris_ant, gris_act, p0, None, **parametros)
    else:
        R_act_ant, K, dist = giro
        prediccion, _ = predecir_con_giro(puntos, R_act_ant, K, dist)
        prediccion = sanear_prediccion(prediccion, puntos, gris_act.shape, R_act_ant)
        volcar_entradas_lk(gris_ant, gris_act, p0, prediccion, R_act_ant)
        # OPTFLOW_USE_INITIAL_FLOW: el cuarto argumento ya no es "nada" sino el punto de partida
        p1, estado_ida, _ = cv2.calcOpticalFlowPyrLK(gris_ant, gris_act, p0, prediccion.reshape(-1, 1, 2).copy(),
                                                     flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **parametros)

    # 2. Vuelta: de la actual a la anterior, partiendo de lo encontrado (o de donde lo
    #    devolvería el giro inverso)
    if giro is None:
        p0_vuelta, estado_vuelta, _ = cv2.calcOpticalFlowPyrLK(gris_act, gris_ant, p1, None, **parametros)
    else:
        prediccion_vuelta, _ = predecir_con_giro(p1.reshape(-1, 2), R_act_ant.T, K, dist)
        prediccion_vuelta = sanear_prediccion(prediccion_vuelta, p1.reshape(-1, 2), gris_ant.shape, R_act_ant.T)
        volcar_entradas_lk(gris_act, gris_ant, p1, prediccion_vuelta, R_act_ant.T)
        p0_vuelta, estado_vuelta, _ = cv2.calcOpticalFlowPyrLK(
            gris_act, gris_ant, p1, prediccion_vuelta.reshape(-1, 1, 2).copy(),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **parametros)

    p1 = p1.reshape(-1, 2)
    error_ida_vuelta = np.linalg.norm(p0_vuelta.reshape(-1, 2) - puntos, axis=1)

    # 3. Válidos: los dos sentidos encontrados, vuelta al mismo sitio y dentro de la imagen
    alto, ancho = gris_act.shape
    m = cfg_seg["margen_borde_px"]
    dentro = ((p1[:, 0] >= m) & (p1[:, 0] < ancho - m) &
              (p1[:, 1] >= m) & (p1[:, 1] < alto - m))
    validos = ((estado_ida.ravel() == 1) & (estado_vuelta.ravel() == 1) &
               (error_ida_vuelta < cfg_seg["ida_vuelta_max_px"]) & dentro)
    return p1, validos, error_ida_vuelta


# =====================================================================
# Paso 8.3 — Rechazo geométrico con la matriz esencial y RANSAC
# =====================================================================
def rechazo_geometrico(xy_ant, xy_act, desplazamiento_px, cfg_geo, focal_px):
    """¿Qué correspondencias encajan con un único movimiento rígido de la cámara?

    xy_ant, xy_act: arrays (N, 2) en coordenadas normalizadas sin distorsión.
    desplazamiento_px: (N,) cuánto se ha movido cada punto en la imagen [px].
    focal_px: focal en píxeles, para pasar el umbral de px a coordenadas normalizadas.
    Devuelve: array (N,) bool, True = encaja (inlier).
    """
    n = len(xy_ant)
    # Con pocos puntos o casi sin movimiento, la matriz esencial no está bien definida
    if n < cfg_geo["puntos_min"] or np.median(desplazamiento_px) < cfg_geo["paralaje_min_px"]:
        return np.ones(n, dtype=bool)

    _, mascara = cv2.findEssentialMat(
        xy_ant.astype(np.float64), xy_act.astype(np.float64),
        focal=1.0, pp=(0.0, 0.0),                      # ya están normalizadas: K = identidad
        method=cv2.RANSAC, prob=cfg_geo["confianza"],
        threshold=cfg_geo["umbral_px"] / focal_px)     # 1 px expresado sin unidades
    if mascara is None:
        return np.ones(n, dtype=bool)
    return mascara.ravel().astype(bool)


# =====================================================================
# Paso 8.4 — Pistas y el front-end completo
# =====================================================================
@dataclass
class Pista:
    """Historia de un punto: en qué imágenes se vio y dónde."""
    id: int
    imagenes: list = field(default_factory=list)   # índice de cada imagen
    t_ns: list = field(default_factory=list)       # instante de cada imagen [ns]
    uv: list = field(default_factory=list)         # posición en píxeles, (u, v)
    xy: list = field(default_factory=list)         # coordenadas normalizadas sin distorsión

    def anadir(self, imagen, t_ns, uv, xy):
        self.imagenes.append(imagen)
        self.t_ns.append(t_ns)
        self.uv.append((float(uv[0]), float(uv[1])))
        self.xy.append((float(xy[0]), float(xy[1])))

    def longitud(self):
        """Número de imágenes en las que se ha visto."""
        return len(self.imagenes)


class FrontEnd:
    """Recibe imágenes una a una y entrega las pistas que terminan."""

    def __init__(self, K, dist, cfg):
        """K (3, 3), dist (5,): intrínsecos. cfg: dict de config/frontend.yaml."""
        self.K = K
        self.dist = dist
        self.cfg = cfg
        # Etapa 31: el pipeline mira este atributo; si es True, le pasa a procesar() el giro de
        # la cámara desde la imagen anterior, medido por el giroscopio
        self.usa_giro = bool(cfg["seguimiento"]["usar_giro"])
        self.gris_ant = None
        self.puntos = np.zeros((0, 2), np.float32)   # posición actual de cada punto seguido
        self.ids = np.zeros(0, dtype=int)            # id de la pista de cada punto
        self.activas = {}                            # id -> Pista
        self.siguiente_id = 0
        self.n_imagenes = 0

    def procesar(self, gris, t_ns, R_act_ant=None):
        """Procesa una imagen.

        R_act_ant: (3, 3) giro de la cámara desde la imagen anterior (d_act = R_act_ant·d_ant),
            o None. Solo se usa si config/frontend.yaml tiene seguimiento.usar_giro: true.
        Devuelve: dict con
            terminadas: lista de Pista que acaban en la imagen anterior
            estadisticas: dict con los recuentos de esta imagen. Etapa 31, además:
                con_giro: 1 si se usó el giro en esta imagen, 0 si no
                giro_px: mediana de lo que movió el giro a los puntos (la predicción) [px]
                resto_px: mediana de lo que tuvo que corregir Lucas-Kanade después [px]
        """
        indice = self.n_imagenes
        terminadas = []
        est = {"seguidos": 0, "perdidos_klt": 0, "rechazados_ransac": 0, "nuevos": 0,
               "con_giro": 0, "giro_px": 0.0, "resto_px": 0.0}

        # --- 1-3. Seguir los puntos existentes y filtrar ---
        if self.gris_ant is not None and len(self.puntos) > 0:
            giro = None
            prediccion = self.puntos                 # sin giro, se busca desde donde estaba
            if self.usa_giro and R_act_ant is not None:
                giro = (R_act_ant, self.K, self.dist)
                est["con_giro"] = 1
                prediccion, _ = predecir_con_giro(self.puntos, R_act_ant, self.K, self.dist)
            puntos_act, validos, _ = seguir(self.gris_ant, gris, self.puntos, self.cfg["seguimiento"], giro)
            est["perdidos_klt"] = int((~validos).sum())
            if validos.any():
                # Para entender qué hace cada parte: cuánto explica el giro y cuánto queda
                est["giro_px"] = float(np.median(np.linalg.norm(prediccion[validos] - self.puntos[validos], axis=1)))
                est["resto_px"] = float(np.median(np.linalg.norm(puntos_act[validos] - prediccion[validos], axis=1)))

            idx = np.flatnonzero(validos)
            if len(idx) > 0:
                xy_ant = mc.normalizar(self.puntos[idx], self.K, self.dist)
                xy_act = mc.normalizar(puntos_act[idx], self.K, self.dist)
                desplazamiento = np.linalg.norm(puntos_act[idx] - self.puntos[idx], axis=1)
                encajan = rechazo_geometrico(xy_ant, xy_act, desplazamiento,
                                             self.cfg["geometria"], self.K[0, 0])
                validos[idx[~encajan]] = False
                est["rechazados_ransac"] = int((~encajan).sum())

            # --- 4. Terminar las pistas de los puntos perdidos ---
            for k in np.flatnonzero(~validos):
                terminadas.append(self.activas.pop(int(self.ids[k])))

            # Los supervivientes se quedan con su nueva posición y una observación más
            self.puntos = puntos_act[validos]
            self.ids = self.ids[validos]
            xy = mc.normalizar(self.puntos, self.K, self.dist) if len(self.puntos) else []
            for k in range(len(self.puntos)):
                self.activas[int(self.ids[k])].anadir(indice, t_ns, self.puntos[k], xy[k])
            est["seguidos"] = len(self.puntos)

        # --- 5. Puntos nuevos donde haya hueco ---
        nuevos = detectar(gris, self.puntos, self.cfg["deteccion"],
                          self.cfg["seguimiento"]["margen_borde_px"])
        if len(nuevos) > 0:
            xy_nuevos = mc.normalizar(nuevos, self.K, self.dist)
            ids_nuevos = []
            for k in range(len(nuevos)):
                pista = Pista(id=self.siguiente_id)
                pista.anadir(indice, t_ns, nuevos[k], xy_nuevos[k])
                self.activas[pista.id] = pista
                ids_nuevos.append(pista.id)
                self.siguiente_id += 1
            self.puntos = np.vstack([self.puntos, nuevos])
            self.ids = np.concatenate([self.ids, np.array(ids_nuevos, dtype=int)])
        est["nuevos"] = len(nuevos)
        est["activos"] = len(self.puntos)

        self.gris_ant = gris
        self.n_imagenes += 1
        return {"terminadas": terminadas, "estadisticas": est}

    def terminar_todas(self):
        """Al acabar la secuencia: entrega todas las pistas que seguían activas."""
        restantes = list(self.activas.values())
        self.activas = {}
        self.puntos = np.zeros((0, 2), np.float32)
        self.ids = np.zeros(0, dtype=int)
        return restantes