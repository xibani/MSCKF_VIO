"""
vio/frontend_lightglue.py — Etapa 22: de parejas a pistas (front-end con SuperPoint + LightGlue).

La etapa 21 da PAREJAS entre dos imágenes: "el punto 37 de la imagen A es el punto 112 de la
imagen B". El MSCKF no quiere parejas, quiere PISTAS: "el punto con identificador 5 se ha
visto en las imágenes 10, 11, 12 y 13, en estas posiciones". Este módulo encadena las parejas
de cada imagen con la anterior para construir esas pistas.

Entrega lo mismo que el front-end FAST + Lucas-Kanade de la etapa 8 (vio/frontend.py), para
que el MSCKF, la vista de la tablet, el mapa y ver_pistas funcionen sin cambios:
    ids (V,), puntos (V, 2) [px], activas (id -> Pista), procesar() y terminar_todas().

Por cada imagen nueva:
    1. SuperPoint: los puntos de la imagen (EmparejadorLightGlue.extraer, etapa 21).
    2. LightGlue: parejas entre los puntos de la imagen ANTERIOR y los de esta.
    3. Cada pista recuerda qué punto de la imagen anterior era el suyo. Si ese punto tiene
       pareja, con confianza suficiente, la pista continúa en la pareja. Si no, termina.
    4. Rechazo geométrico: el mismo RANSAC con la matriz esencial de la etapa 8.
    5. Pistas nuevas con los puntos que no ha usado ninguna pista, repartidos por la imagen.

Este archivo no importa PyTorch: las redes están detrás de EmparejadorLightGlue, que solo lo
importa al crearse. Con un emparejador de mentira (el de los tests), todo esto funciona
también en la Pi.
"""

import numpy as np

from vio import modelo_camara as mc
from vio.emparejador import EmparejadorLightGlue
from vio.frontend import Pista, celda_de, rechazo_geometrico


# =====================================================================
# Paso 22.2 — ¿Dónde sigue cada pista?
# =====================================================================
def continuar_pistas(punto_de_pista, indices, confianza, umbral_confianza):
    """A qué punto de la imagen actual ha ido a parar cada pista.

    punto_de_pista: (V,) int. De cada pista viva, el número de su punto dentro de los puntos
        de la imagen ANTERIOR (su fila en a["puntos"]).
    indices: (K, 2) int, confianza: (K,): lo que devuelve emparejar(a, b). La fila k dice que
        el punto indices[k, 0] de la anterior es el indices[k, 1] de la actual. Cada punto
        sale como mucho en una pareja (LightGlue lo garantiza).
    umbral_confianza: una pareja con menos confianza no continúa la pista.
    Devuelve: (destino (V,) int, confianza_pista (V,)).
        destino[j]: número del punto de la imagen ACTUAL donde sigue la pista j, o -1 si la
            pista se pierde (su punto no tiene pareja, o la tiene con poca confianza).
        confianza_pista[j]: la confianza de la pareja de la pista j, o -1.0 si no tiene pareja.
    """
    # 1. Apuntar las parejas en un diccionario: punto de la anterior -> (punto de la actual, confianza)
    pareja_de = {}
    for k in range(len(indices)):
        pareja_de[int(indices[k, 0])] = (int(indices[k, 1]), float(confianza[k]))

    # 2. Cada pista busca la pareja de su punto
    n_pistas = len(punto_de_pista)
    destino = np.full(n_pistas, -1, dtype=int)
    confianza_pista = np.full(n_pistas, -1.0)
    for j in range(n_pistas):
        punto = int(punto_de_pista[j])
        if punto not in pareja_de:
            continue                                   # sin pareja: la pista se pierde
        punto_actual, c = pareja_de[punto]
        confianza_pista[j] = c
        if c >= umbral_confianza:
            destino[j] = punto_actual
    return destino, confianza_pista


# =====================================================================
# Paso 22.3 — Qué puntos empiezan una pista nueva
# =====================================================================
def elegir_nuevos(candidatos, puntuaciones, existentes, cfg_pistas, tamano):
    """Qué candidatos se convierten en pistas nuevas.

    candidatos: (M, 2) [px], puntos de SuperPoint que no continúan ninguna pista.
    puntuaciones: (M,) lo llamativo que es cada candidato (mayor = mejor).
    existentes: (N, 2) [px], puntos de las pistas que siguen vivas.
    cfg_pistas: sección "pistas" de config/lightglue.yaml. tamano: (ancho, alto) de la imagen.
    Las mismas reglas que detectar() de la etapa 8: de mejor a peor puntuación, fuera de la
    franja del borde, como mucho max_por_celda por celda, a distancia_min_px o más de
    cualquier otro punto, y hasta completar max_puntos entre existentes y nuevos.
    Devuelve: (E,) int, los números de los elegidos dentro de candidatos.
    """
    ancho, alto = tamano
    cx, cy = cfg_pistas["celdas_x"], cfg_pistas["celdas_y"]
    d_min = cfg_pistas["distancia_min_px"]
    margen = cfg_pistas["margen_borde_px"]

    # 1. Cuántos puntos tiene ya cada celda, y la lista de puntos de los que hay que alejarse
    ocupacion = np.zeros((cy, cx), dtype=int)
    ocupados = []
    for u, v in existentes:
        ocupacion[celda_de(u, v, ancho, alto, cx, cy)] += 1
        ocupados.append((float(u), float(v)))

    # 2. Recorrer los candidatos de mayor a menor puntuación
    orden = np.argsort(-np.asarray(puntuaciones))
    hueco = cfg_pistas["max_puntos"] - len(existentes)
    elegidos = []
    for m in orden:
        if len(elegidos) >= hueco:
            break
        u, v = float(candidatos[m, 0]), float(candidatos[m, 1])
        if u < margen or u >= ancho - margen or v < margen or v >= alto - margen:
            continue                                   # en la franja del borde
        celda = celda_de(u, v, ancho, alto, cx, cy)
        if ocupacion[celda] >= cfg_pistas["max_por_celda"]:
            continue                                   # su celda ya está llena
        if any((u - uo) ** 2 + (v - vo) ** 2 < d_min ** 2 for uo, vo in ocupados):
            continue                                   # demasiado cerca de otro punto
        elegidos.append(int(m))
        ocupacion[celda] += 1
        ocupados.append((u, v))
    return np.array(elegidos, dtype=int)


# =====================================================================
# Paso 22.4 — El front-end completo
# =====================================================================
class FrontEndLightGlue:
    """Recibe imágenes una a una y entrega las pistas que terminan (como FrontEnd, etapa 8)."""

    def __init__(self, K, dist, cfg, emparejador=None):
        """K (3, 3), dist (5,): intrínsecos. cfg: dict de config/lightglue.yaml.

        emparejador: cualquier objeto con extraer(gris) y emparejar(a, b) como los de
        EmparejadorLightGlue. Normalmente se deja en None y se crean aquí las redes (hace
        falta PyTorch); los tests pasan uno de mentira.
        """
        if emparejador is None:
            emparejador = EmparejadorLightGlue(cfg)
        self.emparejador = emparejador
        self.K = K
        self.dist = dist
        self.cfg = cfg
        self.extraccion_ant = None                   # lo que devolvió extraer() con la imagen anterior
        self.punto_de_pista = np.zeros(0, dtype=int)   # de cada punto seguido: su número en extraccion_ant
        self.puntos = np.zeros((0, 2), np.float32)   # posición actual de cada punto seguido
        self.ids = np.zeros(0, dtype=int)            # id de la pista de cada punto
        self.activas = {}                            # id -> Pista
        self.siguiente_id = 0
        self.n_imagenes = 0

    def procesar(self, gris, t_ns):
        """Procesa una imagen. Devuelve: dict con
            terminadas: lista de Pista que acaban en la imagen anterior
            estadisticas: dict con los recuentos de esta imagen. Los mismos nombres que en
                FrontEnd (perdidos_klt = pistas que el emparejador no ha continuado), y además
                sin_pareja y poca_confianza (las dos partes de perdidos_klt) y los tiempos
                de las redes, t_extraer_ms y t_emparejar_ms.
        """
        indice = self.n_imagenes
        cfg_pistas = self.cfg["pistas"]
        terminadas = []
        est = {"seguidos": 0, "perdidos_klt": 0, "rechazados_ransac": 0, "nuevos": 0,
               "sin_pareja": 0, "poca_confianza": 0, "t_emparejar_ms": 0.0}

        # --- 1. SuperPoint sobre la imagen nueva ---
        actual = self.emparejador.extraer(gris)
        est["t_extraer_ms"] = actual["tiempo_ms"]
        puntos_sp = actual["puntos"]                          # (M, 2) todos los puntos de esta imagen
        libre = np.ones(len(puntos_sp), dtype=bool)           # los que no usa ninguna pista

        if self.extraccion_ant is not None and len(self.puntos) > 0:
            # --- 2. LightGlue: parejas entre la imagen anterior y esta ---
            indices, confianza, est["t_emparejar_ms"] = self.emparejador.emparejar(
                self.extraccion_ant, actual)

            # --- 3. Dónde sigue cada pista ---
            destino, confianza_pista = continuar_pistas(
                self.punto_de_pista, indices, confianza, cfg_pistas["umbral_confianza"])
            validos = destino >= 0
            est["sin_pareja"] = int((confianza_pista < 0).sum())
            est["poca_confianza"] = int((~validos).sum()) - est["sin_pareja"]
            est["perdidos_klt"] = int((~validos).sum())
            puntos_act = np.zeros((len(self.puntos), 2), np.float32)
            puntos_act[validos] = puntos_sp[destino[validos]]

            # --- 4. Rechazo geométrico (el de la etapa 8, tal cual) ---
            idx = np.flatnonzero(validos)
            if len(idx) > 0:
                xy_ant = mc.normalizar(self.puntos[idx], self.K, self.dist)
                xy_act = mc.normalizar(puntos_act[idx], self.K, self.dist)
                desplazamiento = np.linalg.norm(puntos_act[idx] - self.puntos[idx], axis=1)
                encajan = rechazo_geometrico(xy_ant, xy_act, desplazamiento,
                                             self.cfg["geometria"], self.K[0, 0])
                validos[idx[~encajan]] = False
                est["rechazados_ransac"] = int((~encajan).sum())

            # Las pistas perdidas o rechazadas terminan y se entregan
            for k in np.flatnonzero(~validos):
                terminadas.append(self.activas.pop(int(self.ids[k])))

            # Las supervivientes se quedan con su nueva posición y una observación más
            self.puntos = puntos_act[validos]
            self.ids = self.ids[validos]
            self.punto_de_pista = destino[validos]
            xy = mc.normalizar(self.puntos, self.K, self.dist) if len(self.puntos) else []
            for k in range(len(self.puntos)):
                self.activas[int(self.ids[k])].anadir(indice, t_ns, self.puntos[k], xy[k])
            est["seguidos"] = len(self.puntos)
            libre[self.punto_de_pista] = False                # estos puntos ya tienen pista

        # --- 5. Pistas nuevas con los puntos libres, donde haya hueco ---
        candidatos = np.flatnonzero(libre)
        alto, ancho = gris.shape
        elegidos = elegir_nuevos(puntos_sp[candidatos], actual["puntuaciones"][candidatos],
                                 self.puntos, cfg_pistas, (ancho, alto))
        puntos_nuevos = candidatos[elegidos]                  # sus números dentro de puntos_sp
        if len(puntos_nuevos) > 0:
            nuevos = puntos_sp[puntos_nuevos].astype(np.float32)
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
            self.punto_de_pista = np.concatenate([self.punto_de_pista, puntos_nuevos])
        est["nuevos"] = len(puntos_nuevos)
        est["activos"] = len(self.puntos)

        self.extraccion_ant = actual
        self.n_imagenes += 1
        return {"terminadas": terminadas, "estadisticas": est}

    def terminar_todas(self):
        """Al acabar la secuencia: entrega todas las pistas que seguían activas."""
        restantes = list(self.activas.values())
        self.activas = {}
        self.puntos = np.zeros((0, 2), np.float32)
        self.ids = np.zeros(0, dtype=int)
        self.punto_de_pista = np.zeros(0, dtype=int)
        return restantes
