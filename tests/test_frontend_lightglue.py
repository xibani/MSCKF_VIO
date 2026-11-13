"""tests/test_frontend_lightglue.py — Etapa 22: de parejas a pistas.

Dos grupos:
  - Sin redes (pasan también en la Pi): las dos funciones sueltas y el front-end completo con
    un emparejador DE MENTIRA, que no mira la imagen: se le dice de antemano qué puntos se
    ven en cada una y cuál es cuál. Así se prueba la contabilidad de las pistas, que es lo
    que se escribe en esta etapa, sin depender de que las redes acierten.
  - Con PyTorch (PC): el front-end sobre el EmparejadorLightGlue de la etapa 21 con redes
    falsas y, con las redes de verdad, sobre unas imágenes seguidas del simulador, comparando
    cada pista con la verdad. Se saltan si no hay PyTorch o lightglue.
"""

import copy

import cv2
import numpy as np
import pytest

from vio import emparejador as emp
from vio import escena
from vio import frontend as fe
from vio import frontend_lightglue as fel
from vio import modelo_camara as mc
from vio import simulador as sim
from vio.config import cargar_config

CFG = cargar_config("lightglue")
CFG_PISTAS = CFG["pistas"]
CFG_KLT = cargar_config("frontend")
CFG_SIM = cargar_config("simulador")["vio"]
K = mc.matriz_K(643.0, 643.0, 320.0, 240.0)
SIN_DIST = np.zeros(5)
DIST = np.array([0.168, -0.59, -0.0073, 0.0004, 0.955])
TAMANO = (640, 480)
PASO_NS = 50_000_000                                  # 20 imágenes por segundo


class EmparejadorDeMentira:
    """Hace de EmparejadorLightGlue sin redes y sin mirar la imagen.

    vistas: lista con, de cada imagen de la secuencia, (identidad (N,), uv (N, 2)): qué puntos
        del mundo se ven en ella y dónde. extraer() entrega una vista cada vez, por orden.
    puntuacion: (P,) la puntuación de cada punto del mundo.
    emparejar() une los puntos con la misma identidad: lo que las redes tienen que adivinar.
    Para los casos difíciles:
        poca_confianza: {identidad: confianza} que sustituye a la normal (0,9)
        cambiadas: {identidad: otra identidad}: parejas FALSAS (p se empareja con q)
    """

    def __init__(self, vistas, puntuacion):
        self.vistas = vistas
        self.puntuacion = puntuacion
        self.siguiente = 0                  # qué vista entregará el próximo extraer()
        self.n_emparejar = 0
        self.poca_confianza = {}
        self.cambiadas = {}

    def extraer(self, gris):
        identidad, uv = self.vistas[self.siguiente]
        self.siguiente += 1
        return {"puntos": np.asarray(uv, dtype=np.float64), "puntuaciones": self.puntuacion[identidad],
                "tiempo_ms": 0.0, "_red": identidad}

    def emparejar(self, a, b):
        self.n_emparejar += 1
        fila_en_b = {int(p): k for k, p in enumerate(b["_red"])}      # identidad -> fila en b
        indices, confianza = [], []
        for k, p in enumerate(a["_red"]):
            q = self.cambiadas.get(int(p), int(p))                     # con quién se empareja p
            if q in fila_en_b:
                indices.append((k, fila_en_b[q]))
                confianza.append(self.poca_confianza.get(int(p), 0.9))
        return np.array(indices, dtype=int).reshape(-1, 2), np.array(confianza), 0.0


def vistas_de_ventana(n_imagenes, paso_px, n_puntos=600, semilla=0, dos_distancias=False):
    """Puntos fijos de un dibujo grande, vistos por una ventana de 640 x 480 que avanza
    paso_px a la derecha en cada imagen (los puntos se mueven paso_px a la izquierda).

    dos_distancias: la mitad de los puntos (los de número impar) se mueven el doble, como si
        estuvieran a la mitad de distancia de una cámara que se desplaza de lado.
    Devuelve: (vistas, puntuacion) para EmparejadorDeMentira. Posiciones en píxeles enteros,
    como las de SuperPoint.
    """
    rng = np.random.default_rng(semilla)
    ancho_dibujo = 640 + 2 * paso_px * n_imagenes
    dibujo = np.column_stack([rng.integers(0, ancho_dibujo, n_puntos),
                              rng.integers(0, 480, n_puntos)]).astype(float)
    avance = np.full(n_puntos, float(paso_px))
    if dos_distancias:
        avance[1::2] = 2.0 * paso_px
    vistas = []
    for k in range(n_imagenes):
        uv = dibujo - np.column_stack([avance * k, np.zeros(n_puntos)])
        se_ven = np.flatnonzero((uv[:, 0] >= 0) & (uv[:, 0] <= 639))
        vistas.append((se_ven, uv[se_ven]))
    return vistas, rng.uniform(0.1, 1.0, n_puntos)


def identidad_de_cada_pista(frontend):
    """Qué punto del mundo sigue ahora cada pista (solo con el emparejador de mentira)."""
    return frontend.extraccion_ant["_red"][frontend.punto_de_pista]


def textura(alto=480, ancho=640, semilla=0):
    """La imagen de manchas de tests/test_frontend.py."""
    rng = np.random.default_rng(semilla)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (alto, ancho)).astype(np.float32), (0, 0), 3)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# =====================================================================
# Paso 22.2
# =====================================================================
def test_continuar_pistas():
    punto_de_pista = np.array([5, 2, 9, 7])                 # 4 pistas y su punto en la imagen anterior
    indices = np.array([[2, 11], [9, 4], [7, 30], [0, 1]])  # el punto 5 no tiene pareja
    confianza = np.array([0.9, 0.3, 0.5, 0.99])
    destino, confianza_pista = fel.continuar_pistas(punto_de_pista, indices, confianza, 0.5)
    assert destino.tolist() == [-1, 11, -1, 30]             # sin pareja, sigue, poca confianza, sigue (0.5 vale)
    assert confianza_pista.tolist() == [-1.0, 0.9, 0.3, 0.5]
    assert destino.dtype == int


def test_continuar_pistas_casos_vacios():
    sin_parejas = (np.zeros((0, 2), dtype=int), np.zeros(0))
    destino, confianza_pista = fel.continuar_pistas(np.array([3, 4]), *sin_parejas, 0.5)
    assert destino.tolist() == [-1, -1] and confianza_pista.tolist() == [-1.0, -1.0]
    destino, confianza_pista = fel.continuar_pistas(np.zeros(0, dtype=int), np.array([[0, 1]]),
                                                    np.array([0.9]), 0.5)
    assert destino.shape == (0,) and confianza_pista.shape == (0,)


# =====================================================================
# Paso 22.3
# =====================================================================
def test_elegir_nuevos_reparte_los_puntos():
    rng = np.random.default_rng(1)
    candidatos = np.column_stack([rng.uniform(0, 640, 2000), rng.uniform(0, 480, 2000)])
    puntuaciones = rng.uniform(0, 1, 2000)
    elegidos = fel.elegir_nuevos(candidatos, puntuaciones, np.zeros((0, 2)), CFG_PISTAS, TAMANO)
    p = candidatos[elegidos]
    assert 0 < len(p) <= CFG_PISTAS["max_puntos"] and len(set(elegidos.tolist())) == len(elegidos)
    # Ninguna celda con más del máximo, y casi todas usadas
    cx, cy = CFG_PISTAS["celdas_x"], CFG_PISTAS["celdas_y"]
    ocupacion = np.zeros((cy, cx), dtype=int)
    for u, v in p:
        ocupacion[fe.celda_de(u, v, 640, 480, cx, cy)] += 1
    assert ocupacion.max() <= CFG_PISTAS["max_por_celda"] and (ocupacion > 0).sum() >= 0.9 * ocupacion.size
    # Ningún par de puntos más cerca de la distancia mínima, y ninguno en la franja del borde
    distancias = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(distancias, np.inf)
    assert distancias.min() >= CFG_PISTAS["distancia_min_px"]
    m = CFG_PISTAS["margen_borde_px"]
    assert (p >= m).all() and (p[:, 0] < 640 - m).all() and (p[:, 1] < 480 - m).all()


def test_elegir_nuevos_prefiere_la_mejor_puntuacion():
    # Dos candidatos a 5 px el uno del otro: solo cabe uno, y es el de más puntuación
    candidatos = np.array([[100.0, 100.0], [105.0, 100.0], [400.0, 300.0]])
    elegidos = fel.elegir_nuevos(candidatos, np.array([0.2, 0.8, 0.5]), np.zeros((0, 2)), CFG_PISTAS, TAMANO)
    assert elegidos.tolist() == [1, 2]                      # por orden de puntuación: 0.8 y 0.5
    # En la franja del borde no empieza ninguna pista
    borde = np.array([[3.0, 200.0], [636.0, 200.0], [300.0, 2.0], [300.0, 477.0]])
    assert len(fel.elegir_nuevos(borde, np.ones(4), np.zeros((0, 2)), CFG_PISTAS, TAMANO)) == 0


def test_elegir_nuevos_respeta_los_puntos_existentes():
    rng = np.random.default_rng(2)
    candidatos = np.column_stack([rng.uniform(0, 640, 2000), rng.uniform(0, 480, 2000)])
    puntuaciones = rng.uniform(0, 1, 2000)
    primeros = fel.elegir_nuevos(candidatos, puntuaciones, np.zeros((0, 2)), CFG_PISTAS, TAMANO)
    existentes = candidatos[primeros[:40]]
    nuevos = candidatos[fel.elegir_nuevos(candidatos, puntuaciones, existentes, CFG_PISTAS, TAMANO)]
    assert len(existentes) + len(nuevos) <= CFG_PISTAS["max_puntos"]
    distancias = np.linalg.norm(nuevos[:, None, :] - existentes[None, :, :], axis=2)
    assert distancias.min() >= CFG_PISTAS["distancia_min_px"]     # tampoco se elige a sí mismo
    # Sin hueco (ya hay max_puntos), no se elige ninguno
    lleno = np.zeros((CFG_PISTAS["max_puntos"], 2))
    assert len(fel.elegir_nuevos(candidatos, puntuaciones, lleno, CFG_PISTAS, TAMANO)) == 0
    # Y sin candidatos, un array vacío de enteros (sirve para indexar)
    vacio = fel.elegir_nuevos(np.zeros((0, 2)), np.zeros(0), existentes, CFG_PISTAS, TAMANO)
    assert vacio.shape == (0,) and vacio.dtype == int


# =====================================================================
# Paso 22.4 — El criterio de salida: lo que se le pide a FAST + Lucas-Kanade
# =====================================================================
def comprobar_secuencia(frontend, max_puntos):
    """Las comprobaciones de test_frontend_sobre_una_secuencia (etapa 8), para cualquier front-end.

    Una textura grande vista por una ventana que se mueve 2 px a la derecha por imagen: el
    contenido se desplaza 2 px a la izquierda en cada imagen.
    """
    grande = textura(480, 640 + 2 * 40, semilla=3)
    terminadas = []
    for k in range(40):
        imagen = np.ascontiguousarray(grande[:, 2 * k: 2 * k + 640])
        r = frontend.procesar(imagen, t_ns=k * PASO_NS)
        terminadas += r["terminadas"]
        assert r["estadisticas"]["activos"] <= max_puntos
        assert len(frontend.ids) == len(frontend.puntos) == len(frontend.activas) == r["estadisticas"]["activos"]

    activas = list(frontend.activas.values())
    assert max(p.longitud() for p in activas) == 40              # puntos seguidos de principio a fin
    # Cada observación de cada pista se mueve exactamente (-2, 0) respecto a la anterior
    for pista in activas + terminadas:
        uv = np.array(pista.uv)
        assert pista.imagenes == list(range(pista.imagenes[0], pista.imagenes[0] + len(uv)))   # sin huecos
        if len(uv) > 1:
            assert np.abs(np.diff(uv, axis=0) - [-2.0, 0.0]).max() < 0.1
    # Los puntos que salen por la izquierda se terminan; los ids no se repiten
    assert len(terminadas) > 0
    ids = [p.id for p in activas + terminadas]
    assert len(ids) == len(set(ids))
    # Las coordenadas normalizadas son las de los píxeles con K
    p = activas[0]
    assert p.xy[0] == pytest.approx(((p.uv[0][0] - 320.0) / 643.0, (p.uv[0][1] - 240.0) / 643.0), abs=1e-9)
    # Lo que lee el pipeline: la posición actual de cada pista es su última observación
    for id_pista, punto in zip(frontend.ids, frontend.puntos):
        assert frontend.activas[int(id_pista)].uv[-1] == pytest.approx(tuple(punto))

    restantes = frontend.terminar_todas()
    assert len(restantes) == len(activas) and frontend.activas == {} and len(frontend.ids) == 0


def test_la_comprobacion_la_pasa_fast_klt():
    comprobar_secuencia(fe.FrontEnd(K, SIN_DIST, CFG_KLT), CFG_KLT["deteccion"]["max_puntos"])


def test_la_comprobacion_la_pasa_el_frontend_nuevo():
    falso = EmparejadorDeMentira(*vistas_de_ventana(40, 2))
    frontend = fel.FrontEndLightGlue(K, SIN_DIST, CFG, falso)
    comprobar_secuencia(frontend, CFG_PISTAS["max_puntos"])
    # Una extracción por imagen (no dos) y una llamada a LightGlue por cada par de imágenes seguidas
    assert falso.siguiente == 40 and falso.n_emparejar == 39


def test_cada_pista_sigue_siempre_al_mismo_punto():
    falso = EmparejadorDeMentira(*vistas_de_ventana(30, 2))
    frontend = fel.FrontEndLightGlue(K, SIN_DIST, CFG, falso)
    identidad_de = {}                                    # id de pista -> punto del mundo
    for k in range(30):
        r = frontend.procesar(np.zeros((480, 640), np.uint8), k * PASO_NS)
        est = r["estadisticas"]
        identidades = identidad_de_cada_pista(frontend)
        assert len(set(identidades.tolist())) == len(identidades)       # dos pistas nunca comparten punto
        for id_pista, identidad in zip(frontend.ids, identidades):
            assert identidad_de.setdefault(int(id_pista), int(identidad)) == int(identidad)
        # Las cuentas cuadran: las de antes = las que siguen + perdidas + rechazadas
        if k > 0:
            assert activos_antes == est["seguidos"] + est["perdidos_klt"] + est["rechazados_ransac"]
            assert est["perdidos_klt"] == est["sin_pareja"] + est["poca_confianza"]
            assert len(r["terminadas"]) == est["perdidos_klt"] + est["rechazados_ransac"]
        assert est["activos"] == est["seguidos"] + est["nuevos"]
        activos_antes = est["activos"]
    assert len(identidad_de) > CFG_PISTAS["max_puntos"]  # ha habido relevo de pistas


def test_sin_pareja_o_con_poca_confianza_la_pista_termina():
    # Cámara quieta (paso 0): sin RANSAC de por medio, las cuentas son exactas
    vistas, puntuacion = vistas_de_ventana(8, 0)
    falso = EmparejadorDeMentira(vistas, puntuacion)
    frontend = fel.FrontEndLightGlue(K, SIN_DIST, CFG, falso)
    for k in range(5):
        r = frontend.procesar(np.zeros((480, 640), np.uint8), k * PASO_NS)
    assert r["estadisticas"]["perdidos_klt"] == 0 and r["estadisticas"]["nuevos"] == 0
    seguidos = identidad_de_cada_pista(frontend)
    id_de = dict(zip(seguidos.tolist(), frontend.ids.tolist()))        # punto del mundo -> id de su pista
    dudosos, desaparecen = seguidos[:10], seguidos[10:17]
    # En la imagen 5: 10 puntos se emparejan con confianza 0,3 (< 0,5) y 7 dejan de detectarse
    falso.poca_confianza = {int(p): 0.3 for p in dudosos}
    identidad, uv = vistas[5]
    quedan = ~np.isin(identidad, desaparecen)
    vistas[5] = (identidad[quedan], uv[quedan])
    ultimo_id = frontend.siguiente_id

    r = frontend.procesar(np.zeros((480, 640), np.uint8), 5 * PASO_NS)
    est = r["estadisticas"]
    assert est["poca_confianza"] == 10 and est["sin_pareja"] == 7 and est["perdidos_klt"] == 17
    assert est["rechazados_ransac"] == 0
    # Las 17 pistas se entregan, con sus 5 observaciones (imágenes 0 a 4)
    perdidos = np.concatenate([dudosos, desaparecen])
    assert sorted(p.id for p in r["terminadas"]) == sorted(id_de[int(p)] for p in perdidos)
    assert all(p.imagenes == [0, 1, 2, 3, 4] for p in r["terminadas"])
    # Los huecos se rellenan con pistas NUEVAS (ids sin estrenar), que empiezan en la imagen 5
    assert est["nuevos"] > 0
    for id_pista in frontend.ids[frontend.ids >= ultimo_id]:
        assert frontend.activas[int(id_pista)].imagenes == [5]


def test_las_parejas_falsas_las_quita_el_rechazo_geometrico():
    # Los puntos se mueven 10 o 20 px por imagen (dos distancias). Con 2 px no valdría: un
    # movimiento del tamaño del umbral del RANSAC (2 px) encaja con casi cualquier movimiento
    # de cámara, y entonces el RANSAC elige uno con el que también encaja alguna pareja falsa
    vistas, puntuacion = vistas_de_ventana(8, 10, dos_distancias=True)
    falso = EmparejadorDeMentira(vistas, puntuacion)
    frontend = fel.FrontEndLightGlue(K, SIN_DIST, CFG, falso)
    for k in range(5):
        frontend.procesar(np.zeros((480, 640), np.uint8), k * PASO_NS)
    # En la imagen 5, LightGlue "se equivoca" con 12 puntos seguidos: los cruza de dos en dos.
    # Se eligen parejas de puntos a distinta altura (más de 30 px) y que sigan viéndose
    seguidos = identidad_de_cada_pista(frontend)
    v_de = dict(zip(seguidos.tolist(), frontend.puntos[:, 1].tolist()))
    en_imagen_5 = set(vistas[5][0].tolist())
    orden = sorted((p for p in seguidos.tolist() if p in en_imagen_5), key=lambda p: v_de[p])
    cruzados = []
    for p, q in zip(orden[:6], orden[-6:]):              # los 6 de más arriba con los 6 de más abajo
        assert abs(v_de[p] - v_de[q]) > 30
        falso.cambiadas[p], falso.cambiadas[q] = q, p
        cruzados += [p, q]
    id_de = dict(zip(seguidos.tolist(), frontend.ids.tolist()))

    r = frontend.procesar(np.zeros((480, 640), np.uint8), 5 * PASO_NS)
    assert r["estadisticas"]["rechazados_ransac"] == 12
    ids_terminadas = {p.id for p in r["terminadas"]}
    assert all(id_de[p] in ids_terminadas for p in cruzados)
    # Ninguna pista viva ha dado un salto: todas se mueven en horizontal, 10 o 20 px por imagen
    for pista in frontend.activas.values():
        if pista.longitud() > 1:
            pasos = np.diff(np.array(pista.uv), axis=0)
            assert np.abs(pasos[:, 1]).max() < 0.1 and set(pasos[:, 0].tolist()) <= {-10.0, -20.0}


def test_imagenes_sin_puntos():
    # Una imagen lisa (ningún punto), luego puntos, luego otra lisa: no falla y no inventa pistas
    identidad = np.arange(50)
    uv = np.column_stack([np.linspace(20, 620, 50), np.linspace(20, 460, 50)])
    nada = (np.zeros(0, dtype=int), np.zeros((0, 2)))
    falso = EmparejadorDeMentira([nada, (identidad, uv), nada, (identidad, uv)], np.linspace(0.5, 1.0, 50))
    frontend = fel.FrontEndLightGlue(K, SIN_DIST, CFG, falso)
    gris = np.zeros((480, 640), np.uint8)
    r0 = frontend.procesar(gris, 0)
    assert r0["estadisticas"]["activos"] == 0 and r0["terminadas"] == [] and frontend.puntos.shape == (0, 2)
    r1 = frontend.procesar(gris, PASO_NS)
    n = r1["estadisticas"]["nuevos"]
    assert n > 0 and r1["estadisticas"]["activos"] == n and frontend.puntos.dtype == np.float32
    r2 = frontend.procesar(gris, 2 * PASO_NS)
    assert len(r2["terminadas"]) == n and r2["estadisticas"]["sin_pareja"] == n and frontend.activas == {}
    r3 = frontend.procesar(gris, 3 * PASO_NS)
    assert r3["estadisticas"]["nuevos"] == n and min(frontend.ids) == n            # ids sin repetir


def test_vio_con_pistas_hechas_de_parejas():
    # De principio a fin (como test_vio_de_principio_a_fin, etapa 13) pero con el front-end
    # nuevo: IMU simulada + los puntos de la habitación redondeados a píxeles enteros (como
    # SuperPoint) y con una de cada cinco parejas FALSA. El MSCKF no nota el cambio de front-end
    from herramientas.comparar_trayectoria import alinear
    from vio.pipeline import PipelineVio

    cfgs = {n: cargar_config(n) for n in ("vio", "frontend", "msckf", "triangulacion")}
    ext = cargar_config("camara_imu")
    cfg_vio = copy.deepcopy(CFG_SIM)
    cfg_vio["trayectoria"]["duracion_s"] = 8.0
    cfg_vio["camara"].update(R_i_c=ext["R_i_c"], p_i_c_m=ext["p_i_c_m"], ruido_px=0.0)
    graba = cfg_vio["grabacion_sintetica"]

    def pose(t):
        return sim.pose_con_reposo(t, cfg_vio["trayectoria"], graba["reposo_s"], graba["rampa_s"])
    s = sim.generar_secuencia(cfg_vio, K, TAMANO, np.random.default_rng(3), pose=pose, dist=DIST)

    # 1. Lo que "ve" el emparejador de mentira en cada imagen, y sus fallos
    rng = np.random.default_rng(4)
    n_imagenes = len(s["camaras"]["t_ns"])
    vistas = []
    for j in range(n_imagenes):
        en_imagen = s["obs"]["imagen"] == j
        vistas.append((s["obs"]["id"][en_imagen], np.round(s["obs"]["uv"][en_imagen])))
    falso = EmparejadorDeMentira(vistas, rng.uniform(0.1, 1.0, len(s["puntos"])))

    def emparejar_con_fallos(a, b):
        indices, confianza, tiempo_ms = EmparejadorDeMentira.emparejar(falso, a, b)
        cuales = rng.choice(len(indices), len(indices) // 5, replace=False)
        indices[cuales, 1] = indices[rng.permutation(cuales), 1]      # se barajan sus destinos
        return indices, confianza, tiempo_ms
    falso.emparejar = emparejar_con_fallos

    # 2. El VIO, con la IMU y las imágenes por orden de tiempo (las imágenes van vacías: el
    #    emparejador de mentira no las mira)
    frontend = fel.FrontEndLightGlue(K, DIST, CFG, falso)
    pipeline = PipelineVio(K, DIST, s["R_c_i"], s["t_c_i"], ext["t_d_ns"], cfgs, frontend=frontend)
    gris = np.zeros((480, 640), np.uint8)
    resultados = []
    i = 0
    for j, t_cam in enumerate(s["camaras"]["t_ns"]):
        while i < len(s["imu"]["t_ns"]) and s["imu"]["t_ns"][i] <= t_cam:
            pipeline.nueva_imu(int(s["imu"]["t_ns"][i]), s["imu"]["w"][i], s["imu"]["a"][i])
            i += 1
        falso.siguiente = j                  # el pipeline no pasa al front-end las imágenes del reposo
        r = pipeline.nueva_imagen(int(t_cam), gris)
        if r is not None:
            resultados.append(r)

    # 3. Los mismos límites que con FAST + Lucas-Kanade
    assert len(resultados) >= 8 * 20 - 2 * 20 - 2
    t_est = np.array([r["t_ns"] for r in resultados])
    p_est = np.array([r["p_w"] for r in resultados])
    R_est = np.array([r["R_w_i"] for r in resultados])
    p_ver, _ = alinear(t_est, p_est, R_est, s["verdad"]["t_ns"], s["verdad"]["p_w"], s["verdad"]["R_w_i"])
    error = np.linalg.norm(p_est - p_ver, axis=1)
    assert error.max() < 0.15 and error[-1] < 0.1
    assert sum(r["msckf"]["usadas"] for r in resultados) > 100
    assert sum(r["frontend"]["rechazados_ransac"] for r in resultados) > 500     # las parejas falsas
    assert np.mean([r["frontend"]["activos"] for r in resultados]) > 60


# =====================================================================
# Con PyTorch y redes falsas: el front-end sobre el EmparejadorLightGlue de la etapa 21
# =====================================================================
def test_frontend_sobre_el_emparejador_de_la_etapa_21():
    # Las piezas de verdad encajan: lo que devuelven extraer() y emparejar() de la etapa 21
    # (con las redes falsas de sus tests) es lo que espera el front-end
    torch = pytest.importorskip("torch")
    from test_emparejador import EmparejadorFalso, ExtractorFalso

    puntos = [[100.0, 100.0], [300.0, 200.0], [500.0, 400.0], [200.0, 350.0]]
    # La red falsa "detecta" siempre los mismos 4 puntos y los empareja 0-0, 1-1 y 2-2 con
    # confianzas 0,9, 0,8 y 0,2: el punto 2 tiene poca confianza y el 3 se queda sin pareja
    redes = emp.EmparejadorLightGlue(dict(CFG, dispositivo="cpu"), ExtractorFalso(torch, puntos),
                                     EmparejadorFalso(torch, [[0, 0], [1, 1], [2, 2]], [0.9, 0.8, 0.2]))
    frontend = fel.FrontEndLightGlue(K, SIN_DIST, CFG, redes)
    gris = np.zeros((480, 640), np.uint8)
    r0 = frontend.procesar(gris, 0)
    assert r0["estadisticas"]["nuevos"] == 4 and frontend.ids.tolist() == [0, 1, 2, 3]
    r1 = frontend.procesar(gris, PASO_NS)
    est = r1["estadisticas"]
    assert est["seguidos"] == 2 and est["poca_confianza"] == 1 and est["sin_pareja"] == 1
    assert sorted(p.id for p in r1["terminadas"]) == [2, 3]
    # Las pistas 0 y 1 siguen; los puntos 2 y 3 quedan libres y empiezan las pistas 4 y 5
    assert frontend.ids.tolist() == [0, 1, 4, 5] and est["nuevos"] == 2
    assert frontend.activas[0].uv == [(100.0, 100.0), (100.0, 100.0)]
    assert frontend.puntos.dtype == np.float32 and est["t_extraer_ms"] >= 0.0 and est["t_emparejar_ms"] >= 0.0


# =====================================================================
# Con las redes de verdad (PC con lightglue instalado)
# =====================================================================
N_IMAGENES = 12                                         # 0,6 s de la trayectoria del simulador


@pytest.fixture(scope="module")
def frontend_real():
    pytest.importorskip("torch")
    pytest.importorskip("lightglue")
    return fel.FrontEndLightGlue(K, DIST, CFG)


@pytest.fixture(scope="module")
def secuencia_simulada():
    """12 imágenes seguidas (20 por segundo) de la habitación con textura, con su verdad."""
    rng = np.random.default_rng(7)
    habitacion = escena.crear_habitacion(CFG_SIM["mundo"]["habitacion_m"], CFG_SIM["textura"], rng)
    rayos = escena.rayos_camara(K, DIST, TAMANO)
    R_c_i, t_c_i = sim.extrinseca(CFG_SIM["camara"])
    vistas = []
    for k in range(N_IMAGENES):
        R_w_i, p_w = sim.pose_vio(4.0 + 0.05 * k, CFG_SIM["trayectoria"])
        pose = sim.pose_camara(R_w_i, p_w, R_c_i, t_c_i)
        ideal, profundidad = escena.renderizar_habitacion(habitacion, pose[0], pose[1], rayos)
        vistas.append({"gris": escena.acabar_imagen(ideal, CFG_SIM["textura"], rng),
                       "profundidad": profundidad, "pose": pose})
    return vistas


def test_frontend_con_las_redes_de_verdad(frontend_real, secuencia_simulada):
    # (frontend_real va primero: si no hay redes, el test se salta antes de dibujar las imágenes)
    frontend = frontend_real
    terminadas, activos = [], []
    for k, vista in enumerate(secuencia_simulada):
        r = frontend.procesar(vista["gris"], k * PASO_NS)
        terminadas += r["terminadas"]
        activos.append(r["estadisticas"]["activos"])
        assert len(set(frontend.punto_de_pista.tolist())) == len(frontend.punto_de_pista)
    pistas = terminadas + frontend.terminar_todas()
    ids = [p.id for p in pistas]
    assert len(ids) == len(set(ids))

    # Cada paso de cada pista, comparado con la verdad: el punto de la imagen k, llevado a la
    # imagen k+1 con la profundidad y las poses verdaderas, ¿cae donde dice la pista?
    errores = []
    for pista in pistas:
        for n in range(pista.longitud() - 1):
            a, b = secuencia_simulada[pista.imagenes[n]], secuencia_simulada[pista.imagenes[n + 1]]
            verdad_b, visible = emp.posicion_verdadera_en_b(
                np.array([pista.uv[n]]), a["profundidad"], a["pose"], b["pose"], K, DIST)
            errores.append(np.linalg.norm(np.array(pista.uv[n + 1]) - verdad_b[0]) if visible[0] else np.inf)
    errores = np.array(errores)
    umbral = CFG["evaluacion"]["umbral_correcta_px"]
    assert len(errores) > 300                                    # del orden de 100 pistas x 11 pasos
    assert np.mean(errores < umbral) > 0.9                       # pasos correctos
    assert np.median(errores) < 1.5
    assert np.mean(activos) > 60                                 # como se le pide a FAST + KLT (etapa 20)
    assert max(p.longitud() for p in pistas) >= N_IMAGENES - 2   # hay puntos seguidos casi de principio a fin
    # De las pistas que empiezan en la primera imagen, al menos la mitad le sirve al MSCKF
    # (3 imágenes o más). Las que empiezan después no cuentan: se cortan al acabar la secuencia
    primeras = [p.longitud() for p in pistas if p.imagenes[0] == 0]
    assert np.median(primeras) >= 3
