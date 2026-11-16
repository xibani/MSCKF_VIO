"""tests/test_bucles.py — Etapa 28: detectar que se ha vuelto a un sitio.

Ninguno necesita PyTorch: pasan también en la Pi. Tres grupos:
  - Cuentas con poses y candidatos, con números hechos a mano.
  - PnP y la comprobación de un bucle, con vistas SINTÉTICAS (profundidad exacta del
    trazado de rayos) y un emparejador de mentira que sabe qué punto es cuál.
  - Con la sesión simulada: imágenes de verdad emparejadas con ORB.
"""

import copy

import cv2
import numpy as np
import pytest

from herramientas import buscar_bucles
from herramientas import medir_mapa
from herramientas import nube_densa
from vio import bucles as bu
from vio import escena
from vio import modelo_camara as mc
from vio.config import cargar_config
from vio.emparejador_orb import EmparejadorOrb

CFG = cargar_config("bucles")
DIMENSIONES = (6.0, 6.0, 3.0)
TAMANO = (640, 480)
K = mc.matriz_K(643.0, 643.0, 320.0, 240.0)
DIST = np.array([0.168, -0.59, -0.0073, 0.0004, 0.955])
ABAJO = np.array([0.0, 0.0, -1.0])


def giro_z(grados):
    """Matriz de un giro alrededor de la vertical."""
    return cv2.Rodrigues(np.array([0.0, 0.0, np.radians(grados)]))[0]


def vista_sintetica(posicion, hacia, indice=0, escala=1.0):
    """Una vista hecha a mano (como en tests/test_densificar.py): la cámara en `posicion`
    mirando en horizontal hacia `hacia`, con la profundidad exacta del trazado de rayos."""
    adelante = np.array(hacia, dtype=float) / np.linalg.norm(hacia)
    R_w_c = np.column_stack([np.cross(ABAJO, adelante), ABAJO, adelante])
    p_w_c = np.array(posicion, dtype=float)
    habitacion = {"dimensiones": DIMENSIONES, "px_por_m": 10,
                  "texturas": [np.full((int(10 * alto), int(10 * ancho)), 100.0, np.float32)
                               for ancho, alto in (escena.tamano_de_cara(n, DIMENSIONES) for n in range(escena.N_CARAS))]}
    _, z = escena.renderizar_habitacion(habitacion, R_w_c.T, -R_w_c.T @ p_w_c, escena.rayos_camara(K, DIST, TAMANO))
    kf = {"indice": indice, "t_ns": indice, "tamano": TAMANO, "K": K, "dist": DIST, "R_w_c": R_w_c, "p_w_c": p_w_c}
    return {"kf": kf, "z": z * escala, "gris": np.full((TAMANO[1], TAMANO[0]), 100, np.uint8)}


def parejas_exactas(vista_a, vista_b, n, semilla=0, ruido_px=0.0):
    """n píxeles al azar de A y dónde se ve cada uno, de verdad, en B (solo los que B ve).

    Píxel de A + profundidad -> punto del mundo -> píxel de B. Devuelve: (uv_a (M, 2), uv_b (M, 2)).
    """
    generador = np.random.default_rng(semilla)
    uv_a = np.column_stack([generador.uniform(20, 620, n), generador.uniform(20, 460, n)])
    p_a, _ = bu.puntos_3d_de_pixeles(uv_a, vista_a["z"], K, DIST, CFG["pnp"])
    kf_a, kf_b = vista_a["kf"], vista_b["kf"]
    p_w = p_a @ kf_a["R_w_c"].T + kf_a["p_w_c"]
    p_b = (p_w - kf_b["p_w_c"]) @ kf_b["R_w_c"]
    uv_b = mc.proyectar(p_b, K, DIST)
    dentro = (p_b[:, 2] > 0.3) & (uv_b[:, 0] > 5) & (uv_b[:, 0] < 635) & (uv_b[:, 1] > 5) & (uv_b[:, 1] < 475)
    return uv_a[dentro], uv_b[dentro] + generador.normal(0.0, ruido_px, (int(dentro.sum()), 2))


class EmparejadorDeMentira:
    """Hace de emparejador sin mirar las imágenes: extraer() entrega los puntos que se le han
    dado para esa imagen (por orden) y emparejar() une los que tienen el mismo número de fila."""

    dispositivo = "ninguno (de mentira)"

    def __init__(self, puntos_por_imagen, confianza=0.9):
        self.puntos_por_imagen = puntos_por_imagen
        self.confianza = confianza
        self.siguiente = 0

    def extraer(self, gris):
        puntos = self.puntos_por_imagen[self.siguiente]
        self.siguiente += 1
        return {"puntos": puntos, "puntuaciones": np.ones(len(puntos)), "tiempo_ms": 0.0, "_red": None}

    def emparejar(self, a, b):
        n = min(len(a["puntos"]), len(b["puntos"]))
        return np.column_stack([np.arange(n), np.arange(n)]), np.full(n, self.confianza), 0.0


# =====================================================================
# Paso 28.1 — Poses relativas
# =====================================================================
def test_pose_relativa_e_inversa():
    R_i, p_i = giro_z(30.0), np.array([1.0, 2.0, 0.5])
    R_j, p_j = giro_z(50.0), np.array([1.5, 2.0, 0.5])
    R, p = bu.pose_relativa(R_i, p_i, R_j, p_j)
    # j está girada 20° respecto a i y, vista desde i, a 0,5 m en la dirección x del mundo
    assert R == pytest.approx(giro_z(20.0)) and p == pytest.approx(R_i.T @ [0.5, 0.0, 0.0])
    # Un punto en ejes de j, llevado a i por los dos caminos
    q_j = np.array([0.3, -0.2, 2.0])
    q_w = R_j @ q_j + p_j
    assert R @ q_j + p == pytest.approx(R_i.T @ (q_w - p_i))
    # La inversa deshace
    R_inv, p_inv = bu.invertir_pose(R, p)
    assert R_inv @ (R @ q_j + p) + p_inv == pytest.approx(q_j)
    assert bu.pose_relativa(R_j, p_j, R_i, p_i)[1] == pytest.approx(p_inv)
    assert bu.diferencia_de_poses(R, p, R, p) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert bu.diferencia_de_poses(R_i, p_i, R_j, p_j) == pytest.approx((0.5, 20.0))


# =====================================================================
# Paso 28.2 — Candidatos
# =====================================================================
def test_candidatos():
    # Dos vueltas a un círculo de 1,5 m mirando hacia fuera, con un keyframe cada 18°: el
    # keyframe 20 está donde estuvo el 0, el 21 donde el 1...
    vistas = []
    for k in range(40):
        a = np.radians(18.0 * k)
        adelante = np.array([np.cos(a), np.sin(a), 0.0])
        kf = {"R_w_c": np.column_stack([np.cross(ABAJO, adelante), ABAJO, adelante]), "p_w_c": 1.5 * adelante}
        vistas.append({"kf": kf})
    parejas = bu.candidatos(vistas, CFG["candidatos"])
    assert (0, 20) in parejas and (5, 25) in parejas and (19, 39) in parejas
    assert all(j - i > CFG["candidatos"]["separacion_min"] for i, j in parejas)
    # De cada keyframe, el de una vuelta antes y sus dos vecinos de cada lado (a 0,47 m y 18°).
    # Los de dos puestos (0,93 m, 36°) ya miran demasiado distinto
    assert sorted(i for i, j in parejas if j == 30) == [9, 10, 11]
    # El más cercano va primero; con max_por_keyframe = 1 solo queda ese
    uno = bu.candidatos(vistas, dict(CFG["candidatos"], max_por_keyframe=1))
    assert [i for i, j in uno if j == 30] == [10]
    assert len(uno) == 21                                # del 19 (vecino del 0, casi una vuelta después) al 39
    # Sin mirar la pose: todos con todos los que estén lo bastante separados
    todos = bu.candidatos(vistas, dict(CFG["candidatos"], por_pose=False))
    assert len(todos) == sum(max(j - CFG["candidatos"]["separacion_min"], 0) for j in range(40))


# =====================================================================
# Paso 28.3 — PnP
# =====================================================================
def test_puntos_3d_de_pixeles():
    vista = vista_sintetica((0.0, 0.0, 1.5), (1.0, 0.0, 0.0))
    vista["z"][:, :100] = np.nan
    uv = np.array([[320.0, 240.0], [50.0, 240.0], [600.5, 100.25]])
    p_c, validos = bu.puntos_3d_de_pixeles(uv, vista["z"], K, DIST, CFG["pnp"])
    assert list(validos) == [True, False, True]
    assert p_c[0] == pytest.approx([0.0, 0.0, 3.0], abs=1e-3)            # el centro de la imagen: de frente, a 3 m
    assert mc.proyectar(p_c[[0, 2]], K, DIST) == pytest.approx(uv[[0, 2]], abs=1e-3)      # y vuelven a su píxel


def test_pose_por_pnp_con_ruido_y_parejas_falsas():
    a = vista_sintetica((0.0, 0.0, 1.5), (1.0, 0.2, 0.0))
    b = vista_sintetica((0.1, 0.4, 1.4), (1.0, 0.0, 0.0))
    uv_a, uv_b = parejas_exactas(a, b, 300, ruido_px=0.5)
    falsas = np.arange(0, len(uv_a), 3)                  # una de cada tres parejas, falsa
    uv_b[falsas] = np.random.default_rng(1).uniform(20, 460, (len(falsas), 2))
    p_a, validos = bu.puntos_3d_de_pixeles(uv_a, a["z"], K, DIST, CFG["pnp"])
    assert validos.all()
    r = bu.pose_por_pnp(p_a, uv_b, K, DIST, CFG["pnp"])
    # La pose de A vista desde B, comparada con la verdadera
    R_ver, p_ver = bu.pose_relativa(b["kf"]["R_w_c"], b["kf"]["p_w_c"], a["kf"]["R_w_c"], a["kf"]["p_w_c"])
    error_m, error_grados = bu.diferencia_de_poses(r["R"], r["p"], R_ver, p_ver)
    assert error_m < 0.01 and error_grados < 0.1
    buenas = np.setdiff1d(np.arange(len(uv_a)), falsas)
    assert r["encajan"][buenas].mean() > 0.95 and r["encajan"][falsas].mean() < 0.05
    # Con tres puntos no se intenta
    assert bu.pose_por_pnp(p_a[:3], uv_b[:3], K, DIST, CFG["pnp"]) is None


# =====================================================================
# Paso 28.4 — Un bucle
# =====================================================================
def bucle_sintetico(escala_i=1.0, n=300, falsas=False, confianza=0.9):
    """Dos vistas del mismo sitio, sus parejas exactas y lo que dice comprobar_bucle."""
    vista_i = vista_sintetica((0.0, 0.0, 1.5), (1.0, 0.2, 0.0), indice=10)
    vista_j = vista_sintetica((0.1, 0.4, 1.4), (1.0, 0.0, 0.0), indice=40)
    uv_i, uv_j = parejas_exactas(vista_i, vista_j, n, ruido_px=0.5)
    if falsas:
        uv_j = np.random.default_rng(2).uniform(20, 460, uv_j.shape)
    vista_i["z"] = vista_i["z"] * escala_i
    emparejador = EmparejadorDeMentira([uv_i, uv_j], confianza)
    bucle, motivo = bu.comprobar_bucle(vista_i, vista_j, emparejador.extraer(None), emparejador.extraer(None),
                                       emparejador, CFG)
    return vista_i, vista_j, bucle, motivo


def test_comprobar_bucle_bueno():
    vista_i, vista_j, bucle, motivo = bucle_sintetico()
    assert motivo == "ok" and bucle["parejas"] > 200 and bucle["encajan"] > 0.95 * bucle["parejas"]
    assert bucle["desacuerdo_m"] < 0.01 and bucle["desacuerdo_grados"] < 0.1
    R_ver, p_ver = bu.pose_relativa(vista_i["kf"]["R_w_c"], vista_i["kf"]["p_w_c"],
                                    vista_j["kf"]["R_w_c"], vista_j["kf"]["p_w_c"])
    error_m, error_grados = bu.diferencia_de_poses(bucle["R"], bucle["p"], R_ver, p_ver)
    assert error_m < 0.01 and error_grados < 0.1
    assert bucle["uv_i"].shape == bucle["uv_j"].shape == (bucle["encajan"], 2)


def test_comprobar_bucle_descartado():
    # Pocas parejas (o con poca confianza): no se intenta
    assert bucle_sintetico(n=30)[2:] == (None, "pocas parejas")
    assert bucle_sintetico(confianza=0.3)[2:] == (None, "pocas parejas")
    # Parejas que no tienen nada que ver: ninguna pose las explica
    assert bucle_sintetico(falsas=True)[2:] == (None, "sin pose")
    # La profundidad de una vista, un 30 % larga: cada sentido da una distancia distinta
    # entre las dos cámaras (0,42 m de verdad; un 30 % son 13 cm de diferencia)
    assert bucle_sintetico(escala_i=1.3)[2:] == (None, "no coinciden los dos sentidos")


# =====================================================================
# El emparejador clásico
# =====================================================================
def test_emparejador_orb_con_una_imagen_y_su_copia_desplazada():
    # Una imagen con dibujo (rectángulos de grises al azar, algo suavizada) y la misma
    # desplazada 12 px a la derecha y 5 hacia abajo
    generador = np.random.default_rng(3)
    imagen = np.full((480, 700), 110, np.uint8)
    for _ in range(150):
        c, f = int(generador.uniform(0, 680)), int(generador.uniform(0, 460))
        imagen[f:f + int(generador.uniform(10, 60)), c:c + int(generador.uniform(10, 60))] = int(generador.uniform(20, 235))
    imagen = cv2.GaussianBlur(imagen, (0, 0), 1.0)
    a, b = imagen[5:, 12:652].copy()[:470], imagen[:470, :640].copy()
    orb = EmparejadorOrb(CFG["orb"])
    extraccion_a, extraccion_b = orb.extraer(a), orb.extraer(b)
    assert len(extraccion_a["puntos"]) > 300 and extraccion_a["_red"].shape == (len(extraccion_a["puntos"]), 32)
    indices, confianza, _ = orb.emparejar(extraccion_a, extraccion_b)
    assert len(indices) > 150 and (confianza == 1.0).all()
    # Un punto de A está, en B, 12 px más a la derecha y 5 más abajo
    desplazamiento = extraccion_b["puntos"][indices[:, 1]] - extraccion_a["puntos"][indices[:, 0]]
    correctas = np.linalg.norm(desplazamiento - [12.0, 5.0], axis=1) < 2.0
    assert correctas.mean() > 0.95
    # Una imagen lisa no tiene puntos, y no pasa nada
    lisa = orb.extraer(np.full((480, 640), 100, np.uint8))
    assert len(lisa["puntos"]) == 0 and len(orb.emparejar(lisa, extraccion_b)[0]) == 0
    with pytest.raises(ValueError):
        orb.extraer(imagen.astype(np.float32))


# =====================================================================
# Con la sesión simulada: imágenes de verdad, ORB y profundidad verdadera
# =====================================================================
@pytest.fixture(scope="module")
def sesion(sesion_con_textura):
    """Las vistas de la sesión de la etapa 24 (con la profundidad verdadera), las poses
    verdaderas de la cámara en cada una y los bucles que salen.

    La sesión dura 10 s y la cámara no vuelve a ningún sitio. Para tener bucles se compara
    cada keyframe con los uno o dos anteriores, que aún ven en parte lo mismo (la cámara
    gira unos 15° de uno al siguiente): no hay deriva que corregir, pero el camino del
    código es el mismo.
    """
    carpeta = sesion_con_textura["carpeta"]
    mapa = medir_mapa.leer_mapa(carpeta / "mapa.paquetes")
    vistas = nube_densa.cargar_vistas(mapa.keyframes, carpeta, True, sesion_con_textura["t_d_ns"])
    s = sesion_con_textura["s"]
    cfg_ext = {"R_i_c": s["R_c_i"].T, "p_i_c_m": -s["R_c_i"].T @ s["t_c_i"]}     # la extrínseca del simulador
    poses_verdad = buscar_bucles.verdad_de_la_sesion(carpeta, vistas, cfg_ext)
    cfg = copy.deepcopy(CFG)
    cfg["candidatos"].update(separacion_min=0, distancia_max_m=1.5, angulo_max_grados=40.0)
    bucles, recuentos = bu.buscar_bucles(vistas, EmparejadorOrb(cfg["orb"]), cfg)
    return {"carpeta": carpeta, "vistas": vistas, "poses_verdad": poses_verdad, "bucles": bucles,
            "recuentos": recuentos}


def test_bucles_de_la_sesion(sesion):
    bucles, recuentos = sesion["bucles"], sesion["recuentos"]
    assert recuentos["candidatos"] >= 8 and recuentos["bucles"] == len(bucles) >= 4
    assert recuentos["candidatos"] == len(bucles) + recuentos["pocas parejas"] + recuentos["sin pose"] \
        + recuentos["no coinciden los dos sentidos"]
    errores = np.array(bu.errores_de_bucles(bucles, sesion["poses_verdad"]))
    # Con la profundidad verdadera, cada bucle da la pose relativa con pocos centímetros de error
    assert np.median(errores[:, 0]) < 0.03 and np.median(errores[:, 1]) < 0.5
    assert errores[:, 0].max() < 0.15 and errores[:, 1].max() < 3.0
    for bucle in bucles:
        assert bucle["j"] > bucle["i"] and bucle["encajan"] >= CFG["pnp"]["encajan_min"]
        assert bucle["indice_i"] == sesion["vistas"][bucle["i"]]["kf"]["indice"]
        # El VIO del simulador casi no deriva: el bucle y él dicen lo mismo
        assert bucle["vio_m"] < 0.15 and bucle["vio_grados"] < 3.0


def test_poses_verdaderas_de_camara(sesion):
    # La pose verdadera de la cámara y la que da el VIO (tras pasar de un mundo al otro)
    # tienen que coincidir en pocos centímetros
    R, t = nube_densa.transformacion_de_la_sesion(sesion["carpeta"])
    for vista, (R_ver, p_ver) in zip(sesion["vistas"], sesion["poses_verdad"]):
        kf = vista["kf"]
        error_m, error_grados = bu.diferencia_de_poses(R @ kf["R_w_c"], R @ kf["p_w_c"] + t, R_ver, p_ver)
        assert error_m < 0.05 and error_grados < 1.0


def test_la_herramienta_guarda_y_lee_los_bucles(sesion, tmp_path):
    bucles = sesion["bucles"]
    buscar_bucles.guardar_bucles(tmp_path / "bucles.csv", bucles)
    leidos = buscar_bucles.leer_bucles(tmp_path / "bucles.csv")
    assert len(leidos) == len(bucles)
    for a, b in zip(bucles, leidos):
        assert (a["i"], a["j"], a["indice_i"], a["indice_j"], a["parejas"], a["encajan"]) == \
            (b["i"], b["j"], b["indice_i"], b["indice_j"], b["parejas"], b["encajan"])
        assert b["R"] == pytest.approx(a["R"], abs=1e-8) and b["p"] == pytest.approx(a["p"], abs=1e-8)
        assert b["vio_m"] == pytest.approx(a["vio_m"], abs=1e-6)
    # El informe, con verdad y sin ella
    errores = bu.errores_de_bucles(bucles, sesion["poses_verdad"])
    texto = buscar_bucles.informe(bucles, sesion["recuentos"], errores)
    assert f"Bucles: {len(bucles)}." in texto and "Frente a la verdad" in texto and "error de verdad" in texto
    assert "Frente a la verdad" not in buscar_bucles.informe(bucles, sesion["recuentos"], None)
    assert len(buscar_bucles.informe([], dict(sesion["recuentos"], bucles=0), None).splitlines()) == 1
    # Las figuras
    buscar_bucles.dibujar(tmp_path / "bucles.png", sesion["vistas"], bucles)
    assert (tmp_path / "bucles.png").stat().st_size > 10000
    # El emparejador por su nombre
    assert isinstance(buscar_bucles.crear_emparejador("orb", CFG), EmparejadorOrb)
    with pytest.raises(ValueError):
        buscar_bucles.crear_emparejador("otro", CFG)
    # Una sesión que no es del simulador no tiene verdad
    assert buscar_bucles.verdad_de_la_sesion(tmp_path, sesion["vistas"], {}) is None
