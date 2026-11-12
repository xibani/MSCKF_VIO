"""tests/test_escena.py — Etapa 20: la habitación con textura (vio/escena.py).

De un píxel a su rayo, del rayo a la pared y de la pared a su dibujo; y al final, que el
VIO de siempre (FAST + Lucas-Kanade + MSCKF) funciona igual con estas imágenes.
"""

import copy

import numpy as np
import pytest

from vio import escena
from vio import modelo_camara as mc
from vio import simulador as sim
from vio.config import cargar_config

CFG_SIM = cargar_config("simulador")["vio"]
CFG_TEXTURA = CFG_SIM["textura"]
DIMENSIONES = tuple(CFG_SIM["mundo"]["habitacion_m"])
K = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])
TAMANO = (640, 480)
DIST = np.array([0.168, -0.59, -0.0073, 0.0004, 0.955])     # como la de la cámara real


def pose_de_prueba(t_s):
    """Pose de la cámara (R_c_w, t_c_w) en el instante t_s de la trayectoria del simulador."""
    R_c_i, t_c_i = sim.extrinseca(CFG_SIM["camara"])
    R_w_i, p_w = sim.pose_vio(t_s, CFG_SIM["trayectoria"])
    return sim.pose_camara(R_w_i, p_w, R_c_i, t_c_i)


@pytest.fixture(scope="module")
def habitacion():
    return escena.crear_habitacion(DIMENSIONES, CFG_TEXTURA, np.random.default_rng(7))


# =====================================================================
# Paso 20.1 — El dibujo
# =====================================================================
def test_tamano_de_cara():
    assert escena.tamano_de_cara(0, (6.0, 4.0, 3.0)) == (4.0, 3.0)      # pared x = cte: y por z
    assert escena.tamano_de_cara(3, (6.0, 4.0, 3.0)) == (6.0, 3.0)      # pared y = cte: x por z
    assert escena.tamano_de_cara(4, (6.0, 4.0, 3.0)) == (6.0, 4.0)      # suelo: x por y


def test_textura_aleatoria():
    t = escena.textura_aleatoria(2.0, 1.0, CFG_TEXTURA, np.random.default_rng(1))
    px_m = CFG_TEXTURA["px_por_m"]
    assert t.shape == (px_m, 2 * px_m)                                  # (alto, ancho)
    assert t.min() >= 0.0 and t.max() <= 255.0
    assert t.std() > 20.0                                               # tiene dibujo, no es lisa
    # La misma semilla da el mismo dibujo; otra semilla, otro
    assert np.array_equal(t, escena.textura_aleatoria(2.0, 1.0, CFG_TEXTURA, np.random.default_rng(1)))
    assert not np.array_equal(t, escena.textura_aleatoria(2.0, 1.0, CFG_TEXTURA, np.random.default_rng(2)))


def test_crear_habitacion(habitacion):
    assert len(habitacion["texturas"]) == 6
    px_m = CFG_TEXTURA["px_por_m"]
    lx, ly, lz = DIMENSIONES
    assert habitacion["texturas"][0].shape == (int(lz * px_m), int(ly * px_m))
    assert habitacion["texturas"][4].shape == (int(ly * px_m), int(lx * px_m))


# =====================================================================
# Paso 20.2 — Rayos y choque
# =====================================================================
def test_rayos_camara_sin_distorsion():
    rayos = escena.rayos_camara(K, None, TAMANO)
    assert rayos.shape == (480, 640, 3)
    assert np.allclose(rayos[240, 320], [0.0, 0.0, 1.0])                # el centro mira de frente
    assert np.allclose(rayos[240, 0], [-320 / 643, 0.0, 1.0])           # el borde izquierdo
    assert np.all(rayos[:, :, 2] == 1.0)


def test_rayos_camara_son_la_inversa_de_proyectar():
    # Con distorsión: un punto cualquiera del rayo de un píxel se proyecta en ese píxel
    rayos = escena.rayos_camara(K, DIST, TAMANO)
    p_c = 2.5 * rayos.reshape(-1, 3)                                    # todos a 2,5 m
    uv = mc.proyectar(p_c, K, DIST).reshape(480, 640, 2)
    u, v = np.meshgrid(np.arange(640.0), np.arange(480.0))
    assert np.abs(uv[:, :, 0] - u).max() < 0.01 and np.abs(uv[:, :, 1] - v).max() < 0.01


def test_chocar_con_habitacion():
    origen = np.array([1.0, 0.0, 1.5])
    direcciones = np.array([[1.0, 0.0, 0.0],      # hacia +x: cara 1, a 2 m
                            [-1.0, 0.0, 0.0],     # hacia -x: cara 0, a 4 m
                            [0.0, 2.0, 0.0],      # hacia +y con dirección de longitud 2: s = 1,5
                            [0.0, 0.0, -1.0],     # hacia abajo: suelo, a 1,5 m
                            [0.0, 0.0, 1.0],      # hacia arriba: techo, a 1,5 m
                            [1.0, 1.0, 0.0]])     # en diagonal: llega antes a x = 3 (s = 2)
    s, cara = escena.chocar_con_habitacion(origen, direcciones, (6.0, 6.0, 3.0))
    assert np.allclose(s, [2.0, 4.0, 1.5, 1.5, 1.5, 2.0])
    assert list(cara) == [1, 0, 3, 4, 5, 1]
    with pytest.raises(ValueError):                                     # la cámara, fuera
        escena.chocar_con_habitacion(np.array([5.0, 0.0, 1.0]), direcciones, (6.0, 6.0, 3.0))


def test_coordenadas_en_cara():
    p = np.array([[3.0, -1.0, 2.0]])
    a, b = escena.coordenadas_en_cara(p, 1, (6.0, 6.0, 3.0))            # pared x = +3
    assert np.allclose([a[0], b[0]], [2.0, 2.0])                        # a = y + 3, b = z
    a, b = escena.coordenadas_en_cara(p, 2, (6.0, 6.0, 3.0))            # pared y = cte
    assert np.allclose([a[0], b[0]], [6.0, 2.0])                        # a = x + 3, b = z
    a, b = escena.coordenadas_en_cara(np.array([[1.0, 2.0, 0.0]]), 4, (6.0, 6.0, 3.0))   # suelo
    assert np.allclose([a[0], b[0]], [4.0, 5.0])


def test_muestrear_bilineal():
    textura = np.array([[0.0, 10.0, 20.0],
                        [100.0, 110.0, 120.0]])
    col = np.array([0.0, 1.0, 0.5, 0.5, 1.25])
    fil = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
    # En un píxel exacto, su valor; entre dos, la media; y una rampa se interpola sin error
    assert np.allclose(escena.muestrear_bilineal(textura, col, fil), [0.0, 10.0, 5.0, 55.0, 62.5])
    # Fuera del dibujo se queda en el borde (no falla ni lee del otro lado)
    assert np.allclose(escena.muestrear_bilineal(textura, np.array([-3.0]), np.array([-3.0])), [0.0])


# =====================================================================
# Paso 20.3 — La imagen
# =====================================================================
def test_la_profundidad_cae_en_las_paredes(habitacion):
    R_c_w, t_c_w = pose_de_prueba(4.0)
    rayos = escena.rayos_camara(K, DIST, TAMANO)
    imagen, profundidad = escena.renderizar_habitacion(habitacion, R_c_w, t_c_w, rayos)
    assert imagen.shape == profundidad.shape == (480, 640)
    assert imagen.std() > 15.0 and 0.0 <= imagen.min() and imagen.max() <= 255.0
    # Cada píxel, llevado a 3D con su profundidad y pasado al mundo, está SOBRE una cara:
    # alguna de sus coordenadas vale exactamente un límite de la habitación, y ninguna se sale
    p_c = profundidad.reshape(-1, 1) * rayos.reshape(-1, 3)
    p_w = (p_c - t_c_w) @ R_c_w                                         # R_c_wᵀ·(p_c - t_c_w)
    lx, ly, lz = DIMENSIONES
    limites = np.abs(np.column_stack([p_w[:, 0] + lx / 2, p_w[:, 0] - lx / 2,
                                      p_w[:, 1] + ly / 2, p_w[:, 1] - ly / 2,
                                      p_w[:, 2], p_w[:, 2] - lz]))
    assert limites.min(axis=1).max() < 1e-9
    assert np.all(np.abs(p_w[:, 0]) <= lx / 2 + 1e-9) and np.all(np.abs(p_w[:, 1]) <= ly / 2 + 1e-9)
    assert np.all(p_w[:, 2] >= -1e-9) and np.all(p_w[:, 2] <= lz + 1e-9)
    # Dentro de una habitación de 6 x 6 x 3 nada está más lejos que su diagonal
    assert profundidad.min() > 0.2 and profundidad.max() < np.linalg.norm(DIMENSIONES)


def test_el_dibujo_esta_pegado_a_la_pared(habitacion):
    # El mismo punto de la pared, visto desde dos poses, tiene el mismo brillo: es lo que
    # hace que seguir un punto de una imagen a otra tenga sentido
    rayos = escena.rayos_camara(K, DIST, TAMANO)
    R_a, t_a = pose_de_prueba(4.0)
    R_b, t_b = pose_de_prueba(4.3)                                      # 0,3 s después
    imagen_a, prof_a = escena.renderizar_habitacion(habitacion, R_a, t_a, rayos)
    imagen_b, _ = escena.renderizar_habitacion(habitacion, R_b, t_b, rayos)
    # Una rejilla de píxeles de A -> punto 3D en el mundo -> píxel (con decimales) en B
    v, u = np.mgrid[40:440:20, 40:600:20]
    u, v = u.ravel(), v.ravel()
    p_c_a = prof_a[v, u][:, None] * rayos[v, u]
    p_w = (p_c_a - t_a) @ R_a
    uv_b = mc.proyectar(p_w @ R_b.T + t_b, K, DIST)
    dentro = (uv_b[:, 0] > 1) & (uv_b[:, 0] < 638) & (uv_b[:, 1] > 1) & (uv_b[:, 1] < 478)
    assert dentro.sum() > 200                                           # las dos vistas se solapan
    brillo_a = imagen_a[v, u][dentro]
    brillo_b = escena.muestrear_bilineal(imagen_b, uv_b[dentro, 0], uv_b[dentro, 1])
    diferencia = np.abs(brillo_a - brillo_b)
    # No es exactamente 0 porque B se lee interpolando entre píxeles, pero casi: la mediana
    # es de milésimas de gris (de 0 a 255)
    assert np.median(diferencia) < 0.1 and np.percentile(diferencia, 90) < 3.0
    # Y se han movido de verdad: sin llevar el punto a B, los brillos no coinciden
    assert np.median(np.abs(brillo_a - imagen_b[v, u][dentro])) > 5.0


def test_acabar_imagen():
    ideal = np.full((100, 100), 120.0)
    cfg = dict(CFG_TEXTURA, desenfoque_px=0.0, ruido_gris=0.0)
    limpia = escena.acabar_imagen(ideal, cfg, np.random.default_rng(0))
    assert limpia.dtype == np.uint8 and np.all(limpia == 120)
    con_ruido = escena.acabar_imagen(ideal, dict(cfg, ruido_gris=3.0), np.random.default_rng(0))
    assert 2.5 < con_ruido.std() < 3.5 and abs(con_ruido.mean() - 120) < 0.2
    # Los valores fuera de rango se recortan en vez de dar la vuelta (300 no pasa a ser 44)
    assert np.all(escena.acabar_imagen(np.full((4, 4), 300.0), cfg, np.random.default_rng(0)) == 255)


# =====================================================================
# Paso 20.4 — Grabación con textura, de principio a fin
# =====================================================================
def test_vio_sobre_grabacion_con_textura(tmp_path):
    # La misma prueba de la etapa 13 (test_pipeline), pero con la habitación dibujada y la
    # distorsión de la cámara real: el front-end FAST + Lucas-Kanade y el MSCKF, sin cambios,
    # tienen que dar el mismo error
    import cv2

    from apps.pi_vio import ADELANTO_IMU_NS, eventos_grabacion
    from captura import euroc
    from herramientas.comparar_trayectoria import alinear
    from herramientas.grabar_simulacion import guardar_grabacion_sintetica
    from vio.pipeline import PipelineVio

    cfgs = {n: cargar_config(n) for n in ("vio", "frontend", "msckf", "triangulacion")}
    ext = cargar_config("camara_imu")
    R_c_i = np.array(ext["R_i_c"], dtype=float).T
    t_c_i = -R_c_i @ np.array(ext["p_i_c_m"], dtype=float)
    cfg_vio = copy.deepcopy(CFG_SIM)
    cfg_vio["trayectoria"]["duracion_s"] = 8.0
    s = guardar_grabacion_sintetica(tmp_path, cfg_vio, K, TAMANO, ext, np.random.default_rng(3),
                                    dist=DIST, textura=True)

    # Las imágenes son fotos en gris de 640 x 480, con dibujo en toda la imagen
    _, nombres = euroc.leer_cam(euroc.ruta_cam(tmp_path))
    foto = cv2.imread(str(euroc.carpeta_imagenes(tmp_path) / nombres[100]), cv2.IMREAD_GRAYSCALE)
    assert len(nombres) == 161 and foto.shape == (480, 640) and foto.std() > 20.0
    assert "habitacion" in s

    pipeline = PipelineVio(K, DIST, R_c_i, t_c_i, ext["t_d_ns"], cfgs)
    resultados = []
    for evento in eventos_grabacion(tmp_path, ADELANTO_IMU_NS):
        if evento[0] == "imu":
            pipeline.nueva_imu(*evento[1:])
        else:
            r = pipeline.nueva_imagen(*evento[1:])
            if r is not None:
                resultados.append(r)

    assert len(resultados) >= 8 * 20 - 2 * 20 - 2
    t_est = np.array([r["t_ns"] for r in resultados])
    p_est = np.array([r["p_w"] for r in resultados])
    R_est = np.array([r["R_w_i"] for r in resultados])
    p_ver, _ = alinear(t_est, p_est, R_est, s["verdad"]["t_ns"], s["verdad"]["p_w"], s["verdad"]["R_w_i"])
    error = np.linalg.norm(p_est - p_ver, axis=1)
    assert error.max() < 0.15 and error[-1] < 0.1             # los mismos límites que con manchas
    assert np.mean([r["frontend"]["activos"] for r in resultados]) > 60   # puntos seguidos a la vez
    assert sum(r["msckf"]["usadas"] for r in resultados) > 100