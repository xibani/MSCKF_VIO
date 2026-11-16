"""tests/test_densificar.py — Etapa 26: de la profundidad de cada keyframe a una nube densa.

Ninguno necesita PyTorch: pasan también en la Pi. Dos grupos:
  - Con vistas SINTÉTICAS: una cámara colocada a mano dentro de la habitación del simulador,
    con la profundidad exacta que da el trazado de rayos. Ahí se sabe la respuesta.
  - Con la sesión simulada completa (la fixture de la etapa 24): las poses son las del VIO,
    con su error, y la profundidad es la verdadera o una estropeada a propósito.
"""

import cv2
import numpy as np
import pytest

from captura import euroc
from herramientas import medir_mapa
from herramientas import nube_densa
from vio import densificar as den
from vio import escena
from vio import modelo_camara as mc
from vio.config import cargar_config
from vio.reconstruccion import decodificar_gris, leer_ply

CFG = cargar_config("densificar")
DIMENSIONES = (6.0, 6.0, 3.0)
TAMANO = (640, 480)
K = mc.matriz_K(643.0, 643.0, 320.0, 240.0)
DIST = np.array([0.168, -0.59, -0.0073, 0.0004, 0.955])
ABAJO = np.array([0.0, 0.0, -1.0])


def vista_sintetica(posicion, hacia, escala=1.0):
    """Una vista hecha a mano: la cámara en `posicion` mirando en horizontal hacia `hacia`.

    Los ejes de la cámara en el mundo: z hacia donde mira, y hacia abajo, x a la derecha
    (x = y × z). Son las columnas de R_w_c. La profundidad sale del trazado de rayos de la
    etapa 20, exacta; `escala` la multiplica (una vista con la escala mal puesta).
    """
    adelante = np.array(hacia, dtype=float) / np.linalg.norm(hacia)
    derecha = np.cross(ABAJO, adelante)
    R_w_c = np.column_stack([derecha, ABAJO, adelante])
    p_w_c = np.array(posicion, dtype=float)
    habitacion = {"dimensiones": DIMENSIONES, "px_por_m": 10,
                  "texturas": [np.full((int(10 * alto), int(10 * ancho)), 100.0, np.float32)
                               for ancho, alto in (escena.tamano_de_cara(n, DIMENSIONES) for n in range(escena.N_CARAS))]}
    rayos = escena.rayos_camara(K, DIST, TAMANO)
    _, z = escena.renderizar_habitacion(habitacion, R_w_c.T, -R_w_c.T @ p_w_c, rayos)
    kf = {"indice": 0, "tamano": TAMANO, "K": K, "dist": DIST, "R_w_c": R_w_c, "p_w_c": p_w_c}
    return {"kf": kf, "z": z * escala, "gris": np.full((TAMANO[1], TAMANO[0]), 100, np.uint8)}


# =====================================================================
# Paso 26.1 — De un píxel con profundidad a un punto del mundo
# =====================================================================
def test_rejilla_de_pixeles():
    filas, columnas = den.rejilla_de_pixeles((640, 480), 4)
    assert len(filas) == 160 * 120
    assert filas.max() == 476 and columnas.max() == 636 and filas.min() == 0 and columnas.min() == 0
    # Sin píxeles no hay rayos (y OpenCV no se queja)
    assert den.rayos_de_pixeles(filas[:0], columnas[:0], K, DIST).shape == (0, 3)


def test_los_puntos_de_una_vista_caen_en_las_paredes():
    vista = vista_sintetica((1.0, -0.5, 1.4), (1.0, 0.3, 0.0))
    nube = den.nube_de_vista(vista, CFG)
    assert nube["candidatos"] == 160 * 120 and len(nube["p_w"]) == nube["candidatos"]
    distancia, cara = den.distancia_a_la_habitacion(nube["p_w"], DIMENSIONES)
    assert distancia.max() < 1e-3                        # todos sobre una superficie (error < 1 mm)
    assert set(np.unique(cara)) <= {1, 3, 4, 5}          # mirando hacia x+ con algo de y+: esas paredes, suelo y techo
    assert (cara == 1).mean() > 0.5
    assert nube["gris"].dtype == np.uint8 and len(nube["gris"]) == len(nube["p_w"])


def test_profundidad_fuera_de_rango_no_se_usa():
    vista = vista_sintetica((0.0, 0.0, 1.5), (1.0, 0.0, 0.0))
    vista["z"][:240] = np.nan                            # la mitad de arriba, sin dato
    vista["z"][240:, :320] = 0.1                         # un cuarto, demasiado cerca
    cfg = dict(CFG, bordes={"salto_relativo": 1e9, "margen_px": 0})       # ningún salto es bastante grande
    nube = den.nube_de_vista(vista, cfg)
    assert nube["sin_dato"] == 60 * 160 + 60 * 80
    # Del cuarto que queda, la primera fila toca la zona sin dato: cuenta como borde
    assert nube["en_borde"] == 80 and len(nube["p_w"]) == 60 * 80 - 80


# =====================================================================
# Paso 26.2 — Los saltos de profundidad
# =====================================================================
def test_mascara_de_bordes():
    # Una pared inclinada: la profundidad crece un 0,2 % por píxel. No hay saltos
    z = np.tile(2.0 * 1.002 ** np.arange(100), (50, 1))
    assert den.mascara_de_bordes(z, 0.03, 2).all()
    # Delante, un objeto a 1 m que ocupa las columnas 40 a 59: dos saltos
    z[:, 40:60] = 1.0
    lisos = den.mascara_de_bordes(z, 0.03, 2)
    # Fuera los dos píxeles del salto y 2 de margen a cada lado: columnas 37-42 y 57-62
    assert not lisos[:, 37:43].any() and not lisos[:, 57:63].any()
    assert lisos[:, :37].all() and lisos[:, 43:57].all() and lisos[:, 63:].all()
    # Sin margen, solo los dos píxeles de cada salto
    assert (~den.mascara_de_bordes(z, 0.03, 0)).sum() == 50 * 4
    # Un píxel sin dato cuenta como salto con sus vecinos
    z = np.full((20, 20), 2.0)
    z[10, 10] = np.nan
    lisos = den.mascara_de_bordes(z, 0.03, 0)
    assert (~lisos).sum() == 5 and not lisos[10, 10] and not lisos[9, 10] and not lisos[10, 11]


# =====================================================================
# Paso 26.3 — Cada punto, comprobado desde los demás keyframes
# =====================================================================
def test_campo_de_vision():
    x_min, x_max, y_min, y_max = den.campo_de_vision(K, DIST, TAMANO)
    # Sin distorsión sería ±320/643 = ±0,498 y ±240/643 = ±0,373; la lente lo cambia poco
    assert x_min == pytest.approx(-0.5, abs=0.03) and x_max == pytest.approx(0.5, abs=0.03)
    assert y_min == pytest.approx(-0.373, abs=0.02) and y_max == pytest.approx(0.373, abs=0.02)


def test_comprobar_desde_otra():
    # Dos cámaras a 40 cm una de otra, mirando a la misma pared
    una = vista_sintetica((0.0, 0.0, 1.5), (1.0, 0.0, 0.0))
    otra = vista_sintetica((0.0, 0.4, 1.5), (1.0, 0.05, 0.0))
    p_w = den.nube_de_vista(una, CFG)["p_w"]
    pared = den.distancia_a_la_habitacion(p_w, DIMENSIONES)[1] == 1         # los puntos de la pared x+
    centro = np.array([0.0, 0.2, 1.5])                                       # entre las dos cámaras

    # 1. Los puntos tal cual: la otra confirma los que caen dentro de su imagen (ve la pared
    #    desplazada 40 cm y algo girada: comparten unas cuatro quintas partes) y no contradice ninguno
    confirma, contradice = den.comprobar_desde_otra(p_w[pared], otra, 0.05)
    assert 0.75 < confirma.mean() < 0.85 and not contradice.any()
    # 2. Acercados un 20 % hacia las cámaras: flotan delante de la pared. La otra ve la pared
    #    a través de ellos: los contradice
    flotando = centro + 0.8 * (p_w[pared] - centro)
    confirma, contradice = den.comprobar_desde_otra(flotando, otra, 0.05)
    assert not confirma.any() and contradice.mean() > 0.75
    # 3. Alejados un 20 %: quedan detrás de la pared. La otra no puede decir nada
    detras = centro + 1.2 * (p_w[pared] - centro)
    confirma, contradice = den.comprobar_desde_otra(detras, otra, 0.05)
    assert not confirma.any() and not contradice.any()
    # 4. Puntos a la espalda de la otra cámara: tampoco
    confirma, contradice = den.comprobar_desde_otra(np.array([[-2.0, 0.4, 1.5], [-2.9, 0.0, 1.0]]), otra, 0.05)
    assert not confirma.any() and not contradice.any()


def test_puntos_coherentes_quita_la_vista_con_la_escala_mal():
    # Tres vistas buenas de la misma pared y una cuarta con la profundidad un 15 % larga
    vistas = [vista_sintetica((0.0, y, 1.5), (1.0, 0.0, 0.0)) for y in (-0.15, 0.0, 0.15)]
    vistas.append(vista_sintetica((0.0, 0.05, 1.45), (1.0, 0.0, 0.0), escala=1.15))
    nubes = [den.nube_de_vista(v, CFG) for v in vistas]
    se_quedan = den.puntos_coherentes(nubes, vistas, CFG["coherencia"])
    # La vista mala ve "más lejos" por los rayos de los puntos buenos: los contradice. Pero a
    # cada punto bueno lo confirman las otras dos buenas (donde se solapan): 2 contra 1
    assert all(queda.mean() > 0.8 for queda in se_quedan[:3])
    assert se_quedan[3].mean() < 0.02                    # sus puntos quedan detrás de la pared: nadie los confirma


# =====================================================================
# Paso 26.4 — Un punto por cada cubo ocupado
# =====================================================================
def test_fundir_en_voxeles():
    # Cubos de 10 cm. Tres puntos en el cubo (0, 0, 0), dos en el (1, 0, 0) y uno suelto
    p = np.array([[0.01, 0.02, 0.03], [0.05, 0.05, 0.05], [0.09, 0.08, 0.07],
                  [0.12, 0.02, 0.02], [0.18, 0.04, 0.06], [-0.55, 0.3, 0.3]])
    gris = np.array([10, 20, 30, 100, 200, 77], dtype=np.uint8)
    q, brillo, cuentas = den.fundir_en_voxeles(p, gris, 0.10, 2)
    orden = np.argsort(q[:, 0])
    assert len(q) == 2 and list(cuentas[orden]) == [3, 2]
    assert q[orden[0]] == pytest.approx([0.05, 0.05, 0.05]) and q[orden[1]] == pytest.approx([0.15, 0.03, 0.04])
    assert list(brillo[orden]) == [20, 150]
    # Con puntos_min = 1 el suelto también cuenta
    assert len(den.fundir_en_voxeles(p, gris, 0.10, 1)[0]) == 3
    assert len(den.fundir_en_voxeles(p[:0], gris[:0], 0.10, 1)[0]) == 0


def test_densificar_con_vistas_sinteticas():
    vistas = [vista_sintetica((0.5 * np.cos(a), 0.5 * np.sin(a), 1.5), (np.cos(a + 0.3), np.sin(a + 0.3), 0.0))
              for a in np.linspace(0.0, 1.2, 5)]
    r = den.densificar(vistas, CFG)
    assert len(r["por_vista"]) == 5
    for v in r["por_vista"]:
        assert v["candidatos"] == v["sin_dato"] + v["en_borde"] + v["sin_confirmar"] + v["usados"]
    distancia, _ = den.distancia_a_la_habitacion(r["p_w"], DIMENSIONES)
    # Casi todos sobre una superficie. En las esquinas, un cubo junta puntos de dos paredes y
    # su media queda un poco separada de las dos (como mucho, medio cubo)
    assert len(r["p_w"]) > 3000 and np.median(distancia) < 1e-3 and distancia.max() < 0.03
    assert r["cuentas"].min() >= CFG["voxel"]["puntos_min"]
    assert len(den.densificar([], CFG)["p_w"]) == 0


# =====================================================================
# Paso 26.5 — Comparar con la habitación de verdad
# =====================================================================
def test_distancia_a_la_habitacion():
    p = np.array([[0.0, 0.0, 0.2],        # dentro, a 20 cm del suelo
                  [2.9, 0.0, 1.5],        # dentro, a 10 cm de la pared x+
                  [-3.5, 0.0, 1.5],       # fuera, 50 cm detrás de la pared x-
                  [0.0, 3.3, 3.4],        # fuera por dos lados: 30 cm de y+ y 40 cm del techo -> 50 cm
                  [1.0, -2.8, 2.95]])     # dentro, a 5 cm del techo
    distancia, cara = den.distancia_a_la_habitacion(p, DIMENSIONES)
    assert distancia == pytest.approx([0.2, 0.1, 0.5, 0.5, 0.05])
    assert list(cara[[0, 1, 2, 4]]) == [4, 1, 0, 5]


def test_transformacion_a_la_verdad():
    # Una trayectoria "verdadera" y la misma vista desde otro mundo (girado 40° y desplazado)
    rng = np.random.default_rng(0)
    t_ns = np.arange(50, dtype=np.int64) * 50_000_000
    p_ver = np.cumsum(rng.normal(0.0, 0.05, (50, 3)), axis=0) + [1.5, 0.0, 1.5]
    R_ver = np.array([cv2.Rodrigues(rng.normal(0.0, 0.3, 3))[0] for _ in range(50)])
    giro = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(40.0)]))[0]
    desplazamiento = np.array([0.7, -0.2, 1.1])
    p_est = (p_ver - desplazamiento) @ giro              # p_est = giroᵀ·(p_ver - desplazamiento)
    R_est = np.array([giro.T @ R for R in R_ver])
    # La estimada empieza más tarde (el filtro arranca tras el reposo)
    R, t = den.transformacion_a_la_verdad(t_ns[10:], p_est[10:], R_est[10:], t_ns, p_ver, R_ver)
    assert R == pytest.approx(giro, abs=1e-9) and t == pytest.approx(desplazamiento, abs=1e-9)
    assert p_est @ R.T + t == pytest.approx(p_ver, abs=1e-9)


def test_evaluar_nube():
    # Puntos cada 5 cm sobre media pared x+ (y de -3 a 0, toda la altura), 2 cm por delante
    y, z = np.meshgrid(np.arange(-2.975, 0.0, 0.05), np.arange(0.025, 3.0, 0.05))
    p = np.column_stack([np.full(y.size, 2.98), y.ravel(), z.ravel()])
    e = den.evaluar_nube(p, DIMENSIONES, CFG["evaluacion"])
    assert e["n"] == len(p) and e["error_mediano"] == pytest.approx(0.02) and e["cerca"] == 1.0
    assert e["cubierto_por_cara"][1] == pytest.approx(0.5) and sum(e["cubierto_por_cara"]) == pytest.approx(0.5)
    # La pared x+ es 6 x 3 = 18 de los 144 m² de la habitación: media pared, 9 / 144
    assert e["cubierto"] == pytest.approx(9.0 / 144.0)
    # Los del centro de esa media pared, 20 cm por delante: ni son buenos ni cubren nada
    centro = (p[:, 1] < -0.5) & (p[:, 1] > -2.5) & (p[:, 2] > 0.5) & (p[:, 2] < 2.5)
    e = den.evaluar_nube(p[centro] - [0.18, 0.0, 0.0], DIMENSIONES, CFG["evaluacion"])
    assert e["error_mediano"] == pytest.approx(0.2) and e["cerca"] == 0.0 and e["cubierto"] == 0.0
    assert den.evaluar_nube(p[:0], DIMENSIONES, CFG["evaluacion"])["n"] == 0


# =====================================================================
# Con la sesión simulada completa (poses del VIO)
# =====================================================================
@pytest.fixture(scope="module")
def sesion(sesion_con_textura):
    """La sesión de la etapa 24 con lo que hace falta aquí: keyframes, vistas con la
    profundidad verdadera y la transformación al mundo del simulador."""
    carpeta = sesion_con_textura["carpeta"]
    mapa = medir_mapa.leer_mapa(carpeta / "mapa.paquetes")
    vistas = nube_densa.cargar_vistas(mapa.keyframes, carpeta, True, sesion_con_textura["t_d_ns"])
    R, t = nube_densa.transformacion_de_la_sesion(carpeta)
    dimensiones = cargar_config("simulador")["vio"]["mundo"]["habitacion_m"]
    return {"carpeta": carpeta, "keyframes": mapa.keyframes, "vistas": vistas, "R": R, "t": t,
            "dimensiones": dimensiones}


def test_nube_de_la_sesion_con_la_profundidad_verdadera(sesion):
    assert len(sesion["vistas"]) == len(sesion["keyframes"]) >= 8
    r = den.densificar(sesion["vistas"], CFG)
    e = den.evaluar_nube(r["p_w"] @ sesion["R"].T + sesion["t"], sesion["dimensiones"], CFG["evaluacion"])
    # Con la profundidad perfecta solo queda el error de las poses del VIO: un par de centímetros
    assert e["n"] > 5000 and e["error_mediano"] < 0.03 and e["cerca"] > 0.9
    usados = sum(v["usados"] for v in r["por_vista"])
    assert usados > 0.8 * sum(v["candidatos"] for v in r["por_vista"])
    assert e["cubierto"] > 0.10                          # 10 s no dan para ver la habitación entera


def test_nube_de_la_sesion_con_la_profundidad_estropeada(sesion):
    # Cada keyframe con su escala (±3 %) y un error suave por zonas (3 %), como una red ya
    # ajustada a metros: el error no cambia de un píxel al siguiente
    rng = np.random.default_rng(1)
    vistas = []
    for vista in sesion["vistas"]:
        alto, ancho = vista["z"].shape
        zonas = cv2.resize(rng.normal(0.0, 0.03, (alto // 80 + 2, ancho // 80 + 2)).astype(np.float32),
                           (ancho, alto), interpolation=cv2.INTER_CUBIC)
        z = vista["z"] * (1.0 + rng.normal(0.0, 0.03)) * (1.0 + zonas)
        vistas.append({"kf": vista["kf"], "z": z, "gris": vista["gris"]})
    r = den.densificar(vistas, CFG)
    e = den.evaluar_nube(r["p_w"] @ sesion["R"].T + sesion["t"], sesion["dimensiones"], CFG["evaluacion"])
    # Un 4 % de error a unos 3 m son unos 12 cm: la nube lo refleja, y no más
    assert e["n"] > 5000 and 0.03 < e["error_mediano"] < 0.15
    # La comprobación entre keyframes ha quitado puntos (con la verdadera casi no quita)
    sin_confirmar = sum(v["sin_confirmar"] for v in r["por_vista"])
    assert sin_confirmar > 0.1 * sum(v["candidatos"] for v in r["por_vista"])


def test_la_herramienta_escribe_la_nube(sesion, tmp_path):
    r = den.densificar(sesion["vistas"], CFG)
    e = den.evaluar_nube(r["p_w"] @ sesion["R"].T + sesion["t"], sesion["dimensiones"], CFG["evaluacion"])
    texto = nube_densa.informe(r, e, CFG["evaluacion"]["cerca_m"])
    assert f"Nube densa: {len(r['p_w'])} puntos" in texto and "Frente a la habitación de verdad" in texto
    assert "suelo" in texto and len(texto.splitlines()) == len(sesion["vistas"]) + 6
    assert "Frente a" not in nube_densa.informe(r, None, 0.05)
    # Los archivos
    assert nube_densa.nombre_de_salida("nube_densa", "ply", False) == "nube_densa.ply"
    assert nube_densa.nombre_de_salida("nube_densa", "ply", True) == "nube_densa_verdad.ply"
    nube_densa.guardar_ply(tmp_path / "nube.ply", r["p_w"], np.column_stack([r["gris"]] * 3))
    p, colores = leer_ply(tmp_path / "nube.ply")
    assert p == pytest.approx(r["p_w"], abs=1e-5) and (colores[:, 0] == r["gris"]).all()
    nube_densa.dibujar(tmp_path / "nube.png", r["p_w"], r["gris"], sesion["keyframes"])
    assert (tmp_path / "nube.png").stat().st_size > 10000


def test_cargar_vistas_con_la_profundidad_de_la_etapa_25(sesion, tmp_path):
    # Sin la carpeta profundidad/ no hay vistas; con el PNG de un keyframe, una
    carpeta = sesion["carpeta"]
    assert nube_densa.cargar_vistas(sesion["keyframes"], tmp_path, False, 0) == []
    kf = sesion["keyframes"][2]
    (tmp_path / "profundidad").mkdir()
    cv2.imwrite(str(tmp_path / "profundidad" / f"{kf['indice']:06d}.png"),
                euroc.profundidad_a_mm(sesion["vistas"][2]["z"]))
    vistas = nube_densa.cargar_vistas(sesion["keyframes"], tmp_path, False, 0)
    assert len(vistas) == 1 and vistas[0]["kf"]["indice"] == kf["indice"]
    assert vistas[0]["z"] == pytest.approx(sesion["vistas"][2]["z"], abs=1e-3)       # milímetros
    assert vistas[0]["gris"].shape == (480, 640) and (vistas[0]["gris"] == decodificar_gris(kf)).all()
    # Y una sesión que no es del simulador no tiene transformación a la verdad
    assert nube_densa.transformacion_de_la_sesion(tmp_path) is None
    assert nube_densa.transformacion_de_la_sesion(carpeta) is not None
