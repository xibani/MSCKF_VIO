"""tests/test_emparejador.py — Etapa 21: SuperPoint + LightGlue sobre dos imágenes.

Tres grupos:
  - Sin PyTorch (pasan también en la Pi): la verdad del simulador, la evaluación y el dibujo.
  - Con PyTorch pero con redes FALSAS: que EmparejadorLightGlue convierte bien de NumPy al
    formato de las redes y vuelta. Se saltan si no hay PyTorch.
  - Con las redes de verdad: se saltan si no está instalado lightglue (y la primera vez
    descargan los pesos: hace falta internet).
"""

import numpy as np
import pytest

from vio import emparejador as emp
from vio import escena
from vio import modelo_camara as mc
from vio import simulador as sim
from vio.config import cargar_config

CFG = cargar_config("lightglue")
CFG_SIM = cargar_config("simulador")["vio"]
K = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])
TAMANO = (640, 480)
DIST = np.array([0.168, -0.59, -0.0073, 0.0004, 0.955])


@pytest.fixture(scope="module")
def dos_vistas():
    """Dos vistas de la habitación con textura, 0,5 s y unos 40 cm la una de la otra."""
    rng = np.random.default_rng(7)
    habitacion = escena.crear_habitacion(CFG_SIM["mundo"]["habitacion_m"], CFG_SIM["textura"], rng)
    rayos = escena.rayos_camara(K, DIST, TAMANO)
    R_c_i, t_c_i = sim.extrinseca(CFG_SIM["camara"])
    vistas = []
    for t_s in (4.0, 4.5):
        R_w_i, p_w = sim.pose_vio(t_s, CFG_SIM["trayectoria"])
        pose = sim.pose_camara(R_w_i, p_w, R_c_i, t_c_i)
        ideal, profundidad = escena.renderizar_habitacion(habitacion, pose[0], pose[1], rayos)
        vistas.append({"gris": escena.acabar_imagen(ideal, CFG_SIM["textura"], rng),
                       "ideal": ideal, "profundidad": profundidad, "pose": pose})
    return vistas


# =====================================================================
# Sin PyTorch
# =====================================================================
def test_elegir_dispositivo():
    assert emp.elegir_dispositivo("auto", True) == "cuda"
    assert emp.elegir_dispositivo("auto", False) == "cpu"
    assert emp.elegir_dispositivo("cpu", True) == "cpu"
    with pytest.raises(RuntimeError):                      # se pide la tarjeta y no la hay
        emp.elegir_dispositivo("cuda", False)
    with pytest.raises(ValueError):
        emp.elegir_dispositivo("gpu", True)


def test_posicion_verdadera_misma_camara(dos_vistas):
    # De A a la propia A: cada punto está donde estaba
    a = dos_vistas[0]
    uv = np.array([[100.5, 80.25], [320.0, 240.0], [600.0, 400.0]])
    uv_b, visible = emp.posicion_verdadera_en_b(uv, a["profundidad"], a["pose"], a["pose"], K, DIST)
    assert visible.all() and np.abs(uv_b - uv).max() < 0.01


def test_posicion_verdadera_tiene_el_mismo_brillo(dos_vistas):
    # El punto de A, llevado a B con la verdad, cae sobre el mismo trozo de pared: mismo brillo
    a, b = dos_vistas
    v, u = np.mgrid[40:440:20, 40:600:20]
    uv_a = np.column_stack([u.ravel(), v.ravel()]).astype(float)
    uv_b, visible = emp.posicion_verdadera_en_b(uv_a, a["profundidad"], a["pose"], b["pose"], K, DIST)
    assert 100 < visible.sum() < len(uv_a)                 # unos se ven en B y otros ya no
    assert np.isfinite(uv_b[visible]).all()
    brillo_a = a["ideal"][v.ravel(), u.ravel()][visible]
    brillo_b = escena.muestrear_bilineal(b["ideal"], uv_b[visible, 0], uv_b[visible, 1])
    assert np.median(np.abs(brillo_a - brillo_b)) < 0.1
    # Y se han movido bastante: decenas de píxeles (por eso Lucas-Kanade ya no los sigue)
    assert np.median(np.linalg.norm(uv_b[visible] - uv_a[visible], axis=1)) > 40


def test_posicion_verdadera_detras_de_la_camara(dos_vistas):
    # B mirando hacia el lado contrario que A: el punto queda detrás y no es visible
    a = dos_vistas[0]
    R_a, t_a = a["pose"]
    media_vuelta = np.diag([-1.0, 1.0, -1.0])              # giro de 180° sobre el eje y de la cámara
    pose_b = (media_vuelta @ R_a, media_vuelta @ t_a)
    uv_b, visible = emp.posicion_verdadera_en_b(np.array([[320.0, 240.0]]), a["profundidad"],
                                                a["pose"], pose_b, K, DIST)
    assert not visible[0] and np.isnan(uv_b[0]).all()


def test_evaluar_parejas():
    verdad = np.array([[10.0, 10.0], [50.0, 50.0], [90.0, 90.0], [0.0, 0.0]])
    visible = np.array([True, True, True, False])
    encontrado = verdad + np.array([[0.5, 0.0], [0.0, 2.0], [10.0, 0.0], [0.0, 0.0]])
    ev = emp.evaluar_parejas(encontrado, verdad, visible, umbral_px=3.0)
    assert np.allclose(ev["error_px"][:3], [0.5, 2.0, 10.0]) and np.isinf(ev["error_px"][3])
    assert list(ev["correctas"]) == [True, True, False, False]     # la 4.ª no se ve en B: falsa
    assert ev["n"] == 4 and ev["n_correctas"] == 2 and ev["fraccion_correctas"] == 0.5
    assert ev["error_mediano_px"] == pytest.approx(1.25)
    vacio = emp.evaluar_parejas(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, bool), 3.0)
    assert vacio["n"] == 0 and vacio["fraccion_correctas"] == 0.0 and np.isnan(vacio["error_mediano_px"])


def test_dibujar_parejas():
    gris_a = np.full((480, 640), 50, np.uint8)
    gris_b = np.full((480, 640), 200, np.uint8)
    uv_a = np.array([[100.0, 100.0], [200.0, 300.0]])
    uv_b = np.array([[120.0, 110.0], [220.0, 310.0]])
    lienzo = emp.dibujar_parejas(gris_a, gris_b, uv_a, uv_b, np.array([True, False]))
    assert lienzo.shape == (480, 1280, 3) and lienzo.dtype == np.uint8
    assert tuple(lienzo[100, 100]) == emp.VERDE            # la pareja correcta, en A...
    assert tuple(lienzo[110, 640 + 120]) == emp.VERDE      # ...y en B, desplazada 640 px
    assert tuple(lienzo[300, 200]) == emp.ROJO             # la falsa
    assert tuple(lienzo[5, 5]) == (50, 50, 50) and tuple(lienzo[5, 1275]) == (200, 200, 200)
    sin_verdad = emp.dibujar_parejas(gris_a, gris_b, uv_a, uv_b)
    assert tuple(sin_verdad[100, 100]) == emp.AMARILLO
    # Con muchas parejas solo se dibujan algunas, pero no falla
    muchas = np.column_stack([np.linspace(10, 600, 2000), np.linspace(10, 400, 2000)])
    assert emp.dibujar_parejas(gris_a, gris_b, muchas, muchas, max_lineas=50).shape == (480, 1280, 3)
    with pytest.raises(ValueError):
        emp.dibujar_parejas(gris_a, gris_b[:100], uv_a, uv_b)


def test_fast_klt_falla_donde_hara_falta_emparejar(dos_vistas):
    # El motivo de esta etapa: con 0,5 s entre las dos imágenes los puntos se han movido
    # decenas de píxeles y el front-end actual (FAST + Lucas-Kanade) ya no los sigue
    from herramientas.ver_lightglue import parejas_fast_klt

    a, b = dos_vistas
    uv_a, uv_b, n_detectados = parejas_fast_klt(a["gris"], b["gris"])
    verdad_b, visible = emp.posicion_verdadera_en_b(uv_a, a["profundidad"], a["pose"], b["pose"], K, DIST)
    ev = emp.evaluar_parejas(uv_b, verdad_b, visible, CFG["evaluacion"]["umbral_correcta_px"])
    assert n_detectados > 40 and ev["n_correctas"] < 0.2 * n_detectados


# =====================================================================
# Con PyTorch y redes falsas: la conversión de formatos
# =====================================================================
class ExtractorFalso:
    """Imita a lightglue.SuperPoint: extract(tensor, resize=None) -> dict de tensores por lotes.

    "Detecta" siempre los mismos puntos y guarda lo que recibió, para comprobarlo.
    """

    def __init__(self, torch, puntos):
        self.torch = torch
        self.puntos = puntos
        self.recibido = None
        self.resize = "no se pasó"

    def extract(self, img, **conf):
        torch = self.torch
        self.recibido = img
        self.resize = conf.get("resize", "no se pasó")
        n = len(self.puntos)
        return {
            "keypoints": torch.tensor(self.puntos, dtype=torch.float32)[None],      # (1, N, 2)
            "keypoint_scores": torch.linspace(1.0, 0.5, n)[None],                   # (1, N)
            "descriptors": torch.zeros(1, n, 256),                                  # (1, N, 256)
            "image_size": torch.tensor([[img.shape[-1], img.shape[-2]]], dtype=torch.float32),
        }


class EmparejadorFalso:
    """Imita a lightglue.LightGlue: recibe {"image0", "image1"} y devuelve listas por lote."""

    def __init__(self, torch, parejas, confianzas):
        self.torch = torch
        self.parejas = parejas
        self.confianzas = confianzas
        self.recibido = None

    def __call__(self, datos):
        torch = self.torch
        self.recibido = datos
        return {"matches": [torch.tensor(self.parejas, dtype=torch.int64).reshape(-1, 2)],
                "scores": [torch.tensor(self.confianzas, dtype=torch.float32)],
                "stop": 3}


def test_emparejador_convierte_los_formatos():
    torch = pytest.importorskip("torch")
    puntos = [[10.5, 20.25], [300.0, 400.0], [639.0, 479.0]]
    extractor = ExtractorFalso(torch, puntos)
    falso = EmparejadorFalso(torch, [[0, 2], [2, 1]], [0.9, 0.4])
    emparejador = emp.EmparejadorLightGlue(dict(CFG, dispositivo="cpu"), extractor, falso)
    assert emparejador.dispositivo == "cpu"

    gris = np.zeros((480, 640), np.uint8)
    gris[100, 200] = 255
    a = emparejador.extraer(gris)
    # Lo que le llega a la red: (1, alto, ancho), decimales de 0 a 1 y sin cambiar de tamaño
    assert tuple(extractor.recibido.shape) == (1, 480, 640) and extractor.recibido.dtype == torch.float32
    assert float(extractor.recibido.max()) == 1.0 and float(extractor.recibido[0, 100, 200]) == 1.0
    assert extractor.resize is None
    # Lo que vuelve: NumPy, sin la dimensión del lote
    assert a["puntos"].shape == (3, 2) and a["puntos"].dtype == np.float64
    assert np.allclose(a["puntos"], puntos) and a["puntuaciones"].shape == (3,)
    assert a["tiempo_ms"] >= 0.0

    b = emparejador.extraer(gris)
    indices, confianza, tiempo_ms = emparejador.emparejar(a, b)
    assert falso.recibido["image0"] is a["_red"] and falso.recibido["image1"] is b["_red"]
    assert indices.shape == (2, 2) and indices.tolist() == [[0, 2], [2, 1]]
    assert np.allclose(confianza, [0.9, 0.4]) and tiempo_ms >= 0.0
    # Así se usan: la pareja k une a["puntos"][indices[k, 0]] con b["puntos"][indices[k, 1]]
    assert np.allclose(a["puntos"][indices[:, 0]], [puntos[0], puntos[2]])
    assert np.allclose(b["puntos"][indices[:, 1]], [puntos[2], puntos[1]])


def test_emparejador_sin_puntos_o_sin_parejas():
    torch = pytest.importorskip("torch")
    cfg = dict(CFG, dispositivo="cpu")
    gris = np.zeros((480, 640), np.uint8)
    # Una imagen lisa: ningún punto -> ninguna pareja, sin llamar siquiera a la red
    falso = EmparejadorFalso(torch, [], [])
    sin_puntos = emp.EmparejadorLightGlue(cfg, ExtractorFalso(torch, np.zeros((0, 2))), falso)
    a = sin_puntos.extraer(gris)
    indices, confianza, _ = sin_puntos.emparejar(a, a)
    assert a["puntos"].shape == (0, 2) and indices.shape == (0, 2) and confianza.shape == (0,)
    assert falso.recibido is None
    # Puntos pero ninguna pareja: arrays vacíos con la forma correcta
    con_puntos = emp.EmparejadorLightGlue(cfg, ExtractorFalso(torch, [[1.0, 2.0]]), EmparejadorFalso(torch, [], []))
    a = con_puntos.extraer(gris)
    indices, confianza, _ = con_puntos.emparejar(a, a)
    assert indices.shape == (0, 2) and confianza.shape == (0,)


def test_extraer_rechaza_imagenes_que_no_son_gris_de_8_bits():
    torch = pytest.importorskip("torch")
    emparejador = emp.EmparejadorLightGlue(dict(CFG, dispositivo="cpu"),
                                           ExtractorFalso(torch, [[1.0, 2.0]]), EmparejadorFalso(torch, [], []))
    with pytest.raises(ValueError):
        emparejador.extraer(np.zeros((480, 640, 3), np.uint8))       # en color
    with pytest.raises(ValueError):
        emparejador.extraer(np.zeros((480, 640), np.float32))        # decimales


# =====================================================================
# Con las redes de verdad (PC con lightglue instalado)
# =====================================================================
@pytest.fixture(scope="module")
def redes():
    pytest.importorskip("torch")
    pytest.importorskip("lightglue")
    return emp.EmparejadorLightGlue(CFG)


def test_superpoint_encuentra_puntos(redes, dos_vistas):
    a = redes.extraer(dos_vistas[0]["gris"])
    n = len(a["puntos"])
    assert 200 < n <= CFG["superpoint"]["max_puntos"]
    assert a["puntos"][:, 0].min() >= 0 and a["puntos"][:, 0].max() <= 639
    assert a["puntos"][:, 1].min() >= 0 and a["puntos"][:, 1].max() <= 479
    assert a["puntuaciones"].shape == (n,)


def test_lightglue_una_imagen_consigo_misma(redes, dos_vistas):
    # La prueba más sencilla: cada punto es pareja de sí mismo
    a = redes.extraer(dos_vistas[0]["gris"])
    indices, confianza, _ = redes.emparejar(a, a)
    assert len(indices) > 0.8 * len(a["puntos"])
    assert np.mean(indices[:, 0] == indices[:, 1]) > 0.95
    assert np.median(confianza) > 0.8


def test_lightglue_empareja_donde_klt_no_llega(redes, dos_vistas):
    # Las mismas dos vistas en las que FAST + Lucas-Kanade no sigue casi nada (test de arriba)
    a_vista, b_vista = dos_vistas
    a = redes.extraer(a_vista["gris"])
    b = redes.extraer(b_vista["gris"])
    indices, _, _ = redes.emparejar(a, b)
    uv_a, uv_b = a["puntos"][indices[:, 0]], b["puntos"][indices[:, 1]]
    verdad_b, visible = emp.posicion_verdadera_en_b(uv_a, a_vista["profundidad"], a_vista["pose"],
                                                    b_vista["pose"], K, DIST)
    ev = emp.evaluar_parejas(uv_b, verdad_b, visible, CFG["evaluacion"]["umbral_correcta_px"])
    assert ev["n"] > 50 and ev["fraccion_correctas"] > 0.8
