"""tests/test_profundidad.py — Etapa 25: profundidad de una sola imagen, en metros.

Tres grupos:
  - Sin PyTorch (pasan también en la Pi): el ajuste a metros, la evaluación y la cadena
    completa con un estimador DE MENTIRA, que en lugar de mirar la imagen devuelve la
    profundidad verdadera del simulador disfrazada como la da la red (sin unidades).
  - Con PyTorch y red falsa: que EstimadorProfundidad convierte bien de NumPy al formato de
    la red y vuelta. Se salta si no hay PyTorch.
  - Con Depth Anything V2 de verdad: se salta si no están PyTorch y transformers, o si no se
    pueden descargar los pesos.
"""

import cv2
import numpy as np
import pytest

from captura import euroc
from herramientas import medir_mapa
from herramientas import ver_profundidad
from vio import profundidad as prof
from vio.config import cargar_config
from vio.reconstruccion import decodificar_gris

CFG = cargar_config("profundidad")
CFG_AJUSTE = CFG["ajuste"]
A_VERDAD, B_VERDAD = 0.004, 0.05        # la recta de mentira: 1/z = 0,004·d + 0,05


def rng():
    return np.random.default_rng(CFG_AJUSTE["semilla"])


def puntos_de_mentira(n, semilla=0, ruido=0.0):
    """n puntos con su profundidad z [m] y el valor d que daría una red con la recta de mentira."""
    generador = np.random.default_rng(semilla)
    z = generador.uniform(0.5, 6.0, n)
    d = (1.0 / z - B_VERDAD) / A_VERDAD
    return d, z * (1.0 + generador.normal(0.0, ruido, n))


class EstimadorDeMentira:
    """Hace de EstimadorProfundidad sin red: sabe la profundidad verdadera de cada imagen.

    verdad_de: dict imagen (sus bytes) -> profundidad verdadera (alto, ancho) [m].
    Devuelve lo que daría una red perfecta: d = (1/z - b) / a, sin unidades, más un poco de
    ruido y un suavizado (las redes no dan bordes tan nítidos como la realidad).
    """

    def __init__(self, verdad_de, ruido=0.0, suavizado_px=0.0, semilla=0):
        self.verdad_de = verdad_de
        self.ruido = ruido
        self.suavizado_px = suavizado_px
        self.rng = np.random.default_rng(semilla)
        self.n_llamadas = 0

    def estimar(self, gris):
        self.n_llamadas += 1
        z = self.verdad_de[gris.tobytes()]
        d = (1.0 / z - B_VERDAD) / A_VERDAD
        if self.suavizado_px > 0:
            d = cv2.GaussianBlur(d, (0, 0), self.suavizado_px)
        d = d * (1.0 + self.rng.normal(0.0, self.ruido, d.shape))
        return {"inversa": d.astype(np.float32), "tiempo_ms": 0.0}


@pytest.fixture(scope="module")
def keyframes_con_verdad(sesion_con_textura):
    """Los keyframes del mapa de la sesión simulada y un estimador de mentira que los conoce."""
    carpeta = sesion_con_textura["carpeta"]
    mapa = medir_mapa.leer_mapa(carpeta / "mapa.paquetes")
    verdad_de = {}
    for kf in mapa.keyframes:
        verdad_de[decodificar_gris(kf).tobytes()] = euroc.leer_profundidad(carpeta, kf["t_ns"])
    return carpeta, mapa.keyframes, verdad_de


# =====================================================================
# Paso 25.3 — El ajuste a metros
# =====================================================================
def test_ajuste_exacto():
    d, z = puntos_de_mentira(60)
    ajuste = prof.ajustar_a_metros(d, z, CFG_AJUSTE, rng())
    assert ajuste["a"] == pytest.approx(A_VERDAD, rel=1e-6) and ajuste["b"] == pytest.approx(B_VERDAD, rel=1e-6)
    assert ajuste["encajan"].all() and ajuste["error_mediano"] < 1e-9 and not ajuste["solo_escala"]


def test_ajuste_con_ruido_y_puntos_malos():
    # 2 % de ruido en la profundidad del VIO y uno de cada tres puntos, disparatado
    d, z = puntos_de_mentira(90, semilla=1, ruido=0.02)
    malos = np.arange(0, 90, 3)
    z[malos] = z[malos] * np.random.default_rng(2).uniform(1.5, 3.0, len(malos))
    ajuste = prof.ajustar_a_metros(d, z, CFG_AJUSTE, rng())
    buenos = np.setdiff1d(np.arange(90), malos)
    assert ajuste["encajan"][buenos].mean() > 0.95 and not ajuste["encajan"][malos].any()
    # La recta recuperada da la profundidad verdadera con menos de un 3 % de error
    z_verdad = np.array([0.5, 1.0, 2.0, 4.0, 6.0])
    d_prueba = (1.0 / z_verdad - B_VERDAD) / A_VERDAD
    z_recta = 1.0 / (ajuste["a"] * d_prueba + ajuste["b"])
    assert np.abs(z_recta / z_verdad - 1.0).max() < 0.03


def test_ajuste_solo_con_la_escala():
    # Todos los puntos del VIO casi a la misma distancia (la cámara mira de frente a una
    # pared a 2,9 m): no se pueden separar a y b, y se ajusta solo la escala
    generador = np.random.default_rng(5)
    z = generador.uniform(2.85, 2.95, 60)
    d = (1.0 / z - B_VERDAD) / A_VERDAD * (1.0 + generador.normal(0.0, 0.03, 60))    # la red, con 3 % de ruido
    ajuste = prof.ajustar_a_metros(d, z, CFG_AJUSTE, rng())
    assert ajuste["solo_escala"] and ajuste["b"] == 0.0 and ajuste["encajan"].mean() > 0.95
    # A la distancia de la pared acierta; lejos de ella se nota que falta la b (aquí, un 7 % a 4 m)
    en_la_pared = 1.0 / (ajuste["a"] * (1.0 / 2.9 - B_VERDAD) / A_VERDAD)
    a_4_m = 1.0 / (ajuste["a"] * (1.0 / 4.0 - B_VERDAD) / A_VERDAD)
    assert en_la_pared == pytest.approx(2.9, rel=0.02) and 0.03 < abs(a_4_m / 4.0 - 1.0) < 0.12
    # Con los mismos puntos repartidos entre 1 y 5 m, vuelve a salir la recta entera
    d, z = puntos_de_mentira(60, semilla=6, ruido=0.02)
    assert not prof.ajustar_a_metros(d, z, CFG_AJUSTE, rng())["solo_escala"]


def test_ajuste_que_no_se_puede_hacer():
    d, z = puntos_de_mentira(60)
    assert prof.ajustar_a_metros(d[:5], z[:5], CFG_AJUSTE, rng()) is None          # pocos puntos
    assert prof.ajustar_a_metros(np.full(60, 100.0), z, CFG_AJUSTE, rng()) is None  # la red no distingue nada
    assert prof.ajustar_a_metros(-d, z, CFG_AJUSTE, rng()) is None                  # la red, al revés
    # Profundidades del VIO sin relación con la red: no encaja ni la mitad
    al_azar = np.random.default_rng(3).uniform(0.5, 6.0, 60)
    assert prof.ajustar_a_metros(d, al_azar, CFG_AJUSTE, rng()) is None


def test_error_relativo_y_a_metros():
    d = np.array([[237.5, 112.5], [0.0, -20.0]])           # 1/z = 1; 0,5; 0,05; -0,03 con la recta de mentira
    z = prof.a_metros(d, A_VERDAD, B_VERDAD, CFG_AJUSTE["inversa_min"])
    assert z.dtype == np.float32 and z.shape == (2, 2)
    assert z[0].tolist() == pytest.approx([1.0, 2.0]) and z[1, 0] == pytest.approx(20.0)
    assert np.isnan(z[1, 1])                                # inversa negativa: detrás de la cámara
    assert np.isnan(prof.a_metros(d, A_VERDAD, B_VERDAD, 0.1)[1, 0])    # 20 m con el límite en 10 m
    error = prof.error_relativo(d.ravel(), np.array([1.0, 2.2, 20.0, 5.0]), A_VERDAD, B_VERDAD, 0.02)
    assert error[:3] == pytest.approx([0.0, 0.2 / 2.2, 0.0]) and np.isinf(error[3])


# =====================================================================
# Paso 25.4 — La evaluación
# =====================================================================
def test_evaluar_profundidad():
    z_verdad = np.full((10, 10), 2.0)
    z_verdad[0] = np.nan                                    # 10 píxeles sin verdad
    z = np.full((10, 10), 2.2, dtype=np.float32)            # 10 % de más
    z[1] = 4.0                                              # 10 píxeles al doble
    z[2] = np.nan                                           # 10 píxeles sin estimación
    ev = prof.evaluar_profundidad(z, z_verdad, 1.25)
    assert ev["n"] == 80 and ev["cubierto"] == pytest.approx(80 / 90)
    assert ev["error_rel_mediano"] == pytest.approx(0.1, abs=1e-6)
    assert ev["error_rel_medio"] == pytest.approx((70 * 0.1 + 10 * 1.0) / 80, abs=1e-6)
    assert ev["delta"] == pytest.approx(70 / 80)
    nada = prof.evaluar_profundidad(np.full((2, 2), np.nan), z_verdad[:2, :2], 1.25)
    assert nada["n"] == 0 and np.isnan(nada["delta"])


# =====================================================================
# Paso 25.5 — La cadena completa, con el estimador de mentira
# =====================================================================
def test_profundidad_de_keyframe_con_una_red_perfecta(keyframes_con_verdad):
    # Con una red perfecta, el único error es el de los puntos del VIO que fijan la escala:
    # mide cuánto vale la etapa 24 como ancla
    carpeta, keyframes, verdad_de = keyframes_con_verdad
    estimador = EstimadorDeMentira(verdad_de)
    generador = rng()
    errores = []
    for kf in keyframes:
        gris = decodificar_gris(kf)
        r = prof.profundidad_de_keyframe(kf, gris, estimador, CFG, generador)
        assert r["ajuste"] is not None and r["z"].shape == gris.shape and r["z"].dtype == np.float32
        assert r["n_puntos"] == len(kf["z"])
        assert r["ajuste"]["encajan"].mean() > 0.9
        assert isinstance(r["ajuste"]["solo_escala"], bool)
        ev = prof.evaluar_profundidad(r["z"], verdad_de[gris.tobytes()], 1.25)
        assert ev["cubierto"] > 0.99 and ev["delta"] > 0.99
        errores.append(ev["error_rel_mediano"])
    assert np.median(errores) < 0.03 and max(errores) < 0.08   # la escala, a un 3 % en el keyframe típico


def test_un_keyframe_con_pocos_puntos_se_queda_sin_profundidad(keyframes_con_verdad):
    _, keyframes, verdad_de = keyframes_con_verdad
    kf = dict(keyframes[0])
    kf["ids"], kf["xy"], kf["z"] = kf["ids"][:5], kf["xy"][:5], kf["z"][:5]
    r = prof.profundidad_de_keyframe(kf, decodificar_gris(kf), EstimadorDeMentira(verdad_de), CFG, rng())
    assert r["ajuste"] is None and r["z"] is None and r["n_puntos"] == 5
    assert r["inversa"].shape == (480, 640)                 # la salida de la red sí está


def test_ver_profundidad_con_una_red_que_se_equivoca_un_poco(keyframes_con_verdad, tmp_path):
    # La herramienta entera, con una red de mentira con 3 % de ruido y bordes suavizados
    carpeta, keyframes, verdad_de = keyframes_con_verdad
    estimador = EstimadorDeMentira(verdad_de, ruido=0.03, suavizado_px=4.0)
    filas = ver_profundidad.procesar_keyframes(keyframes, carpeta, estimador, CFG, 0, tmp_path)
    assert len(filas) == len(keyframes) == estimador.n_llamadas and all(f["ajustado"] for f in filas)
    assert np.median([f["error_rel_mediano"] for f in filas]) < 0.05
    assert np.mean([f["delta"] for f in filas]) > 0.97

    # Lo que queda en disco: un PNG de 16 bits en milímetros por keyframe, su vista y el CSV
    kf = keyframes[3]
    z = euroc.mm_a_profundidad(cv2.imread(str(tmp_path / f"{kf['indice']:06d}.png"), cv2.IMREAD_UNCHANGED))
    ev = prof.evaluar_profundidad(z, verdad_de[decodificar_gris(kf).tobytes()], 1.25)
    assert z.shape == (480, 640) and ev["error_rel_mediano"] == pytest.approx(filas[3]["error_rel_mediano"], abs=1e-3)
    assert cv2.imread(str(tmp_path / f"{kf['indice']:06d}_vista.jpg")).shape == (480, 1280, 3)
    lineas = (tmp_path / "profundidad.csv").read_text(encoding="utf-8").strip().split("\n")
    assert lineas[0] + "\n" == ver_profundidad.CABECERA_CSV and len(lineas) == len(keyframes) + 1
    assert len(lineas[1].split(",")) == len(lineas[0].split(","))
    texto = ver_profundidad.informe(filas)
    assert f"{len(keyframes)} de {len(keyframes)} keyframes" in texto and "Frente a la verdad" in texto


def test_vista():
    gris = np.full((480, 640), 80, np.uint8)
    z = np.full((480, 640), np.nan, dtype=np.float32)
    assert (ver_profundidad.vista(gris, z)[:, 640:] == 0).all()             # sin dato: negro
    z[:, :320] = 1.0                                                        # izquierda cerca...
    z[:, 320:600] = 5.0                                                     # ...derecha lejos, y una franja sin dato
    imagen = ver_profundidad.vista(gris, z)
    assert imagen.shape == (480, 1280, 3) and tuple(imagen[10, 10]) == (80, 80, 80)
    assert imagen[240, 640 + 100].sum() > imagen[240, 640 + 400].sum()      # cerca, más claro que lejos
    assert (imagen[:, 640 + 600:] == 0).all()


# =====================================================================
# Con PyTorch y una red falsa: la conversión de formatos
# =====================================================================
class ProcesadorFalso:
    """Imita al procesador de transformers: recibe la imagen en color y la devuelve como tensor
    (1, 3, alto', ancho'), aquí a la mitad de tamaño."""

    def __init__(self, torch):
        self.torch = torch
        self.recibido = None

    def __call__(self, images, return_tensors):
        self.recibido = images
        assert return_tensors == "pt"
        tensor = self.torch.from_numpy(images.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        return {"pixel_values": tensor[:, :, ::2, ::2]}


class RedFalsa:
    """Imita a la red: devuelve un objeto con predicted_depth (1, alto', ancho'). Aquí, el
    brillo de la imagen: así se sabe qué tiene que salir."""

    def __init__(self):
        self.recibido = None

    def __call__(self, pixel_values):
        self.recibido = pixel_values

        class Salida:
            predicted_depth = pixel_values[:, 0] * 100.0
        return Salida()


def test_estimador_convierte_los_formatos():
    torch = pytest.importorskip("torch")
    procesador, red = ProcesadorFalso(torch), RedFalsa()
    estimador = prof.EstimadorProfundidad(dict(CFG, dispositivo="cpu"), procesador, red)
    assert estimador.dispositivo == "cpu"
    gris = np.zeros((480, 640), np.uint8)
    gris[:, 320:] = 255                                     # mitad izquierda negra, derecha blanca
    r = estimador.estimar(gris)
    # Lo que le llega al procesador: la imagen en color (alto, ancho, 3), con el gris repetido
    assert procesador.recibido.shape == (480, 640, 3) and procesador.recibido.dtype == np.uint8
    assert (procesador.recibido[:, :, 0] == gris).all() and (procesador.recibido[:, :, 2] == gris).all()
    assert tuple(red.recibido.shape) == (1, 3, 240, 320)
    # Lo que vuelve: NumPy, al tamaño de la imagen original aunque la red trabajara a la mitad
    assert r["inversa"].shape == (480, 640) and r["inversa"].dtype == np.float32
    assert abs(r["inversa"][240, 100]) < 1.0 and abs(r["inversa"][240, 540] - 100.0) < 1.0
    assert r["tiempo_ms"] >= 0.0
    with pytest.raises(ValueError):
        estimador.estimar(np.zeros((480, 640, 3), np.uint8))


# =====================================================================
# Con Depth Anything V2 de verdad (PC con transformers y los pesos descargados)
# =====================================================================
@pytest.fixture(scope="module")
def red_de_verdad():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        return prof.EstimadorProfundidad(CFG)
    except OSError as error:                               # sin internet y sin pesos en el disco
        pytest.skip(f"no se han podido cargar los pesos de Depth Anything V2: {error}")


def test_depth_anything_sobre_un_keyframe(red_de_verdad, keyframes_con_verdad):
    # Solo se comprueba que la red funciona y que la cadena no se rompe con su salida. Lo bien
    # que acierte con las paredes del simulador no se exige: se mide con ver_profundidad
    _, keyframes, _ = keyframes_con_verdad
    kf = keyframes[len(keyframes) // 2]
    gris = decodificar_gris(kf)
    r = red_de_verdad.estimar(gris)
    assert r["inversa"].shape == gris.shape and r["inversa"].dtype == np.float32
    assert np.isfinite(r["inversa"]).all() and r["inversa"].std() > 0     # distingue cerca de lejos
    resultado = prof.profundidad_de_keyframe(kf, gris, red_de_verdad, CFG, rng())
    assert resultado["n_puntos"] == len(kf["z"])
    if resultado["ajuste"] is not None:
        assert resultado["z"].shape == gris.shape and np.nanmin(resultado["z"]) > 0
