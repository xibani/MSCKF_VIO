"""tests/test_pipeline.py — Tests del VIO completo (etapa 13): vio/pipeline.py y sus piezas."""

import queue

import cv2
import numpy as np
import pytest

from captura import proceso_imu
from vio import msckf as ms
from vio import simulador as sim
from vio import so3
from vio.config import cargar_config
from vio.constantes import G
from vio.estado import EstadoImu
from vio.pipeline import PipelineVio, leer_poses, linea_pose, CABECERA_POSES

K = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
CFGS = {nombre: cargar_config(nombre) for nombre in ("vio", "frontend", "msckf", "triangulacion")}
EXT = cargar_config("camara_imu")
R_C_I = np.array(EXT["R_i_c"], dtype=float).T
T_C_I = -R_C_I @ np.array(EXT["p_i_c_m"], dtype=float)


class Contador:
    """Imita a multiprocessing.Value: un objeto con atributo .value."""
    def __init__(self):
        self.value = 0


class Evento:
    """Imita a multiprocessing.Event."""
    def __init__(self, activo=False):
        self.activo = activo

    def is_set(self):
        return self.activo


# =====================================================================
# Paso 13.4
# =====================================================================
def test_poses_ida_y_vuelta(tmp_path):
    R, p = so3.exp_so3([0.1, 0.2, -0.3]), np.array([1.0, -2.0, 0.5])
    ruta = tmp_path / "poses.csv"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(CABECERA_POSES)
        f.write(linea_pose(123, R, p))
        f.write(linea_pose(456, np.eye(3), np.zeros(3)))
    t, p_leida, R_leida = leer_poses(ruta)
    assert list(t) == [123, 456] and t.dtype == np.int64
    assert p_leida[0] == pytest.approx(p, abs=1e-9) and R_leida[0] == pytest.approx(R, abs=1e-9)


def test_tiempo_con_reposo():
    reposo, rampa = 3.0, 2.0
    assert sim.tiempo_con_reposo(1.0, reposo, rampa) == 0.0
    # tau y su derivada (la "velocidad" de la trayectoria) son continuas en los dos cambios
    h = 1e-6
    for t in (reposo, reposo + rampa):
        assert sim.tiempo_con_reposo(t - h, reposo, rampa) == pytest.approx(sim.tiempo_con_reposo(t + h, reposo, rampa), abs=1e-5)
    velocidad = lambda t: (sim.tiempo_con_reposo(t + h, reposo, rampa) - sim.tiempo_con_reposo(t - h, reposo, rampa)) / (2 * h)
    assert velocidad(reposo + 1e-3) == pytest.approx(0.0, abs=1e-4)
    assert velocidad(reposo + rampa / 2) == pytest.approx(0.5, abs=1e-6)      # 3u² - 2u³ en u = 1/2
    assert velocidad(reposo + rampa + 1.0) == pytest.approx(1.0, abs=1e-6)


def test_renderizar_una_mancha():
    imagen = sim.renderizar(np.array([[100.3, 50.7]]), (640, 480), 1.5, 200)
    assert imagen.dtype == np.uint8 and imagen.shape == (480, 640)
    v, u = np.unravel_index(np.argmax(imagen), imagen.shape)
    assert (u, v) == (100, 51)
    # El centro de masas de la mancha está donde se dibujó
    vs, us = np.mgrid[0:480, 0:640]
    peso = imagen.astype(float)
    assert (peso * us).sum() / peso.sum() == pytest.approx(100.3, abs=0.05)
    assert (peso * vs).sum() / peso.sum() == pytest.approx(50.7, abs=0.05)
    # FAST la ve como una esquina
    esquinas = cv2.FastFeatureDetector_create(threshold=20).detect(imagen)
    assert any(abs(k.pt[0] - 100.3) < 1.5 and abs(k.pt[1] - 50.7) < 1.5 for k in esquinas)


def test_bucle_cola():
    muestras = ((k, np.zeros(3), np.zeros(3), 25.0) for k in range(10))
    cola = queue.Queue(maxsize=6)
    perdidas, errores = Contador(), Contador()
    proceso_imu.bucle_cola(muestras, cola, Evento(), perdidas, errores, {"errores_i2c": 2})
    assert cola.qsize() == 6 and perdidas.value == 4 and errores.value == 2
    assert [cola.get()[0] for _ in range(6)] == [0, 1, 2, 3, 4, 5]
    # Con evento_parar activo, termina tras la primera muestra
    cola2 = queue.Queue()
    proceso_imu.bucle_cola(((k, None, None, 0) for k in range(10)), cola2, Evento(True),
                           Contador(), Contador(), {"errores_i2c": 0})
    assert cola2.qsize() == 1


def muestras_en_reposo(R_w_i, b_g, segundos, rng, ruido_gir=0.001):
    """Muestras de IMU quieta: giroscopio = sesgo + ruido, acelerómetro = la gravedad."""
    n = int(segundos * 200)
    t = np.arange(n, dtype=np.int64) * 5_000_000
    w = b_g + rng.normal(0.0, ruido_gir, (n, 3))
    a = np.tile(R_w_i.T @ np.array([0.0, 0.0, G]), (n, 1)) + rng.normal(0.0, 0.01, (n, 3))
    return t, w, a


def test_inicializacion_en_reposo():
    rng = np.random.default_rng(0)
    R_verdad = so3.exp_so3([0.1, -0.05, 0.0])                # inclinada, rumbo 0
    b_g = np.array([0.005, -0.003, 0.004])
    pipeline = PipelineVio(K, DIST, R_C_I, T_C_I, 0, CFGS)
    for t, w, a in zip(*muestras_en_reposo(R_verdad, b_g, 2.1, rng)):
        pipeline.nueva_imu(t, w, a)
    assert pipeline.filtro is not None and pipeline.problemas == []
    e = pipeline.filtro.estado
    # La inclinación: la vertical vista desde la IMU (el rumbo no se puede saber en reposo)
    arriba_estimado, arriba_verdadero = e.R_w_i.T @ [0, 0, 1], R_verdad.T @ [0, 0, 1]
    assert np.arccos(np.clip(arriba_estimado @ arriba_verdadero, -1, 1)) < np.radians(0.1)
    assert e.b_g == pytest.approx(b_g, abs=2e-4)
    assert np.array_equal(e.v_w, np.zeros(3)) and np.array_equal(e.p_w, np.zeros(3))


def test_inicializacion_si_se_mueve_lo_repite():
    rng = np.random.default_rng(1)
    pipeline = PipelineVio(K, DIST, R_C_I, T_C_I, 0, CFGS)
    for t, w, a in zip(*muestras_en_reposo(np.eye(3), np.zeros(3), 2.1, rng, ruido_gir=0.1)):
        pipeline.nueva_imu(t, w, a)
    assert pipeline.filtro is None and pipeline.intentos == 1 and pipeline.problemas
    assert pipeline.nueva_imagen(10**10, np.zeros((480, 640), np.uint8)) is None


def test_corregir_reposo_anula_la_velocidad():
    estado = EstadoImu(np.eye(3), np.array([0.1, -0.05, 0.02]), np.zeros(3), np.zeros(3), np.zeros(3))
    P0 = ms.covarianza_inicial(CFGS["msckf"]["inicial"])
    P0[3:6, 3:6] = np.eye(3) * 0.1 ** 2                       # se duda mucho de la velocidad
    filtro = ms.Msckf(estado, P0, 0, np.zeros(3), np.array([0.0, 0.0, G]), R_C_I, T_C_I,
                      643.0, CFGS["msckf"], CFGS["triangulacion"])
    filtro.corregir_reposo(0.01)
    assert np.linalg.norm(filtro.estado.v_w) < 0.002
    assert np.sqrt(np.diag(filtro.P)[3:6]).max() < 0.011