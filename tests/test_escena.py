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