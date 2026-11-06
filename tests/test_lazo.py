"""tests/test_lazo.py — Etapa 16: pruebas de trayectoria (lazo cerrado y recta medida).

    vio/lazo.py (métricas) · apps/visor_web.py (las mismas cifras en vivo, botón "Nueva prueba")
    apps/pi_vio.py (orden de reiniciar) · herramientas/informe_lazo.py (informe posterior)
"""

import copy
import http.client
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from apps import pi_vio, visor_web
from vio import lazo, protocolo, so3
from vio.config import RAIZ_PROYECTO, cargar_config
from vio.pipeline import CABECERA_POSES, linea_pose

CFG = cargar_config("vio")["prueba"]


def cuadrado(lado, puntos_por_lado):
    """Lazo cuadrado en el suelo que sale de (0, 0, 0) y vuelve a él. Devuelve: (N, 3)."""
    esquinas = np.array([[0, 0, 0], [lado, 0, 0], [lado, lado, 0], [0, lado, 0], [0, 0, 0]], float)
    tramos = [np.linspace(esquinas[k], esquinas[k + 1], puntos_por_lado, endpoint=False) for k in range(4)]
    return np.vstack(tramos + [esquinas[-1:]])


# =====================================================================
# Paso 16.1 — vio/lazo.py
# =====================================================================
def test_recorrido_no_se_infla_con_el_temblor():
    # 2000 poses quietas con temblor de 2 mm (como el VIO parado) y después un cuadrado de 2 m
    temblor = np.random.default_rng(0).normal(0.0, 0.002, (2000, 3))
    p = np.vstack([temblor, cuadrado(2.0, 200)])
    ingenuo = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    assert ingenuo > 10.0                                      # sumando todo: ¡más de 10 m!
    assert lazo.recorrido(p, CFG["paso_min_m"]) == pytest.approx(8.0, rel=0.02)


def test_metricas_lazo():
    p = cuadrado(2.0, 200)
    p[-1] = [0.06, -0.08, 0.0]                                 # vuelve a 10 cm de la salida
    t = np.arange(len(p)) * 0.05
    m = lazo.metricas_lazo(t, p, CFG["paso_min_m"], CFG["recorrido_min_m"])
    assert m["cierre_m"] == pytest.approx(0.10)
    assert m["cierre_pct"] == pytest.approx(100 * 0.10 / m["recorrido_m"])
    assert m["recorrido_m"] == pytest.approx(8.0, rel=0.02)
    assert m["deriva_m"] == pytest.approx([0.06, -0.08, 0.0])
    assert m["alejamiento_max_m"] == pytest.approx(2.0 * np.sqrt(2.0), rel=1e-3)   # la esquina opuesta
    assert m["duracion_s"] == pytest.approx(t[-1])
    # Si apenas se ha movido, el % no tiene sentido
    corto = lazo.metricas_lazo(t[:20], p[:20], CFG["paso_min_m"], CFG["recorrido_min_m"])
    assert corto["cierre_pct"] is None


def test_error_escala():
    p = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [5.15, 0.0, 0.0]])
    estimada, error = lazo.error_escala(p, 5.0)
    assert estimada == pytest.approx(5.15) and error == pytest.approx(3.0)