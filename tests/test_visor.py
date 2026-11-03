"""tests/test_visor.py — El visor del PC (apps/pc_visor.py) con paquetes de las dos versiones."""

import copy
import socket
import time

import matplotlib
matplotlib.use("Agg")                     # sin ventana: se dibuja en memoria
import numpy as np

from apps import pc_visor
from vio import protocolo, so3
from vio.config import cargar_config


def test_visor_recibe_y_dibuja_las_dos_versiones():
    cfg = copy.deepcopy(cargar_config("orientacion"))
    cfg["red"]["puerto"] = 0                                  # un puerto libre cualquiera
    visor = pc_visor.Visor(cfg)
    destino = ("127.0.0.1", visor.receptor.getsockname()[1])
    emisor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 1. Versión 1: solo orientación
        emisor.sendto(protocolo.empaquetar(1, so3.exp_so3([0.0, 0.0, 0.5])), destino)
        time.sleep(0.1)
        visor.dibujar(0)
        assert visor.version == 1 and visor.p is None and visor.paquetes == 1
        # 2. Versión 2: una vuelta completa de un círculo
        for k in range(50):
            angulo = 2 * np.pi * k / 50
            p = np.array([np.cos(angulo), np.sin(angulo), 0.1 * k / 50])
            emisor.sendto(protocolo.empaquetar_pose(k, so3.exp_so3([0.0, 0.0, angulo]), p), destino)
        time.sleep(0.2)
        visor.dibujar(0)
        assert visor.version == 2 and len(visor.trayectoria) == 50
        assert np.array_equal(visor.p, visor.trayectoria[-1])
        assert len(visor.ax_planta.lines) >= 3                # trayectoria, inicio y posición actual
    finally:
        emisor.close()
        visor.receptor.close()


def test_limites_contienen_la_trayectoria():
    tr = np.array([[0.0, 0.0, 0.0], [4.0, 1.0, 0.5]])
    centro, semilado = pc_visor.limites(tr, 0.5)
    assert np.allclose(centro, [2.0, 0.5, 0.25])
    assert semilado == 2.5                                    # mitad del lado mayor (4) + margen
    assert pc_visor.limites(np.zeros((1, 3)), 0.5)[1] == 1.0  # como mínimo ±1 m