"""tests/conftest.py — Lo que comparten varios archivos de tests.

pytest lee este archivo solo: las "fixtures" que se definen aquí las puede pedir cualquier
test de la carpeta, con poner su nombre entre los argumentos.
"""

import copy

import numpy as np
import pytest

from apps.pi_vio import ADELANTO_IMU_NS, eventos_grabacion, paquetes_mapa
from herramientas.grabar_simulacion import guardar_grabacion_sintetica
from vio import modelo_camara as mc
from vio import protocolo
from vio.config import cargar_config
from vio.pipeline import PipelineVio


@pytest.fixture(scope="session")
def sesion_con_textura(tmp_path_factory):
    """Etapa 24: una grabación con textura de 10 s pasada por el VIO, con su mapa.paquetes.

    scope="session": se hace una sola vez para todos los tests que la pidan (tarda lo suyo:
    dibuja 201 imágenes). No depende de config/camara_imu.yaml: usa la extrínseca del
    simulador y ningún desfase de tiempo.
    Devuelve: dict con carpeta, s (la secuencia del simulador), K, dist y t_d_ns.
    """
    cfg_sim = copy.deepcopy(cargar_config("simulador")["vio"])
    cfg_sim["trayectoria"]["duracion_s"] = 10.0
    K = mc.matriz_K(643.0, 643.0, 320.0, 240.0)
    dist = np.array([0.168, -0.59, -0.0073, 0.0004, 0.955])
    ext = {"R_i_c": cfg_sim["camara"]["R_i_c"], "p_i_c_m": cfg_sim["camara"]["p_i_c_m"], "t_d_ns": 0}
    carpeta = tmp_path_factory.mktemp("sesion")
    s = guardar_grabacion_sintetica(carpeta, cfg_sim, K, (640, 480), ext, np.random.default_rng(3),
                                    dist=dist, textura=True)

    cfgs = {n: cargar_config(n) for n in ("vio", "frontend", "msckf", "triangulacion")}
    pipeline = PipelineVio(K, dist, s["R_c_i"], s["t_c_i"], 0, cfgs)
    with open(carpeta / "mapa.paquetes", "wb") as archivo:
        for evento in eventos_grabacion(carpeta, ADELANTO_IMU_NS):
            if evento[0] == "imu":
                pipeline.nueva_imu(*evento[1:])
                continue
            r = pipeline.nueva_imagen(*evento[1:])
            if r is not None:
                for datos in paquetes_mapa(r, K, dist, cfgs["vio"]["mapeo"]):
                    protocolo.escribir_paquete(archivo, datos)
    return {"carpeta": carpeta, "s": s, "K": K, "dist": dist, "t_d_ns": 0}
