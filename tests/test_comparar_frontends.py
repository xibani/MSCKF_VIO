"""tests/test_comparar_frontends.py — Etapa 23: herramientas/comparar_frontends.py, sin redes."""

import copy

import numpy as np
import pytest

from herramientas import comparar_frontends as cf
from herramientas.grabar_simulacion import guardar_grabacion_sintetica
from vio import modelo_camara as mc
from vio.config import cargar_config
from vio.frontend import FrontEnd, Pista
from vio.frontend_lightglue import FrontEndLightGlue
from vio.pipeline import leer_poses

K = mc.matriz_K(643.0, 643.0, 320.0, 240.0)
DIST = np.zeros(5)


def test_tabla():
    resumenes = {"klt": {"imagenes": 160, "activos": 123.4, "error_final_cm": 1.34},
                 "lg_giro": {"imagenes": 160, "activos": 147.0}}
    lineas = cf.tabla(resumenes).split("\n")
    assert lineas[0].split() == ["FAST+KLT", "LG", "giro"]
    assert lineas[1].startswith("imágenes con pose") and lineas[1].split()[-2:] == ["160", "160"]
    assert lineas[2].split()[-2:] == ["123", "147"]
    # Una medida que solo tiene una variante: guion en la otra. Las que no tiene nadie, fuera
    assert lineas[3].startswith("error de posición final") and lineas[3].split()[-2:] == ["1.3", "-"]
    assert len(lineas) == 4


def test_crear_frontend():
    assert isinstance(cf.crear_frontend("klt", K, DIST), FrontEnd)
    falso = object()                                   # las redes no se usan al crear el front-end
    esencial = cf.crear_frontend("lg_esencial", K, DIST, falso)
    giro = cf.crear_frontend("lg_giro", K, DIST, falso)
    assert isinstance(giro, FrontEndLightGlue) and giro.emparejador is falso
    assert esencial.cfg["geometria"]["metodo"] == "esencial" and giro.cfg["geometria"]["metodo"] == "giro"
    assert cargar_config("lightglue")["geometria"]["metodo"] in ("esencial", "giro")


def test_resumir_sin_resultados():
    assert cf.resumir([], []) == {"imagenes": 0}


def test_comparar_con_fast_klt(tmp_path):
    # Una grabación de manchas de 8 s por el VIO completo, con todas las imágenes y con una
    # de cada 2: mismas medidas, la mitad de imágenes
    cfg_vio = copy.deepcopy(cargar_config("simulador")["vio"])
    cfg_vio["trayectoria"]["duracion_s"] = 8.0
    guardar_grabacion_sintetica(tmp_path, cfg_vio, K, (640, 480), cargar_config("camara_imu"),
                                np.random.default_rng(3))
    verdad = leer_poses(tmp_path / "verdad.csv")

    resumenes = {}
    for cada in (1, 2):
        frontend = cf.crear_frontend("klt", K, DIST)
        resultados, pistas = cf.ejecutar_vio(tmp_path, frontend, cada, K, DIST)
        assert all(isinstance(p, Pista) for p in pistas)
        resumenes[cada] = cf.resumir(resultados, pistas, verdad)

    todas, mitad = resumenes[1], resumenes[2]
    assert todas["imagenes"] >= 8 * 20 - 2 * 20 - 2
    assert mitad["imagenes"] == pytest.approx(todas["imagenes"] / 2, abs=1)
    assert todas["error_max_cm"] < 15.0 and todas["error_final_cm"] < 10.0      # los límites de la etapa 13
    assert todas["usadas"] > 100 and todas["utiles"] <= todas["pistas"]
    # Las pistas son las terminadas (perdidas o rechazadas) más las que seguían vivas al acabar
    vivas_al_acabar = todas["pistas"] - todas["perdidas"] - todas["ransac"]
    assert 0 < vivas_al_acabar <= cargar_config("frontend")["deteccion"]["max_puntos"]
    assert set(todas) == {clave for clave, _, _ in cf.FILAS}                     # todas las filas de la tabla
    assert "error_final_cm" not in cf.resumir(resultados, pistas)                # sin verdad, sin error
    assert "FAST+KLT" in cf.tabla({"klt": todas})
