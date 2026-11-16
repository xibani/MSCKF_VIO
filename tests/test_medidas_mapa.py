"""tests/test_medidas_mapa.py — Etapa 24: la profundidad verdadera de la grabación simulada y
las medidas del mapa disperso (vio/medidas_mapa.py, herramientas/medir_mapa.py).

La grabación de prueba la hace la fixture sesion_con_textura de tests/conftest.py, que no
depende de config/camara_imu.yaml.
"""

import cv2
import numpy as np
import pytest

from captura import euroc
from herramientas import medir_mapa
from vio import medidas_mapa as mm
from vio import modelo_camara as mc
from vio.config import cargar_config

K = mc.matriz_K(643.0, 643.0, 320.0, 240.0)
DIST = np.array([0.168, -0.59, -0.0073, 0.0004, 0.955])


# =====================================================================
# Paso 24.1 — La profundidad verdadera, en disco
# =====================================================================
def test_profundidad_a_milimetros_y_vuelta():
    prof = np.array([[0.5, 1.2345, 64.999], [np.nan, -1.0, 80.0]])
    mm_16 = euroc.profundidad_a_mm(prof)
    assert mm_16.dtype == np.uint16 and mm_16.tolist() == [[500, 1234, 64999], [0, 0, 0]]
    vuelta = euroc.mm_a_profundidad(mm_16)
    assert np.allclose(vuelta[0], [0.5, 1.234, 64.999])           # resolución de 1 mm
    assert np.isnan(vuelta[1]).all()                              # sin dato, negativo y demasiado lejos


def test_leer_profundidad(tmp_path):
    assert euroc.leer_profundidad(tmp_path, 123) is None          # la grabación no trae profundidad
    carpeta = euroc.carpeta_profundidad(tmp_path)
    carpeta.mkdir(parents=True)
    prof = np.full((480, 640), 2.5)
    prof[0, 0] = np.nan
    cv2.imwrite(str(carpeta / euroc.nombre_imagen(123)), euroc.profundidad_a_mm(prof))
    leida = euroc.leer_profundidad(tmp_path, 123)
    assert leida.shape == (480, 640) and leida[10, 10] == 2.5 and np.isnan(leida[0, 0])
    assert euroc.leer_profundidad(tmp_path, 124) is None          # otra imagen


# =====================================================================
# Paso 24.2 — Las cuentas
# =====================================================================
def test_muestrear():
    v, u = np.mgrid[0:5, 0:6]
    plano = (2.0 * u + 10.0 * v).astype(float)                    # en un plano, la interpolación es exacta
    uv = np.array([[1.25, 2.5], [0.0, 0.0], [4.99, 3.99], [5.0, 1.0], [-0.1, 1.0], [2.0, 4.0]])
    valores = mm.muestrear(plano, uv)
    assert np.allclose(valores[:3], [2.0 * 1.25 + 10.0 * 2.5, 0.0, 2.0 * 4.99 + 10.0 * 3.99])
    assert np.isnan(valores[3:]).all()                            # en el borde o fuera: sin vecinos
    con_hueco = plano.copy()
    con_hueco[2, 2] = np.nan
    assert np.isnan(mm.muestrear(con_hueco, np.array([[1.5, 1.5]]))[0])        # un vecino sin dato
    assert np.isfinite(mm.muestrear(con_hueco, np.array([[3.5, 3.5]]))[0])
    assert mm.muestrear(plano, np.zeros((0, 2))).shape == (0,)


def test_pixeles_de_keyframe():
    # Ida y vuelta: píxeles -> coordenadas normalizadas sin distorsión (lo que guarda el
    # keyframe) -> píxeles
    uv = np.array([[320.0, 240.0], [100.5, 80.25], [600.0, 400.0], [20.0, 460.0]])
    kf = {"xy": mc.normalizar(uv, K, DIST), "z": np.ones(4), "K": K, "dist": DIST}
    assert np.abs(mm.pixeles_de_keyframe(kf) - uv).max() < 0.01
    vacio = {"xy": np.zeros((0, 2)), "z": np.zeros(0), "K": K, "dist": DIST}
    assert mm.pixeles_de_keyframe(vacio).shape == (0, 2)


def test_resumen_profundidades():
    r = mm.resumen_profundidades(np.arange(1.0, 12.0))            # 1, 2, ..., 11
    assert r == {"n": 11, "mediana": 6.0, "p10": 2.0, "p90": 10.0}
    vacio = mm.resumen_profundidades(np.zeros(0))
    assert vacio["n"] == 0 and np.isnan(vacio["mediana"])


def test_comparar_con_verdad():
    rng = np.random.default_rng(0)
    z_verdad = rng.uniform(1.0, 5.0, 2000)
    z = 1.05 * z_verdad * (1.0 + rng.normal(0.0, 0.02, 2000))     # 5 % más grande, 2 % de ruido
    z_verdad[:100] = np.nan                                       # 100 puntos sin verdad
    v = mm.comparar_con_verdad(z, z_verdad)
    assert v["n"] == 1900
    assert v["escala"] == pytest.approx(1.05, abs=0.003)
    assert v["dispersion"] == pytest.approx(0.021, abs=0.003)     # 2 % de 1,05
    assert v["dentro_10"] == pytest.approx(0.99, abs=0.01)        # casi todos entre 0,9 y 1,1
    # Un punto muy mal no mueve ni la escala ni la dispersión (mediana y MAD)
    z[200] = 100.0
    assert mm.comparar_con_verdad(z, z_verdad)["escala"] == pytest.approx(v["escala"], abs=0.001)
    nada = mm.comparar_con_verdad(np.ones(3), np.full(3, np.nan))
    assert nada["n"] == 0 and np.isnan(nada["escala"])


def test_error_frente_a_distancia():
    assert mm.error_frente_a_distancia(np.array([1.0, 1.26, 5.0]), 1.2) == pytest.approx(5.0)
    assert mm.error_frente_a_distancia(np.array([0.9]), 1.0) == pytest.approx(-10.0)
    assert np.isnan(mm.error_frente_a_distancia(np.zeros(0), 1.0))


# =====================================================================
# Pasos 24.1 y 24.3 — De la grabación al informe
# =====================================================================
def test_la_grabacion_con_textura_guarda_la_profundidad(sesion_con_textura):
    carpeta = sesion_con_textura["carpeta"]
    t_ns, nombres = euroc.leer_cam(euroc.ruta_cam(carpeta))
    archivos = sorted(p.name for p in euroc.carpeta_profundidad(carpeta).iterdir())
    assert archivos == sorted(nombres) and len(archivos) == 201   # una por imagen, con su nombre
    prof = euroc.leer_profundidad(carpeta, t_ns[100])
    assert prof.shape == (480, 640) and np.isfinite(prof).all()
    # Dentro de una habitación de 6 x 6 x 3 m: ni pegada a la cámara ni más allá de la diagonal
    assert 0.3 < prof.min() and prof.max() < 9.0


def test_medir_el_mapa_del_simulador(sesion_con_textura):
    carpeta = sesion_con_textura["carpeta"]
    mapa = medir_mapa.leer_mapa(carpeta / "mapa.paquetes")
    assert len(mapa.keyframes) >= 8 and len(mapa.disperso) > 200
    medidas = medir_mapa.medir_keyframes(mapa.keyframes, carpeta, 0)
    assert len(medidas) == len(mapa.keyframes)
    for m in medidas:
        assert m["distancias"]["n"] == len(m["z"]) >= cargar_config("vio")["mapeo"]["puntos_min"]
        assert m["z_verdad"].shape == m["z"].shape and m["verdad"]["n"] > 0
        assert 0.9 < m["verdad"]["escala"] < 1.1                  # ningún keyframe se va un 10 %
    # Todos los puntos juntos: la escala del mapa, a menos del 3 % de la verdadera
    z = np.concatenate([m["z"] for m in medidas])
    z_verdad = np.concatenate([m["z_verdad"] for m in medidas])
    v = mm.comparar_con_verdad(z, z_verdad)
    assert v["n"] > 500 and abs(v["escala"] - 1.0) < 0.03
    assert v["dispersion"] < 0.05 and v["dentro_10"] > 0.95

    texto = medir_mapa.informe(medidas, None)
    assert "escala" in texto and "Frente a la verdad" in texto and "pared" not in texto
    assert len(texto.split("\n")) == len(medidas) + 4             # cabecera, keyframes, blanco y 2 de resumen
    medir_mapa.dibujar(carpeta / "medidas_mapa.png", medidas, None)
    assert (carpeta / "medidas_mapa.png").stat().st_size > 10_000


def test_medir_sin_verdad_y_con_una_pared(sesion_con_textura, tmp_path):
    # Como con la placa: el mismo mapa en una carpeta sin profundidad verdadera, y una
    # distancia medida. No hay columnas de escala; sí la del error frente a la pared
    carpeta = sesion_con_textura["carpeta"]
    mapa = medir_mapa.leer_mapa(carpeta / "mapa.paquetes")
    medidas = medir_mapa.medir_keyframes(mapa.keyframes, tmp_path, 0)
    assert all(m["z_verdad"] is None and m["verdad"] is None for m in medidas)
    texto = medir_mapa.informe(medidas, 3.0)
    assert "escala" not in texto and "error pared" in texto and "Frente a la pared a 3.00 m" in texto
    medir_mapa.dibujar(tmp_path / "medidas_mapa.png", medidas, 3.0)
    assert (tmp_path / "medidas_mapa.png").exists()
