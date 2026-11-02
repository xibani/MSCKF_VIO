"""tests/test_euroc.py — Tests de captura/euroc.py."""

import numpy as np
import pytest

from captura import euroc


def test_ruta_imu(tmp_path):
    assert euroc.ruta_imu(tmp_path) == tmp_path / "mav0" / "imu0" / "data.csv"


def test_escribir_y_leer(tmp_path):
    ruta = euroc.ruta_imu(tmp_path)
    ruta.parent.mkdir(parents=True)
    # Un tiempo enorme: en float64 perdería las últimas cifras
    tiempos = [9_000_000_000_000_000_123, 9_000_000_000_005_000_123]
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(euroc.CABECERA_IMU)
        f.write(euroc.linea_imu(tiempos[0], (0.1, 0.2, 0.3), (1.0, 2.0, 9.8)))
        f.write(euroc.linea_imu(tiempos[1], (-0.1, 0.0, 0.0), (0.0, 0.0, -9.8)))

    t_ns, w, a = euroc.leer_imu(ruta)

    assert t_ns.dtype == np.int64
    assert t_ns.tolist() == tiempos            # ni un nanosegundo perdido
    assert w.shape == (2, 3) and a.shape == (2, 3)
    assert w[0] == pytest.approx([0.1, 0.2, 0.3])
    assert a[1] == pytest.approx([0.0, 0.0, -9.8])


def test_rutas_de_camara(tmp_path):
    assert euroc.ruta_cam(tmp_path) == tmp_path / "mav0" / "cam0" / "data.csv"
    assert euroc.carpeta_imagenes(tmp_path) == tmp_path / "mav0" / "cam0" / "data"
    assert euroc.nombre_imagen(123456789) == "123456789.png"


def test_escribir_y_leer_camara(tmp_path):
    euroc.carpeta_imagenes(tmp_path).mkdir(parents=True)
    tiempos = [9_000_000_000_000_000_123, 9_000_000_000_050_000_123]
    with open(euroc.ruta_cam(tmp_path), "w", encoding="utf-8") as f_cam, \
         open(euroc.ruta_diag_cam(tmp_path), "w", encoding="utf-8") as f_diag:
        f_cam.write(euroc.CABECERA_CAM)
        f_diag.write(euroc.CABECERA_DIAG_CAM)
        for t in tiempos:
            f_cam.write(euroc.linea_cam(t, euroc.nombre_imagen(t)))
            f_diag.write(euroc.linea_diag_cam(t, t + 30_000_000, 8000, 3.75))

    t_ns, nombres = euroc.leer_cam(euroc.ruta_cam(tmp_path))
    assert t_ns.tolist() == tiempos
    assert nombres == [f"{t}.png" for t in tiempos]

    diag = euroc.leer_diag_cam(euroc.ruta_diag_cam(tmp_path))
    assert diag["t_sensor_ns"].tolist() == tiempos
    assert (diag["t_llegada_ns"] - diag["t_sensor_ns"]).tolist() == [30_000_000, 30_000_000]
    assert diag["exposicion_us"].tolist() == [8000.0, 8000.0]
    assert diag["ganancia"].tolist() == [3.75, 3.75]