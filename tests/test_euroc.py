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