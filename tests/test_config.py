"""tests/test_config.py — Tests de vio/config.py"""

import numpy as np
import pytest

from vio import config


def test_carga_proyecto():
    cfg = config.cargar_config("proyecto")
    assert isinstance(cfg, dict)
    assert cfg["nombre"] == "VIO-Pi"
    assert cfg["carpeta_datos"] == "datos"


def test_version_es_texto():
    # Entre comillas en el YAML -> str; sin comillas, 0.1 sería un float
    cfg = config.cargar_config("proyecto")
    assert cfg["version"] == "0.1.0"


def test_archivo_inexistente():
    # pytest.raises: el test pasa SOLO si dentro se lanza ese error
    with pytest.raises(FileNotFoundError):
        config.cargar_config("no_existe")


def test_archivo_vacio(tmp_path, monkeypatch):
    # tmp_path: carpeta temporal que pytest crea para este test y luego borra
    # monkeypatch: cambia algo solo mientras dura este test
    (tmp_path / "vacio.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "CARPETA_CONFIG", tmp_path)
    with pytest.raises(ValueError):
        config.cargar_config("vacio")


def test_la_extrinseca_es_un_giro():
    # R_i_c se teclea a mano en config/camara_imu.yaml: un signo cambiado y deja de ser un giro
    # (las columnas ya no son perpendiculares entre sí), y todo lo que la usa da resultados
    # sin sentido. Tolerancia 0,001: el redondeo a 4 decimales da del orden de 0,0001
    R = np.array(config.cargar_config("camara_imu")["R_i_c"], dtype=float)
    assert R.shape == (3, 3)
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-3          # filas de largo 1 y perpendiculares
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-3)  # +1: un giro; -1 sería un espejo
