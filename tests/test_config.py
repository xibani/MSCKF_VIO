"""tests/test_config.py — Tests de vio/config.py"""

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