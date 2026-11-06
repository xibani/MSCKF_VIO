"""tests/test_proceso_imu.py — Tests del bucle de grabación de la IMU (sin hardware)."""

import os
import sys
from types import SimpleNamespace

import pytest

from captura import euroc, mpu6050, proceso_imu
from vio.config import cargar_config


class BusSiempreListo:
    """Bus falso: siempre hay muestra nueva y siempre es 1 g en +z."""

    def read_byte_data(self, direccion, registro):
        return 1                                # INT_STATUS con el bit 0 a 1

    def read_i2c_block_data(self, direccion, registro, n):
        return [0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00]


class EventoTrasN:
    """Evento falso: dice "parar" a partir de la consulta número n."""

    def __init__(self, n):
        self.restantes = n

    def is_set(self):
        self.restantes -= 1
        return self.restantes <= 0


def test_bucle_imu_escribe_hasta_que_se_le_dice_parar(tmp_path):
    cfg = cargar_config("imu")
    ajustes = mpu6050.ajustes_desde_config(cfg)
    n_muestras = SimpleNamespace(value=0)       # imita a multiprocessing.Value
    n_errores = SimpleNamespace(value=0)
    ruta = tmp_path / "imu.csv"

    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write(euroc.CABECERA_IMU)
        proceso_imu.bucle_imu(BusSiempreListo(), cfg, ajustes, archivo,
                              EventoTrasN(25), n_muestras, n_errores)

    t_ns, w, a = euroc.leer_imu(ruta)
    assert len(t_ns) == 25 == n_muestras.value
    assert n_errores.value == 0
    assert (t_ns[1:] >= t_ns[:-1]).all()        # marcas que no retroceden
    assert abs(a[:, 2] - 9.80665).max() < 1e-5


# =====================================================================
# Etapa 7.3 — Un solo lector de la IMU a la vez
# =====================================================================
@pytest.mark.skipif(sys.platform == "win32", reason="pgrep: solo en Linux (en la Pi)")
def test_otros_lectores_imu():
    # Un proceso cualquiera con "-m apps.pi_vio" en su línea de órdenes, como el servicio vio-pi
    import subprocess
    import time
    falso = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)", "-m", "apps.pi_vio"])
    try:
        time.sleep(0.3)
        otros = proceso_imu.otros_lectores_imu(["apps.pi_vio", "apps.pi"])
        assert (falso.pid, "apps.pi_vio") in otros
        assert all(modulo != "apps.pi" for _, modulo in otros)    # "apps.pi" no es "apps.pi_vio"
        assert all(pid != os.getpid() for pid, _ in otros)        # uno mismo no cuenta
    finally:
        falso.kill()
        falso.wait()
    assert (falso.pid, "apps.pi_vio") not in proceso_imu.otros_lectores_imu(["apps.pi_vio"])
