"""tests/test_proceso_imu.py — Tests del bucle de grabación de la IMU (sin hardware)."""

from types import SimpleNamespace

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