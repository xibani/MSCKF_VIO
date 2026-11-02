"""tests/test_mpu6050.py — Tests de captura/mpu6050.py (sin hardware)."""

import math

import pytest

from captura import mpu6050
from vio.config import cargar_config


# =====================================================================
# Paso 1.1 — conversión
# =====================================================================
def test_entero_con_signo():
    assert mpu6050.a_entero_con_signo(0x7F, 0xFF) == 32767
    assert mpu6050.a_entero_con_signo(0x80, 0x00) == -32768
    assert mpu6050.a_entero_con_signo(0xFF, 0xFF) == -1
    assert mpu6050.a_entero_con_signo(0x00, 0x00) == 0


def test_convertir_bloque():
    # AZ = 0x2000 = 8192 cuentas = 1 g; GX = 0x0041 = 65; GZ = 0xFFBF = -65; TEMP = 0
    bloque = [0x00, 0x00, 0x00, 0x00, 0x20, 0x00,
              0x00, 0x00,
              0x00, 0x41, 0x00, 0x00, 0xFF, 0xBF]
    w, a, temp = mpu6050.convertir_bloque(bloque, escala_acc=8192.0, escala_gir=65.5)
    assert a == pytest.approx((0.0, 0.0, 9.80665))
    assert w[0] == pytest.approx(math.radians(65 / 65.5))
    assert w[1] == 0.0
    assert w[2] == pytest.approx(-math.radians(65 / 65.5))
    assert temp == pytest.approx(36.53)
    

# =====================================================================
# Bus falso: imita a smbus2.SMBus y apunta todo lo que se le pide
# =====================================================================
class BusFalso:
    def __init__(self, who_am_i=0x68, bloque=None, avisos=None):
        self.who_am_i = who_am_i
        self.bloque = bloque if bloque is not None else [0] * 14
        # Respuestas sucesivas de INT_STATUS (cuando se acaban, siempre 1)
        self.avisos = list(avisos) if avisos is not None else []
        self.escrituras = []           # lista de (registro, valor) escritos

    def read_byte_data(self, direccion, registro):
        if registro == mpu6050.REG_WHO_AM_I:
            return self.who_am_i
        if registro == mpu6050.REG_INT_STATUS:
            return self.avisos.pop(0) if self.avisos else 1
        return 0

    def write_byte_data(self, direccion, registro, valor):
        self.escrituras.append((registro, valor))

    def read_i2c_block_data(self, direccion, registro, n):
        return list(self.bloque[:n])


@pytest.fixture
def sin_esperas(monkeypatch):
    """Anula time.sleep dentro de mpu6050 para que los tests no esperen."""
    monkeypatch.setattr(mpu6050.time, "sleep", lambda segundos: None)

# =====================================================================
# Paso 1.2 — configuración
# =====================================================================
def test_yaml_lee_hexadecimal():
    cfg = cargar_config("imu")
    assert cfg["direccion"] == 0x68 == 104


def test_ajustes_de_la_configuracion_del_proyecto():
    ajustes = mpu6050.ajustes_desde_config(cargar_config("imu"))
    assert ajustes["smplrt_div"] == 4          # 1000 / (1 + 4) = 200 Hz
    assert ajustes["config"] == 3
    assert ajustes["gyro_config"] == 0x08      # ±500 °/s
    assert ajustes["accel_config"] == 0x08     # ±4 g
    assert ajustes["escala_gir"] == 65.5
    assert ajustes["escala_acc"] == 8192.0
    assert ajustes["frecuencia_hz"] == 200.0


@pytest.mark.parametrize("clave, valor", [
    ("frecuencia_hz", 300),      # 1000 / 300 no es entero
    ("dlpf_cfg", 0),             # con 0 la frecuencia interna sería 8 kHz
    ("rango_gir_dps", 300),
    ("rango_acc_g", 3),
])
def test_configuracion_imposible(clave, valor):
    cfg = cargar_config("imu")
    cfg[clave] = valor
    with pytest.raises(ValueError):
        mpu6050.ajustes_desde_config(cfg)


def test_configurar_escribe_los_registros(sin_esperas):
    bus = BusFalso()
    mpu6050.configurar(bus, cargar_config("imu"))
    assert bus.escrituras == [
        (0x6B, 0x80),   # reinicio
        (0x6B, 0x01),   # despertar, reloj del giroscopio X
        (0x1A, 3),      # filtro paso bajo
        (0x19, 4),      # divisor -> 200 Hz
        (0x1B, 0x08),   # ±500 °/s
        (0x1C, 0x08),   # ±4 g
        (0x38, 0x01),   # aviso de dato listo
    ]


def test_who_am_i_incorrecto_no_toca_el_chip(sin_esperas):
    bus = BusFalso(who_am_i=0x70)
    with pytest.raises(RuntimeError):
        mpu6050.configurar(bus, cargar_config("imu"))
    assert bus.escrituras == []
    

# =====================================================================
# Paso 1.3 — lectura continua
# =====================================================================
def test_esperar_muestra_espera_al_aviso():
    bus = BusFalso(avisos=[0, 0, 0, 1])
    mpu6050.esperar_muestra(bus, 0x68)
    assert bus.avisos == []        # ha leído los cuatro estados antes de volver


def test_leer_continuo():
    bloque = [0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
              0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    bus = BusFalso(bloque=bloque, avisos=[0, 1, 0, 0, 1, 1])
    cfg = cargar_config("imu")
    ajustes = mpu6050.ajustes_desde_config(cfg)
    contador = {"errores_i2c": 0}

    muestras = mpu6050.leer_continuo(bus, cfg, ajustes, contador)
    t1, w1, a1, _ = next(muestras)
    t2, w2, a2, _ = next(muestras)
    t3, _, _, _ = next(muestras)

    assert isinstance(t1, int)
    assert t1 <= t2 <= t3
    assert a1 == pytest.approx((0.0, 0.0, 9.80665))
    assert contador["errores_i2c"] == 0


def test_leer_continuo_cuenta_errores_y_sigue():
    class BusQueFallaUnaVez(BusFalso):
        def __init__(self):
            super().__init__()
            self.fallos_pendientes = 1

        def read_i2c_block_data(self, direccion, registro, n):
            if self.fallos_pendientes > 0:
                self.fallos_pendientes -= 1
                raise OSError("error I2C simulado")
            return super().read_i2c_block_data(direccion, registro, n)

    bus = BusQueFallaUnaVez()
    cfg = cargar_config("imu")
    contador = {"errores_i2c": 0}
    muestras = mpu6050.leer_continuo(bus, cfg, mpu6050.ajustes_desde_config(cfg), contador)
    next(muestras)
    assert contador["errores_i2c"] == 1