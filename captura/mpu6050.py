"""
captura/mpu6050.py — Driver mínimo de la IMU MPU-6050 por I2C.

Referencia: InvenSense, "MPU-6000 and MPU-6050 Register Map and Descriptions",
Rev. 4.2. En los comentarios, "registro 25 (0x19)" remite al apartado de ese
documento titulado con ese número de registro.

Este módulo NO abre el bus I2C: todas las funciones reciben un objeto `bus`
(en la Pi, un smbus2.SMBus). Así se puede probar con un bus falso, sin hardware.
"""

import math
import time

import numpy as np

from captura.reloj import ahora_ns

# ---------------------------------------------------------------------
# Direcciones de los registros que usamos
# ---------------------------------------------------------------------
REG_SMPLRT_DIV   = 0x19   # registro 25: divisor de la frecuencia de muestreo
REG_CONFIG       = 0x1A   # registro 26: filtro paso bajo digital (DLPF)
REG_GYRO_CONFIG  = 0x1B   # registro 27: rango del giroscopio
REG_ACCEL_CONFIG = 0x1C   # registro 28: rango del acelerómetro
REG_INT_ENABLE   = 0x38   # registro 56: qué eventos activan el aviso
REG_INT_STATUS   = 0x3A   # registro 58: bit 0 = hay muestra nueva
REG_ACCEL_XOUT_H = 0x3B   # registro 59: primer byte del bloque de datos
REG_PWR_MGMT_1   = 0x6B   # registro 107: encendido y fuente de reloj
REG_WHO_AM_I     = 0x75   # registro 117: identificador del chip

# ---------------------------------------------------------------------
# Valores de bits
# ---------------------------------------------------------------------
PWR_RESET       = 0x80    # PWR_MGMT_1, bit 7 (DEVICE_RESET): reinicio completo
PWR_RELOJ_GIR_X = 0x01    # PWR_MGMT_1: despierto (SLEEP = 0), reloj = giroscopio X
INT_DATA_RDY    = 0x01    # INT_ENABLE / INT_STATUS, bit 0: dato listo

# Rangos -> (valor de FS_SEL / AFS_SEL, cuentas por unidad). Registros 27 y 28.
RANGOS_GIR = {250: (0, 131.0), 500: (1, 65.5), 1000: (2, 32.8), 2000: (3, 16.4)}   # cuentas por °/s
RANGOS_ACC = {2: (0, 16384.0), 4: (1, 8192.0), 8: (2, 4096.0), 16: (3, 2048.0)}    # cuentas por g

FRECUENCIA_INTERNA_HZ = 1000.0  # frecuencia interna con el DLPF activo (DLPF_CFG de 1 a 6)
N_BYTES_BLOQUE = 14             # acel. (6) + temperatura (2) + gir. (6)
T_ESPERA_S = 0.1                # espera tras reiniciar o despertar el chip
G = 9.80665                     # gravedad estándar [m/s²], para pasar de g a m/s²


# =====================================================================
# Paso 1.1 — Conversión de bytes (funciones puras)
# =====================================================================
def a_entero_con_signo(alto, bajo):
    """Une dos bytes en un entero de 16 bits con signo (complemento a dos).

    alto, bajo: int entre 0 y 255; el byte alto va primero (big-endian).
    Devuelve: int entre -32768 y 32767.
    """
    valor = alto * 256 + bajo
    if valor >= 32768:
        valor = valor - 65536
    return valor


def convertir_bloque(bloque, escala_acc, escala_gir):
    """Convierte el bloque de 14 bytes del chip a unidades del SI.

    bloque: lista de 14 int, en el orden AX AY AZ TEMP GX GY GZ (2 bytes cada uno).
    escala_acc: cuentas por g.  escala_gir: cuentas por °/s.
    Devuelve: (w_rad_s, a_m_s2, temp_c)
        w_rad_s: tupla de 3 float, velocidad angular [rad/s]
        a_m_s2:  tupla de 3 float, fuerza específica [m/s²]
        temp_c:  float, temperatura del chip [°C]
    """
    # Acelerómetro: cuentas -> g -> m/s²
    ax = a_entero_con_signo(bloque[0], bloque[1]) / escala_acc * G
    ay = a_entero_con_signo(bloque[2], bloque[3]) / escala_acc * G
    az = a_entero_con_signo(bloque[4], bloque[5]) / escala_acc * G

    # Temperatura: fórmula del registro 65-66
    temp_c = a_entero_con_signo(bloque[6], bloque[7]) / 340.0 + 36.53

    # Giroscopio: cuentas -> °/s -> rad/s
    gx = math.radians(a_entero_con_signo(bloque[8], bloque[9]) / escala_gir)
    gy = math.radians(a_entero_con_signo(bloque[10], bloque[11]) / escala_gir)
    gz = math.radians(a_entero_con_signo(bloque[12], bloque[13]) / escala_gir)

    return (gx, gy, gz), (ax, ay, az), temp_c


# =====================================================================
# Paso 1.2 — Configuración y lectura de una muestra
# =====================================================================
def ajustes_desde_config(cfg):
    """Traduce la configuración (unidades físicas) a valores de registro.

    cfg: dict leído de config/imu.yaml.
    Devuelve: dict con los bytes a escribir y las escalas de conversión.
    Lanza ValueError si algún valor no es posible en el chip.
    """
    # 1. Filtro paso bajo: solo con DLPF_CFG entre 1 y 6 la frecuencia interna es 1 kHz
    dlpf = cfg["dlpf_cfg"]
    if dlpf < 1 or dlpf > 6:
        raise ValueError(f"dlpf_cfg debe estar entre 1 y 6 (es {dlpf})")
    
    # 2. Divisor de muestreo: frecuencia = 1000 / (1 + SMPLRT_DIV)
    f_pedida = cfg["frecuencia_hz"]
    divisor = round(FRECUENCIA_INTERNA_HZ / f_pedida) - 1
    f_real = FRECUENCIA_INTERNA_HZ / (1 + divisor)
    if divisor < 0 or divisor > 255 or abs(f_real - f_pedida) > 1e-9:
        raise ValueError(f"{f_pedida} Hz no es posible: 1000 / frecuencia_hz debe ser entero")

    # 3. Rangos
    if cfg["rango_gir_dps"] not in RANGOS_GIR:
        raise ValueError(f"rango_gir_dps debe ser uno de {list(RANGOS_GIR)}")
    if cfg["rango_acc_g"] not in RANGOS_ACC:
        raise ValueError(f"rango_acc_g debe ser uno de {list(RANGOS_ACC)}")
    fs_sel, escala_gir = RANGOS_GIR[cfg["rango_gir_dps"]]
    afs_sel, escala_acc = RANGOS_ACC[cfg["rango_acc_g"]]

    return {
        "config": dlpf,
        "smplrt_div": divisor,
        "gyro_config": fs_sel << 3,    # FS_SEL ocupa los bits 4 y 3
        "accel_config": afs_sel << 3,  # AFS_SEL ocupa los bits 4 y 3
        "escala_gir": escala_gir,
        "escala_acc": escala_acc,
        "frecuencia_hz": f_real,
    }

def configurar(bus, cfg):
    """Reinicia el chip y lo deja midiendo según cfg.

    bus: objeto con read_byte_data / write_byte_data (smbus2.SMBus en la Pi).
    cfg: dict de config/imu.yaml.
    Devuelve: el dict de ajustes_desde_config (contiene las escalas).
    Lanza RuntimeError si el chip no responde con el WHO_AM_I esperado.
    """
    direccion = cfg["direccion"]

    # 1. Validar la configuración ANTES de tocar el chip
    ajustes = ajustes_desde_config(cfg)

    # 2. ¿Es el chip que esperamos?
    quien = bus.read_byte_data(direccion, REG_WHO_AM_I)
    if quien != cfg["who_am_i"]:
        raise RuntimeError(f"WHO_AM_I = {hex(quien)}, se esperaba {hex(cfg['who_am_i'])}")

    # 3. Reinicio completo y despertar (tras el reinicio el chip queda dormido)
    bus.write_byte_data(direccion, REG_PWR_MGMT_1, PWR_RESET)
    time.sleep(T_ESPERA_S)
    bus.write_byte_data(direccion, REG_PWR_MGMT_1, PWR_RELOJ_GIR_X)
    time.sleep(T_ESPERA_S)

    # 4. Filtro, frecuencia y rangos
    bus.write_byte_data(direccion, REG_CONFIG, ajustes["config"])
    bus.write_byte_data(direccion, REG_SMPLRT_DIV, ajustes["smplrt_div"])
    bus.write_byte_data(direccion, REG_GYRO_CONFIG, ajustes["gyro_config"])
    bus.write_byte_data(direccion, REG_ACCEL_CONFIG, ajustes["accel_config"])

    # 5. Activar el aviso de "dato listo" y borrar uno que pudiera estar pendiente
    bus.write_byte_data(direccion, REG_INT_ENABLE, INT_DATA_RDY)
    bus.read_byte_data(direccion, REG_INT_STATUS)

    return ajustes


def leer_bloque(bus, direccion):
    """Lee los 14 bytes de datos (acel., temperatura, gir.) de una vez.

    Devuelve: lista de 14 int.
    """
    return bus.read_i2c_block_data(direccion, REG_ACCEL_XOUT_H, N_BYTES_BLOQUE)

# =====================================================================
# Paso 1.3 — Lectura continua con marca de tiempo
# =====================================================================

def esperar_muestra(bus, direccion):
    """Pregunta en bucle al chip hasta que tenga una muestra nueva.

    Leer INT_STATUS borra el aviso, así que cada aviso se ve una sola vez.
    Devuelve: instante (ns) en que se detectó la muestra.
    """
    while True:
        estado = bus.read_byte_data(direccion, REG_INT_STATUS)
        if estado & INT_DATA_RDY:      # ¿bit 0 a 1?
            return ahora_ns()


def leer_continuo(bus, cfg, ajustes, contador):
    """Generador: entrega muestras una a una, al ritmo que marca el chip.

    contador: dict con la clave "errores_i2c"; se incrementa cuando una
        lectura falla (la muestra se descarta y se sigue).
    Entrega: (t_ns, w_rad_s, a_m_s2, temp_c) en cada iteración.
    """
    direccion = cfg["direccion"]
    while True:
        try:
            t_ns = esperar_muestra(bus, direccion)
            bloque = leer_bloque(bus, direccion)
        except OSError:
            contador["errores_i2c"] += 1
            continue
        w, a, temp = convertir_bloque(bloque, ajustes["escala_acc"], ajustes["escala_gir"])
        yield t_ns, w, a, temp


def tomar_muestras(muestras, n):
    """Saca n muestras del generador leer_continuo.

    Devuelve: (t_ns (n,) int64, w (n, 3) [rad/s], a (n, 3) [m/s²]).
    """
    t, w, a = [], [], []
    for _ in range(n):
        t_ns, w_k, a_k, _ = next(muestras)
        t.append(t_ns)
        w.append(w_k)
        a.append(a_k)
    return np.array(t, dtype=np.int64), np.array(w), np.array(a)