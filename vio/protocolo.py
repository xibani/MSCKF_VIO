"""
vio/protocolo.py — Formato de los paquetes UDP que la Pi envía al PC.

Versión 1: orientación.  '<Bq9d' = versión (1 byte) + t_ns (int64) + R_w_i (9 float64)
    '<'  : orden little-endian, sin relleno entre campos
    'B'  : entero sin signo de 1 byte
    'q'  : entero con signo de 8 bytes
    '9d' : 9 números decimales de 8 bytes (la matriz, fila a fila)
"""

import struct

import numpy as np

VERSION = 1
FORMATO = "<Bq9d"
TAMANO = struct.calcsize(FORMATO)     # 1 + 8 + 72 = 81 bytes


def empaquetar(t_ns, R_w_i):
    """Convierte (t_ns, R_w_i) en bytes listos para enviar."""
    return struct.pack(FORMATO, VERSION, int(t_ns), *np.asarray(R_w_i, dtype=float).ravel())


def desempaquetar(datos):
    """Inversa de empaquetar. Devuelve: (t_ns, R_w_i (3, 3)).

    Lanza ValueError si el paquete no tiene el tamaño o la versión esperados.
    """
    if len(datos) != TAMANO:
        raise ValueError(f"paquete de {len(datos)} bytes, se esperaban {TAMANO}")
    campos = struct.unpack(FORMATO, datos)
    if campos[0] != VERSION:
        raise ValueError(f"versión de protocolo {campos[0]}, se esperaba {VERSION}")
    t_ns = campos[1]
    R_w_i = np.array(campos[2:]).reshape(3, 3)
    return t_ns, R_w_i