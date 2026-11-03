"""
vio/protocolo.py — Formato de los paquetes UDP que la Pi envía al PC.

Versión 1: orientación.  '<Bq9d' = versión (1 byte) + t_ns (int64) + R_w_i (9 float64)
    '<'  : orden little-endian, sin relleno entre campos
    'B'  : entero sin signo de 1 byte
    'q'  : entero con signo de 8 bytes
    '9d' : 9 números decimales de 8 bytes (la matriz, fila a fila)
Versión 2 (etapa 13): pose.  '<Bq9d3d' = lo mismo + p_w (3 float64)
El primer byte dice la versión: el visor sabe qué recibe antes de leer el resto.
"""

import struct

import numpy as np

VERSION = 1
FORMATO = "<Bq9d"
TAMANO = struct.calcsize(FORMATO)     # 1 + 8 + 72 = 81 bytes

VERSION_POSE = 2
FORMATO_POSE = "<Bq9d3d"
TAMANO_POSE = struct.calcsize(FORMATO_POSE)     # 81 + 24 = 105 bytes


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


# =====================================================================
# Etapa 13 — Versión 2: orientación y posición
# =====================================================================
def empaquetar_pose(t_ns, R_w_i, p_w):
    """Convierte (t_ns, R_w_i, p_w) en un paquete de la versión 2 (105 bytes)."""
    return struct.pack(FORMATO_POSE, VERSION_POSE, int(t_ns),
                       *np.asarray(R_w_i, dtype=float).ravel(), *np.asarray(p_w, dtype=float))


def leer_paquete(datos):
    """Lee un paquete de cualquier versión. Devuelve: dict con version, t_ns, R_w_i, p_w.

    En la versión 1 no hay posición: p_w es None.
    Lanza ValueError si la versión es desconocida o el tamaño no corresponde a ella.
    """
    if len(datos) == 0:
        raise ValueError("paquete vacío")
    version = datos[0]                              # el primer byte, como entero
    if version == VERSION:
        t_ns, R_w_i = desempaquetar(datos)
        return {"version": version, "t_ns": t_ns, "R_w_i": R_w_i, "p_w": None}
    if version == VERSION_POSE:
        if len(datos) != TAMANO_POSE:
            raise ValueError(f"paquete de {len(datos)} bytes, se esperaban {TAMANO_POSE}")
        campos = struct.unpack(FORMATO_POSE, datos)
        return {"version": version, "t_ns": campos[1],
                "R_w_i": np.array(campos[2:11]).reshape(3, 3), "p_w": np.array(campos[11:14])}
    raise ValueError(f"versión de protocolo {version} desconocida")