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



# =====================================================================
# Etapa 14 — Imagen con sus pistas (en otro puerto: no se mezcla con la pose)
# =====================================================================
VERSION_IMAGEN = 3
FORMATO_CABECERA_IMAGEN = "<Bq"                                  # versión + t_ns
TAMANO_CABECERA_IMAGEN = struct.calcsize(FORMATO_CABECERA_IMAGEN)  # 9 bytes
MAX_BYTES_UDP = 65507                                            # carga máxima de un datagrama UDP (IPv4)
MAX_BYTES_JPEG = MAX_BYTES_UDP - TAMANO_CABECERA_IMAGEN


def empaquetar_imagen(t_ns, jpeg):
    """Cabecera (versión 3, t_ns) + los bytes de un JPEG. Lanza ValueError si no cabe."""
    if len(jpeg) > MAX_BYTES_JPEG:
        raise ValueError(f"JPEG de {len(jpeg)} bytes: no cabe en un paquete UDP ({MAX_BYTES_JPEG})")
    return struct.pack(FORMATO_CABECERA_IMAGEN, VERSION_IMAGEN, int(t_ns)) + bytes(jpeg)


def leer_imagen(datos):
    """Inversa de empaquetar_imagen. Devuelve: (t_ns, bytes del JPEG)."""
    if len(datos) <= TAMANO_CABECERA_IMAGEN or datos[0] != VERSION_IMAGEN:
        raise ValueError("no es un paquete de imagen (versión 3)")
    _, t_ns = struct.unpack(FORMATO_CABECERA_IMAGEN, datos[:TAMANO_CABECERA_IMAGEN])
    return t_ns, datos[TAMANO_CABECERA_IMAGEN:]




# =====================================================================
# Etapa 16 — Órdenes del visor a pi_vio (puerto de control)
# =====================================================================
ORDEN_REINICIAR = b"reiniciar"     # empezar una prueba nueva: el VIO vuelve a inicializarse