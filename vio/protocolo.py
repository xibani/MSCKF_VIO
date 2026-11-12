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
TAMANO = struct.calcsize(FORMATO)  # 1 + 8 + 72 = 81 bytes

VERSION_POSE = 2
FORMATO_POSE = "<Bq9d3d"
TAMANO_POSE = struct.calcsize(FORMATO_POSE)  # 81 + 24 = 105 bytes


def empaquetar(t_ns, R_w_i):
    """Convierte (t_ns, R_w_i) en bytes listos para enviar."""
    return struct.pack(
        FORMATO, VERSION, int(t_ns), *np.asarray(R_w_i, dtype=float).ravel()
    )


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
    return struct.pack(
        FORMATO_POSE,
        VERSION_POSE,
        int(t_ns),
        *np.asarray(R_w_i, dtype=float).ravel(),
        *np.asarray(p_w, dtype=float),
    )


def leer_paquete(datos):
    """Lee un paquete de cualquier versión. Devuelve: dict con version, t_ns, R_w_i, p_w.

    En la versión 1 no hay posición: p_w es None.
    Lanza ValueError si la versión es desconocida o el tamaño no corresponde a ella.
    """
    if len(datos) == 0:
        raise ValueError("paquete vacío")
    version = datos[0]  # el primer byte, como entero
    if version == VERSION:
        t_ns, R_w_i = desempaquetar(datos)
        return {"version": version, "t_ns": t_ns, "R_w_i": R_w_i, "p_w": None}
    if version == VERSION_POSE:
        if len(datos) != TAMANO_POSE:
            raise ValueError(
                f"paquete de {len(datos)} bytes, se esperaban {TAMANO_POSE}"
            )
        campos = struct.unpack(FORMATO_POSE, datos)
        return {
            "version": version,
            "t_ns": campos[1],
            "R_w_i": np.array(campos[2:11]).reshape(3, 3),
            "p_w": np.array(campos[11:14]),
        }
    raise ValueError(f"versión de protocolo {version} desconocida")


# =====================================================================
# Etapa 14 — Imagen con sus pistas (en otro puerto: no se mezcla con la pose)
# =====================================================================
VERSION_IMAGEN = 3
FORMATO_CABECERA_IMAGEN = "<Bq"  # versión + t_ns
TAMANO_CABECERA_IMAGEN = struct.calcsize(FORMATO_CABECERA_IMAGEN)  # 9 bytes
MAX_BYTES_UDP = 65507  # carga máxima de un datagrama UDP (IPv4)
MAX_BYTES_JPEG = MAX_BYTES_UDP - TAMANO_CABECERA_IMAGEN


def empaquetar_imagen(t_ns, jpeg):
    """Cabecera (versión 3, t_ns) + los bytes de un JPEG. Lanza ValueError si no cabe."""
    if len(jpeg) > MAX_BYTES_JPEG:
        raise ValueError(
            f"JPEG de {len(jpeg)} bytes: no cabe en un paquete UDP ({MAX_BYTES_JPEG})"
        )
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
ORDEN_REINICIAR = (
    b"reiniciar"  # empezar una prueba nueva: el VIO vuelve a inicializarse
)


# =====================================================================
# Etapa 17 — Mapa: keyframes (versión 4) y puntos (versión 5), en el puerto 5008
# =====================================================================
# Versión 4, keyframe de mapeo. Cabecera '<BqI9d3d4d5dHHI':
#     versión, t_ns, número de imagen, R_w_c (9), p_w_c (3), fx fy cx cy, distorsión (5),
#     ancho, alto, N
# Después, N puntos de 16 bytes (id int32, x y z float32) y, al final, el JPEG de la imagen.
# Los intrínsecos van en cada paquete: el archivo se puede repetir aunque config/ cambie.
VERSION_KEYFRAME = 4
FORMATO_CABECERA_KEYFRAME = "<BqI9d3d4d5dHHI"
TAMANO_CABECERA_KEYFRAME = struct.calcsize(FORMATO_CABECERA_KEYFRAME)  # 189 bytes
TIPO_PUNTO_KEYFRAME = np.dtype(
    [("id", "<i4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
)

# Versión 5, puntos del mapa disperso. Cabecera '<BqI': versión, t_ns, N.
# Después, N puntos de 16 bytes (id int32, x y z float32 en el mundo [m]).
VERSION_PUNTOS = 5
FORMATO_CABECERA_PUNTOS = "<BqI"
TAMANO_CABECERA_PUNTOS = struct.calcsize(FORMATO_CABECERA_PUNTOS)  # 13 bytes
TIPO_PUNTO_MAPA = np.dtype([("id", "<i4"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
MAX_PUNTOS_POR_PAQUETE = (
    MAX_BYTES_UDP - TAMANO_CABECERA_PUNTOS
) // TIPO_PUNTO_MAPA.itemsize


def espacio_jpeg_keyframe(n_puntos):
    """Bytes que quedan para el JPEG en un paquete de keyframe con n_puntos."""
    return (
        MAX_BYTES_UDP
        - TAMANO_CABECERA_KEYFRAME
        - n_puntos * TIPO_PUNTO_KEYFRAME.itemsize
    )


def empaquetar_keyframe(kf, K, dist, tamano, jpeg):
    """KeyframeMapeo (vio/mapeo.py) + intrínsecos + JPEG -> un paquete de la versión 4.

    tamano: (ancho, alto) de la imagen. Lanza ValueError si no cabe en un paquete UDP.
    """
    n = len(kf.ids)
    cabecera = struct.pack(
        FORMATO_CABECERA_KEYFRAME,
        VERSION_KEYFRAME,
        int(kf.t_ns),
        int(kf.indice),
        *np.asarray(kf.R_w_c, dtype=float).ravel(),
        *np.asarray(kf.p_w_c, dtype=float),
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
        *np.asarray(dist, dtype=float).ravel()[:5],
        int(tamano[0]),
        int(tamano[1]),
        n,
    )
    puntos = np.zeros(n, dtype=TIPO_PUNTO_KEYFRAME)
    puntos["id"] = kf.ids
    puntos["x"], puntos["y"] = kf.xy[:, 0], kf.xy[:, 1]
    puntos["z"] = kf.z
    datos = cabecera + puntos.tobytes() + bytes(jpeg)
    if len(datos) > MAX_BYTES_UDP:
        raise ValueError(
            f"keyframe de {len(datos)} bytes: no cabe en un paquete UDP ({MAX_BYTES_UDP})"
        )
    return datos


def leer_keyframe(datos):
    """Inversa de empaquetar_keyframe. Devuelve: dict con t_ns, indice, R_w_c, p_w_c, K, dist,
    tamano (ancho, alto), ids (N,), xy (N, 2), z (N,) y jpeg (bytes)."""
    if len(datos) < TAMANO_CABECERA_KEYFRAME or datos[0] != VERSION_KEYFRAME:
        raise ValueError("no es un paquete de keyframe (versión 4)")
    c = struct.unpack(FORMATO_CABECERA_KEYFRAME, datos[:TAMANO_CABECERA_KEYFRAME])
    # c: 0 versión, 1 t_ns, 2 índice, 3-11 R, 12-14 p, 15-18 fx fy cx cy, 19-23 dist, 24-25 tamaño, 26 N
    n = c[26]
    fin_puntos = TAMANO_CABECERA_KEYFRAME + n * TIPO_PUNTO_KEYFRAME.itemsize
    if len(datos) <= fin_puntos:
        raise ValueError("paquete de keyframe incompleto")
    puntos = np.frombuffer(
        datos[TAMANO_CABECERA_KEYFRAME:fin_puntos], dtype=TIPO_PUNTO_KEYFRAME
    )
    fx, fy, cx, cy = c[15:19]
    return {
        "t_ns": c[1],
        "indice": c[2],
        "R_w_c": np.array(c[3:12]).reshape(3, 3),
        "p_w_c": np.array(c[12:15]),
        "K": np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        "dist": np.array(c[19:24]),
        "tamano": (c[24], c[25]),
        "ids": puntos["id"].astype(np.int64),
        "xy": np.column_stack([puntos["x"], puntos["y"]]).astype(float),
        "z": puntos["z"].astype(float),
        "jpeg": bytes(datos[fin_puntos:]),
    }


def empaquetar_puntos(t_ns, ids, p_w):
    """Puntos del mapa disperso -> lista de paquetes de la versión 5 (varios si no caben en uno)."""
    paquetes = []
    for inicio in range(0, len(ids), MAX_PUNTOS_POR_PAQUETE):
        trozo_ids = ids[inicio : inicio + MAX_PUNTOS_POR_PAQUETE]
        trozo_p = p_w[inicio : inicio + MAX_PUNTOS_POR_PAQUETE]
        puntos = np.zeros(len(trozo_ids), dtype=TIPO_PUNTO_MAPA)
        puntos["id"] = trozo_ids
        puntos["x"], puntos["y"], puntos["z"] = (
            trozo_p[:, 0],
            trozo_p[:, 1],
            trozo_p[:, 2],
        )
        cabecera = struct.pack(
            FORMATO_CABECERA_PUNTOS, VERSION_PUNTOS, int(t_ns), len(trozo_ids)
        )
        paquetes.append(cabecera + puntos.tobytes())
    return paquetes


def leer_puntos(datos):
    """Inversa de empaquetar_puntos (un paquete). Devuelve: (t_ns, ids (N,) int64, p_w (N, 3))."""
    if len(datos) < TAMANO_CABECERA_PUNTOS or datos[0] != VERSION_PUNTOS:
        raise ValueError("no es un paquete de puntos (versión 5)")
    _, t_ns, n = struct.unpack(FORMATO_CABECERA_PUNTOS, datos[:TAMANO_CABECERA_PUNTOS])
    if len(datos) != TAMANO_CABECERA_PUNTOS + n * TIPO_PUNTO_MAPA.itemsize:
        raise ValueError(f"paquete de puntos de {len(datos)} bytes para {n} puntos")
    puntos = np.frombuffer(datos[TAMANO_CABECERA_PUNTOS:], dtype=TIPO_PUNTO_MAPA)
    p_w = np.column_stack([puntos["x"], puntos["y"], puntos["z"]]).astype(float)
    return t_ns, puntos["id"].astype(np.int64), p_w


# =====================================================================
# Etapa 19 — Prueba nueva (versión 6), en el puerto del mapa
# =====================================================================
# '<BqI': versión, t_ns de la imagen en que empieza la prueba, número de prueba (1, 2, 3...).
# pi_vio lo envía al arrancar y en cada "Nueva prueba": lo que llegue después pertenece a
# OTRO mapa (otro origen, otro rumbo, y los ids de los puntos vuelven a empezar en 0).
VERSION_PRUEBA = 6
FORMATO_PRUEBA = "<BqI"
TAMANO_PRUEBA = struct.calcsize(FORMATO_PRUEBA)  # 13 bytes
REPETICIONES_PRUEBA = 3  # UDP puede perder un paquete: este se envía varias veces


def empaquetar_prueba(t_ns, n_prueba):
    """Aviso de que empieza la prueba n_prueba -> un paquete de la versión 6."""
    return struct.pack(FORMATO_PRUEBA, VERSION_PRUEBA, int(t_ns), int(n_prueba))


def leer_prueba(datos):
    """Inversa de empaquetar_prueba. Devuelve: (t_ns, n_prueba)."""
    if len(datos) != TAMANO_PRUEBA or datos[0] != VERSION_PRUEBA:
        raise ValueError("no es un paquete de prueba nueva (versión 6)")
    _, t_ns, n_prueba = struct.unpack(FORMATO_PRUEBA, datos)
    return t_ns, n_prueba


# ---------------------------------------------------------------------
# Archivo de paquetes: los mismos bytes que se envían, uno detrás de otro, cada uno
# precedido de su longitud ('<I', 4 bytes). Se puede "reproducir" como si llegaran por red.
# ---------------------------------------------------------------------
FORMATO_LONGITUD = "<I"
TAMANO_LONGITUD = struct.calcsize(FORMATO_LONGITUD)


def escribir_paquete(archivo, datos):
    """Añade un paquete a un archivo abierto en binario ('wb' o 'ab')."""
    archivo.write(struct.pack(FORMATO_LONGITUD, len(datos)))
    archivo.write(datos)


def leer_archivo_paquetes(ruta):
    """Generador: los paquetes de un archivo, en orden.

    Si el último está cortado (la Pi se apagó mientras escribía), se ignora sin error.
    """
    with open(ruta, "rb") as f:
        while True:
            cabecera = f.read(TAMANO_LONGITUD)
            if len(cabecera) < TAMANO_LONGITUD:
                return
            (longitud,) = struct.unpack(FORMATO_LONGITUD, cabecera)
            datos = f.read(longitud)
            if len(datos) < longitud:
                return
            yield datos
