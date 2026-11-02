"""
vio/config.py — Lectura de los archivos de configuración del proyecto.

Todos los parámetros (frecuencias, umbrales, rutas...) viven en config/*.yaml.
Este módulo es la única puerta de entrada a ellos.
"""

from pathlib import Path

import yaml

# Carpeta raíz del proyecto (~/svo-pi), calculada a partir de dónde está
# ESTE archivo: vio/config.py -> carpeta vio/ -> carpeta svo-pi/
# Así funciona aunque el programa se lance desde otra carpeta.
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_CONFIG = RAIZ_PROYECTO / "config"


def ruta_config(nombre):
    """Devuelve la ruta completa de config/<nombre>.yaml.

    nombre: str, nombre del archivo sin la extensión (p. ej. "proyecto").
    Devuelve: Path.
    """
    return CARPETA_CONFIG / f"{nombre}.yaml"


def cargar_config(nombre):
    """Lee config/<nombre>.yaml y lo devuelve como diccionario.

    nombre: str, nombre del archivo sin la extensión.
    Devuelve: dict con el contenido del YAML.
    Lanza FileNotFoundError si el archivo no existe, y ValueError si
    está vacío o no contiene pares clave: valor.
    """
    ruta = ruta_config(nombre)

    # 1. Comprobar que existe, con un mensaje que diga dónde se ha buscado
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {ruta}")

    # 2. Leer el texto y convertirlo en estructuras de Python
    with open(ruta, encoding="utf-8") as f:
        contenido = yaml.safe_load(f)

    # 3. Comprobar que el resultado es un diccionario (pares clave: valor)
    if not isinstance(contenido, dict):
        raise ValueError(f"{ruta} está vacío o no contiene pares clave: valor")

    return contenido