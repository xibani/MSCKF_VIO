"""
herramientas/info_entorno.py — Muestra con qué Python y librerías se ejecuta el proyecto.

Uso, desde ~/svo-pi:
    python -m herramientas.info_entorno
"""

import importlib
import sys

from vio.config import RAIZ_PROYECTO, cargar_config

# Librerías que usará el proyecto, por el nombre con el que se importan
LIBRERIAS = ["numpy", "scipy", "cv2", "matplotlib", "yaml", "smbus2", "picamera2"]


def main():
    # --- 1. Python y entorno virtual ---
    print("=== Python ===")
    print(f"  versión:     {sys.version.split()[0]}")
    print(f"  ejecutable:  {sys.executable}")
    # Dentro de un entorno virtual, sys.prefix apunta al entorno y
    # sys.base_prefix al Python del sistema: si son distintos, hay entorno activo
    en_venv = sys.prefix != sys.base_prefix
    print(f"  entorno virtual activo: {'sí' if en_venv else 'NO'} ({sys.prefix})")
    
    # --- 2. Librerías: si se importan, su versión y de dónde se cargan ---
    print("\n=== Librerías ===")
    for nombre in LIBRERIAS:
        try:
            modulo = importlib.import_module(nombre)
        except ImportError as error:
            print(f"  [NO] {nombre:<11} {error}")
            continue
        version = str(getattr(modulo, "__version__", "?"))
        print(f"  [OK] {nombre:<11} {version:<10} {modulo.__file__}")
        
    # --- 3. Configuración del proyecto ---
    print("\n=== Proyecto ===")
    cfg = cargar_config("proyecto")
    print(f"  raíz:        {RAIZ_PROYECTO}")
    print(f"  nombre:      {cfg['nombre']} (versión {cfg['version']})")
    print(f"  datos en:    {RAIZ_PROYECTO / cfg['carpeta_datos']}")



# Este bloque solo se ejecuta si se lanza el archivo como programa,
# no cuando otro módulo lo importa
if __name__ == "__main__":
    main()