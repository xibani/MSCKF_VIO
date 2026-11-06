"""
apps/pc_mapa.py — Etapa 17: recibe el mapa que envía la Pi y lo va construyendo en el PC.

Uso, desde la raíz del proyecto:
    python -m apps.pc_mapa                                    # en directo: escucha UDP 5007
    python -m apps.pc_mapa --archivo datos/vio_XXXX/mapa.paquetes   # repetir una sesión
    python -m apps.pc_mapa --archivo ... --salida datos/mi_mapa

En directo, la Pi tiene que enviar a la IP de este PC:
    python -m apps.pi_vio --ip-mapa <IP de este PC>     (o mapeo.ip_pc en config/vio.yaml)
y los dos tienen que estar en la MISMA red. Windows preguntará la primera vez si deja a
Python recibir por la red: hay que permitirlo en redes privadas.

Al terminar (Ctrl+C en directo, o al acabar el archivo) escribe en la carpeta de salida:
    puntos_3d.ply      mapa disperso (un punto por id), coloreado por altura
    keyframes.csv      pose e intrínsecos de cada keyframe; keyframes/<indice>.jpg
    mapa.png           planta y alzado del mapa con los keyframes
    mapa.paquetes      (solo en directo) copia de lo recibido, para repetirlo con --archivo
"""

import argparse
import socket
import time
from datetime import datetime
from pathlib import Path

from apps.visor_web import ip_local_probable
from vio import protocolo
from vio.config import RAIZ_PROYECTO, cargar_config
from vio.reconstruccion import Reconstructor

TAMANO_BUFER_UDP = 4 << 20  # 4 MB: si el PC se retrasa, caben ~80 keyframes en espera


def carpeta_salida(args):
    """Carpeta de salida: la indicada, junto al archivo repetido, o datos/mapa_<fecha>."""
    if args.salida is not None:
        return Path(args.salida)
    if args.archivo is not None:
        return Path(args.archivo).parent / "mapa"
    return (
        RAIZ_PROYECTO
        / cargar_config("proyecto")["carpeta_datos"]
        / datetime.now().strftime("mapa_%Y%m%d_%H%M%S")
    )


def desde_archivo(reconstructor, ruta):
    """Todos los paquetes de un archivo, como si llegaran por red."""
    for datos in protocolo.leer_archivo_paquetes(ruta):
        reconstructor.procesar(datos)
    print(reconstructor.resumen())


def en_directo(reconstructor, puerto, copia, aviso_cada_s=1.0):
    """Escucha UDP hasta Ctrl+C. Cada paquete se procesa y se copia al archivo `copia`."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TAMANO_BUFER_UDP)
    sock.bind(("0.0.0.0", puerto))
    sock.settimeout(0.5)  # para que Ctrl+C funcione también si no llega nada
    print(
        f"Escuchando UDP {puerto}. En la Pi:  python -m apps.pi_vio --ip-mapa {ip_local_probable()}"
    )
    print("Ctrl+C para terminar y guardar.")
    t_aviso = time.monotonic()
    try:
        while True:
            try:
                datos, _origen = sock.recvfrom(65536)
            except socket.timeout:
                datos = None
            if datos is not None:
                reconstructor.procesar(datos)
                protocolo.escribir_paquete(copia, datos)
            if time.monotonic() - t_aviso >= aviso_cada_s:
                t_aviso = time.monotonic()
                print(" ", reconstructor.resumen())
                copia.flush()
    except KeyboardInterrupt:
        print("\nTerminado")
    finally:
        sock.close()


def main():
    cfg_mapeo = cargar_config("vio")["mapeo"]
    parser = argparse.ArgumentParser(
        description="Mapa en el PC con los paquetes de pi_vio (etapa 17)"
    )
    parser.add_argument(
        "--archivo",
        default=None,
        help="mapa.paquetes de una sesión (sin él: en directo)",
    )
    parser.add_argument("--salida", default=None, help="carpeta donde guardar el mapa")
    parser.add_argument(
        "--puerto", type=int, default=cfg_mapeo["puerto"], help="puerto UDP (5007)"
    )
    args = parser.parse_args()

    salida = carpeta_salida(args)
    salida.mkdir(parents=True, exist_ok=True)
    reconstructor = Reconstructor()
    if args.archivo is not None:
        desde_archivo(reconstructor, args.archivo)
    else:
        with open(salida / cfg_mapeo["archivo"], "wb") as copia:
            en_directo(reconstructor, args.puerto, copia)

    rutas = reconstructor.guardar(salida)
    if rutas:
        print(f"Mapa guardado en {salida}:")
        for ruta in rutas.values():
            print("  ", ruta.name)
    else:
        print(
            "No ha llegado nada: ¿están la Pi y el PC en la misma red y la IP es la de este PC?"
        )


if __name__ == "__main__":
    main()
