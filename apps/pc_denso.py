"""
apps/pc_denso.py — Etapa 30: la reconstrucción densa en el PC, mientras la Pi se mueve.

Hace lo mismo que apps/pc_mapa (recibe el mapa que envía la Pi y lo guarda) y además, con
cada keyframe que llega: saca su profundidad (Depth Anything, etapa 25), la pone en metros
con los puntos del VIO y añade sus puntos a la nube densa (etapa 26, en su versión
keyframe a keyframe). Sustituye a pc_mapa: los dos escuchan en el mismo puerto y no pueden
estar abiertos a la vez.

Uso, desde la raíz del proyecto, EN EL PC:
    python -m apps.pc_denso                                    # en directo: escucha UDP 5008
    python -m apps.pc_denso --ver                              # ...y abre una ventana con la planta
    python -m apps.pc_denso --archivo datos/vio_XXXX/mapa.paquetes      # repetir una sesión
    python -m apps.pc_denso --archivo datos/sim_textura30/mapa.paquetes --con-verdad
En directo, la Pi tiene que enviar a la IP de este PC y estar en su MISMA red:
    python -m apps.pi_vio --ip-mapa <IP de este PC>
--con-verdad usa la profundidad verdadera de una grabación simulada en lugar de la red: no
necesita PyTorch. La grabación es la carpeta del --archivo o la que diga --grabacion.

Al terminar (Ctrl+C en directo, o al acabar el archivo) escribe en la carpeta de salida lo
mismo que pc_mapa y además:
    nube_densa_directo.ply    la nube densa
    nube_densa_directo.png    su planta (se va actualizando durante la sesión)
"""

import argparse
import socket
import time
from pathlib import Path

import matplotlib
import numpy as np

from apps.pc_mapa import TAMANO_BUFER_UDP, carpeta_salida
from apps.visor_web import ip_local_probable
from captura import euroc
from vio import denso_en_directo as dd
from vio import protocolo
from vio.config import cargar_config
from vio.reconstruccion import MapaPorPruebas, decodificar_gris, guardar_ply


# =====================================================================
# Paso 30.3 — De dónde sale la profundidad de cada keyframe
# =====================================================================
def profundidad_con_la_red(cfg_profundidad):
    """Devuelve una función profundidad_de(kf, gris) -> z (alto, ancho) [m] o None, que usa
    Depth Anything y el ajuste a metros de la etapa 25. Carga la red (necesita PyTorch)."""
    from vio import profundidad as prof

    estimador = prof.EstimadorProfundidad(cfg_profundidad)
    print(f"Red cargada en: {estimador.dispositivo}")
    rng = np.random.default_rng(cfg_profundidad["ajuste"]["semilla"])

    def profundidad_de(kf, gris):
        return prof.profundidad_de_keyframe(kf, gris, estimador, cfg_profundidad, rng)["z"]

    return profundidad_de


def profundidad_verdadera(grabacion, desfase_ns):
    """Devuelve una función profundidad_de(kf, gris) que lee la profundidad verdadera de una
    grabación simulada (el archivo se llama como la marca de la imagen = instante - desfase)."""

    def profundidad_de(kf, gris):
        return euroc.leer_profundidad(grabacion, kf["t_ns"] - desfase_ns)

    return profundidad_de


# =====================================================================
# Paso 30.4 — La sesión: el mapa de pc_mapa y, encima, la nube densa
# =====================================================================
class SesionDensa:
    """Recibe los paquetes de la Pi, uno a uno.

    Cada paquete va al mapa de siempre (MapaPorPruebas, etapas 17 y 19). Si es un keyframe,
    además se le saca la profundidad y se le pasa al DensificadorEnDirecto. Si empieza una
    prueba nueva, se guarda la nube de la anterior y se empieza otra.
    """

    def __init__(self, salida, nombre_copia, profundidad_de, cfg_densificar, cfg_directo):
        self.mapa = MapaPorPruebas(salida, nombre_copia=nombre_copia)
        self.profundidad_de = profundidad_de
        self.cfg_densificar = cfg_densificar
        self.cfg_directo = cfg_directo
        self.guardadas = []                               # (carpeta, número de puntos) de cada nube escrita
        self._empezar_nube()

    def _empezar_nube(self):
        self.denso = dd.DensificadorEnDirecto(self.cfg_densificar, self.cfg_directo)
        self.camaras = []                                 # posición de cada keyframe con profundidad
        self.sin_profundidad = 0
        self.tiempos_ms = []                              # (profundidad, nube) de cada keyframe

    def procesar(self, datos):
        """Un paquete (bytes). Devuelve su tipo, como MapaPorPruebas.procesar."""
        n_antes, carpeta_antes = self.mapa.n, self.mapa.carpeta
        tipo = self.mapa.procesar(datos)
        if self.mapa.n != n_antes:                        # prueba nueva: la nube de la anterior, a disco
            self.guardar_nube(carpeta_antes)
            self._empezar_nube()
        if tipo == "keyframe":
            self._nuevo_keyframe(self.mapa.reconstructor.keyframes[-1])
        return tipo

    def _nuevo_keyframe(self, kf):
        gris = decodificar_gris(kf)
        t0 = time.perf_counter()
        z = self.profundidad_de(kf, gris)
        t1 = time.perf_counter()
        if z is None:                                     # sin ajuste a metros, o sin verdad para esa imagen
            self.sin_profundidad += 1
            return
        self.denso.nueva_vista({"kf": kf, "z": z, "gris": gris})
        self.camaras.append(kf["p_w_c"])
        self.tiempos_ms.append((1e3 * (t1 - t0), 1e3 * (time.perf_counter() - t1)))
        if len(self.camaras) % self.cfg_directo["avisar_cada"] == 0:
            print(" ", self.resumen())
            dibujar_planta(self.mapa.carpeta / "nube_densa_directo.png", self.denso.resultado(), self.camaras)

    def resumen(self):
        """Una línea con el estado, para imprimir."""
        linea = (f"keyframes {len(self.camaras):3d} (en la nube {self.denso.n_cerradas:3d},"
                 f" sin profundidad {self.sin_profundidad}) | nube {len(self.denso.nube):6d} cubos")
        if self.tiempos_ms:
            ultimos = np.array(self.tiempos_ms[-10:])
            linea += f" | profundidad {ultimos[:, 0].mean():4.0f} ms, nube {ultimos[:, 1].mean():4.0f} ms por keyframe"
        return linea

    def guardar_nube(self, carpeta):
        """Mete en la nube los keyframes que esperaban y la escribe en `carpeta`."""
        self.denso.terminar()
        resultado = self.denso.resultado()
        if len(resultado["p_w"]) == 0:
            return
        gris = resultado["gris"]
        guardar_ply(Path(carpeta) / "nube_densa_directo.ply", resultado["p_w"], np.column_stack([gris, gris, gris]))
        dibujar_planta(Path(carpeta) / "nube_densa_directo.png", resultado, self.camaras)
        self.guardadas.append((Path(carpeta), len(resultado["p_w"])))

    def cerrar(self):
        """Fin de la sesión: guarda la nube densa y el mapa de siempre."""
        self.guardar_nube(self.mapa.carpeta)
        self.mapa.cerrar()


# =====================================================================
# Paso 30.5 — Verlo
# =====================================================================
VENTANA = {"figura": None, "motor": None}      # la ventana de --ver, si se ha pedido, y con qué la dibuja matplotlib


def dibujar_planta(ruta, resultado, camaras):
    """La planta (x, y) de la nube, con el recorrido de la cámara en rojo: a un PNG y, si
    hay ventana abierta (--ver), también a ella."""
    import matplotlib.pyplot as plt

    if VENTANA["figura"] is not None and matplotlib.get_backend() != VENTANA["motor"]:
        # Al guardar el mapa de una prueba (vio/reconstruccion.py) matplotlib pasa a "sin
        # ventana" y la cierra: se sigue sin ella
        print("  (la ventana se ha cerrado al guardar el mapa de la prueba anterior: se sigue sin ella)")
        VENTANA["figura"] = None
    if VENTANA["figura"] is None:
        figura, ax = plt.subplots(figsize=(8, 8))
    else:
        figura = VENTANA["figura"]
        ax = figura.axes[0]
        ax.clear()
    p_w = resultado["p_w"]
    if len(p_w):
        ax.scatter(p_w[:, 0], p_w[:, 1], s=1, c=resultado["gris"], cmap="gray", vmin=0, vmax=255)
    if len(camaras):
        recorrido = np.array(camaras)
        ax.plot(recorrido[:, 0], recorrido[:, 1], "r.-", lw=0.8, ms=4)
    ax.set_title(f"{len(p_w)} puntos, {len(camaras)} keyframes")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_facecolor("#203040")
    figura.savefig(ruta, dpi=100)
    if VENTANA["figura"] is None:
        plt.close(figura)
    else:
        figura.canvas.draw_idle()
        plt.pause(0.001)                                 # deja que la ventana se repinte


def abrir_ventana():
    """Abre la ventana de --ver. Hay que llamarla antes de dibujar nada."""
    import matplotlib.pyplot as plt

    plt.ion()                                            # modo interactivo: dibujar no bloquea
    figura, _ = plt.subplots(figsize=(8, 8))
    VENTANA["figura"] = figura
    VENTANA["motor"] = matplotlib.get_backend()
    plt.pause(0.001)


# =====================================================================
# Paso 30.6 — Las dos fuentes de paquetes
# =====================================================================
def desde_archivo(sesion, ruta):
    """Todos los paquetes de un archivo, como si llegaran por red."""
    for datos in protocolo.leer_archivo_paquetes(ruta):
        sesion.procesar(datos)
    print(" ", sesion.resumen())


def en_directo(sesion, puerto):
    """Escucha UDP hasta Ctrl+C. Los keyframes que llegan mientras se procesa otro esperan
    en el búfer del sistema (4 MB: unos 80)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TAMANO_BUFER_UDP)
    sock.bind(("0.0.0.0", puerto))
    sock.settimeout(0.5)                                 # para que Ctrl+C funcione también si no llega nada
    print(f"Escuchando UDP {puerto}. En la Pi:  python -m apps.pi_vio --ip-mapa {ip_local_probable()}")
    print("Ctrl+C para terminar y guardar.")
    try:
        while True:
            try:
                datos, _origen = sock.recvfrom(65536)
            except socket.timeout:
                if VENTANA["figura"] is not None:        # sin paquetes, la ventana tiene que seguir viva
                    VENTANA["figura"].canvas.flush_events()
                continue
            n_antes = sesion.mapa.n
            sesion.procesar(datos)
            if sesion.mapa.n != n_antes:
                print(f"Prueba nueva: el mapa sigue en {sesion.mapa.carpeta}")
            sesion.mapa.vaciar_copia()
    except KeyboardInterrupt:
        print("\nTerminado")
    finally:
        sock.close()


def comparar_con_la_verdad(sesion, grabacion):
    """Si la sesión es de una grabación del simulador, el error de la nube frente a la
    habitación de verdad (como en la etapa 26). Devuelve: una línea de texto, o None."""
    # Este import cambia matplotlib a "sin ventana": por eso va aquí, al final, y no arriba
    from herramientas.nube_densa import transformacion_de_la_sesion
    from vio import densificar as den

    transformacion = transformacion_de_la_sesion(grabacion) if grabacion is not None else None
    resultado = sesion.denso.resultado()
    if transformacion is None or len(resultado["p_w"]) == 0:
        return None
    R, t = transformacion
    dimensiones = cargar_config("simulador")["vio"]["mundo"]["habitacion_m"]
    e = den.evaluar_nube(resultado["p_w"] @ R.T + t, dimensiones, sesion.cfg_densificar["evaluacion"])
    return (f"Frente a la habitación de verdad: {e['n']} puntos, error mediano {100 * e['error_mediano']:.1f} cm,"
            f" el 90 % a menos de {100 * e['error_p90']:.1f} cm, cubierto {100 * e['cubierto']:.0f} %")


def main():
    cfg_mapeo = cargar_config("vio")["mapeo"]
    parser = argparse.ArgumentParser(description="Reconstrucción densa en el PC con los paquetes de pi_vio (etapa 30)")
    parser.add_argument("--archivo", default=None, help="mapa.paquetes de una sesión (sin él: en directo)")
    parser.add_argument("--salida", default=None, help="carpeta donde guardar el resultado")
    parser.add_argument("--puerto", type=int, default=cfg_mapeo["puerto"], help="puerto UDP (5008)")
    parser.add_argument("--con-verdad", action="store_true",
                        help="simulador: la profundidad verdadera de la grabación en lugar de la red")
    parser.add_argument("--grabacion", default=None,
                        help="con --con-verdad: carpeta de la grabación simulada (por defecto, la del --archivo)")
    parser.add_argument("--desfase-ns", type=int, default=cargar_config("camara_imu")["t_d_ns"],
                        help="con --con-verdad: desfase cámara-IMU de la grabación (por defecto, el de camara_imu.yaml)")
    parser.add_argument("--ver", action="store_true", help="abrir una ventana con la planta de la nube")
    args = parser.parse_args()

    if not args.ver:
        matplotlib.use("Agg")                            # sin ventana: las figuras solo van a un PNG
    grabacion = args.grabacion
    if grabacion is None and args.archivo is not None:
        grabacion = str(Path(args.archivo).parent)
    if args.con_verdad:
        if grabacion is None:
            parser.error("--con-verdad en directo necesita --grabacion")
        profundidad_de = profundidad_verdadera(grabacion, args.desfase_ns)
    else:
        profundidad_de = profundidad_con_la_red(cargar_config("profundidad"))
    if args.ver:
        abrir_ventana()

    salida = carpeta_salida(args)
    nombre_copia = None if args.archivo is not None else cfg_mapeo["archivo"]
    sesion = SesionDensa(salida, nombre_copia, profundidad_de, cargar_config("densificar"), cargar_config("directo"))
    if args.archivo is not None:
        desde_archivo(sesion, args.archivo)
    else:
        en_directo(sesion, args.puerto)
    sesion.cerrar()

    print(" ", sesion.resumen())
    if not sesion.guardadas:
        print("No hay nube densa: no ha llegado ningún keyframe con profundidad."
              " ¿Están la Pi y el PC en la misma red y la IP es la de este PC?")
    for carpeta, n_puntos in sesion.guardadas:
        print(f"Nube densa de {n_puntos} puntos en {carpeta / 'nube_densa_directo.ply'}")
    linea = comparar_con_la_verdad(sesion, grabacion)
    if linea is not None:
        print(linea)


if __name__ == "__main__":
    main()
