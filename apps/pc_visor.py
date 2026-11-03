"""
apps/pc_visor.py — Visor en el PC de lo que envía la Pi.

Uso, en Windows, desde la carpeta del proyecto y con su entorno activo:
    python -m apps.pc_visor
Entiende las dos versiones del protocolo:
    versión 1 (apps/pi_orientacion): solo orientación -> la placa girando en el origen
    versión 2 (apps/pi_vio, etapa 13): pose -> la placa en su posición, la trayectoria en 3D
              y vista desde arriba
"""

import socket
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from vio import protocolo, so3
from vio.config import cargar_config

COLORES = ["tab:red", "tab:green", "tab:blue"]

# Las 12 aristas de una caja, como pares de índices de sus 8 vértices
ARISTAS = [(0, 1), (1, 3), (3, 2), (2, 0),       # cara inferior
           (4, 5), (5, 7), (7, 6), (6, 4),       # cara superior
           (0, 4), (1, 5), (2, 6), (3, 7)]       # verticales


def vertices_caja(semiejes):
    """Los 8 vértices de una caja centrada en el origen. Devuelve: array (8, 3)."""
    sx, sy, sz = semiejes
    vertices = []
    for z in (-sz, sz):
        for y in (-sy, sy):
            for x in (-sx, sx):
                vertices.append([x, y, z])
    return np.array(vertices)


def limites(trayectoria, margen):
    """Límites cúbicos (mismo tamaño en x, y, z) que contienen la trayectoria con un margen.

    trayectoria: array (N, 3). Devuelve: (centro (3,), semilado).
    """
    minimo, maximo = trayectoria.min(axis=0), trayectoria.max(axis=0)
    centro = (minimo + maximo) / 2
    semilado = max((maximo - minimo).max() / 2 + margen, 1.0)
    return centro, semilado


class Visor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.caja = vertices_caja(cfg["visor"]["semiejes_caja"])
        self.R = np.eye(3)
        self.p = None                         # posición (solo con la versión 2)
        self.trayectoria = []                 # últimas posiciones recibidas
        self.version = None
        self.t_ultimo_paquete = None          # instante (de este PC) del último paquete
        self.paquetes = 0

        # Socket UDP que escucha en todas las interfaces del PC, sin bloquear
        self.receptor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receptor.bind(("0.0.0.0", cfg["red"]["puerto"]))
        self.receptor.setblocking(False)

        self.fig = plt.figure(figsize=(14, 7))
        self.ax = self.fig.add_subplot(1, 2, 1, projection="3d")
        self.ax_planta = self.fig.add_subplot(1, 2, 2)

    def recibir(self):
        """Lee TODOS los paquetes que hayan llegado; guarda la trayectoria y el último."""
        while True:
            try:
                datos, _ = self.receptor.recvfrom(1024)
            except BlockingIOError:           # no queda ninguno
                return
            try:
                paquete = protocolo.leer_paquete(datos)
            except ValueError as error:
                print("Paquete descartado:", error)
                continue
            self.version = paquete["version"]
            self.R = paquete["R_w_i"]
            self.p = paquete["p_w"]
            if self.p is not None:
                self.trayectoria.append(self.p)
                del self.trayectoria[:-self.cfg["visor"]["trayectoria_max"]]   # solo las últimas
            self.t_ultimo_paquete = time.monotonic()
            self.paquetes += 1

    def dibujar_orientacion(self, ax):
        """Versión 1: la placa girada en el origen, con sus ejes."""
        caja_w = (self.R @ self.caja.T).T
        for i, j in ARISTAS:
            ax.plot(*zip(caja_w[i], caja_w[j]), color="0.3", lw=1)
        for k in range(3):
            eje = 1.5 * self.R[:, k]
            ax.plot([0, eje[0]], [0, eje[1]], [0, eje[2]], color=COLORES[k], lw=3)
            ax.text(*(1.1 * eje), "xyz"[k], color=COLORES[k])
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

    def dibujar_pose(self, ax):
        """Versión 2: la trayectoria y la placa (pequeña) en su posición actual."""
        tr = np.array(self.trayectoria)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="tab:blue", lw=1)
        escala = self.cfg["visor"]["escala_caja_pose_m"]
        caja_w = (self.R @ (escala * self.caja).T).T + self.p
        for i, j in ARISTAS:
            ax.plot(*zip(caja_w[i], caja_w[j]), color="0.3", lw=1)
        for k in range(3):
            eje = self.p + 2 * escala * self.R[:, k]
            ax.plot(*zip(self.p, eje), color=COLORES[k], lw=2)
        centro, semilado = limites(tr, self.cfg["visor"]["margen_m"])
        ax.set_xlim(centro[0] - semilado, centro[0] + semilado)
        ax.set_ylim(centro[1] - semilado, centro[1] + semilado)
        ax.set_zlim(centro[2] - semilado, centro[2] + semilado)

        # Vista desde arriba
        planta = self.ax_planta
        planta.plot(tr[:, 0], tr[:, 1], color="tab:blue", lw=1)
        planta.plot(tr[0, 0], tr[0, 1], "ko", label="inicio")
        planta.plot(self.p[0], self.p[1], "ro", label="ahora")
        planta.arrow(self.p[0], self.p[1], escala * self.R[0, 0], escala * self.R[1, 0],
                     width=0.01, color="tab:red")      # hacia dónde mira el eje x de la placa
        planta.set_xlim(centro[0] - semilado, centro[0] + semilado)
        planta.set_ylim(centro[1] - semilado, centro[1] + semilado)
        planta.set_aspect("equal")
        planta.set_xlabel("x [m]")
        planta.set_ylabel("y [m]")
        planta.set_title("Vista desde arriba")
        planta.legend(loc="upper right")
        planta.grid(True)

    def dibujar(self, _fotograma):
        """Función que FuncAnimation llama periódicamente."""
        self.recibir()
        ax = self.ax
        ax.cla()                              # borrar y redibujar todo (sencillo, no optimizado)
        self.ax_planta.cla()
        if self.p is not None and self.trayectoria:
            self.dibujar_pose(ax)
        else:
            self.dibujar_orientacion(ax)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x mundo")
        ax.set_ylabel("y mundo")
        ax.set_zlabel("z mundo (arriba)")

        sin_datos = (self.t_ultimo_paquete is None or
                     time.monotonic() - self.t_ultimo_paquete > self.cfg["visor"]["timeout_sin_datos_s"])
        if sin_datos:
            ax.set_title("SIN DATOS: ¿está la Pi enviando a la IP de este PC?", color="red")
        else:
            al, ca, gu = np.degrees(so3.angulos_zyx(self.R))
            titulo = f"alabeo {al:+6.1f}°  cabeceo {ca:+6.1f}°  guiñada {gu:+6.1f}°"
            if self.p is not None:
                titulo += f"\np = ({self.p[0]:+.2f}, {self.p[1]:+.2f}, {self.p[2]:+.2f}) m"
            ax.set_title(f"{titulo}   (v{self.version}, {self.paquetes} paquetes)")

    def ejecutar(self):
        # La animación debe guardarse en una variable: si no, Python la borra y se para
        self.animacion = FuncAnimation(self.fig, self.dibujar,
                                       interval=self.cfg["visor"]["intervalo_refresco_ms"],
                                       cache_frame_data=False)
        plt.show()
        self.receptor.close()


def main():
    cfg = cargar_config("orientacion")
    print(f"Escuchando en el puerto UDP {cfg['red']['puerto']}. Cierra la ventana para terminar.")
    Visor(cfg).ejecutar()


if __name__ == "__main__":
    main()