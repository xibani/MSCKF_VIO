"""
apps/pc_visor.py — Visor en el PC de la orientación que envía la Pi.

Uso, en Windows, desde la carpeta del proyecto y con su entorno activo:
    python -m apps.pc_visor
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


class Visor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.caja = vertices_caja(cfg["visor"]["semiejes_caja"])
        self.R = np.eye(3)
        self.t_ultimo_paquete = None          # instante (de este PC) del último paquete
        self.paquetes = 0

        # Socket UDP que escucha en todas las interfaces del PC, sin bloquear
        self.receptor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receptor.bind(("0.0.0.0", cfg["red"]["puerto"]))
        self.receptor.setblocking(False)

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(projection="3d")

    def recibir(self):
        """Lee TODOS los paquetes que hayan llegado y se queda con el último."""
        while True:
            try:
                datos, _ = self.receptor.recvfrom(1024)
            except BlockingIOError:           # no queda ninguno
                return
            try:
                _, self.R = protocolo.desempaquetar(datos)
            except ValueError as error:
                print("Paquete descartado:", error)
                continue
            self.t_ultimo_paquete = time.monotonic()
            self.paquetes += 1

    def dibujar(self, _fotograma):
        """Función que FuncAnimation llama periódicamente."""
        self.recibir()
        ax = self.ax
        ax.cla()                              # borrar y redibujar todo (sencillo, no optimizado)

        # Caja girada: cada vértice v (en ejes de la placa) se ve en el mundo como R @ v
        caja_w = (self.R @ self.caja.T).T
        for i, j in ARISTAS:
            ax.plot(*zip(caja_w[i], caja_w[j]), color="0.3", lw=1)

        # Ejes de la placa: las columnas de R
        for k in range(3):
            eje = 1.5 * self.R[:, k]
            ax.plot([0, eje[0]], [0, eje[1]], [0, eje[2]], color=COLORES[k], lw=3)
            ax.text(*(1.1 * eje), "xyz"[k], color=COLORES[k])

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
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
            ax.set_title(f"alabeo {al:+6.1f}°  cabeceo {ca:+6.1f}°  guiñada {gu:+6.1f}°"
                         f"   ({self.paquetes} paquetes)")

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