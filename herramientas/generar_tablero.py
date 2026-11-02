"""
herramientas/generar_tablero.py — PDF con el tablero de ajedrez, a tamaño real, en A4 apaisado.

Uso, desde ~/svo-pi:
    python -m herramientas.generar_tablero
Imprímelo al 100 % ("tamaño real", SIN "ajustar a la página"), pégalo sobre algo
rígido y plano, y MIDE con una regla el lado de un cuadro: ese valor va en
config/calibracion.yaml -> tablero.lado_m.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from vio.config import RAIZ_PROYECTO, cargar_config

A4_ANCHO_MM, A4_ALTO_MM = 297.0, 210.0
MM_POR_PULGADA = 25.4


def main():
    t = cargar_config("calibracion")["tablero"]
    lado_mm = t["lado_m"] * 1000.0
    cuadros_x, cuadros_y = t["esquinas_x"] + 1, t["esquinas_y"] + 1   # esquinas interiores + 1
    ancho_mm, alto_mm = cuadros_x * lado_mm, cuadros_y * lado_mm
    if ancho_mm > A4_ANCHO_MM - 20 or alto_mm > A4_ALTO_MM - 20:
        print("ERROR: el tablero no cabe en un A4 con márgenes; reduce lado_m.")
        return

    # Figura del tamaño exacto de un A4; los ejes ocupan toda la hoja y miden en mm
    fig = plt.figure(figsize=(A4_ANCHO_MM / MM_POR_PULGADA, A4_ALTO_MM / MM_POR_PULGADA))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_ANCHO_MM)
    ax.set_ylim(0, A4_ALTO_MM)
    ax.axis("off")

    x0 = (A4_ANCHO_MM - ancho_mm) / 2           # centrado en la hoja
    y0 = (A4_ALTO_MM - alto_mm) / 2
    for fila in range(cuadros_y):
        for col in range(cuadros_x):
            if (fila + col) % 2 == 0:
                ax.add_patch(Rectangle((x0 + col * lado_mm, y0 + fila * lado_mm),
                                       lado_mm, lado_mm, color="black", lw=0))
    ax.text(A4_ANCHO_MM / 2, 6, f"{t['esquinas_x']} x {t['esquinas_y']} esquinas interiores, "
            f"cuadro de {lado_mm:.1f} mm. Imprimir al 100 % y MEDIR.", ha="center", fontsize=8)

    ruta = RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"] / "tablero.pdf"
    fig.savefig(ruta)
    print(f"Tablero de {cuadros_x} x {cuadros_y} cuadros de {lado_mm:.1f} mm: {ruta}")


if __name__ == "__main__":
    main()