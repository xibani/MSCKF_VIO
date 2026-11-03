"""
herramientas/ver_perfil.py — Muestra las funciones más costosas de un perfil de cProfile.

Uso, desde ~/svo-pi:
    python -m cProfile -o datos/perfil.out -m apps.pi_vio --grabacion datos/sim_vio --sin-red
    python -m herramientas.ver_perfil datos/perfil.out
"""

import argparse
import pstats

FUNCIONES = 20     # cuántas se muestran en cada lista


def main():
    parser = argparse.ArgumentParser(description="Resumen de un perfil de cProfile")
    parser.add_argument("perfil", help="archivo creado con python -m cProfile -o")
    args = parser.parse_args()
    estadisticas = pstats.Stats(args.perfil)
    estadisticas.strip_dirs()                        # solo el nombre del archivo, sin la ruta
    print("=== Por tiempo ACUMULADO (la función y todo lo que llama) ===")
    estadisticas.sort_stats("cumulative").print_stats(FUNCIONES)
    print("=== Por tiempo PROPIO (solo el código de la función) ===")
    estadisticas.sort_stats("tottime").print_stats(FUNCIONES)


if __name__ == "__main__":
    main()