"""
captura/reloj.py — Reloj común a todos los sensores del proyecto.

Todas las marcas de tiempo del proyecto son enteros de nanosegundos del
reloj CLOCK_MONOTONIC: cuenta desde el arranque y nunca salta hacia atrás.
"""

import time


def ahora_ns():
    """Instante actual en ns (int), reloj CLOCK_MONOTONIC."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def desfase_boottime_ns():
    """Diferencia CLOCK_BOOTTIME - CLOCK_MONOTONIC, en ns.

    Los dos relojes solo se separan mientras el sistema está suspendido;
    en la Pi, que no se suspende, la diferencia es prácticamente 0.
    Como dos lecturas nunca son simultáneas, se lee MONOTONIC antes y
    después de BOOTTIME y se compara con la media de las dos.
    """
    t_mono_antes = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    t_boot = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    t_mono_despues = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    t_mono_medio = (t_mono_antes + t_mono_despues) // 2
    return t_boot - t_mono_medio