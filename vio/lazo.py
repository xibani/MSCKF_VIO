"""
vio/lazo.py — Métricas de una prueba de trayectoria con datos reales (etapa 16).

Sin una verdad de referencia, dos pruebas sencillas dicen mucho de un VIO:

  1. Lazo cerrado: salir de una marca en el suelo, dar una vuelta y volver EXACTAMENTE a ella.
     La verdad se conoce aunque no se haya medido nada: posición final = posición inicial.
     Todo lo que el VIO diga de más es deriva acumulada:
         error de cierre = |p_final - p_inicial| / recorrido        (en % del recorrido)
  2. Recta medida: ir de una marca a otra separadas una distancia medida con cinta:
         error de escala = (|p_final - p_inicial| - distancia_real) / distancia_real

El recorrido se suma con un paso mínimo. Si se sumaran todos los pasitos entre imágenes, el
temblor de unos milímetros de la estimación (también con la placa parada) inflaría el recorrido
y, con él, haría parecer más pequeño el error de cierre.
"""

import numpy as np


def recorrido(p, paso_min_m):
    """Longitud del camino [m] de p (N, 3), sumando solo desplazamientos de al menos paso_min_m.

    Se lleva un "ancla" (el último punto contado): cuando la posición se aleja de ella
    paso_min_m o más, se suma esa distancia y el ancla pasa al punto actual.
    """
    total = 0.0
    ancla = p[0]
    for q in p[1:]:
        d = float(np.linalg.norm(q - ancla))
        if d >= paso_min_m:
            total += d
            ancla = q
    return total


def metricas_lazo(t_s, p, paso_min_m, recorrido_min_m):
    """Métricas de una prueba. t_s (N,) [s], p (N, 3) [m].

    Devuelve: dict con duracion_s, recorrido_m, cierre_m, cierre_pct (None si el recorrido es
    menor que recorrido_min_m), deriva_m (3,) = p_final - p_inicial y alejamiento_max_m.
    """
    longitud = recorrido(p, paso_min_m)
    deriva = p[-1] - p[0]
    cierre = float(np.linalg.norm(deriva))
    return {"duracion_s": float(t_s[-1] - t_s[0]),
            "recorrido_m": longitud,
            "cierre_m": cierre,
            "cierre_pct": 100.0 * cierre / longitud if longitud >= recorrido_min_m else None,
            "deriva_m": deriva,
            "alejamiento_max_m": float(np.max(np.linalg.norm(p - p[0], axis=1)))}


def error_escala(p, distancia_real_m):
    """Prueba de la recta medida. Devuelve: (distancia estimada [m], error de escala [%])."""
    estimada = float(np.linalg.norm(p[-1] - p[0]))
    return estimada, 100.0 * (estimada - distancia_real_m) / distancia_real_m