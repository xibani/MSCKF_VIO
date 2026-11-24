"""
tests/test_comparar_estimadores.py — Referencia común de la prueba en comparar_estimadores.

Los dos estimadores tienen que medir el cierre (y la recta) desde la misma posición física:
la primera imagen quieta a partir del instante común, aunque uno arranque más tarde.
"""

import numpy as np

from herramientas.comparar_estimadores import indice_referencia


def ejecucion(t_s, quietas):
    """Ejecución mínima: tiempos [s] y qué imágenes están quietas (ZUPT)."""
    t_ns = (np.asarray(t_s) * 1e9).astype(np.int64)
    res = [{"msckf": {"quieto": k in quietas}} for k in range(len(t_s))]
    return {"resultados": res, "t_ns": t_ns}


def test_sin_instante_comun_es_la_primera_quieta():
    e = ejecucion([0, 1, 2, 3, 4], quietas={0, 1, 3})
    assert indice_referencia(e) == 0


def test_con_instante_comun_salta_las_quietas_anteriores():
    """El MSCKF está quieto desde el principio, pero VINS no ha arrancado hasta los 2,5 s:
    el MSCKF mide desde la siguiente imagen quieta (3 s), igual que VINS."""
    msckf = ejecucion([0, 1, 2, 3, 4], quietas={0, 1, 3})
    vins = ejecucion([2.5, 3, 4], quietas={1})
    t_ref = max(e["t_ns"][indice_referencia(e)] for e in (msckf, vins))
    assert t_ref == 3_000_000_000
    assert msckf["t_ns"][indice_referencia(msckf, t_ref)] == t_ref
    assert vins["t_ns"][indice_referencia(vins, t_ref)] == t_ref


def test_sin_quietas_despues_usa_la_primera_imagen_posterior():
    e = ejecucion([0, 1, 2, 3], quietas={0})
    assert indice_referencia(e, 1_500_000_000) == 2