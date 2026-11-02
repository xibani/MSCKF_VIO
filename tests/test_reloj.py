"""tests/test_reloj.py — Tests de captura/reloj.py."""

from captura import reloj


def test_ahora_ns_es_entero_y_avanza():
    t1 = reloj.ahora_ns()
    t2 = reloj.ahora_ns()
    assert isinstance(t1, int)
    assert t2 >= t1


def test_boottime_y_monotonic_casi_iguales():
    # Sin suspensiones, la diferencia es despreciable: menos de 1 ms
    assert abs(reloj.desfase_boottime_ns()) < 1_000_000