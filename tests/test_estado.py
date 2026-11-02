"""tests/test_estado.py — Tests de vio/estado.py."""

import numpy as np

from vio.estado import estado_inicial


def test_estado_inicial():
    e = estado_inicial(np.eye(3))
    assert np.array_equal(e.v_w, np.zeros(3)) and np.array_equal(e.p_w, np.zeros(3))
    assert np.array_equal(e.b_g, np.zeros(3)) and np.array_equal(e.b_a, np.zeros(3))


def test_copia_es_independiente():
    e = estado_inicial(np.eye(3))
    c = e.copia()
    c.p_w[0] = 5.0
    c.R_w_i[0, 0] = -1.0
    assert e.p_w[0] == 0.0 and e.R_w_i[0, 0] == 1.0