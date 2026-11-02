"""tests/test_protocolo.py — Tests de vio/protocolo.py."""

import socket

import numpy as np
import pytest

from vio import protocolo, so3


def test_tamano():
    assert protocolo.TAMANO == 81


def test_ida_y_vuelta():
    R = so3.exp_so3([0.1, -0.2, 0.3])
    t_ns = 1_234_567_890_123_456
    paquete = protocolo.empaquetar(t_ns, R)
    assert len(paquete) == 81
    t2, R2 = protocolo.desempaquetar(paquete)
    assert t2 == t_ns
    assert np.array_equal(R2, R)                     # bit a bit: float64 no pierde nada


def test_paquetes_incorrectos():
    paquete = protocolo.empaquetar(0, np.eye(3))
    with pytest.raises(ValueError):
        protocolo.desempaquetar(paquete[:-1])        # le falta un byte
    with pytest.raises(ValueError):
        protocolo.desempaquetar(bytes([99]) + paquete[1:])   # versión desconocida


def test_por_un_socket_udp_real():
    # Emisor y receptor en el mismo ordenador (127.0.0.1)
    receptor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receptor.bind(("127.0.0.1", 0))                  # puerto 0: el sistema elige uno libre
    receptor.settimeout(2.0)
    emisor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        R = so3.exp_so3([0.0, 0.0, 1.0])
        emisor.sendto(protocolo.empaquetar(42, R), receptor.getsockname())
        datos, _ = receptor.recvfrom(1024)
        t_ns, R_recibida = protocolo.desempaquetar(datos)
        assert t_ns == 42
        assert np.array_equal(R_recibida, R)
    finally:
        emisor.close()
        receptor.close()