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



# =====================================================================
# Etapa 13 — Versión 2: pose
# =====================================================================
def test_pose_tamano_e_ida_y_vuelta():
    assert protocolo.TAMANO_POSE == 105
    R, p = so3.exp_so3([0.3, -0.1, 2.0]), np.array([1.25, -0.5, 0.75])
    paquete = protocolo.empaquetar_pose(987_654_321, R, p)
    assert len(paquete) == 105 and paquete[0] == 2
    leido = protocolo.leer_paquete(paquete)
    assert leido["version"] == 2 and leido["t_ns"] == 987_654_321
    assert np.array_equal(leido["R_w_i"], R) and np.array_equal(leido["p_w"], p)


def test_leer_paquete_entiende_la_version_1():
    R = so3.exp_so3([0.0, 0.2, 0.0])
    leido = protocolo.leer_paquete(protocolo.empaquetar(5, R))
    assert leido["version"] == 1 and leido["p_w"] is None and np.array_equal(leido["R_w_i"], R)


def test_leer_paquete_rechaza_lo_incorrecto():
    paquete = protocolo.empaquetar_pose(0, np.eye(3), np.zeros(3))
    for malo in (b"", paquete[:-1], bytes([7]) + paquete[1:]):
        with pytest.raises(ValueError):
            protocolo.leer_paquete(malo)

# =====================================================================
# Etapa 16 — Órdenes del visor a pi_vio (puerto de control)
# =====================================================================
ORDEN_REINICIAR = b"reiniciar"     # empezar una prueba nueva: el VIO vuelve a inicializarse