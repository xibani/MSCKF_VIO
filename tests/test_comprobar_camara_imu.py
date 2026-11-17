"""
tests/test_comprobar_camara_imu.py — herramientas/comprobar_camara_imu.py recupera una extrínseca
y un desfase conocidos a partir de giros sintéticos.

    python -m pytest tests/test_comprobar_camara_imu.py -v
"""

import numpy as np

from herramientas.comprobar_camara_imu import (errores_rotacion, estimar_t_d, giro_integrado,
                                               integral_acumulada, refinar_rotacion, wahba)
from vio import so3


def test_integral_de_un_giro_constante():
    t = np.arange(0, 2, 0.005)
    w = np.tile([0.1, -0.2, 0.3], (len(t), 1))
    C = integral_acumulada(t, w)
    theta = giro_integrado(t, C, np.array([0.5]), np.array([0.6]))
    assert np.allclose(theta[0], [0.01, -0.02, 0.03], atol=1e-12)


def test_wahba_recupera_la_rotacion():
    rng = np.random.default_rng(3)
    R = so3.exp_so3(np.radians([3.0, -2.0, 90.0]))
    a = rng.normal(size=(50, 3))
    b = a @ R.T + rng.normal(scale=1e-3, size=(50, 3))
    R_est = wahba(a, b)
    assert np.degrees(np.linalg.norm(so3.log_so3(R_est @ R.T))) < 0.1
    assert abs(np.linalg.det(R_est) - 1.0) < 1e-9


def test_t_d_recuperado():
    # Giroscopio con velocidad angular variable; la cámara ve el giro desplazado 27 ms
    t = np.arange(0, 20, 0.005)
    w = np.column_stack([np.sin(2.1 * t), 0.5 * np.cos(1.3 * t), 0.8 * np.sin(0.7 * t + 1)])
    C = integral_acumulada(t, w)
    t_a = np.arange(1, 18, 0.1)
    t_b = t_a + 0.1
    t_d_verdad = 0.027
    angulo = np.linalg.norm(giro_integrado(t, C, t_a + t_d_verdad, t_b + t_d_verdad), axis=1)
    t_d, _ = estimar_t_d(t, C, t_a, t_b, angulo, np.arange(-0.1, 0.1, 0.0005))
    assert abs(t_d - t_d_verdad) < 0.001


def test_error_cero_con_la_extrinseca_correcta_y_no_con_otra():
    R_c_i = so3.exp_so3(np.radians([0.0, 0.0, 90.0])) @ so3.exp_so3(np.radians([-90.0, 0.0, 0.0]))
    rng = np.random.default_rng(5)
    theta_imu = rng.normal(scale=0.05, size=(40, 3))
    R_vision = np.array([R_c_i @ so3.exp_so3(th) @ R_c_i.T for th in theta_imu])
    assert np.max(errores_rotacion(R_c_i, theta_imu, R_vision)) < 1e-9
    R_mal = so3.exp_so3(np.radians([3.0, 0.0, 0.0])) @ R_c_i     # 3° de error en la extrínseca
    assert np.median(errores_rotacion(R_mal, theta_imu, R_vision)) > 1e-3


def test_refinado_de_la_rotacion_con_sampson():
    # Puntos 3D delante de la cámara a; la cámara b gira 3° y se desplaza 2 cm
    rng = np.random.default_rng(7)
    P_a = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(-0.8, 0.8, 200), rng.uniform(1, 3, 200)])
    R_b_a = so3.exp_so3(np.radians([1.0, 2.5, -1.2]))
    t_b_a = np.array([0.02, 0.0, 0.005])
    P_b = P_a @ R_b_a.T + t_b_a
    ruido = 0.3 / 620                                        # 0,3 px con fx = 620
    xy_a = P_a[:, :2] / P_a[:, 2:3] + rng.normal(scale=ruido, size=(200, 2))
    xy_b = P_b[:, :2] / P_b[:, 2:3] + rng.normal(scale=ruido, size=(200, 2))
    # Punto de partida con 0,5° de error, como el de un RANSAC de 5 puntos
    R0 = so3.exp_so3(np.radians([0.3, -0.3, 0.2])) @ R_b_a
    R = refinar_rotacion(R0, t_b_a + np.array([0.003, 0.002, 0.0]), xy_a, xy_b, 1.0 / 620)
    assert np.degrees(np.linalg.norm(so3.log_so3(R @ R_b_a.T))) < 0.05