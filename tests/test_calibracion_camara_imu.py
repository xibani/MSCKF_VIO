"""tests/test_calibracion_camara_imu.py — Calibración cámara-IMU con el giroscopio (etapa 7.4).

Se inventa un movimiento de la placa (giros suaves por los tres ejes) con un montaje un
poco torcido, un desfase de tiempo y un sesgo conocidos, y se comprueba que se recuperan:
    - con las posturas del tablero calculadas directamente (rápido, sin imágenes);
    - de principio a fin: se DIBUJAN las imágenes del tablero, se escribe una grabación
      EuRoC como la de herramientas.grabar y se pasa por la herramienta.
"""

import cv2
import numpy as np
import pytest
import yaml

from captura import euroc
from herramientas import calibrar_camara_imu as herramienta
from vio import calibracion_camara as cc
from vio import calibracion_camara_imu as cci
from vio import so3
from vio.config import cargar_config

CFG = cargar_config("calibracion")["camara_imu"]
TABLERO = {"esquinas_x": 9, "esquinas_y": 6, "lado_m": 0.025}     # el de la 7.1

# El montaje "de verdad" del simulacro: el de regla girado ~1,4° (lo que se quiere medir)
R_REGLA = np.array(cargar_config("camara_imu")["R_i_c"], dtype=float)
R_I_C = so3.exp_so3(np.radians([0.6, -1.0, 0.8])) @ R_REGLA
P_I_C = np.array([0.024, 0.0, 0.035])
B_G = np.array([0.012, -0.020, 0.008])          # sesgo del giroscopio [rad/s]
F_IMU = 200.0                                   # [Hz]
F_CAM = 20.0
K = np.array([[620.0, 0.0, 320.0], [0.0, 620.0, 240.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)


# =====================================================================
# Movimiento inventado
# =====================================================================
def vector_giro(t, amplitud):
    """Orientación de la IMU como vector de rotación: senos de frecuencias distintas por eje."""
    return amplitud * np.array([np.sin(2 * np.pi * 0.9 * t),
                                np.sin(2 * np.pi * 1.3 * t + 1.0),
                                np.sin(2 * np.pi * 0.7 * t + 2.0)])


def R_w_i(t, amplitud):
    return so3.exp_so3(vector_giro(t, amplitud))


def w_i_verdad(t, amplitud, h=1e-4):
    """Velocidad de giro en ejes de la IMU: log(R(t-h)ᵀ·R(t+h)) / 2h (diferencia centrada)."""
    return so3.log_so3(R_w_i(t - h, amplitud).T @ R_w_i(t + h, amplitud)) / (2 * h)


def giroscopio(duracion_s, amplitud, ruido_rad_s, rng):
    """Muestras del giroscopio a 200 Hz: verdad + sesgo + ruido. Devuelve: (t_s (N,), w (N, 3))."""
    t = np.arange(0.0, duracion_s, 1.0 / F_IMU)
    w = np.array([w_i_verdad(tk, amplitud) for tk in t]) + B_G
    return t, w + rng.normal(0.0, ruido_rad_s, w.shape)


def tablero_en_el_mundo(distancia_m):
    """Ejes del tablero (b) en el mundo: de frente a la cámara en reposo, centrado, a esa distancia."""
    R_w_c0 = R_I_C                                    # en t = 0 la IMU no está girada
    p_w_c0 = P_I_C
    ancho = (TABLERO["esquinas_x"] - 1) * TABLERO["lado_m"]
    alto = (TABLERO["esquinas_y"] - 1) * TABLERO["lado_m"]
    p_w_b = p_w_c0 + R_w_c0 @ np.array([-ancho / 2, -alto / 2, distancia_m])
    return R_w_c0, p_w_b


def postura_verdad(tau, amplitud, R_w_b, p_w_b):
    """Postura del tablero vista desde la cámara en el instante VERDADERO tau: (R_c_b, t_c_b)."""
    R_w_c = R_w_i(tau, amplitud) @ R_I_C
    p_w_c = R_w_i(tau, amplitud) @ P_I_C               # la placa gira alrededor de la IMU
    return R_w_c.T @ R_w_b, R_w_c.T @ (p_w_b - p_w_c)


def instantes_camara(duracion_s, t_d_s):
    """Instantes verdaderos de las imágenes y sus sellos (sello + t_d = instante verdadero)."""
    tau = np.arange(0.3, duracion_s - 0.3, 1.0 / F_CAM)
    return tau, tau - t_d_s


# =====================================================================
# Piezas sueltas
# =====================================================================
def test_alinear_giros_recupera_giro_y_sesgo():
    rng = np.random.default_rng(1)
    w_c = rng.normal(0.0, 1.0, (500, 3))
    w_i = w_c @ R_I_C.T + B_G
    R, b, residuos = cci.alinear_giros(w_c, w_i)
    assert so3.angulo_entre(R, R_I_C) < 1e-9
    assert np.allclose(b, B_G) and cci.rms(residuos) < 1e-9


def test_alinear_giros_da_un_giro_y_no_un_espejo():
    # Girando solo en un plano, H tiene rango 2 y la SVD podría dar un espejo (det = -1)
    rng = np.random.default_rng(2)
    w_c = np.column_stack([rng.normal(0, 1, 200), rng.normal(0, 1, 200), np.zeros(200)])
    R, _, _ = cci.alinear_giros(w_c, w_c @ R_I_C.T)
    assert so3.es_rotacion(R, tolerancia=1e-9)


def test_velocidades_camara_con_giro_constante():
    w = np.array([0.3, -0.5, 0.2])                     # rad/s, en ejes de la cámara
    t = np.arange(0.0, 1.0, 0.05)
    # Orientación de la cámara en el tablero: R_b_c(t) = exp(w·t); R_c_b = su traspuesta
    R_c_b = [so3.exp_so3(w * tk).T for tk in t]
    t0, t1, w_c = cci.velocidades_camara(t, R_c_b, 0.08, np.radians(400))
    assert len(t0) == len(t) - 1
    assert np.allclose(w_c, w, atol=1e-12)
    assert np.allclose(t1 - t0, 0.05)


def test_velocidades_camara_salta_huecos_y_saltos_falsos():
    w = np.array([0.0, 0.0, 1.0])
    t = np.array([0.00, 0.05, 0.10, 0.30, 0.35, 0.40])   # entre 0,10 y 0,30 faltan imágenes
    R_c_b = [so3.exp_so3(w * tk).T for tk in t]
    R_c_b[5] = so3.exp_so3([0.0, 0.0, np.pi - 0.01]) @ R_c_b[5]   # tablero "al revés"
    t0, _, w_c = cci.velocidades_camara(t, R_c_b, 0.08, np.radians(400))
    assert list(np.round(t0, 2)) == [0.00, 0.05, 0.30]   # fuera el hueco y los 2 del salto...
    assert np.allclose(w_c, w)


def test_velocidad_media_giroscopio_exacta():
    # ω que crece en línea recta: el trapecio la integra sin error
    t = np.arange(0.0, 2.0, 0.005)
    w = np.column_stack([1.0 + 2.0 * t, -0.5 * t, np.full_like(t, 0.3)])
    Omega = cci.integral_giroscopio(t, w)
    ta, tb = np.array([0.1234, 1.0]), np.array([0.1734, 1.05])
    medias = cci.velocidad_media_giroscopio(t, Omega, ta, tb)
    centro = (ta + tb) / 2                               # media de una recta = su valor en el centro
    assert np.allclose(medias, np.column_stack([1 + 2 * centro, -0.5 * centro, [0.3, 0.3]]), atol=1e-12)


def test_minimo_parabola():
    x = np.array([1.0, 2.0, 3.0])
    assert cci.minimo_parabola(x, (x - 2.3) ** 2) == pytest.approx(2.3)


# =====================================================================
# Calibración con posturas directas (sin imágenes)
# =====================================================================
def calibrar_simulacro(t_d_s, rng, ruido_postura_deg=0.05, malas=0, duracion_s=30.0, amplitud=0.3):
    t_imu, w = giroscopio(duracion_s, amplitud, 3.5e-3, rng)    # ruido de una MPU-6050 a 200 Hz
    R_w_b, p_w_b = tablero_en_el_mundo(0.4)
    tau, sellos = instantes_camara(duracion_s, t_d_s)
    indices_malas = set(rng.choice(len(tau), malas, replace=False))   # repartidas al azar
    R_c_b = []
    for k, tk in enumerate(tau):
        R, _ = postura_verdad(tk, amplitud, R_w_b, p_w_b)
        sigma = 5.0 if k in indices_malas else ruido_postura_deg    # malas: muy mal estimadas
        R_c_b.append(so3.exp_so3(rng.normal(0.0, np.radians(sigma), 3)) @ R)
    return cci.calibrar_giros(sellos, R_c_b, t_imu, w, CFG)


@pytest.mark.parametrize("t_d_ms", [12.3, -27.6])
def test_calibrar_giros_recupera_montaje_desfase_y_sesgo(t_d_ms):
    r = calibrar_simulacro(t_d_ms / 1000, np.random.default_rng(3))
    assert r["t_d_s"] * 1000 == pytest.approx(t_d_ms, abs=0.3)
    assert np.degrees(so3.angulo_entre(r["R_i_c"], R_I_C)) < 0.1
    assert np.allclose(r["b_g"], B_G, atol=2e-3)
    assert np.degrees(r["rms_rad_s"]) < 3.0
    # Un desfase equivocado debe costar claramente más que el bueno (la curva tiene valle)
    candidatos, costes = r["curva"]
    assert costes[np.argmin(np.abs(candidatos - (r["t_d_s"] + 0.03)))] > 2 * costes.min()


def test_calibrar_giros_descarta_posturas_malas():
    r = calibrar_simulacro(0.015, np.random.default_rng(4), malas=20)
    assert r["descartados"] >= 20
    assert r["t_d_s"] * 1000 == pytest.approx(15.0, abs=0.3)
    assert np.degrees(so3.angulo_entre(r["R_i_c"], R_I_C)) < 0.1


def test_poco_giro_en_un_eje_no_vale():
    # Girando casi solo alrededor de un eje, R_i_c queda mal determinada: la herramienta lo dice
    r = calibrar_simulacro(0.01, np.random.default_rng(5), amplitud=0.02)
    motivos = herramienta.motivos_no_valida(r, R_REGLA, CFG)
    assert any("poco giro" in m for m in motivos)


def test_combinar_varias_grabaciones():
    R1 = so3.exp_so3(np.radians([0.1, 0.0, 0.0])) @ R_I_C
    R2 = so3.exp_so3(np.radians([-0.1, 0.0, 0.0])) @ R_I_C
    m = cci.combinar([{"R_i_c": R1, "t_d_s": 0.010}, {"R_i_c": R2, "t_d_s": 0.012},
                      {"R_i_c": R_I_C, "t_d_s": 0.050}])        # la tercera, con un t_d raro
    assert np.degrees(so3.angulo_entre(m["R_i_c"], R_I_C)) < 1e-6
    assert m["t_d_s"] == pytest.approx(0.012)                     # la mediana no se deja arrastrar
    assert np.degrees(m["dispersion_R_rad"]) == pytest.approx(0.1, abs=1e-6)
    assert m["dispersion_td_s"] > 0.02


def test_guardar_camara_imu_se_lee_como_antes(tmp_path):
    ruta = tmp_path / "camara_imu.yaml"
    cci.guardar_camara_imu(ruta, R_I_C, P_I_C, 12_345_678, {"provisional": False})
    with open(ruta, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    # Lo que lee apps/pi_vio: R_i_c, p_i_c_m, t_d_ns y info.provisional
    assert np.allclose(np.array(d["R_i_c"]), R_I_C) and d["p_i_c_m"] == list(P_I_C)
    assert d["t_d_ns"] == 12_345_678 and d["info"]["provisional"] is False


# =====================================================================
# De principio a fin: imágenes dibujadas y grabación EuRoC
# =====================================================================
PX_CUADRO = 40                                   # lado de un cuadro en la imagen del tablero [px]


def imagen_tablero():
    """Tablero de 10 x 7 cuadros (9 x 6 esquinas interiores) con un cuadro de margen blanco."""
    columnas, filas = TABLERO["esquinas_x"] + 1, TABLERO["esquinas_y"] + 1
    img = np.full(((filas + 2) * PX_CUADRO, (columnas + 2) * PX_CUADRO), 255, np.uint8)
    for f in range(filas):
        for c in range(columnas):
            if (f + c) % 2 == 0:
                y, x = (f + 1) * PX_CUADRO, (c + 1) * PX_CUADRO
                img[y:y + PX_CUADRO, x:x + PX_CUADRO] = 0
    return img


def dibujar_vista(textura, R_c_b, t_c_b):
    """Lo que ve la cámara: homografía del plano del tablero (Z = 0) a la imagen.

    Esquina interior (0, 0) del tablero = píxel (2·PX_CUADRO, 2·PX_CUADRO) de la textura.
    De píxel de textura a metros del tablero: A; de metros (Z = 0) a la imagen: K·[r1 r2 t].
    """
    escala = TABLERO["lado_m"] / PX_CUADRO
    A = np.array([[escala, 0.0, -2 * PX_CUADRO * escala],
                  [0.0, escala, -2 * PX_CUADRO * escala],
                  [0.0, 0.0, 1.0]])
    H = K @ np.column_stack([R_c_b[:, 0], R_c_b[:, 1], t_c_b]) @ A
    vista = cv2.warpPerspective(textura, H, (640, 480), flags=cv2.INTER_LINEAR, borderValue=110)
    return cv2.GaussianBlur(vista, (3, 3), 0)


def test_de_principio_a_fin_con_imagenes(tmp_path):
    rng = np.random.default_rng(6)
    duracion_s, amplitud, t_d = 14.0, 0.15, 0.0187
    origen_ns = 10 ** 12                                  # los relojes reales no empiezan en 0

    # 1. IMU
    t_imu, w = giroscopio(duracion_s, amplitud, 3.5e-3, rng)
    euroc.ruta_imu(tmp_path).parent.mkdir(parents=True)
    with open(euroc.ruta_imu(tmp_path), "w", encoding="utf-8") as f:
        f.write(euroc.CABECERA_IMU)
        for tk, wk in zip(t_imu, w):
            f.write(euroc.linea_imu(origen_ns + round(tk * 1e9), wk, [0.0, 0.0, 9.81]))

    # 2. Cámara: imagen del tablero en el instante verdadero, sellada t_d antes
    textura = imagen_tablero()
    R_w_b, p_w_b = tablero_en_el_mundo(0.4)
    tau, sellos = instantes_camara(duracion_s, t_d)
    euroc.carpeta_imagenes(tmp_path).mkdir(parents=True)
    with open(euroc.ruta_cam(tmp_path), "w", encoding="utf-8") as f:
        f.write(euroc.CABECERA_CAM)
        for tk, sello in zip(tau, sellos):
            t_ns = origen_ns + round(sello * 1e9)
            R_c_b, t_c_b = postura_verdad(tk, amplitud, R_w_b, p_w_b)
            cv2.imwrite(str(euroc.carpeta_imagenes(tmp_path) / euroc.nombre_imagen(t_ns)),
                        dibujar_vista(textura, R_c_b, t_c_b))
            f.write(euroc.linea_cam(t_ns, euroc.nombre_imagen(t_ns)))

    # 3. La herramienta, como si fuera una grabación de verdad
    r, cuenta = herramienta.calibrar_grabacion(tmp_path, K, DIST, TABLERO, CFG)
    assert cuenta["sin_tablero"] + cuenta["movidas"] + cuenta["reproyeccion"] < 0.05 * cuenta["imagenes"]
    assert r["t_d_s"] * 1000 == pytest.approx(t_d * 1000, abs=0.5)
    assert np.degrees(so3.angulo_entre(r["R_i_c"], R_I_C)) < 0.2
    assert herramienta.motivos_no_valida(r, R_REGLA, dict(CFG, intervalos_min=200)) == []
    herramienta.dibujar(r, tmp_path / "figura.png", "prueba")
    assert (tmp_path / "figura.png").stat().st_size > 10_000