"""
vio/inicializacion.py — Estado inicial a partir de un tramo de IMU en reposo.

Marco del mundo (w): z hacia arriba (contra la gravedad); x e y horizontales,
con x en la dirección horizontal que mira la placa al arrancar (rumbo = 0).
Origen: donde está la placa al arrancar.
"""

import numpy as np

from vio import so3
from vio.constantes import G


def estimar_reposo(w, a):
    """Estadística de un tramo de IMU en reposo.

    w: array (N, 3) giroscopio [rad/s].  a: array (N, 3) acelerómetro [m/s²].
    Devuelve: dict con
        b_g:        (3,) media del giroscopio = sesgo [rad/s]
        sigma_gir:  (3,) desviación típica del giroscopio [rad/s]
        a_media:    (3,) media del acelerómetro [m/s²] (apunta hacia arriba)
        norma_a:    |a_media| [m/s²]
    """
    a_media = a.mean(axis=0)
    return {
        "b_g": w.mean(axis=0),
        "sigma_gir": w.std(axis=0),
        "a_media": a_media,
        "norma_a": float(np.linalg.norm(a_media)),
    }


def comprobar_reposo(reposo, cfg_ini):
    """¿Estaba de verdad quieto? Devuelve: lista de problemas (vacía = todo bien)."""
    problemas = []
    sigma_max = np.radians(cfg_ini["sigma_gir_max_dps"])
    if reposo["sigma_gir"].max() > sigma_max:
        problemas.append(f"giroscopio con σ = {np.degrees(reposo['sigma_gir'].max()):.2f} °/s"
                         f" (máx {cfg_ini['sigma_gir_max_dps']}): se ha movido")
    error_norma = abs(reposo["norma_a"] - G)
    if error_norma > cfg_ini["error_norma_acc_max"]:
        problemas.append(f"|a| = {reposo['norma_a']:.2f} m/s², lejos de g: ¿se ha movido?")
    return problemas


def inclinacion_desde_acelerometro(a):
    """Alabeo y cabeceo [rad] a partir de una medida del acelerómetro en reposo.

    a: array (3,) [m/s²]. En reposo a = R_w_i^T @ (0, 0, g), y despejando:
        alabeo  = atan2(a_y, a_z)
        cabeceo = atan2(-a_x, sqrt(a_y² + a_z²))
    (El rumbo no se puede saber: girar sobre la vertical no cambia a.)
    """
    alabeo = np.arctan2(a[1], a[2])
    cabeceo = np.arctan2(-a[0], np.sqrt(a[1] ** 2 + a[2] ** 2))
    return alabeo, cabeceo


def rotacion_desde_gravedad(a_media):
    """R_w_i inicial: inclinación del acelerómetro y rumbo 0.

    a_media: array (3,) media del acelerómetro en reposo [m/s²].
    Devuelve: array (3, 3) tal que R_w_i @ a_media apunta a +z del mundo.
    """
    alabeo, cabeceo = inclinacion_desde_acelerometro(a_media)
    R_y = so3.exp_so3([0.0, cabeceo, 0.0])     # giro de cabeceo sobre y
    R_x = so3.exp_so3([alabeo, 0.0, 0.0])      # giro de alabeo sobre x
    return R_y @ R_x                            # = Rz(0) @ Ry(cabeceo) @ Rx(alabeo)