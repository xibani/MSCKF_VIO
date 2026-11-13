"""
herramientas/calibrar_camara_imu.py — Giro cámara-IMU (R_i_c) y desfase t_d con el tablero (etapa 7.4).

Uso, desde ~/proyectos/MSCKF_VIO, con grabaciones de herramientas.grabar moviendo la placa
delante de un tablero quieto: la plantilla AprilGrid de Kalibr o el de ajedrez de la 7.1
(config/calibracion.yaml -> camara_imu.objetivo):
    python -m herramientas.calibrar_camara_imu datos/20261113_180000
    python -m herramientas.calibrar_camara_imu datos/20261113_180000 datos/20261113_180300 datos/20261113_180600
    python -m herramientas.calibrar_camara_imu datos/2026111* --guardar   # escribe config/camara_imu.yaml
Con varias grabaciones da cada resultado, su dispersión (repetibilidad) y la media.
Guarda una figura calibracion_camara_imu.png dentro de cada grabación.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                     # sin ventana: la figura se guarda en un PNG
import matplotlib.pyplot as plt
import numpy as np

from captura import euroc
from vio import calibracion_camara as cc
from vio import calibracion_camara_imu as cci
from vio import modelo_camara as mc
from vio import so3
from vio.config import cargar_config, ruta_config

AVISO_CADA_IMAGENES = 200                 # cada cuántas imágenes se imprime el progreso


# =====================================================================
# 1. Leer una grabación
# =====================================================================
def preparar_objetivo(cfg_cal):
    """Función que busca el tablero elegido en config/calibracion.yaml (camara_imu.objetivo).

    "tablero": el de ajedrez de la 7.1 (sección tablero). "aprilgrid": la plantilla de Kalibr
    (sección aprilgrid). Devuelve: función gris -> (esquinas (N, 2), puntos_3d (N, 3)), o
        (None, None) si no lo encuentra.
    """
    objetivo = cfg_cal["camara_imu"]["objetivo"]
    if objetivo == "tablero":
        t = cfg_cal["tablero"]
        puntos_3d = cc.puntos_tablero(t["esquinas_x"], t["esquinas_y"], t["lado_m"])

        def detectar(gris):
            esquinas = cc.detectar_esquinas(gris, t["esquinas_x"], t["esquinas_y"])
            return (None, None) if esquinas is None else (esquinas, puntos_3d)
        return detectar

    if objetivo == "aprilgrid":
        a = cfg_cal["aprilgrid"]
        detector = cci.crear_detector_aprilgrid(a["bits_borde"])
        modelo = cci.puntos_aprilgrid(a["tags_x"], a["tags_y"], a["lado_tag_m"], a["espacio_relativo"])

        def detectar(gris):
            return cci.detectar_aprilgrid(gris, detector, modelo, a["tags_min"])
        return detectar

    raise ValueError(f'camara_imu.objetivo = "{objetivo}": tiene que ser "tablero" o "aprilgrid"')


def posturas_grabacion(carpeta, K, dist, detectar, cfg):
    """Detecta el tablero en cada imagen y calcula su postura.

    detectar: función de preparar_objetivo.
    Devuelve: (t_ns (M,) int64, R_c_b lista de M (3, 3), cuenta) con solo las imágenes
        válidas. cuenta: dict con cuántas imágenes había y por qué se descartó cada una.
    """
    t_ns, nombres = euroc.leer_cam(euroc.ruta_cam(carpeta))
    carpeta_png = euroc.carpeta_imagenes(carpeta)
    cuenta = {"imagenes": len(nombres), "movidas": 0, "sin_tablero": 0, "reproyeccion": 0}
    t_validas, R_validas = [], []
    for k, (t, nombre) in enumerate(zip(t_ns, nombres)):
        if k % AVISO_CADA_IMAGENES == 0:
            print(f"  imagen {k} de {len(nombres)}")
        gris = cv2.imread(str(carpeta_png / nombre), cv2.IMREAD_GRAYSCALE)
        if gris is None or cc.nitidez(gris) < cfg["nitidez_min"]:
            cuenta["movidas"] += 1
            continue
        esquinas, puntos_3d = detectar(gris)
        if esquinas is None:
            cuenta["sin_tablero"] += 1
            continue
        R_c_b, _, error = cci.postura_tablero(esquinas, puntos_3d, K, dist)
        if R_c_b is None or error > cfg["reproyeccion_max_px"]:
            cuenta["reproyeccion"] += 1
            continue
        t_validas.append(int(t))
        R_validas.append(R_c_b)
    return np.array(t_validas, dtype=np.int64), R_validas, cuenta


def calibrar_grabacion(carpeta, K, dist, detectar, cfg):
    """Calibra con una grabación. Devuelve: (resultado de cci.calibrar_giros o None, cuenta)."""
    t_cam_ns, R_c_b, cuenta = posturas_grabacion(carpeta, K, dist, detectar, cfg)
    t_imu_ns, w, _ = euroc.leer_imu(euroc.ruta_imu(carpeta))
    if len(t_cam_ns) < 3:
        return None, cuenta
    # Origen de tiempos común: la primera muestra de la IMU (en segundos, sin perder precisión)
    origen = t_imu_ns[0]
    t_imu_s = (t_imu_ns - origen) / 1e9
    t_cam_s = (t_cam_ns - origen) / 1e9
    return cci.calibrar_giros(t_cam_s, R_c_b, t_imu_s, w, cfg), cuenta


# =====================================================================
# 2. Comprobaciones e informe
# =====================================================================
def motivos_no_valida(r, R_regla, cfg):
    """Por qué una calibración no es de fiar. Devuelve: lista de textos (vacía = vale)."""
    motivos = []
    if r["intervalos"] < cfg["intervalos_min"]:
        motivos.append(f"solo {r['intervalos']} intervalos buenos (mín. {cfg['intervalos_min']}): "
                       "graba más tiempo con el tablero a la vista")
    for eje, s in zip("xyz", np.degrees(r["excitacion"])):
        if s < cfg["excitacion_min_dps"]:
            motivos.append(f"poco giro alrededor del eje {eje} de la cámara ({s:.1f} °/s, "
                           f"mín. {cfg['excitacion_min_dps']})")
    if np.degrees(r["rms_rad_s"]) > cfg["residuo_max_dps"]:
        motivos.append(f"residuo {np.degrees(r['rms_rad_s']):.2f} °/s (máx. {cfg['residuo_max_dps']}): "
                       "cámara y giroscopio no cuadran")
    angulo = np.degrees(so3.angulo_entre(R_regla, r["R_i_c"]))
    if angulo > cfg["angulo_max_regla_deg"]:
        motivos.append(f"{angulo:.1f}° de diferencia con la R_i_c de regla (máx. "
                       f"{cfg['angulo_max_regla_deg']}°): ¿ejes de la IMU o del tablero mal?")
    return motivos


def imprimir_resultado(r, R_regla):
    print(f"  t_d   = {r['t_d_s'] * 1000:+.2f} ms   (t_cámara_corregido = t_cámara + t_d)")
    print(f"  R_i_c = {np.array2string(r['R_i_c'], precision=4, suppress_small=True, prefix='          ')}")
    # Diferencia con la de regla, como giro pequeño en ejes de la IMU: "la cámara está girada
    # tantos grados alrededor de cada eje de la IMU respecto a como se suponía"
    giro = np.degrees(so3.log_so3(r["R_i_c"] @ R_regla.T))
    print(f"  respecto a la de regla: {np.linalg.norm(giro):.2f}° "
          f"(alrededor de x, y, z de la IMU: {giro[0]:+.2f}°, {giro[1]:+.2f}°, {giro[2]:+.2f}°)")
    print(f"  sesgo del giroscopio = {np.degrees(r['b_g']).round(3)} °/s")
    print(f"  residuo RMS = {np.degrees(r['rms_rad_s']):.2f} °/s con {r['intervalos']} intervalos "
          f"({r['descartados']} descartados)")
    print(f"  giro en cada eje de la cámara (desv. típica): {np.degrees(r['excitacion']).round(1)} °/s")


def dibujar(r, ruta_png, titulo):
    """Curva del coste frente a t_d y las velocidades de cámara y giroscopio superpuestas."""
    fig, ejes = plt.subplots(4, 1, figsize=(12, 12))
    candidatos, costes = r["curva"]
    ejes[0].plot(candidatos * 1000, np.degrees(costes))
    ejes[0].axvline(r["t_d_s"] * 1000, color="r", ls="--", label=f"t_d = {r['t_d_s'] * 1000:+.2f} ms")
    ejes[0].set_xlabel("t_d de prueba [ms]")
    ejes[0].set_ylabel("residuo RMS [°/s]")
    ejes[0].set_title(titulo)
    ejes[0].legend()
    for eje in range(3):
        ejes[eje + 1].plot(r["t_medio"], np.degrees(r["w_i"][:, eje]), lw=1, label="giroscopio")
        ejes[eje + 1].plot(r["t_medio"], np.degrees(r["w_c_en_i"][:, eje]), ".", ms=2,
                           label="cámara (pasada a ejes de la IMU)")
        ejes[eje + 1].set_ylabel(f"ω_{'xyz'[eje]} IMU [°/s]")
        ejes[eje + 1].legend(loc="upper right")
    ejes[3].set_xlabel("tiempo desde el inicio de la IMU [s]")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=90)
    plt.close(fig)


# =====================================================================
# 3. Programa
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Calibración cámara-IMU con el giroscopio (etapa 7.4)")
    parser.add_argument("carpetas", nargs="+", help="grabaciones de herramientas.grabar (contienen mav0/)")
    parser.add_argument("--guardar", action="store_true",
                        help="escribir config/camara_imu.yaml con el resultado (si es válido)")
    args = parser.parse_args()

    cfg_cal = cargar_config("calibracion")
    cfg = cfg_cal["camara_imu"]
    detectar = preparar_objetivo(cfg_cal)
    print(f"Tablero: {cfg['objetivo']} (config/calibracion.yaml -> camara_imu.objetivo)")
    K, dist, _ = mc.cargar_intrinsecos()
    actual = cargar_config("camara_imu")
    R_regla = np.array(actual["R_i_c"], dtype=float)

    validos = []
    for carpeta in [Path(c) for c in args.carpetas]:
        print(f"\n=== {carpeta} ===")
        r, cuenta = calibrar_grabacion(carpeta, K, dist, detectar, cfg)
        print(f"  imágenes: {cuenta['imagenes']} | movidas {cuenta['movidas']} | sin tablero "
              f"{cuenta['sin_tablero']} | con mucho error {cuenta['reproyeccion']}")
        if r is None:
            print(f"  NO VALE: casi ninguna imagen con el tablero (¿es el de camara_imu.objetivo = "
                  f"{cfg['objetivo']}?)")
            continue
        imprimir_resultado(r, R_regla)
        dibujar(r, carpeta / "calibracion_camara_imu.png", str(carpeta.name))
        print(f"  figura: {carpeta / 'calibracion_camara_imu.png'}")
        motivos = motivos_no_valida(r, R_regla, cfg)
        for motivo in motivos:
            print(f"  NO VALE: {motivo}")
        if not motivos:
            print("  >>> VÁLIDA")
            validos.append((carpeta, r))

    if not validos:
        print("\nNinguna grabación válida: no se guarda nada.")
        return

    # --- Media y repetibilidad ---
    media = cci.combinar([r for _, r in validos])
    print(f"\n=== RESULTADO con {len(validos)} grabación(es) válida(s) ===")
    print(f"  t_d   = {media['t_d_s'] * 1000:+.2f} ms (mediana)")
    print(f"  R_i_c = {np.array2string(media['R_i_c'], precision=4, suppress_small=True, prefix='          ')}")
    repetible = True
    if len(validos) > 1:
        disp_td = media["dispersion_td_s"] * 1000
        disp_R = np.degrees(media["dispersion_R_rad"])
        print(f"  dispersión: t_d ± {disp_td:.2f} ms | R hasta {disp_R:.2f}° de la media")
        repetible = disp_td <= cfg["dispersion_td_max_ms"] and disp_R <= cfg["dispersion_R_max_deg"]
        print(">>> REPETIBLE" if repetible else
              f">>> NO REPETIBLE (máx. {cfg['dispersion_td_max_ms']} ms y {cfg['dispersion_R_max_deg']}°)")
    else:
        print("  (con una sola grabación no se puede medir la repetibilidad: mejor 3)")

    if not args.guardar:
        print("\nPara guardarlo: repite la orden con --guardar")
        return
    if not repetible:
        print("\nNo se guarda: las grabaciones no coinciden entre sí.")
        return
    info = {"provisional": False,
            "metodo": "etapa 7.4: giroscopio frente a postura del tablero (R_i_c y t_d); p_i_c_m con regla",
            "grabaciones": [str(c) for c, _ in validos],
            "rms_dps": [round(float(np.degrees(r["rms_rad_s"])), 3) for _, r in validos],
            "t_d_ms": [round(r["t_d_s"] * 1000, 3) for _, r in validos],
            "fecha": datetime.now().isoformat(timespec="seconds")}
    ruta = ruta_config("camara_imu")
    cci.guardar_camara_imu(ruta, media["R_i_c"], actual["p_i_c_m"],
                           round(media["t_d_s"] * 1e9), info)
    print(f"\nGuardado en {ruta}")


if __name__ == "__main__":
    main()