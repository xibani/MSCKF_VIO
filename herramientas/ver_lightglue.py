"""
herramientas/ver_lightglue.py — Etapa 21: SuperPoint + LightGlue sobre dos imágenes.

Uso, desde la raíz del proyecto, EN EL PC (necesita PyTorch y lightglue):
    python -m herramientas.ver_lightglue                          # dos vistas del simulador
    python -m herramientas.ver_lightglue --instantes 4 6          # más separadas en el tiempo
    python -m herramientas.ver_lightglue --imagenes a.png b.png   # dos fotos tuyas
    python -m herramientas.ver_lightglue --sin-red                # solo FAST + Lucas-Kanade

Con el simulador (etapa 20) se conoce la verdad: de cada pareja se sabe si es correcta, y se
compara con lo que hace el front-end actual (FAST + Lucas-Kanade) con las mismas dos imágenes.
Con fotos reales no hay verdad: se dibujan las parejas y se cuentan.
Guarda datos/lightglue_parejas.png (y datos/lightglue_klt.png con el simulador).
"""

import argparse

import cv2
import numpy as np

from vio import emparejador as emp
from vio import escena
from vio import frontend
from vio import modelo_camara as mc
from vio import simulador as sim
from vio.config import RAIZ_PROYECTO, cargar_config


def vistas_del_simulador(instantes, K, dist, tamano):
    """Dos imágenes de la habitación con textura, con su profundidad y su pose verdaderas.

    instantes: (t_a, t_b) [s] de la trayectoria del simulador.
    Devuelve: lista de dos dict con gris, profundidad y pose (R_c_w, t_c_w).
    """
    cfg = cargar_config("simulador")["vio"]
    rng = np.random.default_rng(cfg["semilla"])
    habitacion = escena.crear_habitacion(cfg["mundo"]["habitacion_m"], cfg["textura"], rng)
    rayos = escena.rayos_camara(K, dist, tamano)
    R_c_i, t_c_i = sim.extrinseca(cfg["camara"])
    vistas = []
    for t_s in instantes:
        R_w_i, p_w = sim.pose_vio(t_s, cfg["trayectoria"])
        pose = sim.pose_camara(R_w_i, p_w, R_c_i, t_c_i)
        ideal, profundidad = escena.renderizar_habitacion(habitacion, pose[0], pose[1], rayos)
        vistas.append({"gris": escena.acabar_imagen(ideal, cfg["textura"], rng),
                       "profundidad": profundidad, "pose": pose})
    return vistas


def parejas_fast_klt(gris_a, gris_b):
    """Lo que hace el front-end actual con dos imágenes: FAST en A y Lucas-Kanade hasta B.

    Devuelve: (uv_a (K, 2), uv_b (K, 2), n_detectados): solo los puntos que Lucas-Kanade da
    por buenos (ida y vuelta, etapa 8).
    """
    cfg = cargar_config("frontend")
    margen = cfg["seguimiento"]["margen_borde_px"]
    puntos = frontend.detectar(gris_a, np.zeros((0, 2), np.float32), cfg["deteccion"], margen)
    seguidos, validos, _ = frontend.seguir(gris_a, gris_b, puntos, cfg["seguimiento"])
    return puntos[validos].astype(float), seguidos[validos].astype(float), len(puntos)


def linea_resumen(nombre, n_puntos, evaluacion):
    return (f"  {nombre:16s} puntos {n_puntos:5d} | parejas {evaluacion['n']:5d} | correctas "
            f"{evaluacion['n_correctas']:5d} ({100 * evaluacion['fraccion_correctas']:5.1f} %)"
            f" | error mediano {evaluacion['error_mediano_px']:.2f} px")


def main():
    cfg = cargar_config("lightglue")
    parser = argparse.ArgumentParser(description="SuperPoint + LightGlue sobre dos imágenes")
    parser.add_argument("--imagenes", nargs=2, default=None, metavar=("A", "B"),
                        help="dos fotos (sin ellas: dos vistas del simulador)")
    parser.add_argument("--instantes", type=float, nargs=2, default=cfg["evaluacion"]["instantes_s"],
                        metavar=("T_A", "T_B"), help="segundos de la trayectoria simulada")
    parser.add_argument("--sin-red", action="store_true",
                        help="no usar SuperPoint ni LightGlue (para probar sin PyTorch)")
    args = parser.parse_args()
    carpeta = RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"]
    carpeta.mkdir(parents=True, exist_ok=True)
    umbral = cfg["evaluacion"]["umbral_correcta_px"]

    # --- 1. Las dos imágenes ---
    vistas = None
    if args.imagenes is not None:
        grises = [cv2.imread(ruta, cv2.IMREAD_GRAYSCALE) for ruta in args.imagenes]
        for ruta, gris in zip(args.imagenes, grises):
            if gris is None:
                raise SystemExit(f"No puedo leer la imagen {ruta}")
        print(f"Imágenes: {args.imagenes[0]} y {args.imagenes[1]} ({grises[0].shape[1]} x {grises[0].shape[0]})")
    else:
        K, dist, tamano = mc.cargar_intrinsecos()
        vistas = vistas_del_simulador(args.instantes, K, dist, tamano)
        grises = [v["gris"] for v in vistas]
        centro_a = -vistas[0]["pose"][0].T @ vistas[0]["pose"][1]
        centro_b = -vistas[1]["pose"][0].T @ vistas[1]["pose"][1]
        print(f"Simulador: t = {args.instantes[0]} s y {args.instantes[1]} s; la cámara se ha "
              f"movido {np.linalg.norm(centro_b - centro_a):.2f} m entre las dos")

    # --- 2. El front-end actual, para comparar (solo con el simulador) ---
    if vistas is not None:
        uv_a, uv_b, n_detectados = parejas_fast_klt(grises[0], grises[1])
        verdad_b, visible = emp.posicion_verdadera_en_b(
            uv_a, vistas[0]["profundidad"], vistas[0]["pose"], vistas[1]["pose"], K, dist)
        ev_klt = emp.evaluar_parejas(uv_b, verdad_b, visible, umbral)
        print(linea_resumen("FAST + KLT", n_detectados, ev_klt))
        cv2.imwrite(str(carpeta / "lightglue_klt.png"),
                    emp.dibujar_parejas(grises[0], grises[1], uv_a, uv_b, ev_klt["correctas"]))

    if args.sin_red:
        print("Sin red (--sin-red): no se ejecutan SuperPoint ni LightGlue")
        return

    # --- 3. SuperPoint + LightGlue ---
    emparejador = emp.EmparejadorLightGlue(cfg)
    print(f"Redes cargadas en: {emparejador.dispositivo}")
    # La primera vez que se usa una red es más lenta (reserva memoria): se repite y se mide
    # la segunda vuelta, que es la representativa
    for _ in range(2):
        a = emparejador.extraer(grises[0])
        b = emparejador.extraer(grises[1])
        indices, confianza, t_emparejar_ms = emparejador.emparejar(a, b)
    uv_a = a["puntos"][indices[:, 0]]
    uv_b = b["puntos"][indices[:, 1]]

    correctas = None
    if vistas is not None:
        verdad_b, visible = emp.posicion_verdadera_en_b(
            uv_a, vistas[0]["profundidad"], vistas[0]["pose"], vistas[1]["pose"], K, dist)
        ev = emp.evaluar_parejas(uv_b, verdad_b, visible, umbral)
        correctas = ev["correctas"]
        print(linea_resumen("SuperPoint + LG", len(a["puntos"]), ev))
    else:
        print(f"  SuperPoint: {len(a['puntos'])} y {len(b['puntos'])} puntos | LightGlue: {len(indices)} parejas")
    if len(confianza) > 0:
        print(f"  confianza de las parejas: mediana {np.median(confianza):.2f}, mínima {confianza.min():.2f}")
    print(f"  tiempos: SuperPoint {a['tiempo_ms']:.0f} ms por imagen, LightGlue {t_emparejar_ms:.0f} ms por pareja")

    ruta = carpeta / "lightglue_parejas.png"
    cv2.imwrite(str(ruta), emp.dibujar_parejas(grises[0], grises[1], uv_a, uv_b, correctas))
    print(f"Guardado {ruta}")


if __name__ == "__main__":
    main()
