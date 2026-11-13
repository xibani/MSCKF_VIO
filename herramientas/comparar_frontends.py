"""
herramientas/comparar_frontends.py — Etapa 23: el mismo VIO con cada front-end, sobre la misma grabación.

Uso, desde la raíz del proyecto:
    python -m herramientas.comparar_frontends datos/sim_textura --sin-calibracion-acc
    python -m herramientas.comparar_frontends datos/sim_textura --sin-calibracion-acc --cada 6
    python -m herramientas.comparar_frontends datos/sim_textura --sin-calibracion-acc --solo klt
Pasa la grabación por el VIO completo (apps/pi_vio, sin red ni mapa) tres veces:
    klt          FAST + Lucas-Kanade (etapa 8)
    lg_esencial  SuperPoint + LightGlue con el rechazo de la matriz esencial (etapa 22)
    lg_giro      SuperPoint + LightGlue con el rechazo del giroscopio (etapa 23)
y escribe una tabla con las pistas, lo que usa y rechaza el filtro, los tiempos y, si la
grabación trae verdad.csv (las del simulador), el error.
--cada N usa solo una imagen de cada N (la IMU, entera): es la manera de tener un caso
difícil sin grabar nada nuevo. Con N = 6, de 20 imágenes por segundo quedan 3,3 y entre dos
imágenes seguidas todo se mueve seis veces más.
Las dos variantes lg_ necesitan PyTorch (PC); con --solo klt funciona también en la Pi.
La tabla se guarda además en la carpeta de la grabación (comparacion_frontends_cadaN.txt).
"""

import argparse
import copy
from pathlib import Path

import numpy as np

from apps.pi_vio import ADELANTO_IMU_NS, cargar_calibracion_acc, eventos_grabacion
from herramientas.comparar_trayectoria import alinear
from vio import modelo_camara as mc
from vio import so3
from vio.config import cargar_config, ruta_config
from vio.frontend import FrontEnd
from vio.frontend_lightglue import FrontEndLightGlue
from vio.pipeline import PipelineVio, leer_poses

VARIANTES = ("klt", "lg_esencial", "lg_giro")
TITULOS = {"klt": "FAST+KLT", "lg_esencial": "LG esencial", "lg_giro": "LG giro"}


def crear_frontend(variante, K, dist, emparejador=None):
    """El front-end de una variante.

    emparejador: las redes ya creadas (EmparejadorLightGlue), para no cargarlas dos veces;
    solo hace falta con las variantes lg_.
    """
    if variante == "klt":
        return FrontEnd(K, dist, cargar_config("frontend"))
    cfg = copy.deepcopy(cargar_config("lightglue"))
    cfg["geometria"]["metodo"] = "giro" if variante == "lg_giro" else "esencial"
    return FrontEndLightGlue(K, dist, cfg, emparejador)


def ejecutar_vio(carpeta, frontend, cada, K, dist, calib_acc=None):
    """Pasa una grabación por el VIO completo con el front-end dado.

    cada: se usa una imagen de cada `cada` (1 = todas). La IMU se usa entera.
    Devuelve: (resultados, pistas): lo que devuelve PipelineVio.nueva_imagen en cada imagen
    con pose, y todas las pistas del front-end (las terminadas y las que seguían vivas).
    """
    cfgs = {nombre: cargar_config(nombre) for nombre in ("vio", "frontend", "msckf", "triangulacion")}
    ext = cargar_config("camara_imu")
    R_c_i = np.array(ext["R_i_c"], dtype=float).T
    t_c_i = -R_c_i @ np.array(ext["p_i_c_m"], dtype=float)
    pipeline = PipelineVio(K, dist, R_c_i, t_c_i, ext["t_d_ns"], cfgs, calib_acc, frontend)

    resultados, pistas = [], []
    n_imagen = 0
    for evento in eventos_grabacion(carpeta, max(ext["t_d_ns"], 0) + ADELANTO_IMU_NS):
        if evento[0] == "imu":
            pipeline.nueva_imu(evento[1], evento[2], evento[3])
            continue
        usar = n_imagen % cada == 0
        n_imagen += 1
        if not usar:
            continue                                   # imagen saltada (--cada)
        r = pipeline.nueva_imagen(evento[1], evento[2])
        if r is not None:
            resultados.append(r)
            pistas += r["pistas_terminadas"]
    pistas += pipeline.frontend.terminar_todas()
    return resultados, pistas


def resumir(resultados, pistas, verdad=None):
    """Los números de una ejecución, en un diccionario (una fila de la tabla cada uno).

    verdad: (t_ns, p_w, R_w_i) de verdad.csv, o None si la grabación no la trae.
    """
    if len(resultados) == 0:
        return {"imagenes": 0}
    longitudes = np.array([p.longitud() for p in pistas])
    fe = [r["frontend"] for r in resultados]
    ms = [r["msckf"] for r in resultados]
    resumen = {
        "imagenes": len(resultados),
        "activos": float(np.mean([e["activos"] for e in fe])),
        "pistas": len(longitudes),
        "longitud_mediana": float(np.median(longitudes)),
        "utiles": int((longitudes >= 3).sum()),
        "perdidas": sum(e["perdidos_klt"] for e in fe),
        "ransac": sum(e["rechazados_ransac"] for e in fe),
        "usadas": sum(e["usadas"] for e in ms),
        "rechazadas_filtro": sum(e["reproyeccion"] + e["chi2"] for e in ms),
        "t_frontend_ms": float(np.median([r["tiempos_ms"]["frontend"] for r in resultados])),
        "t_total_ms": float(np.median([r["tiempos_ms"]["total"] for r in resultados])),
    }
    if verdad is not None:
        t_est = np.array([r["t_ns"] for r in resultados])
        p_est = np.array([r["p_w"] for r in resultados])
        R_est = np.array([r["R_w_i"] for r in resultados])
        p_ver, R_ver = alinear(t_est, p_est, R_est, *verdad)
        error = np.linalg.norm(p_est - p_ver, axis=1)
        giro = np.degrees([np.linalg.norm(so3.log_so3(Rv.T @ Re)) for Re, Rv in zip(R_est, R_ver)])
        resumen["error_final_cm"] = 100.0 * float(error[-1])
        resumen["error_max_cm"] = 100.0 * float(error.max())
        resumen["giro_max_grados"] = float(giro.max())
    return resumen


FILAS = [                                             # (clave, texto, decimales)
    ("imagenes", "imágenes con pose", 0),
    ("activos", "puntos activos por imagen (media)", 0),
    ("pistas", "pistas", 0),
    ("longitud_mediana", "longitud mediana [imágenes]", 0),
    ("utiles", "pistas de 3 imágenes o más", 0),
    ("perdidas", "perdidas por el seguimiento", 0),
    ("ransac", "rechazadas por geometría", 0),
    ("usadas", "usadas por el filtro", 0),
    ("rechazadas_filtro", "rechazadas por el filtro", 0),
    ("error_final_cm", "error de posición final [cm]", 1),
    ("error_max_cm", "error de posición máximo [cm]", 1),
    ("giro_max_grados", "error de orientación máximo [°]", 2),
    ("t_frontend_ms", "front-end por imagen [ms] (mediana)", 1),
    ("t_total_ms", "total por imagen [ms] (mediana)", 1),
]


def tabla(resumenes):
    """La tabla en texto: una columna por variante y una fila por medida.

    resumenes: dict variante -> lo que devuelve resumir(). Si a una variante le falta una
    medida (p. ej. el error, sin verdad.csv), en su casilla va un guion.
    """
    lineas = [f"{'':38s}" + "".join(f"{TITULOS.get(v, v):>14s}" for v in resumenes)]
    for clave, texto, decimales in FILAS:
        if not any(clave in r for r in resumenes.values()):
            continue                                   # nadie tiene esta medida: fila fuera
        casillas = ""
        for r in resumenes.values():
            casillas += f"{r[clave]:14.{decimales}f}" if clave in r else f"{'-':>14s}"
        lineas.append(f"{texto:38s}" + casillas)
    return "\n".join(lineas)


def main():
    parser = argparse.ArgumentParser(description="Compara los front-ends sobre una grabación")
    parser.add_argument("carpeta", help="carpeta de la grabación (la que contiene mav0/)")
    parser.add_argument("--cada", type=int, default=1, help="usar una imagen de cada N")
    parser.add_argument("--solo", nargs="+", choices=VARIANTES, default=list(VARIANTES),
                        help="variantes que se ejecutan (por defecto, las tres)")
    parser.add_argument("--sin-calibracion-acc", action="store_true",
                        help="no corregir el acelerómetro (grabaciones del simulador)")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)

    K, dist, _ = mc.cargar_intrinsecos()
    calib_acc = cargar_calibracion_acc(ruta_config("imu_calibracion"), args.sin_calibracion_acc)
    verdad = leer_poses(carpeta / "verdad.csv") if (carpeta / "verdad.csv").exists() else None
    fps = cargar_config("camara")["fps"]

    # Las redes se cargan una sola vez, y solo si alguna variante las necesita
    emparejador = None
    if any(v != "klt" for v in args.solo):
        from vio.emparejador import EmparejadorLightGlue

        emparejador = EmparejadorLightGlue(cargar_config("lightglue"))
        print(f"Redes cargadas en: {emparejador.dispositivo}")

    resumenes = {}
    for variante in args.solo:
        print(f"Ejecutando {variante}...")
        frontend = crear_frontend(variante, K, dist, emparejador)
        resultados, pistas = ejecutar_vio(carpeta, frontend, args.cada, K, dist, calib_acc)
        resumenes[variante] = resumir(resultados, pistas, verdad)

    cabecera = (f"Grabación: {carpeta} | una imagen de cada {args.cada}"
                f" ({fps / args.cada:.1f} por segundo)")
    texto = cabecera + "\n" + tabla(resumenes)
    print()
    print(texto)
    ruta = carpeta / f"comparacion_frontends_cada{args.cada}.txt"
    ruta.write_text(texto + "\n", encoding="utf-8")
    print(f"Guardado {ruta}")


if __name__ == "__main__":
    main()
