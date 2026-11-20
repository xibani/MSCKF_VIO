"""
herramientas/comparar_estimadores.py — Etapa 37: MSCKF frente a VINS sobre la misma grabación.

Uso, desde la raíz del repositorio:
    python -m herramientas.comparar_estimadores datos/vio_20261120_101500
    python -m herramientas.comparar_estimadores datos/X --sin-calibracion-acc        # sintéticas
    python -m herramientas.comparar_estimadores datos/X --distancia-real 2.0         # recta medida
    python -m herramientas.comparar_estimadores datos/X --vins-reposo                # VINS arrancando quieto
Pasa la grabación por el VIO completo (front-end + estimador, como apps/pi_vio sin red ni mapa)
dos veces, con el MSCKF y con VINS, y escribe una tabla:
    - arranque (s desde la primera imagen), imágenes con pose;
    - prueba del protocolo: recorrido, cierre del lazo [m y %] (vio/lazo.py), alejamiento máximo
      y, con --distancia-real, error de escala de la recta medida;
    - si la grabación trae verdad.csv (simulador): error RMS y final de posición tras alinear la
      primera pose, y escala (recorrido estimado / verdadero) en el mismo intervalo;
    - recuentos y tiempos por imagen.
Guarda la tabla y las trayectorias (vio_trayectoria_msckf.csv, vio_trayectoria_vins.csv) en la
carpeta de la grabación.
"""

import argparse
import copy
from pathlib import Path

import numpy as np

from apps.pi_vio import ADELANTO_IMU_NS, cargar_calibracion_acc, eventos_grabacion
from herramientas.comparar_trayectoria import alinear
from vio import lazo
from vio import modelo_camara as mc
from vio.config import cargar_config, ruta_config
from vio.pipeline import CABECERA_POSES, PipelineVio, leer_poses, linea_pose


def ejecutar(carpeta, tipo, arranque_vins, calib_acc):
    """Pasa la grabación por el VIO con el estimador dado.

    Devuelve: dict con t_ns (N,), p (N, 3), R (N, 3, 3) de las imágenes con pose, los
    resultados de nueva_imagen, el instante de la primera imagen y la de arranque.
    """
    cfgs = {n: copy.deepcopy(cargar_config(n)) for n in ("vio", "frontend", "msckf", "triangulacion", "vins")}
    cfgs["vio"].setdefault("estimador", {})["tipo"] = tipo
    cfgs["vio"]["estimador"]["inicializacion_vins"] = arranque_vins
    K, dist, _ = mc.cargar_intrinsecos()
    ext = cargar_config("camara_imu")
    R_c_i = np.array(ext["R_i_c"], dtype=float).T
    t_c_i = -R_c_i @ np.array(ext["p_i_c_m"], dtype=float)
    pipeline = PipelineVio(K, dist, R_c_i, t_c_i, ext["t_d_ns"], cfgs, calib_acc)

    resultados, t_primera = [], None
    for evento in eventos_grabacion(carpeta, max(ext["t_d_ns"], 0) + ADELANTO_IMU_NS):
        if evento[0] == "imu":
            pipeline.nueva_imu(evento[1], evento[2], evento[3])
            continue
        if t_primera is None:
            t_primera = evento[1] + ext["t_d_ns"]
        r = pipeline.nueva_imagen(evento[1], evento[2])
        if r is not None:
            resultados.append(r)
    if not resultados:
        return {"resultados": [], "t_primera": t_primera}
    return {"resultados": resultados, "t_primera": t_primera,
            "t_ns": np.array([r["t_ns"] for r in resultados], dtype=np.int64),
            "p": np.array([r["p_w"] for r in resultados]),
            "R": np.array([r["R_w_i"] for r in resultados])}


def resumir(e, cfg_prueba, verdad=None, distancia_real=None, t_desde=None):
    """Números de una ejecución (dict). t_desde: comparar con la verdad solo desde ese instante
    (el arranque del estimador que arranca más tarde: mismo intervalo para los dos)."""
    res = e["resultados"]
    if not res:
        return {"imagenes": 0}
    t_s = (e["t_ns"] - e["t_ns"][0]) / 1e9
    m = lazo.metricas_lazo(t_s, e["p"], cfg_prueba["paso_min_m"], cfg_prueba["recorrido_min_m"])
    fila = {"arranque_s": (e["t_ns"][0] - e["t_primera"]) / 1e9, "imagenes": len(res),
            "recorrido_m": m["recorrido_m"], "cierre_m": m["cierre_m"], "cierre_pct": m["cierre_pct"],
            "alejamiento_max_m": m["alejamiento_max_m"],
            "usadas": float(np.mean([r["msckf"]["usadas"] for r in res])),
            "rechazadas": float(np.mean([r["msckf"]["chi2"] + r["msckf"]["reproyeccion"] for r in res])),
            "ms_media": float(np.mean([r["tiempos_ms"]["total"] for r in res])),
            "ms_p95": float(np.percentile([r["tiempos_ms"]["total"] for r in res], 95))}
    if distancia_real is not None:
        fila["distancia_estimada_m"], fila["error_escala_pct"] = lazo.error_escala(e["p"], distancia_real)
    if verdad is not None:
        sel = e["t_ns"] >= (t_desde if t_desde is not None else e["t_ns"][0])
        t_ns, p, R = e["t_ns"][sel], e["p"][sel], e["R"][sel]
        p_v, _ = alinear(t_ns, p, R, *verdad)
        err = np.linalg.norm(p - p_v, axis=1)
        fila.update({"rms_verdad_m": float(np.sqrt(np.mean(err ** 2))), "final_verdad_m": float(err[-1]),
                     "escala_verdad": lazo.recorrido(p, cfg_prueba["paso_min_m"])
                     / lazo.recorrido(p_v, cfg_prueba["paso_min_m"])})
    return fila


FILAS = (("arranque_s", "arranque [s desde la 1.ª imagen]", 2), ("imagenes", "imágenes con pose", 0),
         ("recorrido_m", "recorrido [m]", 3), ("cierre_m", "cierre del lazo [m]", 3),
         ("cierre_pct", "cierre del lazo [%]", 2), ("alejamiento_max_m", "alejamiento máximo [m]", 3),
         ("distancia_estimada_m", "recta: distancia estimada [m]", 3), ("error_escala_pct", "recta: error de escala [%]", 2),
         ("rms_verdad_m", "verdad: error RMS [m]", 3), ("final_verdad_m", "verdad: error final [m]", 3),
         ("escala_verdad", "verdad: recorrido / verdadero", 3),
         ("usadas", "puntos/pistas usadas por imagen", 1), ("rechazadas", "rechazadas por imagen", 2),
         ("ms_media", "tiempo medio [ms/imagen]", 1), ("ms_p95", "tiempo p95 [ms/imagen]", 1))


def tabla(filas):
    lineas = [f"{'':34s} {'MSCKF':>10s} {'VINS':>10s}"]
    for clave, titulo, dec in FILAS:
        valores = [f.get(clave) for f in filas]
        if all(v is None for v in valores):
            continue
        texto = [("—" if v is None else f"{v:.{dec}f}") for v in valores]
        lineas.append(f"{titulo:34s} {texto[0]:>10s} {texto[1]:>10s}")
    return "\n".join(lineas)


def main():
    parser = argparse.ArgumentParser(description="MSCKF frente a VINS en una grabación")
    parser.add_argument("carpeta")
    parser.add_argument("--sin-calibracion-acc", action="store_true")
    parser.add_argument("--distancia-real", type=float, default=None, help="recta medida con cinta [m]")
    parser.add_argument("--vins-reposo", action="store_true", help="VINS arrancando en reposo (etapa 5.1)")
    args = parser.parse_args()

    carpeta = Path(args.carpeta)
    calib_acc = cargar_calibracion_acc(ruta_config("imu_calibracion"), args.sin_calibracion_acc)
    cfg_prueba = cargar_config("vio")["prueba"]
    verdad = leer_poses(carpeta / "verdad.csv") if (carpeta / "verdad.csv").exists() else None
    arranque = "reposo" if args.vins_reposo else "visual_inercial"

    ejecuciones = {}
    for tipo in ("msckf", "vins"):
        print(f"{tipo.upper()}...")
        ejecuciones[tipo] = ejecutar(carpeta, tipo, arranque, calib_acc)
        e = ejecuciones[tipo]
        if e["resultados"]:
            with open(carpeta / f"vio_trayectoria_{tipo}.csv", "w", encoding="utf-8") as f:
                f.write(CABECERA_POSES)
                for t, R, p in zip(e["t_ns"], e["R"], e["p"]):
                    f.write(linea_pose(t, R, p))
    # Con verdad: los dos en el mismo intervalo (desde el arranque del que arranca más tarde)
    t_desde = max(e["t_ns"][0] for e in ejecuciones.values() if e["resultados"])
    filas = [resumir(ejecuciones[t], cfg_prueba, verdad, args.distancia_real, t_desde) for t in ("msckf", "vins")]
    texto = tabla(filas)
    print(texto)
    (carpeta / "comparacion_estimadores.txt").write_text(texto + "\n", encoding="utf-8")
    print(f"Tabla: {carpeta / 'comparacion_estimadores.txt'}")


if __name__ == "__main__":
    main()