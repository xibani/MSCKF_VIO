"""
herramientas/comparar_estimadores.py — Etapa 37: MSCKF frente a VINS sobre la misma grabación.

Uso, desde la raíz del repositorio:
    python -m herramientas.comparar_estimadores datos/vio_20261120_101500
    python -m herramientas.comparar_estimadores datos/X --sin-calibracion-acc        # sintéticas
    python -m herramientas.comparar_estimadores datos/X --distancia-real 2.0         # recta medida
    python -m herramientas.comparar_estimadores datos/X --vins-reposo                # VINS arrancando quieto
    python -m herramientas.comparar_estimadores datos/X --solo vins                  # uno solo (más rápido)
Cada --aviso imágenes (100 por defecto) escribe por dónde va. De VINS, además, por qué rechaza
cada intento de inicializar (y un resumen al final).
Si el estado deja de ser finito (NaN o infinito), lo dice y para ese estimador. Si una imagen
tarda más de --volcado-s segundos (30 por defecto), escribe dónde está cada hilo (faulthandler),
aunque esté atascado dentro de una función de C (LAPACK, OpenCV).
Pasa la grabación por el VIO completo (front-end + estimador, como apps/pi_vio sin red ni mapa)
dos veces, con el MSCKF y con VINS, y escribe una tabla:
    - arranque (s desde la primera imagen), imágenes con pose;
    - prueba del protocolo: recorrido, cierre del lazo [m y %] (vio/lazo.py), alejamiento máximo
      y, con --distancia-real, error de escala de la recta medida. Se miden desde la primera imagen
      QUIETA tras el arranque del estimador (la vuelta a la marca después del vaivén: VINS arranca
      a mitad del vaivén y su primera pose no está en la marca); si no hay ninguna, desde la primera;
    - si la grabación trae verdad.csv (simulador): error RMS y final de posición tras alinear la
      primera pose, y escala (recorrido estimado / verdadero) en el mismo intervalo;
    - recuentos y tiempos por imagen.
Guarda la tabla y las trayectorias (vio_trayectoria_msckf.csv, vio_trayectoria_vins.csv) en la
carpeta de la grabación.
"""

import argparse
import copy
import faulthandler
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from apps.pi_vio import ADELANTO_IMU_NS, cargar_calibracion_acc, eventos_grabacion
from herramientas.comparar_trayectoria import alinear
from vio import lazo
from vio import modelo_camara as mc
from vio.config import cargar_config, ruta_config
from vio.pipeline import CABECERA_POSES, PipelineVio, leer_poses, linea_pose


def motivo_corto(motivo):
    """El motivo de un intento de inicializar sin los números (para agruparlos)."""
    return motivo.split(" (")[0].split(":")[0]


def ejecutar(carpeta, tipo, arranque_vins, calib_acc, aviso=100, volcado_s=30.0):
    """Pasa la grabación por el VIO con el estimador dado.

    aviso: cada cuántas imágenes se escribe por dónde va.
    Devuelve: dict con t_ns (N,), p (N, 3), R (N, 3, 3) de las imágenes con pose, los
    resultados de nueva_imagen, el instante de la primera imagen, la de arranque y, con VINS,
    los motivos de los intentos de inicializar rechazados.
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
    motivos, ultimo_motivo, n_imagenes, n_imu = Counter(), None, 0, 0
    t0 = time.perf_counter()
    print(f"  {tipo.upper()}: leyendo la grabación...", flush=True)
    for evento in eventos_grabacion(carpeta, max(ext["t_d_ns"], 0) + ADELANTO_IMU_NS):
        if evento[0] == "imu":
            pipeline.nueva_imu(evento[1], evento[2], evento[3])
            n_imu += 1
            continue
        if t_primera is None:
            t_primera = evento[1] + ext["t_d_ns"]
            print(f"  {tipo.upper()}: primera imagen (tras {n_imu} muestras de IMU)", flush=True)
        # Si esta imagen se atasca, a los volcado_s segundos se escribe la pila de cada hilo
        faulthandler.dump_traceback_later(volcado_s, repeat=True, file=sys.stderr)
        r = pipeline.nueva_imagen(evento[1], evento[2])
        faulthandler.cancel_dump_traceback_later()
        n_imagenes += 1
        if r is not None:
            if not (np.all(np.isfinite(r["p_w"])) and np.all(np.isfinite(r["v_w"])) and np.all(np.isfinite(r["R_w_i"]))):
                print(f"  {tipo.upper()}: ESTADO NO FINITO en la imagen {n_imagenes} "
                      f"({(r['t_ns'] - t_primera) / 1e9:.2f} s): p = {r['p_w']}, v = {r['v_w']}. Se para aquí.", flush=True)
                break
            resultados.append(r)
        # VINS sin inicializar: por qué se rechazó el último intento (los de espera no cuentan)
        motivo = getattr(pipeline, "ultimo_intento", None)
        if r is None and motivo not in (None, "esperando", "acumulando frames", "esperando reintento"):
            if motivo != ultimo_motivo:
                motivos[motivo_corto(motivo)] += 1
                ultimo_motivo = motivo
        if n_imagenes % aviso == 0:
            t_grab = (evento[1] + ext["t_d_ns"] - t_primera) / 1e9
            extra = f", último rechazo: {ultimo_motivo}" if (ultimo_motivo and not resultados) else ""
            print(f"  {tipo.upper()}: {n_imagenes} imágenes ({t_grab:.1f} s de grabación), {len(resultados)} con pose, "
                  f"{1000 * (time.perf_counter() - t0) / n_imagenes:.0f} ms/imagen{extra}", flush=True)
    print(f"  {tipo.upper()}: fin, {n_imagenes} imágenes y {n_imu} muestras de IMU", flush=True)
    if motivos:
        print(f"  {tipo.upper()}: intentos de inicializar rechazados: "
              + "; ".join(f"{m} ×{n}" for m, n in motivos.most_common()), flush=True)
    if not resultados:
        return {"resultados": [], "t_primera": t_primera, "motivos": motivos}
    return {"resultados": resultados, "t_primera": t_primera, "motivos": motivos,
            "t_ns": np.array([r["t_ns"] for r in resultados], dtype=np.int64),
            "p": np.array([r["p_w"] for r in resultados]),
            "R": np.array([r["R_w_i"] for r in resultados])}


def indice_referencia(e, t_ref_ns=None):
    """Índice de la imagen desde la que se mide la prueba (cierre, recta): la primera imagen
    quieta (ZUPT) con t >= t_ref_ns; si no hay ninguna quieta, la primera con t >= t_ref_ns.
    Sin t_ref_ns: desde el arranque del estimador."""
    res = e["resultados"]
    t_min = e["t_ns"][0] if t_ref_ns is None else t_ref_ns
    quietas = [k for k, r in enumerate(res) if r["msckf"].get("quieto") and e["t_ns"][k] >= t_min]
    if quietas:
        return quietas[0]
    despues = np.nonzero(e["t_ns"] >= t_min)[0]
    return int(despues[0]) if len(despues) else 0


def resumir(e, cfg_prueba, verdad=None, distancia_real=None, t_desde=None, t_ref_ns=None):
    """Números de una ejecución (dict). t_desde: comparar con la verdad solo desde ese instante
    (el arranque del estimador que arranca más tarde: mismo intervalo para los dos).
    t_ref_ns: instante común desde el que se mide la prueba (ver indice_referencia)."""
    res = e["resultados"]
    if not res:
        return {"imagenes": 0}
    # Referencia de la prueba: la primera imagen quieta a partir del instante común. Así los dos
    # estimadores miden el cierre y la recta desde la MISMA posición física de la placa
    i0 = indice_referencia(e, t_ref_ns)
    t_s = (e["t_ns"][i0:] - e["t_ns"][i0]) / 1e9
    m = lazo.metricas_lazo(t_s, e["p"][i0:], cfg_prueba["paso_min_m"], cfg_prueba["recorrido_min_m"])
    fila = {"arranque_s": (e["t_ns"][0] - e["t_primera"]) / 1e9, "imagenes": len(res),
            "referencia_s": (e["t_ns"][i0] - e["t_primera"]) / 1e9,
            "recorrido_m": m["recorrido_m"], "cierre_m": m["cierre_m"], "cierre_pct": m["cierre_pct"],
            "alejamiento_max_m": m["alejamiento_max_m"],
            "usadas": float(np.mean([r["msckf"]["usadas"] for r in res])),
            "rechazadas": float(np.mean([r["msckf"]["chi2"] + r["msckf"]["reproyeccion"] for r in res])),
            "ms_media": float(np.mean([r["tiempos_ms"]["total"] for r in res])),
            "ms_p95": float(np.percentile([r["tiempos_ms"]["total"] for r in res], 95))}
    if distancia_real is not None:
        fila["distancia_estimada_m"], fila["error_escala_pct"] = lazo.error_escala(e["p"][i0:], distancia_real)
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
         ("referencia_s", "inicio de la prueba (quieta) [s]", 2),
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
    parser.add_argument("--solo", choices=("msckf", "vins"), default=None, help="solo un estimador")
    parser.add_argument("--aviso", type=int, default=100, help="cada cuántas imágenes informar")
    parser.add_argument("--volcado-s", type=float, default=30.0,
                        help="si una imagen tarda más, escribir dónde está atascado")
    args = parser.parse_args()

    carpeta = Path(args.carpeta)
    calib_acc = cargar_calibracion_acc(ruta_config("imu_calibracion"), args.sin_calibracion_acc)
    cfg_prueba = cargar_config("vio")["prueba"]
    verdad = leer_poses(carpeta / "verdad.csv") if (carpeta / "verdad.csv").exists() else None
    arranque = "reposo" if args.vins_reposo else "visual_inercial"

    tipos = (args.solo,) if args.solo else ("msckf", "vins")
    ejecuciones = {}
    for tipo in tipos:
        print(f"{tipo.upper()}...")
        ejecuciones[tipo] = ejecutar(carpeta, tipo, arranque, calib_acc, args.aviso, args.volcado_s)
        e = ejecuciones[tipo]
        if e["resultados"]:
            with open(carpeta / f"vio_trayectoria_{tipo}.csv", "w", encoding="utf-8") as f:
                f.write(CABECERA_POSES)
                for t, R, p in zip(e["t_ns"], e["R"], e["p"]):
                    f.write(linea_pose(t, R, p))
    # Con verdad: los dos en el mismo intervalo (desde el arranque del que arranca más tarde)
    arrancados = [e["t_ns"][0] for e in ejecuciones.values() if e["resultados"]]
    t_desde = max(arrancados) if arrancados else None
    # Referencia común de la prueba: la primera imagen quieta de cada estimador tras SU arranque, y
    # de esos instantes el más tardío. Si VINS arranca a los 10 s, el MSCKF tampoco cuenta lo de antes
    referencias = [e["t_ns"][indice_referencia(e)] for e in ejecuciones.values() if e["resultados"]]
    t_ref_ns = max(referencias) if referencias else None
    filas = [resumir(ejecuciones[t], cfg_prueba, verdad, args.distancia_real, t_desde, t_ref_ns)
             if t in ejecuciones else {"imagenes": None} for t in ("msckf", "vins")]
    texto = tabla(filas)
    print(texto)
    (carpeta / "comparacion_estimadores.txt").write_text(texto + "\n", encoding="utf-8")
    print(f"Tabla: {carpeta / 'comparacion_estimadores.txt'}")


if __name__ == "__main__":
    main()