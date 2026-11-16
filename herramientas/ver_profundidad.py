"""
herramientas/ver_profundidad.py — Etapa 25: la profundidad en metros de cada keyframe del mapa.

Uso, desde la raíz del proyecto, EN EL PC (necesita PyTorch y transformers):
    python -m herramientas.ver_profundidad datos/sim_textura
    python -m herramientas.ver_profundidad datos/20261116_101500 --archivo mapa_2.paquetes
La carpeta es la de la sesión: la que tiene el mapa.paquetes que escribe apps/pi_vio.

Por cada keyframe de mapeo: Depth Anything V2 sobre su imagen, ajuste a metros con los
puntos del VIO que trae el propio keyframe, y a disco. En <carpeta>/profundidad/ deja:
    <indice>.png          la profundidad en milímetros (PNG de 16 bits; 0 = sin dato)
    <indice>_vista.jpg    la imagen y, al lado, su profundidad en colores (claro = cerca)
    profundidad.csv       una fila por keyframe: el ajuste (a, b) y sus errores
Con una grabación del simulador (trae mav0/prof0) compara además con la profundidad
verdadera de cada píxel.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from captura import euroc
from herramientas.medir_mapa import leer_mapa
from vio import profundidad as prof
from vio.config import cargar_config
from vio.reconstruccion import decodificar_gris

CABECERA_CSV = ("#indice,t_ns,ajustado,a,b,solo_escala,puntos,encajan,error_puntos,"
                "error_rel_mediano,error_rel_medio,delta,cubierto,archivo\n")


def vista(gris, z):
    """La imagen y, a su derecha, la profundidad en colores: claro = cerca, negro = sin dato.

    Los colores siguen a 1/z (como ve la red): así lo cercano no se queda todo del mismo color.
    Devuelve: (alto, 2·ancho, 3) uint8.
    """
    color = np.zeros(gris.shape + (3,), dtype=np.uint8)
    con_dato = np.isfinite(z)
    if con_dato.any():
        inversa = 1.0 / z[con_dato]
        minimo, maximo = np.percentile(inversa, [2, 98])
        escala = np.clip((inversa - minimo) / max(maximo - minimo, 1e-9), 0.0, 1.0)
        niveles = np.zeros(gris.shape, dtype=np.uint8)
        niveles[con_dato] = (255 * escala).astype(np.uint8)
        color = cv2.applyColorMap(niveles, cv2.COLORMAP_INFERNO)
        color[~con_dato] = 0
    return np.hstack([cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR), color])


def procesar_keyframes(keyframes, carpeta, estimador, cfg, t_d_ns, salida):
    """La profundidad de cada keyframe, a disco. Devuelve: lista de dict, uno por keyframe.

    carpeta: la de la sesión (para la profundidad verdadera, si la hay). salida: carpeta
    donde se escriben los PNG, las vistas y el CSV. t_d_ns: desfase cámara-IMU (el nombre del
    archivo de profundidad verdadera es la marca de la imagen = instante del keyframe - t_d).
    Cada dict lleva indice, t_ns, ajustado (bool), a, b, solo_escala, puntos, encajan,
    error_puntos, tiempo_ms y, si hay verdad, la evaluación (error_rel_mediano,
    error_rel_medio, delta, cubierto).
    """
    salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["ajuste"]["semilla"])
    filas = []
    for kf in keyframes:
        gris = decodificar_gris(kf)
        r = prof.profundidad_de_keyframe(kf, gris, estimador, cfg, rng)
        fila = {"indice": kf["indice"], "t_ns": kf["t_ns"], "ajustado": r["ajuste"] is not None,
                "puntos": r["n_puntos"], "tiempo_ms": r["tiempo_ms"], "archivo": ""}
        if r["ajuste"] is not None:
            fila.update(a=r["ajuste"]["a"], b=r["ajuste"]["b"],
                        solo_escala=int(r["ajuste"]["solo_escala"]),
                        encajan=int(r["ajuste"]["encajan"].sum()),
                        error_puntos=r["ajuste"]["error_mediano"],
                        archivo=f"{kf['indice']:06d}.png")
            cv2.imwrite(str(salida / fila["archivo"]), euroc.profundidad_a_mm(r["z"]))
            cv2.imwrite(str(salida / f"{kf['indice']:06d}_vista.jpg"), vista(gris, r["z"]))
            verdad = euroc.leer_profundidad(carpeta, kf["t_ns"] - t_d_ns)
            if verdad is not None:
                fila.update(prof.evaluar_profundidad(r["z"], verdad, cfg["evaluacion"]["umbral_delta"]))
        filas.append(fila)

    with open(salida / "profundidad.csv", "w", encoding="utf-8") as f:
        f.write(CABECERA_CSV)
        for fila in filas:
            campos = [fila["indice"], fila["t_ns"], int(fila["ajustado"])]
            for clave in ("a", "b", "solo_escala", "puntos", "encajan", "error_puntos",
                          "error_rel_mediano", "error_rel_medio", "delta", "cubierto"):
                valor = fila.get(clave, "")
                campos.append(f"{valor:.6g}" if isinstance(valor, float) else valor)
            campos.append(fila["archivo"])
            f.write(",".join(str(c) for c in campos) + "\n")
    return filas


def informe(filas):
    """El texto del informe: una línea por keyframe y el resumen."""
    hay_verdad = any("error_rel_mediano" in fila for fila in filas)
    cabecera = f"{'keyframe':>8s} {'puntos':>6s} {'encajan':>7s} {'error puntos':>12s} {'ajuste':>7s}"
    if hay_verdad:
        cabecera += f" {'error denso':>11s} {'buenos':>7s} {'cubierto':>8s}"
    lineas = [cabecera]
    for fila in filas:
        if not fila["ajustado"]:
            lineas.append(f"{fila['indice']:8d} {fila['puntos']:6d}   sin ajuste")
            continue
        linea = (f"{fila['indice']:8d} {fila['puntos']:6d} {fila['encajan']:7d}"
                 f" {100 * fila['error_puntos']:10.1f} % {'escala' if fila['solo_escala'] else 'recta':>7s}")
        if "error_rel_mediano" in fila:
            linea += (f" {100 * fila['error_rel_mediano']:9.1f} % {100 * fila['delta']:5.0f} %"
                      f" {100 * fila['cubierto']:6.0f} %")
        lineas.append(linea)
    ajustados = [fila for fila in filas if fila["ajustado"]]
    lineas.append("")
    lineas.append(f"{len(ajustados)} de {len(filas)} keyframes con profundidad en metros"
                  f" ({sum(f['solo_escala'] for f in ajustados)} de ellos, solo con la escala)")
    if ajustados:
        lineas.append(f"Puntos del VIO que encajan con la red: "
                      f"{100 * sum(f['encajan'] for f in ajustados) / sum(f['puntos'] for f in ajustados):.0f} %"
                      f" | error mediano en esos puntos: "
                      f"{100 * np.median([f['error_puntos'] for f in ajustados]):.1f} %")
    con_verdad = [fila for fila in ajustados if "error_rel_mediano" in fila]
    if con_verdad:
        lineas.append(f"Frente a la verdad, en todos los píxeles: error relativo mediano "
                      f"{100 * np.median([f['error_rel_mediano'] for f in con_verdad]):.1f} %,"
                      f" medio {100 * np.mean([f['error_rel_medio'] for f in con_verdad]):.1f} %,"
                      f" píxeles buenos {100 * np.mean([f['delta'] for f in con_verdad]):.0f} %")
    lineas.append(f"Red: {np.median([f['tiempo_ms'] for f in filas]):.0f} ms por imagen (mediana)")
    return "\n".join(lineas)


def main():
    parser = argparse.ArgumentParser(description="Profundidad en metros de cada keyframe del mapa")
    parser.add_argument("carpeta", help="carpeta de la sesión (la que tiene mapa.paquetes)")
    parser.add_argument("--archivo", default=cargar_config("vio")["mapeo"]["archivo"],
                        help="nombre del archivo de paquetes dentro de la carpeta")
    args = parser.parse_args()
    carpeta = Path(args.carpeta)
    cfg = cargar_config("profundidad")

    mapa = leer_mapa(carpeta / args.archivo)
    print(mapa.resumen())
    if not mapa.keyframes:
        print("El mapa no tiene ningún keyframe: no hay nada que hacer.")
        return
    estimador = prof.EstimadorProfundidad(cfg)
    print(f"Red cargada en: {estimador.dispositivo}")
    salida = carpeta / "profundidad"
    filas = procesar_keyframes(mapa.keyframes, carpeta, estimador, cfg,
                               cargar_config("camara_imu")["t_d_ns"], salida)
    print(informe(filas))
    print(f"Resultados en {salida}")


if __name__ == "__main__":
    main()
