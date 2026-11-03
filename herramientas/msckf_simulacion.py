"""
herramientas/msckf_simulacion.py — El MSCKF sobre el simulador: precisión y consistencia.

Uso, desde ~/svo-pi:
    python -m herramientas.msckf_simulacion                     una secuencia, con figura
    python -m herramientas.msckf_simulacion --nees 10           además, NEES medio en 10 secuencias
    python -m herramientas.msckf_simulacion --nees 20 --duracion 10
El NEES con muchas secuencias tarda: en la Pi, unos 2 minutos por cada secuencia de 30 s.
Guarda datos/msckf_simulacion.png.
"""

import argparse
import copy
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from vio import modelo_camara as mc
from vio import msckf as ms
from vio import simulador as sim
from vio.config import RAIZ_PROYECTO, cargar_config
from vio.estado import EstadoImu
from vio.propagacion import propagar_secuencia

POSE = np.r_[0:3, 6:9]          # índices de [δθ, δp] en el error de la IMU


def una_secuencia(cfg_sim, cfg, cfg_tri, K, tamano, rng):
    """Simula, arranca con un error sorteado con P0 y pasa el MSCKF y la IMU sola.

    Devuelve: dict con la secuencia, los resultados, el error real en cada imagen y el tiempo.
    """
    s = sim.generar_secuencia(cfg_sim, K, tamano, rng)
    v = s["verdad"]

    def verdad(k):
        return EstadoImu(v["R_w_i"][k], v["v_w"][k], v["p_w"][k], v["b_g"][k], v["b_a"][k])

    P0 = ms.covarianza_inicial(cfg["inicial"])
    inicial = ms.inyectar_imu(verdad(0), -(np.linalg.cholesky(P0) @ rng.normal(0.0, 1.0, 15)))
    filtro = ms.Msckf(inicial, P0, s["imu"]["t_ns"][0], s["imu"]["w"][0], s["imu"]["a"][0],
                      s["R_c_i"], s["t_c_i"], K[0, 0], cfg, cfg_tri)
    t0 = time.perf_counter()
    res = ms.ejecutar_secuencia(filtro, s["imu"], s["obs"], s["camaras"]["t_ns"])
    segundos = time.perf_counter() - t0
    solo_imu = propagar_secuencia(inicial, s["imu"]["t_ns"], s["imu"]["w"], s["imu"]["a"],
                                  cargar_config("orientacion")["integracion"])
    errores = np.array([ms.diferencia_imu(verdad(k), e)
                        for k, e in zip(s["camaras"]["indice_imu"], res["estados"])])
    return {"s": s, "res": res, "solo_imu": solo_imu, "errores": errores, "segundos": segundos}


def nees_por_imagen(errores, P_imu, indices):
    """NEES de las componentes 'indices' del error en cada imagen."""
    return np.array([e[indices] @ np.linalg.solve(P[np.ix_(indices, indices)], e[indices])
                     for e, P in zip(errores, P_imu)])


def main():
    parser = argparse.ArgumentParser(description="MSCKF sobre el simulador")
    parser.add_argument("--nees", type=int, default=0, help="secuencias para medir el NEES medio")
    parser.add_argument("--duracion", type=float, default=None, help="duración de cada secuencia [s]")
    args = parser.parse_args()

    cfg_sim = copy.deepcopy(cargar_config("simulador")["vio"])
    if args.duracion is not None:
        cfg_sim["trayectoria"]["duracion_s"] = args.duracion
    cfg = cargar_config("msckf")
    cfg_tri = cargar_config("triangulacion")
    K, _, tamano = mc.cargar_intrinsecos()
    rng = np.random.default_rng(cfg_sim["semilla"])

    # --- 1. Una secuencia, con todos los detalles ---
    r = una_secuencia(cfg_sim, cfg, cfg_tri, K, tamano, rng)
    s, errores, res = r["s"], r["errores"], r["res"]
    p_verdad = s["verdad"]["p_w"]
    distancia = np.linalg.norm(np.diff(p_verdad, axis=0), axis=1).sum()
    error_final = np.linalg.norm(errores[-1, ms.P])
    error_imu = np.linalg.norm(r["solo_imu"]["p"][-1] - p_verdad[-1])
    n_img = len(res["estados"])
    recuentos = {k: sum(x[k] for x in res["recuentos"]) for k in ("usadas", "cortas", "sin_triangular", "chi2")}
    nees_pose = nees_por_imagen(errores, res["P_imu"], POSE)

    print(f"secuencia: {cfg_sim['trayectoria']['duracion_s']} s, {n_img} imágenes, {distancia:.1f} m recorridos")
    print(f"error de posición final: MSCKF {error_final:.3f} m ({100 * error_final / distancia:.2f} % de la distancia)"
          f" | solo IMU {error_imu:.1f} m")
    print(f"error de orientación final: {np.degrees(np.linalg.norm(errores[-1, ms.TH])):.3f}°")
    print(f"error final de los sesgos: giroscopio {np.degrees(np.linalg.norm(errores[-1, ms.BG])):.4f} °/s,"
          f" acelerómetro {np.linalg.norm(errores[-1, ms.BA]):.4f} m/s²")
    print(f"pistas: usadas {recuentos['usadas']}, cortas {recuentos['cortas']}, "
          f"sin triangular {recuentos['sin_triangular']}, rechazadas por chi² {recuentos['chi2']}")
    print(f"tiempo del filtro: {1000 * r['segundos'] / n_img:.1f} ms por imagen")
    print(f"NEES de la pose (esta secuencia, ideal 6): {nees_pose[20:].mean():.2f}")

    # --- 2. NEES medio en muchas secuencias ---
    nees_medio = None
    if args.nees > 0:
        suma_pose, suma_todo = 0.0, 0.0
        for k in range(args.nees):
            rk = una_secuencia(cfg_sim, cfg, cfg_tri, K, tamano, rng)
            suma_pose = suma_pose + nees_por_imagen(rk["errores"], rk["res"]["P_imu"], POSE)
            suma_todo = suma_todo + nees_por_imagen(rk["errores"], rk["res"]["P_imu"], np.arange(15))
            print(f"  secuencia {k + 1}/{args.nees}: error final {np.linalg.norm(rk['errores'][-1, ms.P]):.3f} m")
        nees_medio = suma_pose / args.nees
        for nombre, valor, dim in (("pose [δθ, δp]", nees_medio, 6), ("estado de la IMU (15)", suma_todo / args.nees, 15)):
            inferior = chi2.ppf(0.025, dim * args.nees) / args.nees
            superior = chi2.ppf(0.975, dim * args.nees) / args.nees
            print(f"NEES medio de {nombre}: {valor[20:].mean():.2f}   (ideal {dim}, "
                  f"intervalo del 95 % con {args.nees} secuencias: [{inferior:.2f}, {superior:.2f}])")

    # --- 3. Figura ---
    t = s["camaras"]["t_ns"] / 1e9
    P_imu = res["P_imu"]
    p_est = np.array([e.p_w for e in res["estados"]])
    fig, ejes = plt.subplots(2, 3, figsize=(18, 10))

    ax = ejes[0, 0]
    ax.plot(p_verdad[:, 0], p_verdad[:, 1], "k", lw=2, label="verdad")
    ax.plot(p_est[:, 0], p_est[:, 1], "b", lw=1, label="MSCKF")
    ax.plot(r["solo_imu"]["p"][:, 0], r["solo_imu"]["p"][:, 1], "r", lw=0.8, label="solo IMU")
    lx, ly, _ = cfg_sim["mundo"]["habitacion_m"]
    ax.set_xlim(-lx / 2, lx / 2)
    ax.set_ylim(-ly / 2, ly / 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Vista desde arriba")
    ax.legend()

    # Errores con sus bandas de ±3 sigma: posición y orientación (ésta, en ejes del mundo)
    error_th_w = np.array([e.R_w_i @ d[ms.TH] for e, d in zip(res["estados"], errores)])
    sigma_th_w = np.array([np.sqrt(np.diag(e.R_w_i @ P[ms.TH, ms.TH] @ e.R_w_i.T))
                           for e, P in zip(res["estados"], P_imu)])
    sigma_p = np.sqrt(P_imu[:, ms.P, ms.P].diagonal(axis1=1, axis2=2))
    for ax, error, sigma, factor, nombre, unidad in (
            (ejes[0, 1], errores[:, ms.P], sigma_p, 1.0, "posición", "m"),
            (ejes[0, 2], error_th_w, sigma_th_w, np.degrees(1.0), "orientación (ejes del mundo)", "°")):
        for eje, color in zip(range(3), "rgb"):
            ax.fill_between(t, -3 * factor * sigma[:, eje], 3 * factor * sigma[:, eje], color=color, alpha=0.1)
            ax.plot(t, factor * error[:, eje], color, lw=0.8, label="xyz"[eje])
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"error [{unidad}]")
        ax.set_title(f"Error de {nombre} y ±3 sigma")
        ax.legend()

    for ax, clave, sl, factor, nombre, unidad in ((ejes[1, 0], "b_g", ms.BG, np.degrees(1.0), "giroscopio", "°/s"),
                                                  (ejes[1, 1], "b_a", ms.BA, 1.0, "acelerómetro", "m/s²")):
        estimado = np.array([getattr(e, clave) for e in res["estados"]])
        verdadero = s["verdad"][clave][s["camaras"]["indice_imu"]]
        for eje, color in zip(range(3), "rgb"):
            ax.plot(t, factor * verdadero[:, eje], color, lw=2, alpha=0.4)
            ax.plot(t, factor * estimado[:, eje], color, lw=0.8, label="xyz"[eje])
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"sesgo [{unidad}]")
        ax.set_title(f"Sesgo del {nombre}: estimado (fino) y verdadero (grueso)")
        ax.legend()

    ax = ejes[1, 2]
    if nees_medio is None:
        ax.plot(t, nees_pose, lw=0.8, label="esta secuencia")
        ax.set_title("NEES de la pose (ideal 6)")
    else:
        inferior = chi2.ppf(0.025, 6 * args.nees) / args.nees
        superior = chi2.ppf(0.975, 6 * args.nees) / args.nees
        ax.axhspan(inferior, superior, color="0.85", label="intervalo 95 %")
        ax.plot(t, nees_medio, lw=0.8, label=f"media de {args.nees} secuencias")
        ax.set_title("NEES medio de la pose (ideal 6)")
    ax.axhline(6, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.legend()

    fig.tight_layout()
    ruta = RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"] / "msckf_simulacion.png"
    fig.savefig(ruta, dpi=90)
    print(f"Figura: {ruta}")


if __name__ == "__main__":
    main()