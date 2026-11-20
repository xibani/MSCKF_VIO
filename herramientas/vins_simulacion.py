"""
herramientas/vins_simulacion.py — Etapa 35: VINS (ventana deslizante) frente al MSCKF en el
simulador, con la misma secuencia, el mismo error inicial y los mismos modelos de ruido.

Uso, desde la raíz del repositorio:
    python -m herramientas.vins_simulacion                          5 s, con figura
    python -m herramientas.vins_simulacion --duracion 60            la secuencia del criterio
    python -m herramientas.vins_simulacion --nees 10 --duracion 20  NEES medio en 10 secuencias
Opciones: --puntos (por superficie de la habitación), --iteraciones (de LM por imagen),
--perdida (huber | cauchy | ninguna), --semilla.
VINS en Python puro es lento (~0,2 s por imagen en el PC): 60 s a 20 Hz son unos 4 minutos.
Guarda datos/vins_simulacion.png.
"""

import argparse
import copy
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from vio import msckf as ms
from vio import simulador as sim
from vio import so3
from vio.config import RAIZ_PROYECTO, cargar_config
from vio.estado import EstadoImu
from vins import factores as fa
from vins import ventana as vt

POSE = np.r_[0:3, 6:9]           # [δθ, δp] dentro del error de 15
K_SIM = np.array([[643.0, 0.0, 320.0], [0.0, 643.0, 240.0], [0.0, 0.0, 1.0]])
TAMANO = (640, 480)
CLAVES_RUIDO = ("ruido_gir", "ruido_acc", "paseo_sesgo_gir", "paseo_sesgo_acc")


def configuraciones(cfg_sim, iteraciones=None, perdida="ninguna"):
    """vins.yaml y msckf.yaml con el ruido VERDADERO del simulador (IMU y píxel), para que
    los dos estimadores tengan el modelo correcto y el NEES signifique algo."""
    cfg_v = copy.deepcopy(cargar_config("vins"))
    cfg_v["visual"]["ruido_px"] = cfg_sim["camara"]["ruido_px"]
    cfg_v["visual"]["perdida"] = perdida
    if iteraciones is not None:
        cfg_v["ventana"]["iteraciones_max"] = iteraciones
    cfg_m = copy.deepcopy(cargar_config("msckf"))
    cfg_m["imu"].update({k: cfg_sim["imu"][k] for k in CLAVES_RUIDO})
    cfg_m["correccion"]["ruido_px"] = cfg_sim["camara"]["ruido_px"]
    cfg_m["inicial"] = dict(cfg_v["inicial"])             # el mismo P0 en los dos
    return cfg_v, cfg_m


def una_secuencia(cfg_sim, cfg_v, cfg_m, rng, con_msckf=True):
    """Simula, sortea el error inicial con P0, y pasa VINS (y el MSCKF).

    Devuelve: dict con la secuencia s, y para cada estimador ("vins", "msckf"): estados,
    P (15x15 por imagen), errores (C, 15), recuentos y segundos.
    """
    s = sim.generar_secuencia(cfg_sim, K_SIM, TAMANO, rng)
    v = s["verdad"]

    def verdad(k):
        return EstadoImu(v["R_w_i"][k], v["v_w"][k], v["p_w"][k], v["b_g"][k], v["b_a"][k])

    sigmas = fa.sigmas_iniciales(cfg_v["inicial"])
    P0 = np.diag(sigmas ** 2)
    inicial = ms.inyectar_imu(verdad(0), -(np.linalg.cholesky(P0) @ rng.normal(0.0, 1.0, 15)))
    imu, t_cam, idx = s["imu"], s["camaras"]["t_ns"], s["camaras"]["indice_imu"]
    ruido_imu = {k: cfg_sim["imu"][k] for k in CLAVES_RUIDO}
    salida = {"s": s}

    # VINS
    ventana = vt.VentanaDeslizante(inicial, sigmas, imu["t_ns"][0], imu["w"][0], imu["a"][0],
                                   s["R_c_i"], s["t_c_i"], K_SIM[0, 0], cfg_v, ruido_imu)
    t0 = time.perf_counter()
    r = vt.ejecutar_secuencia(ventana, imu, s["obs"], t_cam, con_covarianza=True)
    salida["vins"] = {"estados": r["estados"], "P": r["P"], "recuentos": r["recuentos"],
                      "segundos": time.perf_counter() - t0,
                      "errores": np.array([ms.diferencia_imu(verdad(k), e) for k, e in zip(idx, r["estados"])])}

    # MSCKF
    if con_msckf:
        filtro = ms.Msckf(inicial, P0, imu["t_ns"][0], imu["w"][0], imu["a"][0], s["R_c_i"], s["t_c_i"],
                          K_SIM[0, 0], cfg_m, cargar_config("triangulacion"))
        t0 = time.perf_counter()
        r = ms.ejecutar_secuencia(filtro, imu, s["obs"], t_cam)
        salida["msckf"] = {"estados": r["estados"], "P": r["P_imu"], "recuentos": r["recuentos"],
                           "segundos": time.perf_counter() - t0,
                           "errores": np.array([ms.diferencia_imu(verdad(k), e) for k, e in zip(idx, r["estados"])])}
    return salida


def nees(errores, P, indices):
    """NEES de las componentes 'indices' del error, en cada imagen."""
    return np.array([e[indices] @ np.linalg.solve(Pk[np.ix_(indices, indices)], e[indices])
                     for e, Pk in zip(errores, P)])


def resumen(nombre, res, s, descartar):
    """Métricas de un estimador en una secuencia (dict) y una línea de texto."""
    e = res["errores"]
    # Error de orientación en ejes del MUNDO: z = rumbo (no observable), x-y = inclinación
    th_w = np.array([est.R_w_i @ d[0:3] for est, d in zip(res["estados"], e)])
    m = {"rms_pos": float(np.sqrt(np.mean(np.sum(e[:, 6:9] ** 2, axis=1)))),
         "final_pos": float(np.linalg.norm(e[-1, 6:9])),
         "rumbo_final": float(np.degrees(abs(th_w[-1, 2]))),
         "inclinacion_final": float(np.degrees(np.linalg.norm(th_w[-1, 0:2]))),
         "nees_pose": float(nees(e, res["P"], POSE)[descartar:].mean()),
         "ms_por_imagen": 1000.0 * res["segundos"] / len(e)}
    texto = (f"{nombre:6s} RMS pos {100 * m['rms_pos']:6.2f} cm | final {100 * m['final_pos']:6.2f} cm | "
             f"rumbo final {m['rumbo_final']:.3f}° | inclinación final {m['inclinacion_final']:.3f}° | "
             f"NEES pose {m['nees_pose']:5.2f} (ideal 6) | {m['ms_por_imagen']:.0f} ms/imagen")
    return m, texto


def main():
    parser = argparse.ArgumentParser(description="VINS frente al MSCKF en el simulador")
    parser.add_argument("--duracion", type=float, default=5.0)
    parser.add_argument("--nees", type=int, default=0, help="secuencias para el NEES medio")
    parser.add_argument("--puntos", type=int, default=100, help="puntos por superficie de la habitación")
    parser.add_argument("--iteraciones", type=int, default=None, help="iteraciones de LM por imagen")
    parser.add_argument("--perdida", default="ninguna")
    parser.add_argument("--semilla", type=int, default=11)
    args = parser.parse_args()

    cfg_sim = copy.deepcopy(cargar_config("simulador")["vio"])
    cfg_sim["trayectoria"]["duracion_s"] = args.duracion
    cfg_sim["mundo"]["puntos_por_superficie"] = args.puntos
    cfg_v, cfg_m = configuraciones(cfg_sim, args.iteraciones, args.perdida)
    rng = np.random.default_rng(args.semilla)
    descartar = 20                       # primeras imágenes (arranque) fuera de las medias

    # --- 1. Una secuencia, con detalle ---
    r = una_secuencia(cfg_sim, cfg_v, cfg_m, rng)
    s = r["s"]
    distancia = np.linalg.norm(np.diff(s["verdad"]["p_w"], axis=0), axis=1).sum()
    print(f"secuencia: {args.duracion} s, {len(s['camaras']['t_ns'])} imágenes, {distancia:.1f} m recorridos")
    for nombre in ("vins", "msckf"):
        print(resumen(nombre, r[nombre], s, descartar)[1])
    rec = r["vins"]["recuentos"]
    marg = [x["marginalizacion"] for x in rec]
    print(f"VINS: {marg.count('antiguo')} marginalizaciones del más antiguo, {marg.count('segundo')} del 2.º "
          f"más reciente | puntos activos medios {np.mean([x['puntos_activos'] for x in rec]):.0f} | "
          f"iteraciones medias {np.mean([x['iteraciones'] for x in rec]):.1f}")

    # --- 2. NEES medio en varias secuencias ---
    nees_medio = None
    if args.nees > 0:
        sumas = {"vins": 0.0, "msckf": 0.0}
        for k in range(args.nees):
            rk = una_secuencia(cfg_sim, cfg_v, cfg_m, rng)
            for nombre in sumas:
                sumas[nombre] = sumas[nombre] + nees(rk[nombre]["errores"], rk[nombre]["P"], POSE)
            print(f"  secuencia {k + 1}/{args.nees}: RMS pos VINS {100 * resumen('', rk['vins'], rk['s'], descartar)[0]['rms_pos']:.2f} cm,"
                  f" MSCKF {100 * resumen('', rk['msckf'], rk['s'], descartar)[0]['rms_pos']:.2f} cm")
        inferior = chi2.ppf(0.025, 6 * args.nees) / args.nees
        superior = chi2.ppf(0.975, 6 * args.nees) / args.nees
        nees_medio = {n: v / args.nees for n, v in sumas.items()}
        for nombre, valor in nees_medio.items():
            dentro = np.mean((valor[descartar:] >= inferior) & (valor[descartar:] <= superior))
            print(f"NEES medio de la pose, {nombre}: {valor[descartar:].mean():.2f} (ideal 6; banda 95 % con "
                  f"{args.nees} secuencias: [{inferior:.2f}, {superior:.2f}]; {100 * dentro:.0f} % de las imágenes dentro)")

    # --- 3. Figura ---
    t = s["camaras"]["t_ns"] / 1e9
    fig, ejes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = ejes[0]
    ax.plot(s["verdad"]["p_w"][:, 0], s["verdad"]["p_w"][:, 1], "k", lw=2, label="verdad")
    for nombre, color in (("vins", "b"), ("msckf", "r")):
        p = np.array([e.p_w for e in r[nombre]["estados"]])
        ax.plot(p[:, 0], p[:, 1], color, lw=1, label=nombre.upper())
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Vista desde arriba")
    ax.legend()

    ax = ejes[1]
    for nombre, color in (("vins", "b"), ("msckf", "r")):
        e = r[nombre]["errores"]
        sigma = np.sqrt(np.trace(r[nombre]["P"][:, 6:9, 6:9], axis1=1, axis2=2))
        ax.plot(t, 100 * np.linalg.norm(e[:, 6:9], axis=1), color, lw=1, label=f"{nombre.upper()} |error|")
        ax.plot(t, 300 * sigma, color, lw=0.8, ls="--", label=f"{nombre.upper()} 3σ")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("error de posición [cm]")
    ax.set_title("Error de posición y su 3σ")
    ax.legend()

    ax = ejes[2]
    if nees_medio is None:
        for nombre, color in (("vins", "b"), ("msckf", "r")):
            ax.plot(t, nees(r[nombre]["errores"], r[nombre]["P"], POSE), color, lw=0.8, label=nombre.upper())
        ax.set_title("NEES de la pose (esta secuencia; ideal 6)")
    else:
        ax.axhspan(inferior, superior, color="0.85", label="banda 95 %")
        for nombre, color in (("vins", "b"), ("msckf", "r")):
            ax.plot(t, nees_medio[nombre], color, lw=0.8, label=f"{nombre.upper()} (media de {args.nees})")
        ax.set_title("NEES medio de la pose (ideal 6)")
    ax.axhline(6, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.legend()

    fig.tight_layout()
    ruta = RAIZ_PROYECTO / cargar_config("proyecto")["carpeta_datos"] / "vins_simulacion.png"
    fig.savefig(ruta, dpi=90)
    print(f"Figura: {ruta}")


if __name__ == "__main__":
    main()