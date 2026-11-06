"""
herramientas/analizar_allan.py — Ruido de la IMU (N y K) por varianza de Allan (etapa 7.3).

Uso, desde ~/proyectos/MSCKF_VIO, cuando la grabación haya terminado:
    python -m herramientas.analizar_allan datos/imu_20261105_120000
    python -m herramientas.analizar_allan datos/imu_20261105_120000 --forzar   # guardar aunque no sea válida
Imprime N y K de cada eje y guarda allan.png dentro de la carpeta de la grabación. Si la
grabación es válida (sin muestras perdidas ni perturbaciones periódicas), escribe
config/imu_ruido.yaml y dice qué copiar en config/msckf.yaml.
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # sin ventana: la figura se guarda en un PNG
import matplotlib.pyplot as plt
import numpy as np

from captura import euroc
from captura.verificar import analizar_intervalos
from vio import allan
from vio.config import cargar_config, ruta_config
from vio.constantes import G

EJES = "xyz"
# (clave en los YAML, nombre para imprimir, unidad de la señal)
SENSORES = [("gir", "giroscopio", "rad/s"), ("acc", "acelerómetro", "m/s²")]


def preparar_datos(t_ns, w, a, cfg, f_nominal_hz, factor_hueco):
    """Quita el principio de la grabación y mide sus pérdidas respecto a la frecuencia NOMINAL.

    (No vale la mediana de los Δt: si se pierde más de la mitad de las muestras, la mediana
    es 2 periodos y las pérdidas salen a la mitad de las reales.)
    Devuelve: (dt [s], t_s, w, a, informe). t_s: instantes reales desde el principio [s];
        informe: dict de duración, muestras y pérdidas.
    """
    quedan = t_ns >= t_ns[0] + int(cfg["descartar_inicio_s"] * 1e9)    # la placa aún se asentaba
    t_ns, w, a = t_ns[quedan], w[quedan], a[quedan]
    r = analizar_intervalos(t_ns, f_nominal_hz, factor_hueco)          # lo mismo que analizar_imu
    informe = {"duracion_s": float(r["duracion_s"]),
               "muestras": int(r["n"]),
               "frecuencia_real_hz": float(r["frecuencia_hz"]),
               "perdidas": int(r["perdidas"]),
               "perdidas_pct": float(r["perdidas_pct"])}
    t_s = (t_ns - t_ns[0]) / 1e9
    return 1.0 / f_nominal_hz, t_s, w, a, informe


def imprimir_sensor(nombre, clave, resultados, cfg):
    """Tabla con N, K y B de los tres ejes de un sensor, en SI y en unidades de hoja de datos."""
    print(f"\n=== {nombre.upper()} ===")
    for eje, r in zip(EJES, resultados):
        if clave == "gir":
            legible = (f"N = {np.degrees(r['N']):.4f} °/s/√Hz | "
                       f"B = {np.degrees(r['B']) * 3600:.1f} °/h")
        else:
            legible = (f"N = {r['N'] / G * 1e6:.0f} µg/√Hz | "
                       f"B = {r['B'] / G * 1e6:.0f} µg")
        if r["periodica"]:
            paseo = f"NO medible: perturbación periódica de ~{r['periodo_s']:.0f} s"
        elif r["K_observado"]:
            paseo = "visto"
        else:
            paseo = "NO visto: es una cota superior"
        print(f"  eje {eje}: N = {r['N']:.3e} | K = {r['K']:.3e} ({paseo}) | "
              f"mínimo en τ = {r['tau_B']:.0f} s")
        print(f"          {legible}")
    referencia = cfg[f"referencia_ruido_{clave}"]
    peor = max(r["N"] for r in resultados)
    print(f"  N del peor eje = {peor / referencia:.2f} × la hoja de datos")


def motivos_no_valida(informe, sensores, cfg):
    """Por qué la grabación no sirve para calibrar. Devuelve: lista de textos (vacía = válida)."""
    motivos = []
    if informe["perdidas_pct"] > cfg["perdidas_max_pct"]:
        motivos.append(f"{informe['perdidas_pct']:.1f} % de muestras perdidas (máx {cfg['perdidas_max_pct']} %,"
                       f" frecuencia real {informe['frecuencia_real_hz']:.1f} Hz):"
                       " ¿otro programa leyendo la IMU a la vez?")
    for clave, nombre, _ in SENSORES:
        for eje, r in zip(EJES, sensores[clave]):
            if r["periodica"]:
                motivos.append(f"{nombre} {eje}: perturbación periódica de ~{r['periodo_s']:.0f} s"
                               " (¿temperatura que sube y baja con el ventilador?)")
    return motivos


def dibujar(ruta_png, sensores, informe, motivos):
    """Curvas σ(τ) de los 6 ejes con las rectas de N (-1/2) y K (+1/2) del peor eje."""
    figura, ejes = plt.subplots(1, 2, figsize=(13, 5.5))
    colores = ["tab:red", "tab:green", "tab:blue"]
    for ax, (clave, nombre, unidad) in zip(ejes, SENSORES):
        resultados = sensores[clave]
        for eje, r, color in zip(EJES, resultados, colores):
            ax.loglog(r["tau"], r["sigma"], color=color, label=f"eje {eje}")
        N = max(r["N"] for r in resultados)
        K = max(r["K"] for r in resultados)
        tau = resultados[0]["tau"]
        ax.loglog(tau, N / np.sqrt(tau), "k--", lw=1, label=f"N/√τ  (N = {N:.2e})")
        ax.loglog(tau, K * np.sqrt(tau / 3.0), "k:", lw=1.5, label=f"K·√(τ/3)  (K = {K:.2e})")
        ax.plot([1.0], [N], "ko")                     # N se lee en τ = 1 s
        ax.plot([3.0], [K], "ks")                     # K se lee en τ = 3 s
        ax.set_ylim(min(r["sigma"].min() for r in resultados) / 3,
                    max(r["sigma"].max() for r in resultados) * 3)
        ax.set_xlabel("τ [s]")
        ax.set_ylabel(f"desviación de Allan [{unidad}]")
        ax.set_title(nombre)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    titulo = (f"Varianza de Allan: {informe['duracion_s'] / 3600:.2f} h en reposo, "
              f"{informe['perdidas_pct']:.3f} % de muestras perdidas")
    if motivos:
        titulo += "  —  GRABACIÓN NO VÁLIDA"
    figura.suptitle(titulo)
    figura.tight_layout()
    figura.savefig(ruta_png, dpi=120)
    plt.close(figura)


def analizar_grabacion(carpeta, cfg, ruta_yaml, f_nominal_hz, factor_hueco, forzar=False):
    """Todo el análisis de una grabación.

    Devuelve: (adoptado, motivos). adoptado = dict con los 4 valores, o None si la grabación
    no es válida (y no se ha forzado); motivos = lista de por qué no es válida.
    """
    carpeta = Path(carpeta)

    # 1. Leer y preparar
    t_ns, w, a = euroc.leer_imu(euroc.ruta_imu(carpeta))
    dt, t_s, w, a, informe = preparar_datos(t_ns, w, a, cfg, f_nominal_hz, factor_hueco)
    print(f"Grabación: {informe['duracion_s'] / 3600:.2f} h útiles, {informe['muestras']} muestras,"
          f" {informe['frecuencia_real_hz']:.1f} Hz (nominal {f_nominal_hz:g}),"
          f" perdidas {informe['perdidas']} ({informe['perdidas_pct']:.3f} %)")
    if informe["duracion_s"] < cfg["duracion_min_s"]:
        print(f"AVISO: menos de {cfg['duracion_min_s'] / 3600:.1f} h: K saldrá como cota superior")

    # 2. Allan de los 6 ejes
    senales = {"gir": w, "acc": a}
    sensores = {}
    for clave, nombre, _ in SENSORES:
        sensores[clave] = [allan.analizar_eje(senales[clave][:, i], dt, cfg, t_s) for i in range(3)]
        imprimir_sensor(nombre, clave, sensores[clave], cfg)

    # 3. ¿Sirve? La figura se dibuja siempre: es lo que permite ver qué ha pasado
    motivos = motivos_no_valida(informe, sensores, cfg)
    dibujar(carpeta / "allan.png", sensores, informe, motivos)
    print(f"\nFigura: {carpeta / 'allan.png'}")
    if motivos:
        print("\nGRABACIÓN NO VÁLIDA para calibrar:")
        for motivo in motivos:
            print(f"  - {motivo}")
        if not forzar:
            print("No se escribe config/imu_ruido.yaml. Corrige la causa y repite la grabación.")
            return None, motivos

    # 4. Valor adoptado: el del PEOR eje (el filtro usa un solo número por sensor)
    adoptado = {}
    for clave, _, _ in SENSORES:
        adoptado[f"ruido_{clave}"] = max(r["N"] for r in sensores[clave])
        adoptado[f"paseo_sesgo_{clave}"] = max(r["K"] for r in sensores[clave])

    # 5. Guardar el YAML
    por_eje = {clave: {eje: {"N": float(r["N"]), "K": float(r["K"]),
                             "K_observado": bool(r["K_observado"]),
                             "periodica": bool(r["periodica"]),
                             "B": float(r["B"]), "tau_B_s": float(r["tau_B"])}
                       for eje, r in zip(EJES, sensores[clave])}
               for clave, _, _ in SENSORES}
    info = {"fecha": datetime.now().isoformat(timespec="seconds"),
            "grabacion": carpeta.name,
            "descartado_inicio_s": cfg["descartar_inicio_s"],
            "motivos_no_valida": motivos,                     # vacía salvo con --forzar
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in informe.items()}}
    allan.guardar_ruido_imu(ruta_yaml, adoptado, por_eje, info)
    print(f"Resultado: {ruta_yaml}")
    return adoptado, motivos


def main():
    parser = argparse.ArgumentParser(description="Ruido de la IMU por varianza de Allan")
    parser.add_argument("carpeta", help="carpeta de herramientas.grabar_imu (la que contiene mav0/)")
    parser.add_argument("--forzar", action="store_true",
                        help="guardar config/imu_ruido.yaml aunque la grabación no sea válida")
    args = parser.parse_args()
    cfg = cargar_config("calibracion")["allan"]
    f_nominal_hz = cargar_config("imu")["frecuencia_hz"]
    factor_hueco = cargar_config("verificacion")["factor_hueco"]

    adoptado, motivos = analizar_grabacion(args.carpeta, cfg, ruta_config("imu_ruido"),
                                           f_nominal_hz, factor_hueco, args.forzar)
    if adoptado is None:
        return

    # Lo que va al filtro: lo medido × factor_filtro (en reposo se mide menos ruido del que hay
    # en movimiento: vibración, errores de escala, cambios de temperatura)
    if motivos:
        print("\nOJO: grabación NO válida guardada con --forzar: no copies esto en el filtro.")
    f = cfg["factor_filtro"]
    print(f"\nCopia en config/msckf.yaml, sección imu (medido × {f:g}):")
    print(f"  ruido_gir: {adoptado['ruido_gir'] * f:.2e}            # N_g [rad/s/√Hz]")
    print(f"  ruido_acc: {adoptado['ruido_acc'] * f:.2e}            # N_a [m/s²/√Hz]")
    print(f"  paseo_sesgo_gir: {adoptado['paseo_sesgo_gir'] * f:.2e}      # K_g [rad/s²/√Hz]")
    print(f"  paseo_sesgo_acc: {adoptado['paseo_sesgo_acc'] * f:.2e}      # K_a [m/s³/√Hz]")
    print(f"(Para Kalibr, en la etapa 7.4, se usará lo medido × {cfg['factor_kalibr']:g}.)")


if __name__ == "__main__":
    main()
