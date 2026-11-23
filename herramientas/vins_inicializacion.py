"""
herramientas/vins_inicializacion.py — Etapa 36: probar la inicialización visual-inercial.

Uso, desde la raíz del repositorio:
    python -m herramientas.vins_inicializacion                       simulador: 4 escenarios x 5 semillas
    python -m herramientas.vins_inicializacion --grabacion datos/X   una grabación EuRoC (real o sintética)
Opciones de --grabacion: --frames 11 (frames por ventana), --cada 2 (una imagen de cada N entre
frames), --paso 5 (imágenes entre dos intentos), --sin-calibracion-acc, y --ajustar clave=valor
(repetible) para probar otros umbrales de la sección "inicializacion" de config/vins.yaml sin
tocarlo, p. ej.:
    python -m herramientas.vins_inicializacion --grabacion datos/X --cada 3 --ajustar incertidumbre_escala_max=0.06
De cada intento se escribe el motivo y los números que lo explican (excitación, paralaje, error
del ajuste solo visual, |g|, incertidumbre de la escala), y al final un resumen de los motivos.
Sesgo del giroscopio con que se preintegra: por defecto, el del último reposo antes de la ventana
(inicializacion.SesgoEnReposo, lo mismo que hace VINS en el pipeline); --sesgo-giro-msckf usa el
que estima el MSCKF en ese instante y --sin-sesgo-reposo, 0 (como antes de la etapa 37).

Con una grabación, el VIO de siempre (front-end + MSCKF, apps/pi_vio) da las pistas de cada
imagen y su propia estimación. Se intenta inicializar con ventanas que van avanzando y, para cada
intento aceptado, se compara con el MSCKF en los mismos frames (cociente de longitudes de la
trayectoria = escala relativa, diferencia de inclinación y de sesgo del giroscopio) y, si la
grabación trae verdad.csv (las del simulador), con la verdad. La escala frente a una recta
medida con cinta queda para la etapa 37, con el estimador completo.
"""

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vio import modelo_camara as mc
from vio import so3
from vio.config import cargar_config, ruta_config
from vins import inicializacion as ini
from vins.preintegracion import ruido_desde_config


# =====================================================================
# Simulador
# =====================================================================
def simulador(semillas):
    """Los escenarios de tests/test_vins_inicializacion.py con varias semillas."""
    from tests.test_vins_inicializacion import (K, cfg_test, errores, secuencia, ventana_de_frames)
    R0, p0 = so3.exp_so3([0.1, -0.05, 0.3]), np.array([0.0, 0.0, 1.5])
    escenarios = {
        "excitado": dict(excitada=True, duracion_s=4.0),
        "suave": dict(),
        "reposo": dict(pose=lambda t: (R0, p0)),
        "giro puro": dict(pose=lambda t: (so3.exp_so3([0.0, 0.0, 0.8 * t]) @ so3.exp_so3([0.2 * np.sin(2 * t), 0.0, 0.0]) @ R0, p0)),
    }
    print(f"{'escenario':10s} {'sem':>3s}  resultado")
    for nombre, opciones in escenarios.items():
        for semilla in semillas:
            s, cfg_sim = secuencia(np.random.default_rng(semilla), **opciones)
            frames, pres, idx = ventana_de_frames(s, cfg_sim)
            r = ini.inicializar(frames, pres, s["R_c_i"], s["t_c_i"], K[0, 0], cfg_test())
            if r["ok"]:
                e_s, e_g = errores(r, s, idx)
                texto = (f"OK  escala {100 * e_s:+.2f} %  gravedad {e_g:.2f}°  σ_L/L {100 * r['incertidumbre_escala']:.1f} %"
                         f"  b_g err {np.degrees(np.linalg.norm(r['b_g'] - s['verdad']['b_g'][idx[0]])):.3f} °/s")
            else:
                texto = f"rechazada: {r['motivo']}"
            print(f"{nombre:10s} {semilla:3d}  {texto}")


# =====================================================================
# Grabación
# =====================================================================
def recorrer_grabacion(carpeta, sin_calibracion_acc):
    """Pasa la grabación por el VIO de siempre y guarda, por imagen con pose del MSCKF: el
    instante corregido (t + t_d), las observaciones (id -> xy normalizadas) y el estado.
    Devuelve: (imagenes, imu (t, w, a) con el acelerómetro corregido, R_c_i, t_c_i, K)."""
    from apps.pi_vio import ADELANTO_IMU_NS, cargar_calibracion_acc, eventos_grabacion
    from vio import calibracion_imu as ci
    from vio.pipeline import PipelineVio

    cfgs = {n: cargar_config(n) for n in ("vio", "frontend", "msckf", "triangulacion")}
    K, dist, _ = mc.cargar_intrinsecos()
    ext = cargar_config("camara_imu")
    R_c_i = np.array(ext["R_i_c"], dtype=float).T
    t_c_i = -R_c_i @ np.array(ext["p_i_c_m"], dtype=float)
    calib_acc = cargar_calibracion_acc(ruta_config("imu_calibracion"), sin_calibracion_acc)
    pipeline = PipelineVio(K, dist, R_c_i, t_c_i, ext["t_d_ns"], cfgs, calib_acc)

    imagenes, t_imu, w_imu, a_imu = [], [], [], []
    for evento in eventos_grabacion(carpeta, max(ext["t_d_ns"], 0) + ADELANTO_IMU_NS):
        if evento[0] == "imu":
            _, t, w, a = evento
            pipeline.nueva_imu(t, w, a)
            a = np.asarray(a, dtype=float)
            if calib_acc is not None:
                a = ci.corregir_acelerometro(a, calib_acc[0], calib_acc[1])
            t_imu.append(int(t))
            w_imu.append(np.asarray(w, dtype=float))
            a_imu.append(a)
            continue
        r = pipeline.nueva_imagen(evento[1], evento[2])
        if r is None:
            continue
        ids = pipeline.frontend.ids
        xy = mc.normalizar(pipeline.frontend.puntos, K, dist) if len(ids) else np.zeros((0, 2))
        imagenes.append(SimpleNamespace(t_ns=r["t_ns"], obs={int(i): p for i, p in zip(ids, xy)},
                                        R_w_i=r["R_w_i"], p_w=r["p_w"], b_g=pipeline.filtro.estado.b_g.copy()))
    imu = (np.array(t_imu, dtype=np.int64), np.array(w_imu), np.array(a_imu))
    return imagenes, imu, R_c_i, t_c_i, K


def inclinacion_grados(R_a, R_b):
    """Ángulo entre las verticales (eje z del mundo) vistas desde dos orientaciones de la IMU."""
    za, zb = R_a.T @ [0.0, 0.0, 1.0], R_b.T @ [0.0, 0.0, 1.0]
    return float(np.degrees(np.arccos(np.clip(za @ zb, -1.0, 1.0))))


def detalle(r):
    """Los números de un intento (los que haya llegado a calcular)."""
    partes = []
    if r.get("excitacion") is not None:
        partes.append(f"exc {r['excitacion']:.2f} m/s²")
    if r.get("paralaje_px"):
        partes.append(f"paralaje {r['paralaje_px']:.0f} px")
    if r.get("rms_px") is not None:
        partes.append(f"rms {r['rms_px']:.2f} px")
    if r.get("g_lineal") is not None:
        partes.append(f"|g| {r['g_lineal']:.2f}")
    if r.get("incertidumbre_escala") is not None:
        partes.append(f"σ_L/L {100 * r['incertidumbre_escala']:.1f} %")
    return ", ".join(partes)


def aplicar_ajustes(cfg, ajustes):
    """--ajustar clave=valor sobre cfg["inicializacion"] (el valor se convierte al tipo del YAML)."""
    for ajuste in ajustes:
        clave, valor = ajuste.split("=", 1)
        if clave not in cfg["inicializacion"]:
            raise SystemExit(f"--ajustar: {clave} no está en la sección inicializacion de vins.yaml")
        tipo = type(cfg["inicializacion"][clave])
        cfg["inicializacion"][clave] = (valor.lower() in ("1", "true", "si", "sí")) if tipo is bool else tipo(valor)
        print(f"inicializacion.{clave} = {cfg['inicializacion'][clave]}")


def sesgos_en_reposo(t_imu, w_imu, a_imu, cfg):
    """Recorre la IMU con SesgoEnReposo. Devuelve: (instantes (K,), sesgos (K, 3)) de cada reposo."""
    detector = ini.SesgoEnReposo(cfg["estimador"])
    t_r, b_r = [], []
    for t, w, a in zip(t_imu, w_imu, a_imu):
        if detector.actualizar(t, w, a):
            t_r.append(int(t))
            b_r.append(detector.b_g.copy())
    return np.array(t_r, dtype=np.int64), np.array(b_r).reshape(-1, 3)


def grabacion(carpeta, n_frames, cada, paso, sin_calibracion_acc, ajustes=(), sesgo_msckf=False,
              sin_sesgo_reposo=False):
    from vio.pipeline import leer_poses
    carpeta = Path(carpeta)
    print(f"Pasando {carpeta} por el VIO de siempre (MSCKF)...")
    imagenes, (t_imu, w_imu, a_imu), R_c_i, t_c_i, K = recorrer_grabacion(carpeta, sin_calibracion_acc)
    print(f"{len(imagenes)} imágenes con pose del MSCKF")
    verdad = leer_poses(carpeta / "verdad.csv") if (carpeta / "verdad.csv").exists() else None
    cfg = copy.deepcopy(cargar_config("vins"))
    aplicar_ajustes(cfg, ajustes)
    ruido = ruido_desde_config(cfg["preintegracion"])

    from collections import Counter
    t_rep, b_rep = sesgos_en_reposo(t_imu, w_imu, a_imu, cfg)
    if len(t_rep):
        print(f"Reposos detectados: {len(t_rep)} muestras quietas; primer sesgo del giroscopio "
              f"{np.round(np.degrees(b_rep[0]), 3)} °/s")
    aceptadas, motivos = 0, Counter()
    for inicio in range(0, len(imagenes) - (n_frames - 1) * cada, paso):
        frames = [imagenes[inicio + cada * q] for q in range(n_frames)]
        b_g0 = np.zeros(3)
        if sesgo_msckf:
            b_g0 = frames[0].b_g
        elif not sin_sesgo_reposo:
            antes = np.flatnonzero(t_rep <= frames[0].t_ns)
            if len(antes):
                b_g0 = b_rep[antes[-1]]
        pres = [ini.preintegrar_entre(t_imu, w_imu, a_imu, frames[q].t_ns, frames[q + 1].t_ns,
                                      b_g0, np.zeros(3), ruido) for q in range(n_frames - 1)]
        r = ini.inicializar(frames, pres, R_c_i, t_c_i, K[0, 0], cfg)
        t_s = (frames[0].t_ns - imagenes[0].t_ns) / 1e9
        if not r["ok"]:
            print(f"t = {t_s:6.2f} s  rechazada: {r['motivo']}   [{detalle(r)}]")
            motivos[r["motivo"].split(" (")[0]] += 1
            continue
        aceptadas += 1
        L_vins = ini.longitud_trayectoria(r["estados"])
        L_msckf = float(sum(np.linalg.norm(b.p_w - a.p_w) for a, b in zip(frames[:-1], frames[1:])))
        texto = (f"t = {t_s:6.2f} s  OK  σ_L/L {100 * r['incertidumbre_escala']:.1f} %  "
                 f"escala/MSCKF {L_vins / L_msckf:.3f}  inclinación−MSCKF {inclinacion_grados(r['estados'][0].R_w_i, frames[0].R_w_i):.2f}°  "
                 f"b_g−MSCKF {np.degrees(np.linalg.norm(r['b_g'] - frames[0].b_g)):.2f} °/s")
        if verdad is not None:
            t_v, p_v, R_v = verdad
            k = [int(np.argmin(np.abs(t_v - f.t_ns))) for f in frames]
            L_v = float(np.sum(np.linalg.norm(np.diff(p_v[k], axis=0), axis=1)))
            texto += (f"  | escala/verdad {L_vins / L_v:.3f}  inclinación−verdad "
                      f"{inclinacion_grados(r['estados'][0].R_w_i, R_v[k[0]]):.2f}°")
        print(texto)
    print(f"{aceptadas} inicializaciones aceptadas")
    print("Rechazos: " + "; ".join(f"{m} ×{n}" for m, n in motivos.most_common()))


def main():
    parser = argparse.ArgumentParser(description="Inicialización visual-inercial (etapa 36)")
    parser.add_argument("--grabacion", default=None)
    parser.add_argument("--frames", type=int, default=11)
    parser.add_argument("--cada", type=int, default=2)
    parser.add_argument("--paso", type=int, default=5)
    parser.add_argument("--semillas", type=int, default=5)
    parser.add_argument("--sin-calibracion-acc", action="store_true")
    parser.add_argument("--sesgo-giro-msckf", action="store_true",
                        help="preintegrar con el sesgo del giroscopio del MSCKF (experimento)")
    parser.add_argument("--sin-sesgo-reposo", action="store_true",
                        help="preintegrar con sesgo 0 (como antes de la etapa 37)")
    parser.add_argument("--ajustar", action="append", default=[], metavar="CLAVE=VALOR",
                        help="otro valor para un umbral de inicializacion (repetible)")
    args = parser.parse_args()
    if args.grabacion is None:
        simulador(range(1, args.semillas + 1))
    else:
        grabacion(args.grabacion, args.frames, args.cada, args.paso, args.sin_calibracion_acc, args.ajustar,
                  args.sesgo_giro_msckf, args.sin_sesgo_reposo)


if __name__ == "__main__":
    main()