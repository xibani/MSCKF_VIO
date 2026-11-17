"""
herramientas/comprobar_camara_imu.py — ¿Cuadran los giros de la cámara con los del giroscopio?

El MSCKF predice dónde caerá cada punto usando la rotación del giroscopio llevada a la cámara con
la extrínseca R_c_i y el desfase t_d de config/camara_imu.yaml. Si R_c_i o t_d están mal, la
predicción falla unos píxeles en cuanto la cámara gira, y la triangulación descarta las pistas por
"error de reproyección alto" (umbral de config/triangulacion.yaml) aunque las imágenes sean buenas.

Esta herramienta lo mide en una grabación, SIN el filtro:
  1. Giro de la cámara entre dos imágenes, solo con la visión: esquinas GFTT, KLT ida-vuelta,
     matriz esencial (USAC_ACCURATE) y recoverPose -> R_b_a (puntos de la cámara a a la cámara b),
     refinada minimizando el error de Sampson con todos los inliers (el modelo de RANSAC sale de
     5 puntos y su rotación tiene errores de décimas de grado: varios píxeles).
  2. Giro de la IMU en el mismo intervalo, desplazado t_d: integral del giroscopio (menos su sesgo,
     medido al principio, en reposo). En intervalos de 0,1 s la integral del vector ω es el vector
     de rotación con error de segundo orden.
  3. t_d: el desfase que mejor iguala el TAMAÑO de los dos giros (no depende de R_c_i).
  4. R_c_i: la rotación que mejor lleva los vectores de giro de la IMU a los de la cámara
     (problema de Wahba, resuelto con SVD como en Kabsch). Después se afina t_d con el giro
     completo (no solo su tamaño) y se repite Wahba.
  5. Error de cada intervalo entre el giro visual y el predicho, en grados y en píxeles
     (ángulo · fx), con la extrínseca de config y con la estimada, por tramos de velocidad angular.
Si con la estimada el error baja mucho, la de config es la que hace fallar al MSCKF.

Uso, desde la raíz del repositorio:
    python -m herramientas.comprobar_camara_imu datos/20261116_162115
Imprime el resumen y guarda <carpeta>/camara_imu.png
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                     # sin ventana: la figura se guarda en un PNG
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from herramientas.diagnostico_grabacion import leer_camara, leer_imu
from vio import modelo_camara as mc
from vio import so3
from vio.config import cargar_config


# ----------------------------------------------------------------------------------------------
# IMU: integral del giroscopio
# ----------------------------------------------------------------------------------------------

def integral_acumulada(t_s, w):
    """C[k] = ∫ w dt desde t_s[0] hasta t_s[k] (regla del trapecio). Devuelve (N, 3) [rad]."""
    C = np.zeros_like(w)
    for k in range(1, len(t_s)):
        C[k] = C[k - 1] + 0.5 * (w[k - 1] + w[k]) * (t_s[k] - t_s[k - 1])
    return C


def giro_integrado(t_s, C, t_a, t_b):
    """∫ w dt entre t_a y t_b (arrays de M intervalos), interpolando la integral acumulada.
    Devuelve (M, 3): vector de rotación de la IMU en ese intervalo, en ejes de la IMU."""
    theta = np.zeros((len(t_a), 3))
    for eje in range(3):
        theta[:, eje] = np.interp(t_b, t_s, C[:, eje]) - np.interp(t_a, t_s, C[:, eje])
    return theta


# ----------------------------------------------------------------------------------------------
# Visión: giro entre dos imágenes
# ----------------------------------------------------------------------------------------------

def refinar_rotacion(R0, t0, xy_a, xy_b, escala_huber):
    """Afina (R, dirección de t) minimizando el error de Sampson de x_bᵀ·E·x_a = 0, E = [t]x·R,
    con todos los puntos (coordenadas normalizadas). Parámetros: corrección de la rotación
    R = exp(δθ)·R0 (3) y la dirección de t en esféricas (2). Pérdida de Huber (escala en unidades
    normalizadas, ≈ píxeles / fx) por si queda algún atípico. Devuelve R refinada."""
    t0 = np.asarray(t0, dtype=float).ravel()
    t0 = t0 / np.linalg.norm(t0)
    angulos0 = [np.arctan2(t0[1], t0[0]), np.arcsin(np.clip(t0[2], -1.0, 1.0))]
    X_a = np.column_stack([xy_a, np.ones(len(xy_a))])
    X_b = np.column_stack([xy_b, np.ones(len(xy_b))])

    def direccion(angulos):
        acimut, elevacion = angulos
        return np.array([np.cos(elevacion) * np.cos(acimut), np.cos(elevacion) * np.sin(acimut),
                         np.sin(elevacion)])

    def residuos(q):
        R = so3.exp_so3(q[:3]) @ R0
        E = so3.hat(direccion(q[3:])) @ R
        Ex_a = X_a @ E.T                   # E·x_a de cada punto
        Etx_b = X_b @ E                    # Eᵀ·x_b de cada punto
        numerador = np.sum(X_b * Ex_a, axis=1)
        denominador = np.sqrt(Ex_a[:, 0] ** 2 + Ex_a[:, 1] ** 2 + Etx_b[:, 0] ** 2 + Etx_b[:, 1] ** 2)
        return numerador / denominador

    r = least_squares(residuos, np.r_[0.0, 0.0, 0.0, angulos0], loss="huber", f_scale=escala_huber)
    return so3.exp_so3(r.x[:3]) @ R0


def rotacion_vision(gris_a, gris_b, K, dist, p):
    """Rotación R_b_a (x_b = R_b_a · x_a + t) entre dos imágenes, con la matriz esencial.
    p: dict de parámetros. Devuelve (R_b_a, nº de inliers, flujo mediano [px]) o (None, n, flujo)."""
    esquinas = cv2.goodFeaturesToTrack(gris_a, maxCorners=p["max_esquinas"],
                                       qualityLevel=p["calidad_esquinas"], minDistance=p["dist_esquinas"])
    if esquinas is None or len(esquinas) < p["inliers_min"]:
        return None, 0, np.nan
    klt = dict(winSize=(p["klt_ventana"], p["klt_ventana"]), maxLevel=p["klt_niveles"],
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    # KLT de a a b y vuelta: solo cuentan los puntos que regresan a menos de ida_vuelta_px
    pb, st1, _ = cv2.calcOpticalFlowPyrLK(gris_a, gris_b, esquinas, None, **klt)
    pa_vuelta, st2, _ = cv2.calcOpticalFlowPyrLK(gris_b, gris_a, pb, None, **klt)
    vuelta = np.linalg.norm((pa_vuelta - esquinas).reshape(-1, 2), axis=1)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (vuelta < p["ida_vuelta_px"])
    if ok.sum() < p["inliers_min"]:
        return None, int(ok.sum()), np.nan
    uv_a = esquinas.reshape(-1, 2)[ok]
    uv_b = pb.reshape(-1, 2)[ok]
    flujo = float(np.median(np.linalg.norm(uv_b - uv_a, axis=1)))
    # Coordenadas normalizadas sin distorsión: la matriz esencial con focal 1 y centro (0, 0)
    xy_a = mc.normalizar(uv_a, K, dist)
    xy_b = mc.normalizar(uv_b, K, dist)
    umbral = p["umbral_esencial_px"] / K[0, 0]
    E, mascara = cv2.findEssentialMat(xy_a, xy_b, focal=1.0, pp=(0.0, 0.0), method=cv2.USAC_ACCURATE,
                                      prob=p["confianza_ransac"], threshold=umbral)
    if E is None or E.shape != (3, 3):
        return None, 0, flujo
    n_inliers, R_b_a, t_b_a, mascara_pose = cv2.recoverPose(E, xy_a, xy_b, focal=1.0, pp=(0.0, 0.0),
                                                           mask=mascara)
    if n_inliers < p["inliers_min"]:
        return None, int(n_inliers), flujo
    dentro = mascara_pose.ravel() > 0
    R_b_a = refinar_rotacion(R_b_a, t_b_a, xy_a[dentro], xy_b[dentro], umbral)
    return R_b_a, int(n_inliers), flujo


# ----------------------------------------------------------------------------------------------
# Estimación de t_d y R_c_i
# ----------------------------------------------------------------------------------------------

def estimar_t_d(t_imu, C, t_a, t_b, angulo_vision, candidatos_s):
    """El desfase que mejor iguala el tamaño del giro de la IMU con el de la cámara.
    Convención de Kalibr: la imagen con sello t_cam se tomó en el instante t_cam + t_d del reloj
    de la IMU. Devuelve (t_d [s], error mediano [rad] de cada candidato)."""
    errores = np.zeros(len(candidatos_s))
    for j, t_d in enumerate(candidatos_s):
        theta = giro_integrado(t_imu, C, t_a + t_d, t_b + t_d)
        errores[j] = np.median(np.abs(np.linalg.norm(theta, axis=1) - angulo_vision))
    return float(candidatos_s[np.argmin(errores)]), errores


def wahba(a, b):
    """R (3x3) que minimiza Σ |b_k - R·a_k|² (a, b: (M, 3)), por SVD (Kabsch)."""
    H = np.zeros((3, 3))
    for k in range(len(a)):
        H += np.outer(a[k], b[k])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def errores_rotacion(R_c_i, theta_imu, R_vision_cuerpo):
    """Ángulo [rad] entre el giro de la cámara medido (R_vision_cuerpo = R_a_b) y el predicho con
    la IMU: R_a_b = R_c_i · exp(θ_imu) · R_c_iᵀ."""
    errores = np.zeros(len(theta_imu))
    for k in range(len(theta_imu)):
        R_pred = R_c_i @ so3.exp_so3(theta_imu[k]) @ R_c_i.T
        errores[k] = np.linalg.norm(so3.log_so3(R_pred.T @ R_vision_cuerpo[k]))
    return errores


# ----------------------------------------------------------------------------------------------
# Programa
# ----------------------------------------------------------------------------------------------

def imprimir_por_velocidad(nombre, w_dps, err_px, tramos_dps):
    print(f"  {nombre}")
    print("      |ω| [°/s]      intervalos   error mediano [px]   p90 [px]")
    for lo, hi in zip(tramos_dps[:-1], tramos_dps[1:]):
        en = (w_dps >= lo) & (w_dps < hi)
        if en.sum() == 0:
            continue
        print(f"      {lo:4.0f} – {hi:4.0f}      {int(en.sum()):6d}        {np.median(err_px[en]):8.2f}"
              f"          {np.percentile(err_px[en], 90):7.2f}")
    print(f"      todos          {len(err_px):6d}        {np.median(err_px):8.2f}"
          f"          {np.percentile(err_px, 90):7.2f}")


def main():
    parser = argparse.ArgumentParser(description="Comprueba la extrínseca cámara-IMU y t_d con una grabación")
    parser.add_argument("carpeta", help="carpeta datos/<grabación> con mav0/")
    parser.add_argument("--paso", type=int, default=2, help="intervalo entre las dos imágenes, en imágenes")
    parser.add_argument("--max-esquinas", type=int, default=300)
    parser.add_argument("--calidad-esquinas", type=float, default=0.01)
    parser.add_argument("--dist-esquinas", type=float, default=10.0)
    parser.add_argument("--inliers-min", type=int, default=30)
    parser.add_argument("--umbral-esencial-px", type=float, default=1.0)
    parser.add_argument("--confianza-ransac", type=float, default=0.999)
    parser.add_argument("--flujo-min-px", type=float, default=1.0,
                        help="con menos movimiento de los puntos el intervalo no se usa (quieto)")
    parser.add_argument("--giro-min-grados", type=float, default=1.0,
                        help="para estimar R_c_i solo se usan intervalos con más giro que esto")
    parser.add_argument("--t-d-busqueda-ms", type=float, default=100.0, help="t_d se busca en ± esto")
    parser.add_argument("--t-d-paso-ms", type=float, default=0.5)
    parser.add_argument("--t-d-fino-ms", type=float, default=20.0,
                        help="el afinado de t_d busca en ± esto alrededor del primero")
    args = parser.parse_args()

    carpeta = Path(args.carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"no existe la carpeta {carpeta}")
    mav0 = carpeta / "mav0"
    cfg_seg = cargar_config("frontend")["seguimiento"]
    p = {"max_esquinas": args.max_esquinas, "calidad_esquinas": args.calidad_esquinas,
         "dist_esquinas": args.dist_esquinas, "inliers_min": args.inliers_min,
         "umbral_esencial_px": args.umbral_esencial_px, "confianza_ransac": args.confianza_ransac,
         "klt_ventana": cfg_seg["ventana_px"], "klt_niveles": cfg_seg["niveles_piramide"],
         "ida_vuelta_px": cfg_seg["ida_vuelta_max_px"]}

    # --- Configuración actual ---
    K, dist, _ = mc.cargar_intrinsecos()
    ext = cargar_config("camara_imu")
    R_c_i_config = np.array(ext["R_i_c"], dtype=float).T
    t_d_config_s = ext["t_d_ns"] / 1e9
    t_reposo_s = cargar_config("vio")["inicializacion"]["t_reposo_s"]

    # --- IMU: sesgo del giroscopio en el reposo inicial e integral acumulada ---
    t_imu_ns, w, _ = leer_imu(mav0 / "imu0" / "data.csv")
    t_cam_ns, nombres = leer_camara(mav0 / "cam0" / "data.csv")
    t0 = min(t_imu_ns[0], t_cam_ns[0])
    t_imu = (t_imu_ns - t0) / 1e9
    t_cam = (t_cam_ns - t0) / 1e9
    reposo = t_imu < t_imu[0] + t_reposo_s
    sesgo = w[reposo].mean(axis=0)
    C = integral_acumulada(t_imu, w - sesgo)
    print(f"Grabación: {carpeta}")
    print(f"Sesgo del giroscopio en los primeros {t_reposo_s} s: {np.degrees(sesgo).round(3)} °/s")

    # --- Visión: giro entre la imagen k y la k + paso ---
    print(f"Giros visuales entre imágenes separadas {args.paso} ({len(nombres)} imágenes)...")
    t_a, t_b, R_vis, inliers, flujos = [], [], [], [], []
    for k in range(0, len(nombres) - args.paso, args.paso):
        gris_a = cv2.imread(str(mav0 / "cam0" / "data" / nombres[k]), cv2.IMREAD_GRAYSCALE)
        gris_b = cv2.imread(str(mav0 / "cam0" / "data" / nombres[k + args.paso]), cv2.IMREAD_GRAYSCALE)
        if gris_a is None or gris_b is None:
            continue
        R_b_a, n, flujo = rotacion_vision(gris_a, gris_b, K, dist, p)
        if R_b_a is None or not flujo >= args.flujo_min_px:
            continue
        t_a.append(t_cam[k])
        t_b.append(t_cam[k + args.paso])
        R_vis.append(R_b_a.T)              # R_a_b: giro del CUERPO de la cámara de a a b
        inliers.append(n)
        flujos.append(flujo)
        if k % 500 == 0:
            print(f"  {k}/{len(nombres)}")
    t_a, t_b, inliers = np.array(t_a), np.array(t_b), np.array(inliers)
    R_vis = np.array(R_vis)
    theta_vis = np.array([so3.log_so3(R) for R in R_vis])     # en ejes de la cámara
    angulo_vis = np.linalg.norm(theta_vis, axis=1)
    dt = t_b - t_a
    print(f"  intervalos válidos: {len(t_a)} (inliers medianos {np.median(inliers):.0f})")

    # --- 1. t_d: iguala el tamaño de los giros (independiente de R_c_i) ---
    candidatos = np.arange(-args.t_d_busqueda_ms, args.t_d_busqueda_ms + 1e-9, args.t_d_paso_ms) / 1e3
    t_d_est, curva = estimar_t_d(t_imu, C, t_a, t_b, angulo_vis, candidatos)
    j_config = int(np.argmin(np.abs(candidatos - t_d_config_s)))
    print("\nDesfase temporal t_d (t_cam_corregido = t_cam + t_d):")
    print(f"  config:   {t_d_config_s * 1e3:7.2f} ms  -> | |θ_IMU| − |θ_cámara| | mediano"
          f" {np.degrees(curva[j_config]):.3f}°")
    print(f"  estimado: {t_d_est * 1e3:7.2f} ms  -> {np.degrees(curva.min()):.3f}°")

    # --- 2. R_c_i: Wahba con los giros grandes, y una segunda pasada sin los intervalos malos ---
    theta_imu = giro_integrado(t_imu, C, t_a + t_d_est, t_b + t_d_est)
    grandes = (angulo_vis > np.radians(args.giro_min_grados))
    R_c_i_est = wahba(theta_imu[grandes], theta_vis[grandes])
    err = errores_rotacion(R_c_i_est, theta_imu, R_vis)
    buenos = grandes & (err < 3.0 * np.median(err[grandes]))
    R_c_i_est = wahba(theta_imu[buenos], theta_vis[buenos])
    # t_d afinado con el giro completo (mediana del error con R_c_i estimada) cerca del primero,
    # y Wahba otra vez con ese t_d
    finos = t_d_est + np.arange(-args.t_d_fino_ms, args.t_d_fino_ms + 1e-9, args.t_d_paso_ms) / 1e3
    medianas = np.zeros(len(finos))
    for j, t_d in enumerate(finos):
        theta_j = giro_integrado(t_imu, C, t_a[buenos] + t_d, t_b[buenos] + t_d)
        medianas[j] = np.median(errores_rotacion(R_c_i_est, theta_j, R_vis[buenos]))
    t_d_est = float(finos[np.argmin(medianas)])
    # ¿Está determinado? Si la grabación gira a velocidad casi constante, desplazar el tiempo
    # apenas cambia el giro y muchos t_d dan el mismo error: se informa del intervalo compatible
    compatibles = finos[medianas <= 1.10 * medianas.min()]
    theta_imu = giro_integrado(t_imu, C, t_a + t_d_est, t_b + t_d_est)
    err = errores_rotacion(R_c_i_est, theta_imu, R_vis)
    buenos = grandes & (err < 3.0 * np.median(err[grandes]))
    R_c_i_est = wahba(theta_imu[buenos], theta_vis[buenos])
    print(f"  afinado con el giro completo: {t_d_est * 1e3:7.2f} ms"
          f" (compatible, error < mínimo + 10 %: {compatibles.min() * 1e3:.1f} – {compatibles.max() * 1e3:.1f} ms;"
          f" si el intervalo es ancho, esta grabación no determina t_d)")
    diferencia = np.degrees(np.linalg.norm(so3.log_so3(R_c_i_est @ R_c_i_config.T)))
    print(f"\nRotación de la extrínseca R_c_i (con {int(buenos.sum())} intervalos de más de"
          f" {args.giro_min_grados}°):")
    print("  config (R_i_c, columnas = ejes de la cámara en ejes de la IMU):")
    print("   ", np.array2string(R_c_i_config.T, precision=4, suppress_small=True).replace("\n", "\n    "))
    print("  estimada (R_i_c):")
    print("   ", np.array2string(R_c_i_est.T, precision=4, suppress_small=True).replace("\n", "\n    "))
    print(f"  diferencia entre las dos: {diferencia:.2f}°")

    # --- 3. Error de la predicción con la IMU, con config y con lo estimado ---
    theta_imu_config = giro_integrado(t_imu, C, t_a + t_d_config_s, t_b + t_d_config_s)
    err_config_px = errores_rotacion(R_c_i_config, theta_imu_config, R_vis) * K[0, 0]
    err_est_px = errores_rotacion(R_c_i_est, theta_imu, R_vis) * K[0, 0]
    # Solo t_d estimado o solo R_c_i estimada: cuál de las dos pesa más
    err_solo_td_px = errores_rotacion(R_c_i_config, theta_imu, R_vis) * K[0, 0]
    err_solo_R_px = errores_rotacion(R_c_i_est, theta_imu_config, R_vis) * K[0, 0]
    w_dps = np.degrees(angulo_vis / dt)
    tramos = [0, 20, 40, 60, 90, 1000]
    print(f"\nError del giro predicho con la IMU respecto al visual, en píxeles (ángulo · fx = {K[0, 0]:.0f}),"
          f" en {args.paso} imágenes ({np.median(dt) * 1e3:.0f} ms):")
    imprimir_por_velocidad(f"CONFIG (R_c_i de config, t_d = {t_d_config_s * 1e3:.2f} ms)", w_dps, err_config_px, tramos)
    imprimir_por_velocidad(f"ESTIMADO (R_c_i estimada, t_d = {t_d_est * 1e3:.2f} ms)", w_dps, err_est_px, tramos)
    print(f"  solo cambiando t_d:   mediana {np.median(err_solo_td_px):.2f} px")
    print(f"  solo cambiando R_c_i: mediana {np.median(err_solo_R_px):.2f} px")
    print("  (El MSCKF acumula este error a lo largo de su ventana de clones y la triangulación descarta"
          " la pista si supera error_max_px de config/triangulacion.yaml.)")

    # --- Figura ---
    figura, ejes = plt.subplots(3, 1, figsize=(15, 10))
    ax = ejes[0]
    w_imu_config = np.degrees(np.linalg.norm(theta_imu_config, axis=1) / dt)
    w_imu_est = np.degrees(np.linalg.norm(theta_imu, axis=1) / dt)
    ax.plot(t_a, w_dps, ".", ms=3, color="k", label="cámara (visión)")
    ax.plot(t_a, w_imu_config, lw=0.8, color="tab:red", label=f"IMU con t_d de config ({t_d_config_s * 1e3:.1f} ms)")
    ax.plot(t_a, w_imu_est, lw=0.8, color="tab:green", label=f"IMU con t_d estimado ({t_d_est * 1e3:.1f} ms)")
    ax.set_ylabel("|ω| [°/s]")
    ax.set_xlabel("t [s]")
    ax.set_title("Velocidad angular: cámara frente a IMU")
    ax.legend(fontsize=8)
    ax = ejes[1]
    ax.plot(t_a, err_config_px, ".", ms=3, color="tab:red", label="config")
    ax.plot(t_a, err_est_px, ".", ms=3, color="tab:green", label="estimado")
    ax.set_yscale("log")
    ax.set_ylabel("error [px]")
    ax.set_xlabel("t [s]")
    ax.set_title(f"Error del giro predicho con la IMU (cada {np.median(dt) * 1e3:.0f} ms)")
    ax.legend(fontsize=8)
    ax = ejes[2]
    ax.plot(w_dps, err_config_px, ".", ms=3, color="tab:red", alpha=0.6, label="config")
    ax.plot(w_dps, err_est_px, ".", ms=3, color="tab:green", alpha=0.6, label="estimado")
    ax.set_yscale("log")
    ax.set_xlabel("|ω| de la cámara [°/s]")
    ax.set_ylabel("error [px]")
    ax.set_title("Error frente a velocidad angular (crece con |ω| si la extrínseca o t_d están mal)")
    ax.legend(fontsize=8)
    for ax in ejes:
        ax.grid(True, alpha=0.3)
    figura.suptitle(f"{carpeta.name} — giroscopio frente a cámara")
    figura.tight_layout()
    ruta_png = carpeta / "camara_imu.png"
    figura.savefig(ruta_png, dpi=110)
    plt.close(figura)
    print(f"\nFigura: {ruta_png}")


if __name__ == "__main__":
    main()