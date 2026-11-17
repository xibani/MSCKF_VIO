"""
herramientas/diagnostico_grabacion.py — Por qué diverge el VIO en una grabación.

Pone en el mismo eje de tiempo la trayectoria del VIO y los datos crudos de la grabación:
  - huecos en los timestamps de cámara y de IMU (frames perdidos, FIFO del MPU-6050 desbordada),
  - saturación del MPU-6050 (giróscopo o acelerómetro en el fondo de escala),
  - IMU "congelada" (muestras idénticas repetidas: se está leyendo siempre el mismo registro),
  - imagen congelada (frames idénticos), oscura o movida (nitidez = varianza del laplaciano),
  - pérdida de textura: esquinas (GFTT) y puntos seguidos con KLT ida-vuelta entre frames consecutivos.
Si existe vio_recuentos.csv (lo escribe apps/pi_vio), añade lo que hizo el MSCKF en cada imagen:
pistas usadas en la corrección, rechazadas (chi², reproyección, paralaje), ZUPT y la incertidumbre
que el filtro cree tener (sigma de p, v y θ).
La divergencia se fecha así: primera vez que la trayectoria pasa de --dist-max m del origen y,
desde ahí hacia atrás, el último mínimo de la distancia (donde empezó a crecer). Con recuentos,
además, el primer tramo de --sin-correccion-s segundos seguidos sin ninguna pista usada.

Uso, desde la raíz del repositorio:
    python -m herramientas.diagnostico_grabacion datos/20261116_162115
    python -m herramientas.diagnostico_grabacion datos/20261116_162115 --gyro-max-dps 500 --acc-max-g 4
    python -m herramientas.diagnostico_grabacion datos/20261116_162115 --sin-imagenes   # sin analizar los PNG
Imprime el resumen y guarda <carpeta>/diagnostico.png
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                     # sin ventana: la figura se guarda en un PNG
import matplotlib.pyplot as plt
import numpy as np

from herramientas.informe_lazo import elegir_csv
from vio.pipeline import leer_poses, leer_recuentos

G = 9.80665                               # gravedad estándar [m/s²], para pasar el fondo de escala de g a m/s²


# ----------------------------------------------------------------------------------------------
# Lectura de la grabación (formato EuRoC/ASL)
# ----------------------------------------------------------------------------------------------

def leer_imu(ruta_csv):
    """imu0/data.csv: t [ns], w_x w_y w_z [rad/s], a_x a_y a_z [m/s²].
    El timestamp se lee aparte como int64 para no perder precisión al pasar por float64."""
    t_ns = np.loadtxt(ruta_csv, delimiter=",", comments="#", usecols=0, dtype=np.int64)
    valores = np.loadtxt(ruta_csv, delimiter=",", comments="#", usecols=range(1, 7))
    return t_ns, valores[:, 0:3], valores[:, 3:6]


def leer_camara(ruta_csv):
    """cam0/data.csv: t [ns], nombre del PNG."""
    t_ns, nombres = [], []
    with open(ruta_csv, newline="") as f:
        for fila in csv.reader(f):
            if not fila or fila[0].strip().startswith("#"):
                continue
            t_ns.append(int(fila[0]))
            nombres.append(fila[1].strip())
    return np.array(t_ns, dtype=np.int64), nombres


# ----------------------------------------------------------------------------------------------
# Detectores (funciones puras: se prueban en tests/test_diagnostico_grabacion.py)
# ----------------------------------------------------------------------------------------------

def detectar_huecos(t_ns, factor):
    """Intervalos entre muestras consecutivas mayores que factor × Δt mediano.
    Devuelve: dt_s (N-1), lista de (índice i del hueco entre i e i+1, t_i [ns], duración [s]),
              Δt mediano [s] y nº de Δt <= 0 (timestamps repetidos o hacia atrás)."""
    dt_s = np.diff(t_ns) / 1e9
    dt_mediano = float(np.median(dt_s))
    huecos = []
    for i in range(len(dt_s)):
        if dt_s[i] > factor * dt_mediano:
            huecos.append((i, int(t_ns[i]), float(dt_s[i])))
    no_monotonos = int(np.sum(dt_s <= 0))
    return dt_s, huecos, dt_mediano, no_monotonos


def detectar_saturacion(valores, limite, fraccion):
    """True en las muestras en que algún eje llega a fraccion × fondo de escala."""
    umbral = fraccion * limite
    marcado = np.zeros(len(valores), dtype=bool)
    for k in range(len(valores)):
        marcado[k] = np.max(np.abs(valores[k])) >= umbral
    return marcado


def rachas_identicas(valores, longitud_min):
    """Rachas de filas consecutivas idénticas de al menos longitud_min muestras.
    Un sensor real siempre tiene ruido: varias muestras exactamente iguales significan que el
    grabador está repitiendo la última lectura (FIFO vacía, I2C colgado...).
    Devuelve: lista de (índice de inicio, longitud)."""
    rachas = []
    inicio = 0
    for i in range(1, len(valores) + 1):
        if i < len(valores) and np.array_equal(valores[i], valores[i - 1]):
            continue
        if i - inicio >= longitud_min:
            rachas.append((inicio, i - inicio))
        inicio = i
    return rachas


def agrupar_intervalos(t_s, marcado, separacion_s):
    """Une las muestras marcadas en intervalos [t_ini, t_fin] si distan menos de separacion_s."""
    intervalos = []
    for t, m in zip(t_s, marcado):
        if not m:
            continue
        if intervalos and t - intervalos[-1][1] <= separacion_s:
            intervalos[-1][1] = t
        else:
            intervalos.append([t, t])
    return intervalos


def analizar_imagenes(carpeta_imagenes, nombres, max_esquinas, calidad_esquinas, dist_esquinas,
                      klt_ventana, klt_niveles, klt_ida_vuelta_px):
    """Métricas por frame. La entrada k de 'seguidos' y 'flujo_px' se refiere al par (k-1, k).
    Devuelve un dict de arrays (NaN donde no aplica):
      media       intensidad media (0-255)
      nitidez     varianza del laplaciano (baja = imagen movida o desenfocada)
      diferencia  |I_k - I_{k-1}| medio; 0 exacto = frame repetido (cámara congelada)
      esquinas    nº de esquinas GFTT del frame k
      seguidos    esquinas de k-1 seguidas en k con KLT y comprobación ida-vuelta
      flujo_px    mediana del desplazamiento de los puntos seguidos [px]"""
    n = len(nombres)
    res = {clave: np.full(n, np.nan) for clave in
           ["media", "nitidez", "diferencia", "esquinas", "seguidos", "flujo_px"]}
    ilegibles = []
    parametros_klt = dict(winSize=(klt_ventana, klt_ventana), maxLevel=klt_niveles,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    previa = None
    puntos_previos = None
    for k in range(n):
        img = cv2.imread(str(Path(carpeta_imagenes) / nombres[k]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            ilegibles.append(k)
            previa, puntos_previos = None, None
            continue
        res["media"][k] = img.mean()
        res["nitidez"][k] = cv2.Laplacian(img, cv2.CV_64F).var()

        if previa is not None and previa.shape == img.shape:
            res["diferencia"][k] = np.mean(np.abs(img.astype(np.int16) - previa.astype(np.int16)))
            if puntos_previos is None or len(puntos_previos) == 0:
                res["seguidos"][k] = 0
            else:
                # KLT del frame anterior al actual y vuelta: solo cuentan los que regresan a < 1 px
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(previa, img, puntos_previos, None, **parametros_klt)
                p0_vuelta, st2, _ = cv2.calcOpticalFlowPyrLK(img, previa, p1, None, **parametros_klt)
                error_vuelta = np.linalg.norm((p0_vuelta - puntos_previos).reshape(-1, 2), axis=1)
                ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (error_vuelta < klt_ida_vuelta_px)
                res["seguidos"][k] = int(ok.sum())
                if ok.any():
                    desplazamiento = np.linalg.norm((p1 - puntos_previos).reshape(-1, 2), axis=1)
                    res["flujo_px"][k] = float(np.median(desplazamiento[ok]))

        puntos = cv2.goodFeaturesToTrack(img, maxCorners=max_esquinas, qualityLevel=calidad_esquinas,
                                         minDistance=dist_esquinas)
        res["esquinas"][k] = 0 if puntos is None else len(puntos)
        puntos_previos = puntos
        previa = img
        if k % 500 == 0:
            print(f"  imágenes: {k}/{n}")
    res["ilegibles"] = ilegibles
    return res


def velocidad_en_ventana(t_s, p, ventana_s):
    """Velocidad media de la trayectoria en [t_i, t_i + ventana_s] (más robusta que la diferencia
    entre dos poses seguidas, que se dispara con cualquier pose ruidosa)."""
    v = np.full(len(t_s), np.nan)
    j = 0
    for i in range(len(t_s)):
        if j < i:
            j = i
        while j < len(t_s) - 1 and t_s[j] < t_s[i] + ventana_s:
            j += 1
        if t_s[j] > t_s[i]:
            v[i] = np.linalg.norm(p[j] - p[i]) / (t_s[j] - t_s[i])
    return v


def primer_sostenido(t_s, condicion, duracion_s):
    """Primer t en que la condición se cumple SIN INTERRUPCIÓN durante al menos duracion_s
    (un pico suelto de una sola pose no cuenta). Devuelve t o None."""
    inicio = None
    for k in range(len(t_s)):
        if condicion[k]:
            if inicio is None:
                inicio = k
            if t_s[k] - t_s[inicio] >= duracion_s:
                return float(t_s[inicio])
        else:
            inicio = None
    return None


def inicio_divergencia(t_s, distancia, dist_max, ventana_atras_s, tolerancia_m):
    """Cuándo empezó a irse la trayectoria.

    t_dist: primera vez a más de dist_max del origen (inequívoco, pero tardío: el error ya es grande).
    t_inicio: desde t_dist hacia atrás, mientras la distancia siga bajando, se guarda el mínimo;
              se para al volver a subir más de tolerancia_m por encima de él (se ha cruzado el valle)
              o al llegar a ventana_atras_s. Es el último momento antes de crecer sin volver.
              (El mínimo absoluto de la ventana no vale: puede ser un valle anterior más hondo.)
    Devuelve (t_dist, t_inicio), o (None, None) si nunca pasa de dist_max.
    """
    indices = np.flatnonzero(distancia > dist_max)
    if len(indices) == 0:
        return None, None
    k_dist = indices[0]
    k_minimo = k_dist
    k = k_dist
    while k > 0 and t_s[k - 1] >= t_s[k_dist] - ventana_atras_s:
        k -= 1
        if distancia[k] < distancia[k_minimo]:
            k_minimo = k
        elif distancia[k] > distancia[k_minimo] + tolerancia_m:
            break
    return float(t_s[k_dist]), float(t_s[k_minimo])


def tramos_sin_correccion(t_s, usadas, duracion_min_s):
    """Tramos de imágenes seguidas sin NINGUNA pista usada en la corrección del MSCKF (en ellos
    el filtro solo integra la IMU) de al menos duracion_min_s. Devuelve: lista de [t_ini, t_fin]."""
    tramos = []
    inicio = None
    for k in range(len(t_s) + 1):
        sin_pistas = k < len(t_s) and usadas[k] == 0
        if sin_pistas and inicio is None:
            inicio = k
        elif not sin_pistas and inicio is not None:
            if t_s[k - 1] - t_s[inicio] >= duracion_min_s:
                tramos.append([float(t_s[inicio]), float(t_s[k - 1])])
            inicio = None
    return tramos


# ----------------------------------------------------------------------------------------------
# Informe
# ----------------------------------------------------------------------------------------------

def imprimir_intervalos(nombre, intervalos, max_lineas=12):
    print(f"  {nombre}: {len(intervalos)}")
    for t_ini, t_fin in intervalos[:max_lineas]:
        print(f"      t = {t_ini:7.2f} – {t_fin:7.2f} s")
    if len(intervalos) > max_lineas:
        print(f"      ... y {len(intervalos) - max_lineas} más")


def tabla_por_segundo(t_div, t_ini, t_fin, t_cam, t_imu, w, a, img, rec, t_tray, distancia):
    """Lo que había en los datos (y lo que hizo el MSCKF), segundo a segundo, de t_ini a t_fin.
    La fila del segundo t_div (inicio estimado de la divergencia) lleva la marca <==."""
    print(f"\nSegundo a segundo (la marca <== es el inicio estimado de la divergencia, t = {t_div:.2f} s):")
    print("   t [s] | frames  Δt_cam máx | IMU  Δt_imu máx | |w| máx  |a| máx |"
          " esquinas  seguidos  flujo | nitidez  media | dist "
          + ("| usadas  chi2 reproy paral quieto  σ_p" if rec is not None else ""))
    print("         |          [ms]     |       [ms]     | [°/s]   [m/s²]  |"
          "  (mín)     (mín)    [px] |               |  [m] "
          + ("| (suma en el segundo)          [m]" if rec is not None else ""))
    for s in range(max(int(math.floor(t_ini)), 0), int(math.floor(t_fin)) + 1):
        en_cam = (t_cam >= s) & (t_cam < s + 1)
        en_imu = (t_imu >= s) & (t_imu < s + 1)
        en_tray = (t_tray >= s) & (t_tray < s + 1)
        # Δt máximo: incluye el salto desde la última muestra del segundo anterior
        def dt_max_ms(t, mascara):
            idx = np.flatnonzero(mascara)
            if len(idx) == 0:
                return float("nan")
            desde = max(idx[0] - 1, 0)
            return float(np.max(np.diff(t[desde:idx[-1] + 1]), initial=0.0) * 1000)
        w_max = np.degrees(np.max(np.abs(w[en_imu]))) if en_imu.any() else float("nan")
        a_max = np.max(np.abs(a[en_imu])) if en_imu.any() else float("nan")
        if img is not None and en_cam.any():
            esq = np.nanmin(img["esquinas"][en_cam]) if np.any(~np.isnan(img["esquinas"][en_cam])) else np.nan
            seg = np.nanmin(img["seguidos"][en_cam]) if np.any(~np.isnan(img["seguidos"][en_cam])) else np.nan
            flu = np.nanmedian(img["flujo_px"][en_cam]) if np.any(~np.isnan(img["flujo_px"][en_cam])) else np.nan
            nit = np.nanmean(img["nitidez"][en_cam]) if np.any(~np.isnan(img["nitidez"][en_cam])) else np.nan
            med = np.nanmean(img["media"][en_cam]) if np.any(~np.isnan(img["media"][en_cam])) else np.nan
        else:
            esq = seg = flu = nit = med = float("nan")
        dist = distancia[en_tray][-1] if en_tray.any() else float("nan")
        marca = " <==" if s == int(math.floor(t_div)) else ""
        linea = (f"  {s:6d} | {int(en_cam.sum()):5d}  {dt_max_ms(t_cam, en_cam):9.1f} |"
                 f" {int(en_imu.sum()):4d}  {dt_max_ms(t_imu, en_imu):9.1f} |"
                 f" {w_max:6.0f}  {a_max:6.1f}  | {esq:7.0f}  {seg:7.0f}  {flu:6.1f} |"
                 f" {nit:7.0f}  {med:5.0f} | {dist:5.2f} ")
        if rec is not None:
            en_rec = (rec["t_s"] >= s) & (rec["t_s"] < s + 1)
            suma = {c: int(rec[c][en_rec].sum()) for c in ("usadas", "chi2", "reproyeccion",
                                                           "sin_paralaje", "quieto")}
            sigma_p = rec["sigma_p_m"][en_rec][-1] if en_rec.any() else float("nan")
            linea += (f"| {suma['usadas']:5d} {suma['chi2']:5d} {suma['reproyeccion']:5d}"
                      f" {suma['sin_paralaje']:5d} {suma['quieto']:5d}  {sigma_p:6.3f}")
        print(linea + marca)


def sumas_por_segundo(t_s, valores):
    """Suma de 'valores' en cada segundo entero. Devuelve (centros de los segundos, sumas)."""
    segundos = np.floor(t_s).astype(int)
    eje = np.arange(segundos.min(), segundos.max() + 1)
    sumas = np.array([valores[segundos == s].sum() for s in eje])
    return eje + 0.5, sumas


def dibujar(ruta_png, titulo, t_tray, distancia, velocidad, t_cam, dt_cam, t_imu, dt_imu,
            w, a, gyro_max, acc_max, img, rec, t_div, t_dist, intervalos_huecos_cam,
            intervalos_huecos_imu, intervalos_sat, intervalos_sin_correccion):
    filas = 3 + (2 if img is not None else 0) + (1 if rec is not None else 0)
    figura, ejes = plt.subplots(filas, 1, figsize=(15, 2.6 * filas), sharex=True)

    ax = ejes[0]
    ax.plot(t_tray, distancia, color="tab:blue", label="distancia al origen")
    ax.set_ylabel("dist. [m]")
    ax2 = ax.twinx()
    ax2.plot(t_tray, velocidad, color="tab:gray", lw=0.8, label="velocidad (ventana)")
    ax2.set_ylabel("v [m/s]")
    ax.set_title("Trayectoria del VIO")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper center", fontsize=8)

    ax = ejes[1]
    ax.semilogy(t_cam[1:], dt_cam * 1000, ".", ms=2, color="tab:green", label="Δt cámara")
    ax.semilogy(t_imu[1:], dt_imu * 1000, ".", ms=1, color="tab:purple", label="Δt IMU")
    for t_ini, t_fin in intervalos_huecos_cam:
        ax.axvspan(t_ini, t_fin + 0.05, color="tab:green", alpha=0.25)
    for t_ini, t_fin in intervalos_huecos_imu:
        ax.axvspan(t_ini, t_fin + 0.05, color="tab:purple", alpha=0.25)
    ax.set_ylabel("Δt [ms]")
    ax.set_title("Timestamps (franjas = huecos)")
    ax.legend(loc="upper left", fontsize=8)

    ax = ejes[2]
    ax.plot(t_imu, np.degrees(np.max(np.abs(w), axis=1)), lw=0.6, color="tab:red", label="máx |w_i| [°/s]")
    ax.axhline(np.degrees(gyro_max), color="tab:red", ls="--", lw=1)
    ax.set_ylabel("giro [°/s]")
    ax2 = ax.twinx()
    ax2.plot(t_imu, np.max(np.abs(a), axis=1), lw=0.6, color="tab:orange", label="máx |a_i| [m/s²]")
    ax2.axhline(acc_max, color="tab:orange", ls="--", lw=1)
    ax2.set_ylabel("acel. [m/s²]")
    for t_ini, t_fin in intervalos_sat:
        ax.axvspan(t_ini, t_fin + 0.02, color="red", alpha=0.2)
    ax.set_title("IMU (líneas discontinuas = fondo de escala; franjas = saturación)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper center", fontsize=8)

    if img is not None:
        ax = ejes[3]
        ax.plot(t_cam, img["esquinas"], lw=0.8, color="tab:blue", label="esquinas GFTT")
        ax.plot(t_cam, img["seguidos"], lw=0.8, color="tab:green", label="seguidos KLT (k-1→k)")
        ax.set_ylabel("nº puntos")
        ax2 = ax.twinx()
        ax2.plot(t_cam, img["flujo_px"], lw=0.6, color="tab:gray", label="flujo mediano [px]")
        ax2.set_ylabel("flujo [px]")
        ax.set_title("Textura y seguimiento")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper center", fontsize=8)

        ax = ejes[4]
        ax.semilogy(t_cam, img["nitidez"], lw=0.8, color="tab:brown", label="nitidez (var. laplaciano)")
        ax.set_ylabel("nitidez")
        ax2 = ax.twinx()
        ax2.plot(t_cam, img["media"], lw=0.8, color="k", label="intensidad media")
        ax2.plot(t_cam, img["diferencia"], lw=0.6, color="tab:cyan", label="|I_k − I_k-1|")
        ax2.set_ylabel("nivel de gris")
        ax.set_title("Imagen")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper center", fontsize=8)

    if rec is not None:
        ax = ejes[-1]
        for clave, color, nombre in [("usadas", "tab:green", "usadas"), ("chi2", "tab:red", "rechazo χ²"),
                                     ("reproyeccion", "tab:orange", "rechazo reproyección"),
                                     ("sin_paralaje", "tab:gray", "sin paralaje")]:
            x, y = sumas_por_segundo(rec["t_s"], rec[clave])
            ax.step(x, y, where="mid", color=color, lw=1, label=nombre)
        quietos = rec["quieto"] > 0
        ax.plot(rec["t_s"][quietos], np.zeros(quietos.sum()), "|", color="k", ms=8, label="ZUPT")
        ax.set_ylabel("pistas / s")
        ax2 = ax.twinx()
        ax2.semilogy(rec["t_s"], rec["sigma_p_m"], color="tab:blue", lw=1, label="σ_p del filtro [m]")
        ax2.semilogy(rec["t_s"], rec["sigma_v_m_s"], color="tab:cyan", lw=1, label="σ_v del filtro [m/s]")
        ax2.set_ylabel("σ (log)")
        for t_ini, t_fin in intervalos_sin_correccion:
            ax.axvspan(t_ini, t_fin, color="red", alpha=0.15)
        ax.set_title("MSCKF (franjas = sin ninguna pista usada; | = ZUPT)")
        ax.legend(loc="upper left", fontsize=8, ncol=5)
        ax2.legend(loc="upper right", fontsize=8)

    for ax in ejes:
        ax.grid(True, alpha=0.3)
        if t_div is not None:
            ax.axvline(t_div, color="red", ls="--", lw=1.2)
        if t_dist is not None:
            ax.axvline(t_dist, color="red", ls=":", lw=1.2)
    ejes[-1].set_xlabel("t desde el inicio de la grabación [s]")
    figura.suptitle(titulo)
    figura.tight_layout()
    figura.savefig(ruta_png, dpi=110)
    plt.close(figura)


def diagnostico(carpeta, args):
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"no existe la carpeta {carpeta}")
    mav0 = carpeta / "mav0"
    t_cam_ns, nombres = leer_camara(mav0 / "cam0" / "data.csv")
    t_imu_ns, w, a = leer_imu(mav0 / "imu0" / "data.csv")
    ruta_tray = elegir_csv(carpeta)
    t_tray_ns, p, _ = leer_poses(ruta_tray)
    # Recuentos del MSCKF de la misma prueba: vio_trayectoria_2.csv <-> vio_recuentos_2.csv
    ruta_rec = ruta_tray.with_name(ruta_tray.name.replace("vio_trayectoria", "vio_recuentos"))
    rec = leer_recuentos(ruta_rec) if ruta_rec.is_file() else None

    # Origen de tiempos común: la primera muestra de cámara o IMU
    t0 = min(t_cam_ns[0], t_imu_ns[0])
    t_cam = (t_cam_ns - t0) / 1e9
    t_imu = (t_imu_ns - t0) / 1e9
    if abs(int(t_tray_ns[0]) - int(t_cam_ns[0])) > 3600 * 10**9:
        # La trayectoria no está en la misma base de reloj: se alinea su inicio con el 1er frame
        print("AVISO: la trayectoria no comparte base de tiempos con la grabación; "
              "se alinea su primera pose con el primer frame (el eje de tiempo será aproximado).")
        t_tray = (t_tray_ns - t_tray_ns[0]) / 1e9 + t_cam[0]
        if rec is not None:
            rec["t_s"] = (rec["t_ns"] - t_tray_ns[0]) / 1e9 + t_cam[0]
    else:
        t_tray = (t_tray_ns - t0) / 1e9
        if rec is not None:
            rec["t_s"] = (rec["t_ns"] - t0) / 1e9

    gyro_max = np.radians(args.gyro_max_dps)
    acc_max = args.acc_max_g * G

    # --- Tiempos ---
    dt_cam, huecos_cam, dt_med_cam, nm_cam = detectar_huecos(t_cam_ns, args.factor_hueco)
    dt_imu, huecos_imu, dt_med_imu, nm_imu = detectar_huecos(t_imu_ns, args.factor_hueco)
    print(f"Grabación: {carpeta}")
    print(f"Trayectoria: {ruta_tray}")
    print(f"Recuentos del MSCKF: {ruta_rec if rec is not None else 'no hay (vuelve a pasar apps.pi_vio)'}")
    print(f"\nCámara: {len(t_cam)} frames en {t_cam[-1] - t_cam[0]:.1f} s, {1 / dt_med_cam:.1f} Hz (mediana)")
    print(f"  Δt <= 0: {nm_cam}   huecos (Δt > {args.factor_hueco}×{dt_med_cam * 1000:.1f} ms): {len(huecos_cam)}")
    for _, t_h, d in huecos_cam[:12]:
        print(f"      t = {(t_h - t0) / 1e9:7.2f} s   hueco de {d * 1000:7.1f} ms"
              f" (~{round(d / dt_med_cam) - 1} frames perdidos)")
    print(f"IMU: {len(t_imu)} muestras en {t_imu[-1] - t_imu[0]:.1f} s, {1 / dt_med_imu:.1f} Hz (mediana)")
    print(f"  Δt <= 0: {nm_imu}   huecos (Δt > {args.factor_hueco}×{dt_med_imu * 1000:.2f} ms): {len(huecos_imu)}")
    for _, t_h, d in huecos_imu[:12]:
        print(f"      t = {(t_h - t0) / 1e9:7.2f} s   hueco de {d * 1000:7.1f} ms")

    # --- IMU ---
    sat_gyro = detectar_saturacion(w, gyro_max, args.frac_saturacion)
    sat_acc = detectar_saturacion(a, acc_max, args.frac_saturacion)
    print(f"\nIMU (fondo de escala supuesto: ±{args.gyro_max_dps:.0f} °/s, ±{args.acc_max_g:.0f} g)")
    print(f"  |w| máx = {np.degrees(np.max(np.abs(w))):.0f} °/s,  |a| máx = {np.max(np.abs(a)):.1f} m/s²")
    int_sat_gyro = agrupar_intervalos(t_imu, sat_gyro, 0.1)
    int_sat_acc = agrupar_intervalos(t_imu, sat_acc, 0.1)
    imprimir_intervalos("saturación del giróscopo", int_sat_gyro)
    imprimir_intervalos("saturación del acelerómetro", int_sat_acc)
    rachas = rachas_identicas(np.hstack([w, a]), args.racha_imu)
    print(f"  rachas de ≥ {args.racha_imu} muestras idénticas: {len(rachas)}")
    for i, n in rachas[:12]:
        print(f"      t = {t_imu[i]:7.2f} s   {n} muestras ({n * dt_med_imu * 1000:.0f} ms)")
    # Al principio (quieto, si arrancas quieto) |a| debe ser ~g: si no, el escalado está mal
    quieto = t_imu < t_imu[0] + args.segundos_quieto
    print(f"  |a| medio en los primeros {args.segundos_quieto:.0f} s: "
          f"{np.mean(np.linalg.norm(a[quieto], axis=1)):.3f} m/s² (debe ser ≈ {G:.2f} si estabas quieto)")

    # --- Imágenes ---
    img = None
    if not args.sin_imagenes:
        print("\nAnalizando imágenes...")
        img = analizar_imagenes(mav0 / "cam0" / "data", nombres, args.max_esquinas, args.calidad_esquinas,
                                args.dist_esquinas, args.klt_ventana, args.klt_niveles, args.klt_ida_vuelta_px)
        print(f"  ilegibles: {len(img['ilegibles'])}")
        repetidos = img["diferencia"] == 0
        imprimir_intervalos("frames idénticos al anterior (cámara congelada)",
                            agrupar_intervalos(t_cam, repetidos, 0.2))
        imprimir_intervalos(f"pocos puntos seguidos (< {args.min_seguidos})",
                            agrupar_intervalos(t_cam, img["seguidos"] < args.min_seguidos, 0.2))

    # --- MSCKF ---
    sin_correccion = []
    if rec is not None:
        print("\nMSCKF:")
        for clave, nombre in [("usadas", "pistas usadas"), ("chi2", "rechazadas por χ²"),
                              ("reproyeccion", "sin triangular: reproyección"),
                              ("sin_paralaje", "sin triangular: paralaje"), ("quieto", "imágenes con ZUPT")]:
            print(f"  {nombre:30s} {int(rec[clave].sum()):7d}")
        sin_correccion = tramos_sin_correccion(rec["t_s"], rec["usadas"], args.sin_correccion_s)
        print(f"  tramos de ≥ {args.sin_correccion_s} s sin NINGUNA pista usada (solo IMU): {len(sin_correccion)}")
        for t_ini, t_fin in sin_correccion[:12]:
            print(f"      t = {t_ini:7.2f} – {t_fin:7.2f} s ({t_fin - t_ini:5.1f} s)")

    # --- Divergencia ---
    distancia = np.linalg.norm(p - p[0], axis=1)
    velocidad = velocidad_en_ventana(t_tray, p, args.ventana_vel)
    t_vel = primer_sostenido(t_tray, velocidad > args.vel_max, args.duracion_vel_s)
    t_dist, t_inicio = inicio_divergencia(t_tray, distancia, args.dist_max, args.ventana_atras_s,
                                          args.tolerancia_minimo_m)
    print("\nTrayectoria:")
    print(f"  v > {args.vel_max} m/s durante ≥ {args.duracion_vel_s} s seguidos desde: "
          f"{'—' if t_vel is None else f'{t_vel:.2f} s'}")
    print(f"  primera vez a más de {args.dist_max} m del origen: {'—' if t_dist is None else f'{t_dist:.2f} s'}")
    if t_inicio is not None:
        print(f"  último mínimo de la distancia antes (empieza a crecer): {t_inicio:.2f} s")
    saltos = np.flatnonzero(np.linalg.norm(np.diff(p, axis=0), axis=1) > args.salto_m)
    for i in saltos[:12]:
        texto = (f"  salto de {np.linalg.norm(p[i + 1] - p[i]):6.2f} m entre poses seguidas"
                 f" en t = {t_tray[i + 1]:.2f} s")
        if rec is not None:
            # ¿En esa imagen hubo ZUPT? Tras mucho tiempo solo con IMU, p y v están muy
            # correladas en P y un "v = 0" corrige también la posición decenas de metros
            k = int(np.argmin(np.abs(rec["t_s"] - t_tray[i + 1])))
            texto += f"  (ZUPT en esa imagen: {'sí' if rec['quieto'][k] else 'no'},"
            texto += f" pistas usadas: {int(rec['usadas'][k])})"
        print(texto)

    # Inicio estimado: si hay recuentos, el comienzo del ÚLTIMO tramo sin correcciones que
    # empieza antes de t_dist y llega más allá del mínimo de la distancia (es la prueba directa
    # de que el filtro dejó de ver); si no, el mínimo de la distancia; si no, la velocidad sostenida
    t_div = t_inicio if t_inicio is not None else t_vel
    if t_inicio is not None:
        candidatos = [t_ini for t_ini, t_fin in sin_correccion if t_ini <= t_dist and t_fin >= t_inicio]
        if candidatos:
            t_div = max(candidatos)
            print(f"  inicio estimado: {t_div:.2f} s (empieza el tramo sin correcciones visuales)")
    if t_div is not None:
        t_fin_tabla = (t_dist if t_dist is not None else t_div) + args.segundos_despues
        tabla_por_segundo(t_div, t_div - args.segundos_antes, t_fin_tabla, t_cam, t_imu, w, a,
                          img, rec, t_tray, distancia)
    else:
        print("  la trayectoria no supera los umbrales: no hay divergencia que analizar")

    ruta_png = carpeta / "diagnostico.png"
    dibujar(ruta_png, f"{carpeta.name} — diagnóstico", t_tray, distancia, velocidad,
            t_cam, dt_cam, t_imu, dt_imu, w, a, gyro_max, acc_max, img, rec, t_div, t_dist,
            agrupar_intervalos(t_cam[1:], dt_cam > args.factor_hueco * dt_med_cam, 0.2),
            agrupar_intervalos(t_imu[1:], dt_imu > args.factor_hueco * dt_med_imu, 0.2),
            int_sat_gyro + int_sat_acc, sin_correccion)
    print(f"\nFigura: {ruta_png}")


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico de la divergencia del VIO en una grabación")
    parser.add_argument("carpeta", help="carpeta datos/<grabación> con mav0/ y vio_trayectoria*.csv")
    # Fondo de escala del MPU-6050: ±250 °/s y ±2 g son los valores por defecto del chip;
    # pon los que configure el grabador (FS_SEL / AFS_SEL)
    parser.add_argument("--gyro-max-dps", type=float, default=250.0)
    parser.add_argument("--acc-max-g", type=float, default=2.0)
    parser.add_argument("--frac-saturacion", type=float, default=0.98,
                        help="fracción del fondo de escala a partir de la cual se considera saturado")
    parser.add_argument("--factor-hueco", type=float, default=2.5, help="hueco = Δt > factor × Δt mediano")
    parser.add_argument("--racha-imu", type=int, default=5, help="nº mínimo de muestras IMU idénticas seguidas")
    parser.add_argument("--segundos-quieto", type=float, default=2.0)
    parser.add_argument("--vel-max", type=float, default=1.0, help="velocidad imposible en tu habitación [m/s]")
    parser.add_argument("--ventana-vel", type=float, default=0.5, help="ventana para la velocidad media [s]")
    parser.add_argument("--duracion-vel-s", type=float, default=2.0,
                        help="la velocidad imposible debe mantenerse al menos esto [s]")
    parser.add_argument("--dist-max", type=float, default=3.0, help="distancia imposible en tu habitación [m]")
    parser.add_argument("--ventana-atras-s", type=float, default=15.0,
                        help="desde la 1.ª vez a más de --dist-max, cuánto mirar atrás buscando el mínimo [s]")
    parser.add_argument("--tolerancia-minimo-m", type=float, default=0.2,
                        help="al buscar hacia atrás el mínimo de la distancia, subida que se tolera [m]")
    parser.add_argument("--sin-correccion-s", type=float, default=1.0,
                        help="tramo mínimo sin pistas usadas que se reporta [s]")
    parser.add_argument("--salto-m", type=float, default=1.0, help="salto entre poses seguidas a reportar [m]")
    parser.add_argument("--segundos-antes", type=int, default=6, help="filas de la tabla antes del inicio")
    parser.add_argument("--segundos-despues", type=int, default=2, help="filas después de pasar --dist-max")
    parser.add_argument("--sin-imagenes", action="store_true", help="no analiza los PNG (mucho más rápido)")
    parser.add_argument("--min-seguidos", type=int, default=30)
    parser.add_argument("--max-esquinas", type=int, default=300)
    parser.add_argument("--calidad-esquinas", type=float, default=0.01)
    parser.add_argument("--dist-esquinas", type=float, default=10.0)
    parser.add_argument("--klt-ventana", type=int, default=21)
    parser.add_argument("--klt-niveles", type=int, default=3)
    parser.add_argument("--klt-ida-vuelta-px", type=float, default=1.0)
    args = parser.parse_args()
    diagnostico(args.carpeta, args)


if __name__ == "__main__":
    main()