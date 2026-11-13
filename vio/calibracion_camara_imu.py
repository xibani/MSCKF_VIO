"""
vio/calibracion_camara_imu.py — Giro cámara-IMU (R_i_c) y desfase de tiempo (t_d) con el giroscopio.

Idea (etapa 7.4). Cámara e IMU van atornilladas a la misma placa, así que giran juntas:
la velocidad de giro es LA MISMA, solo que cada sensor la ve en sus propios ejes,
    ω_i(t) = R_i_c · ω_c(t) + b_g                         (1)
donde ω_i es lo que mide el giroscopio (ejes de la IMU), ω_c la velocidad de giro en ejes
de la cámara, R_i_c el giro buscado y b_g el sesgo del giroscopio.

    - ω_c sale de las imágenes: con un tablero de ajedrez quieto delante, cada imagen da
      la orientación de la cámara respecto al tablero (solvePnP); entre dos imágenes
      seguidas, el giro dividido por el tiempo es la velocidad media de giro.
    - ω_i la da el giroscopio, pero con su reloj. La imagen sellada en t_cam corresponde
      en realidad al instante t_cam + t_d del reloj de la IMU (convención de Kalibr).

Se busca el t_d con el que (1) cuadra mejor: para cada t_d candidato, R_i_c y b_g salen
con una fórmula cerrada (Kabsch, más abajo) y queda un residuo; el t_d bueno es el del
residuo mínimo. La posición de la cámara (p_i_c) NO se calcula: con giros apenas influye
y se queda la medida con regla.
"""

import cv2
import numpy as np
import yaml

from vio import so3


# =====================================================================
# 1. Orientación de la cámara respecto al tablero, en cada imagen
# =====================================================================
def postura_tablero(esquinas, puntos_3d, K, dist):
    """Postura del tablero vista desde la cámara (solvePnP) y su error de reproyección.

    esquinas: array (N, 2) [px] de calibracion_camara.detectar_esquinas.
    puntos_3d: array (N, 3) [m] de calibracion_camara.puntos_tablero (mismo orden).
    Devuelve: (R_c_b (3, 3), t_c_b (3,) [m], error [px]) donde p_c = R_c_b·p_b + t_c_b
        (b = ejes del tablero). error = raíz del error cuadrático medio de las esquinas.
    """
    ok, rvec, tvec = cv2.solvePnP(puntos_3d.astype(np.float64), esquinas.astype(np.float64),
                                  K, np.asarray(dist, dtype=np.float64), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, np.inf
    R_c_b = cv2.Rodrigues(rvec)[0]
    reproyectadas, _ = cv2.projectPoints(puntos_3d.astype(np.float64), rvec, tvec, K,
                                         np.asarray(dist, dtype=np.float64))
    diferencias = reproyectadas.reshape(-1, 2) - esquinas
    error = float(np.sqrt((diferencias ** 2).sum(axis=1).mean()))
    return R_c_b, tvec.ravel(), error


# =====================================================================
# 2. Velocidad de giro de la cámara entre imágenes seguidas
# =====================================================================
def velocidades_camara(t_s, R_c_b, hueco_max_s, velocidad_max_rad_s):
    """Velocidad media de giro en ejes de la cámara entre cada par de imágenes seguidas.

    Con el tablero quieto, el giro de la cámara entre las imágenes k y k+1, visto en ejes
    de la cámara k, es
        R_ck_ck1 = R_ck_b · R_b_ck1 = R_c_b[k] · R_c_b[k+1]ᵀ = exp(ω_c · Δt)
    luego ω_c = log(R_ck_ck1) / Δt (velocidad media en el intervalo, si el eje cambia poco).

    t_s: array (M,) instantes de las imágenes válidas [s], crecientes.
    R_c_b: lista de M matrices (3, 3).
    hueco_max_s: si dos imágenes válidas están más separadas (imágenes descartadas entre
        medias), ese intervalo no se usa: la velocidad media sería poco representativa.
    velocidad_max_rad_s: intervalos más rápidos se descartan (el tablero detectado al revés
        en una imagen da un salto falso de 180°).
    Devuelve: (t0 (L,), t1 (L,), w_c (L, 3) [rad/s]) inicio, fin y velocidad de cada intervalo.
    """
    t0, t1, w_c = [], [], []
    for k in range(len(t_s) - 1):
        dt = t_s[k + 1] - t_s[k]
        if dt <= 0.0 or dt > hueco_max_s:
            continue
        w = so3.log_so3(R_c_b[k] @ R_c_b[k + 1].T) / dt
        if np.linalg.norm(w) > velocidad_max_rad_s:
            continue
        t0.append(t_s[k])
        t1.append(t_s[k + 1])
        w_c.append(w)
    return np.array(t0), np.array(t1), np.array(w_c).reshape(-1, 3)


# =====================================================================
# 3. Velocidad media del giroscopio en un intervalo cualquiera
# =====================================================================
def integral_giroscopio(t_s, w):
    """Integral acumulada del giroscopio (regla del trapecio), eje a eje.

    Ω(t) = ∫ ω dt desde la primera muestra. Con ella, la velocidad media entre ta y tb es
    (Ω(tb) - Ω(ta)) / (tb - ta), para cualquier ta, tb (interpolando): así el giroscopio
    se promedia en EXACTAMENTE el mismo intervalo que la cámara, aunque esté desplazado t_d.
    t_s: array (N,) [s]; w: array (N, 3) [rad/s]. Devuelve: array (N, 3) [rad].
    """
    Omega = np.zeros_like(w)
    for k in range(1, len(t_s)):
        Omega[k] = Omega[k - 1] + 0.5 * (w[k] + w[k - 1]) * (t_s[k] - t_s[k - 1])
    return Omega


def velocidad_media_giroscopio(t_s, Omega, ta, tb):
    """Velocidad media del giroscopio entre ta y tb (arrays (L,)). Devuelve: array (L, 3)."""
    medias = np.zeros((len(ta), 3))
    for eje in range(3):
        diferencia = np.interp(tb, t_s, Omega[:, eje]) - np.interp(ta, t_s, Omega[:, eje])
        medias[:, eje] = diferencia / (tb - ta)
    return medias


# =====================================================================
# 4. Kabsch: el giro (y el sesgo) que mejor lleva unos vectores a otros
# =====================================================================
def alinear_giros(w_c, w_i):
    """R_i_c y b_g que minimizan Σ |w_i - R_i_c·w_c - b_g|²  (ecuación (1) del principio).

    1. El sesgo es una constante: con R fija, el mejor b_g es b_g = media(w_i) - R·media(w_c).
       Sustituyéndolo, el problema queda con los vectores centrados (menos su media):
           Σ |ŵ_i - R·ŵ_c|²   con   ŵ = w - media(w)
    2. Kabsch: con H = Σ ŵ_c·ŵ_iᵀ = U·S·Vᵀ (SVD), la R que lo minimiza es
           R = V · diag(1, 1, d) · Uᵀ,   d = signo(det(V·Uᵀ))
       (d evita que salga un espejo en vez de un giro).
    w_c, w_i: arrays (L, 3) [rad/s]. Devuelve: (R_i_c (3, 3), b_g (3,), residuos (L, 3)).
    """
    media_c = w_c.mean(axis=0)
    media_i = w_i.mean(axis=0)
    centrados_c = w_c - media_c
    centrados_i = w_i - media_i
    H = centrados_c.T @ centrados_i
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R_i_c = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    b_g = media_i - R_i_c @ media_c
    residuos = w_i - (w_c @ R_i_c.T + b_g)
    return R_i_c, b_g, residuos


def rms(residuos):
    """Raíz del error cuadrático medio de unos residuos (L, 3), en la unidad de entrada."""
    return float(np.sqrt((residuos ** 2).sum(axis=1).mean()))


# =====================================================================
# 5. Búsqueda del desfase t_d
# =====================================================================
def coste_desfase(t_d_s, t0, t1, w_c, t_imu_s, Omega):
    """Residuo RMS de (1) si la cámara va t_d por detrás del reloj de la IMU.

    Imagen sellada en t  <->  instante t + t_d de la IMU: el giroscopio se promedia en
    [t0 + t_d, t1 + t_d]. Los intervalos que caen fuera de la grabación de la IMU no cuentan.
    Devuelve: (rms [rad/s], R_i_c, b_g, residuos, dentro (L,) bool).
    """
    ta, tb = t0 + t_d_s, t1 + t_d_s
    dentro = (ta >= t_imu_s[0]) & (tb <= t_imu_s[-1])
    w_i = velocidad_media_giroscopio(t_imu_s, Omega, ta[dentro], tb[dentro])
    R_i_c, b_g, residuos = alinear_giros(w_c[dentro], w_i)
    return rms(residuos), R_i_c, b_g, residuos, dentro


def minimo_parabola(x, y):
    """Vértice de la parábola por 3 puntos equiespaciados alrededor del menor. x, y: (3,)."""
    denominador = y[0] - 2.0 * y[1] + y[2]
    if denominador <= 0.0:
        return x[1]                                  # no es un mínimo: el del centro
    return x[1] + 0.5 * (x[1] - x[0]) * (y[0] - y[2]) / denominador


def buscar_desfase(t0, t1, w_c, t_imu_s, Omega, cfg):
    """Barrido del t_d: grueso en ±desfase_max_ms, fino alrededor del mejor y parábola.

    cfg: sección "camara_imu" de config/calibracion.yaml.
    Devuelve: (t_d [s], curva) con curva = (t_d de prueba [s] (P,), coste [rad/s] (P,)) del
        barrido grueso, para dibujarla.
    """
    # 1. Barrido grueso
    maximo = cfg["desfase_max_ms"] / 1000.0
    paso = cfg["paso_grueso_ms"] / 1000.0
    candidatos = np.arange(-maximo, maximo + paso / 2, paso)
    costes = np.array([coste_desfase(t, t0, t1, w_c, t_imu_s, Omega)[0] for t in candidatos])
    mejor = candidatos[int(np.argmin(costes))]

    # 2. Barrido fino alrededor del mejor
    paso_fino = cfg["paso_fino_ms"] / 1000.0
    margen = cfg["margen_fino_ms"] / 1000.0
    finos = np.arange(mejor - margen, mejor + margen + paso_fino / 2, paso_fino)
    costes_finos = np.array([coste_desfase(t, t0, t1, w_c, t_imu_s, Omega)[0] for t in finos])
    k = int(np.argmin(costes_finos))

    # 3. Parábola por el mínimo y sus dos vecinos (resolución por debajo del paso fino)
    if 0 < k < len(finos) - 1:
        t_d = minimo_parabola(finos[k - 1:k + 2], costes_finos[k - 1:k + 2])
    else:
        t_d = finos[k]                               # en el borde: sin refinar
    return float(t_d), (candidatos, costes)


# =====================================================================
# 6. Todo junto: de las posturas y el giroscopio al resultado
# =====================================================================
def excitacion(w_c):
    """Desviación típica de la velocidad de giro de la cámara en cada eje [rad/s] (3,).

    Para fijar R_i_c hace falta girar alrededor de los TRES ejes: si un eje casi no gira,
    el giro alrededor de él queda mal determinado.
    """
    return w_c.std(axis=0)


def calibrar_giros(t_cam_s, R_c_b, t_imu_s, w_imu, cfg):
    """R_i_c, b_g y t_d a partir de las posturas del tablero y del giroscopio.

    t_cam_s: (M,) instantes de las imágenes válidas [s] (mismo origen que t_imu_s).
    R_c_b: lista de M orientaciones del tablero en ejes de la cámara.
    t_imu_s: (N,) [s]; w_imu: (N, 3) [rad/s] del giroscopio.
    cfg: sección "camara_imu" de config/calibracion.yaml.
    Devuelve: dict con R_i_c, b_g, t_d_s, rms_rad_s, intervalos, descartados, excitacion
        (3,), curva, y para dibujar: t_medio, w_i (lo que midió el giroscopio) y w_c_en_i
        (la cámara pasada a ejes de la IMU con R_i_c y b_g).
    """
    t0, t1, w_c = velocidades_camara(t_cam_s, R_c_b, cfg["hueco_max_s"],
                                     np.radians(cfg["velocidad_max_dps"]))
    Omega = integral_giroscopio(t_imu_s, w_imu)

    # 1. Primera búsqueda con todos los intervalos
    t_d, _ = buscar_desfase(t0, t1, w_c, t_imu_s, Omega, cfg)

    # 2. Fuera los intervalos que no cuadran (tablero mal detectado, imagen movida...)
    _, _, _, residuos, dentro = coste_desfase(t_d, t0, t1, w_c, t_imu_s, Omega)
    normas = np.linalg.norm(residuos, axis=1)
    buenos_dentro = normas <= cfg["factor_descarte"] * np.median(normas)
    indices = np.flatnonzero(dentro)[buenos_dentro]
    descartados = len(t0) - len(indices)
    t0, t1, w_c = t0[indices], t1[indices], w_c[indices]

    # 3. Búsqueda definitiva solo con los buenos
    t_d, curva = buscar_desfase(t0, t1, w_c, t_imu_s, Omega, cfg)
    error, R_i_c, b_g, residuos, dentro = coste_desfase(t_d, t0, t1, w_c, t_imu_s, Omega)

    w_i = velocidad_media_giroscopio(t_imu_s, Omega, t0[dentro] + t_d, t1[dentro] + t_d)
    return {"R_i_c": R_i_c, "b_g": b_g, "t_d_s": t_d, "rms_rad_s": error,
            "intervalos": int(dentro.sum()), "descartados": int(descartados),
            "excitacion": excitacion(w_c[dentro]), "curva": curva,
            "t_medio": 0.5 * (t0[dentro] + t1[dentro]), "w_i": w_i,
            "w_c_en_i": w_c[dentro] @ R_i_c.T + b_g}


# =====================================================================
# 7. Varias grabaciones: media, dispersión y guardado
# =====================================================================
def combinar(resultados):
    """Une las calibraciones de varias grabaciones.

    R media: la rotación más cercana a la suma de las R (media "por cuerdas"; con R muy
    parecidas es prácticamente la media de verdad). t_d: la mediana (un mal resultado no
    la arrastra).
    Devuelve: dict con R_i_c, t_d_s, dispersion_R_rad (mayor ángulo entre una R y la media)
        y dispersion_td_s (desviación típica de los t_d; 0 con una sola grabación).
    """
    suma = np.zeros((3, 3))
    for r in resultados:
        suma += r["R_i_c"]
    R_media = so3.reortonormalizar(suma)
    t_d = np.array([r["t_d_s"] for r in resultados])
    return {"R_i_c": R_media,
            "t_d_s": float(np.median(t_d)),
            "dispersion_R_rad": max(so3.angulo_entre(R_media, r["R_i_c"]) for r in resultados),
            "dispersion_td_s": float(t_d.std(ddof=1)) if len(t_d) > 1 else 0.0}


def guardar_camara_imu(ruta, R_i_c, p_i_c_m, t_d_ns, info):
    """Escribe config/camara_imu.yaml con el mismo formato que la versión medida con regla."""
    datos = {"R_i_c": [[float(x) for x in fila] for fila in R_i_c],
             "p_i_c_m": [float(x) for x in p_i_c_m],
             "t_d_ns": int(t_d_ns),
             "info": info}
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# config/camara_imu.yaml — Extrínseca cámara-IMU (T_c_i) y desfase temporal (t_d).\n")
        f.write("# Generado por herramientas/calibrar_camara_imu.py (etapa 7.4): R_i_c y t_d calibrados\n")
        f.write("# con el giroscopio; p_i_c_m sigue siendo la medida con regla.\n")
        f.write("# Columnas de R_i_c = ejes x, y, z de la cámara expresados en ejes de la IMU.\n")
        f.write("# t_cámara_corregido = t_cámara + t_d  [ns]\n")
        yaml.safe_dump(datos, f, allow_unicode=True, sort_keys=False)