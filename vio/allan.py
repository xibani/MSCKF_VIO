"""
vio/allan.py — Varianza de Allan de una señal de IMU en reposo (etapa 7.3).

Idea: se integra la señal (giroscopio -> ángulo, acelerómetro -> velocidad), se trocea en
"clusters" de duración τ y se mira cuánto cambia la media de un cluster al siguiente.
Cada tipo de ruido deja una pendiente distinta en la curva σ(τ) en escala log-log:

    ruido blanco (densidad N):          σ(τ) = N / √τ          pendiente -1/2   -> N = σ(τ = 1 s)
    inestabilidad del sesgo (B):        σ(τ) ≈ 0,664 · B       pendiente  0     (el mínimo)
    paseo aleatorio del sesgo (K):      σ(τ) = K · √(τ / 3)    pendiente +1/2   -> K = σ(τ = 3 s)

Son los mismos N y K que usan el simulador (vio/simulador.py: σ = N·√f por muestra y
b <- b + K·√Δt·ξ), el MSCKF (config/msckf.yaml) y Kalibr (noise_density, random_walk).
"""

import numpy as np
import yaml

FACTOR_INESTABILIDAD = 0.664    # mínimo de σ(τ) = 0,664 · B con ruido 1/f (IEEE Std 952)


def tamanos_cluster(n, puntos_por_decada, fraccion_max):
    """Tamaños de cluster m (en muestras), repartidos en escala logarítmica.

    n: muestras de la grabación. fraccion_max: el más largo es fraccion_max · n
    (más largo, quedan muy pocos clusters y σ sale con mucho error).
    Devuelve: array de enteros crecientes, empezando en 1.
    """
    m_max = int(n * fraccion_max)
    if m_max < 2:
        raise ValueError(f"grabación demasiado corta: {n} muestras")
    decadas = np.log10(m_max)
    candidatos = np.logspace(0.0, decadas, int(decadas * puntos_por_decada) + 1)
    return np.unique(np.round(candidatos).astype(int))


def desviacion_allan(x, dt, m):
    """Desviación de Allan con solapamiento (IEEE Std 952-1997).

    x: array (n,) muestras de UN eje (velocidad angular [rad/s] o aceleración [m/s²]).
    dt: periodo de muestreo [s]. m: array de tamaños de cluster (tamanos_cluster).
    Devuelve: (tau (len(m),) [s], sigma (len(m),) [unidad de x]).

        θ_k  = dt · (x_0 + ... + x_{k-1})          (integral: ángulo o velocidad)
        σ²(τ) = media_k[ (θ_{k+2m} - 2·θ_{k+m} + θ_k)² ] / (2·τ²),   τ = m·dt

    (θ_{k+2m} - θ_{k+m}) / τ y (θ_{k+m} - θ_k) / τ son las medias de dos clusters seguidos:
    σ² es la mitad de la varianza de su diferencia.
    """
    theta = np.concatenate([[0.0], np.cumsum(x) * dt])          # n + 1 valores
    tau = np.asarray(m, dtype=float) * dt
    sigma = np.zeros(len(m))
    for i, mi in enumerate(m):
        n_terminos = len(theta) - 2 * mi
        diferencia = theta[2 * mi:] - 2.0 * theta[mi:mi + n_terminos] + theta[:n_terminos]
        sigma[i] = np.sqrt(np.mean(diferencia ** 2) / (2.0 * tau[i] ** 2))
    return tau, sigma


def pendiente_local(tau, sigma):
    """Pendiente de la curva en log-log en cada punto, suavizada con sus dos vecinos."""
    p = np.gradient(np.log(sigma), np.log(tau))
    suavizada = p.copy()
    suavizada[1:-1] = (p[:-2] + p[1:-1] + p[2:]) / 3.0
    return suavizada


def ruido_blanco(tau, sigma, tau_ini, tau_fin):
    """Densidad del ruido blanco N: recta de pendiente -1/2 ajustada en [tau_ini, tau_fin].

    log σ = log N - ½·log τ  ->  log N = media(log σ + ½·log τ). Es el valor de la recta en τ = 1 s.
    """
    dentro = (tau >= tau_ini) & (tau <= tau_fin)
    if not dentro.any():
        raise ValueError(f"no hay puntos de la curva entre {tau_ini} y {tau_fin} s")
    return float(np.exp(np.mean(np.log(sigma[dentro]) + 0.5 * np.log(tau[dentro]))))


def paseo_sesgo(tau, sigma, pendiente_min):
    """Paseo aleatorio del sesgo K: recta de pendiente +1/2 en el tramo que sube tras el mínimo.

    log σ = log K + ½·log(τ/3)  ->  log K = media(log σ - ½·log(τ/3)). Es el valor en τ = 3 s.
    Devuelve: (K, observado). Si la grabación es corta y la curva no llega a subir,
    observado = False y K es una COTA SUPERIOR: como σ²(τ) ≥ K²·τ/3 en todo τ, el último
    punto da K ≤ σ(τ_max)·√(3/τ_max). Para el filtro, pasarse es el lado seguro.
    """
    p = pendiente_local(tau, sigma)
    indice = np.arange(len(tau))
    tramo = (indice > np.argmin(sigma)) & (p >= pendiente_min)
    if tramo.sum() >= 2:
        K = np.exp(np.mean(np.log(sigma[tramo]) - 0.5 * np.log(tau[tramo] / 3.0)))
        return float(K), True
    return float(sigma[-1] * np.sqrt(3.0 / tau[-1])), False


def perturbacion_periodica(tau, sigma, tau_desde, caida):
    """¿Sube la curva y vuelve a bajar? Es la firma de una perturbación periódica.

    Un paseo aleatorio solo hace SUBIR σ(τ). Una oscilación de periodo T0 (p. ej. la
    temperatura cuando el ventilador se enciende y se apaga) da una joroba: para una
    sinusoide σ(τ) ∝ sin²(π·τ/T0) / (π·τ/T0), con el máximo en τ ≈ 0,37·T0, ceros en
    τ = k·T0 y caída como 1/τ.
    Para cada punto j (con τ ≥ tau_desde) se mira cuántas veces es más alto que lo más bajo
    ANTES de él (subida) y que lo más bajo DESPUÉS (bajada). Hay joroba si algún punto es
    `caida` veces más alto por los dos lados. (No vale buscar solo el máximo: con mucho ruido
    blanco, σ(1 s) puede ser mayor que la joroba.)
    Devuelve: True si hay joroba.
    """
    zona = tau >= tau_desde
    s = sigma[zona]
    if len(s) < 3:
        return False
    minimo_hasta = np.minimum.accumulate(s)                 # minimo_hasta[j] = min(s[0..j])
    minimo_desde = np.minimum.accumulate(s[::-1])[::-1]     # minimo_desde[j] = min(s[j..fin])
    subida = s[1:-1] / minimo_hasta[:-2]                    # punto j = 1..n-2 frente a lo anterior
    bajada = s[1:-1] / minimo_desde[2:]                     # y frente a lo posterior
    return bool(np.max(np.minimum(subida, bajada)) >= caida)


def periodo_dominante(t_s, x, bloque_s, periodo_min_s):
    """Periodo de la oscilación más fuerte de la señal [s], para saber de dónde viene la joroba.

    t_s: array (n,) instantes REALES de cada muestra [s] (si faltan muestras, contar muestras
    en vez de segundos encogería el tiempo y el periodo). x: array (n,) la señal.
    (De la curva de Allan el periodo sale mal: el lóbulo principal puede quedar tapado por el
    ruido blanco. En el espectro, una oscilación es una raya.)
    1. Medias por bloques de bloque_s segundos: quitan casi todo el ruido blanco. Un bloque
       sin muestras se rellena interpolando entre sus vecinos.
    2. Fuera la recta de tendencia (una deriva lenta no es una oscilación).
    3. Pico del espectro entre periodo_min_s y la mitad de la duración.
    """
    bloque = np.floor((t_s - t_s[0]) / bloque_s).astype(int)
    n_bloques = int(bloque[-1]) + 1
    suma = np.bincount(bloque, weights=x, minlength=n_bloques)
    cuenta = np.bincount(bloque, minlength=n_bloques)
    centros = (np.arange(n_bloques) + 0.5) * bloque_s
    llenos = cuenta > 0
    medias = np.interp(centros, centros[llenos], suma[llenos] / cuenta[llenos])
    medias = medias - np.polyval(np.polyfit(centros, medias, 1), centros)
    potencia = np.abs(np.fft.rfft(medias)) ** 2
    frecuencias = np.fft.rfftfreq(n_bloques, d=bloque_s)
    duracion = n_bloques * bloque_s
    validas = np.flatnonzero((frecuencias >= 2.0 / duracion) & (frecuencias <= 1.0 / periodo_min_s))
    k = validas[np.argmax(potencia[validas])]
    return float(1.0 / frecuencias[k])


def inestabilidad_sesgo(tau, sigma):
    """Inestabilidad del sesgo B (solo informativa: el filtro no la usa) y dónde ocurre.

    Devuelve: (B, tau_del_minimo).
    """
    i = int(np.argmin(sigma))
    return float(sigma[i] / FACTOR_INESTABILIDAD), float(tau[i])


def analizar_eje(x, dt, cfg, t_s=None):
    """Curva de Allan y parámetros de un eje.

    cfg: sección "allan" de config/calibracion.yaml.
    t_s: instantes reales de las muestras [s], solo para el periodo de una perturbación
        (si no se dan, se suponen equiespaciadas: t = k·dt).
    Devuelve: dict con tau, sigma, N, K, K_observado, periodica, periodo_s, B y tau_B.
    """
    m = tamanos_cluster(len(x), cfg["puntos_por_decada"], cfg["fraccion_tau_max"])
    tau, sigma = desviacion_allan(x, dt, m)
    tau_ini, tau_fin = cfg["tau_ruido_blanco_s"]
    N = ruido_blanco(tau, sigma, tau_ini, tau_fin)
    periodica = perturbacion_periodica(tau, sigma, tau_fin, cfg["caida_joroba"])
    if periodica:
        # Lo que sube es la oscilación, no un paseo: K no se puede medir. Solo queda la cota
        # del último punto (σ² ≥ K²·τ/3 sigue valiendo con la oscilación encima)
        K, observado = float(sigma[-1] * np.sqrt(3.0 / tau[-1])), False
        if t_s is None:
            t_s = np.arange(len(x)) * dt
        periodo = periodo_dominante(t_s, x, cfg["bloque_periodo_s"], cfg["periodo_min_s"])
    else:
        K, observado = paseo_sesgo(tau, sigma, cfg["pendiente_paseo_min"])
        periodo = None
    B, tau_B = inestabilidad_sesgo(tau, sigma)
    return {"tau": tau, "sigma": sigma, "N": N, "K": K, "K_observado": observado,
            "periodica": periodica, "periodo_s": periodo, "B": B, "tau_B": tau_B}


def guardar_ruido_imu(ruta, adoptado, por_eje, info):
    """Guarda el resultado en YAML (config/imu_ruido.yaml).

    adoptado: dict con ruido_gir, ruido_acc, paseo_sesgo_gir, paseo_sesgo_acc (medidos, sin inflar).
    por_eje: dict con los valores de cada eje, para consultarlos. info: fecha, grabación...
    """
    datos = {clave: float(valor) for clave, valor in adoptado.items()}
    datos["por_eje"] = por_eje
    datos["info"] = info
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# Ruido de la IMU por varianza de Allan: generado por herramientas/analizar_allan.py\n"
                "# Valores MEDIDOS en reposo (sin inflar). Unidades: N [unidad/√Hz], K [unidad/s/√Hz]\n")
        yaml.safe_dump(datos, f, allow_unicode=True, sort_keys=False)


def cargar_ruido_imu(ruta):
    """Lee config/imu_ruido.yaml. Devuelve: dict con las 4 claves adoptadas."""
    with open(ruta, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return {clave: d[clave] for clave in ("ruido_gir", "ruido_acc", "paseo_sesgo_gir", "paseo_sesgo_acc")}
