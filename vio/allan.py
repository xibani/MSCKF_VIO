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


def inestabilidad_sesgo(tau, sigma):
    """Inestabilidad del sesgo B (solo informativa: el filtro no la usa) y dónde ocurre.

    Devuelve: (B, tau_del_minimo).
    """
    i = int(np.argmin(sigma))
    return float(sigma[i] / FACTOR_INESTABILIDAD), float(tau[i])


def analizar_eje(x, dt, cfg):
    """Curva de Allan y parámetros de un eje.

    cfg: sección "allan" de config/calibracion.yaml.
    Devuelve: dict con tau, sigma, N, K, K_observado, B y tau_B.
    """
    m = tamanos_cluster(len(x), cfg["puntos_por_decada"], cfg["fraccion_tau_max"])
    tau, sigma = desviacion_allan(x, dt, m)
    tau_ini, tau_fin = cfg["tau_ruido_blanco_s"]
    N = ruido_blanco(tau, sigma, tau_ini, tau_fin)
    K, observado = paseo_sesgo(tau, sigma, cfg["pendiente_paseo_min"])
    B, tau_B = inestabilidad_sesgo(tau, sigma)
    return {"tau": tau, "sigma": sigma, "N": N, "K": K, "K_observado": observado,
            "B": B, "tau_B": tau_B}


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