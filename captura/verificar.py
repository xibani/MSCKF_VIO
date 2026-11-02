"""
captura/verificar.py — Análisis de la regularidad de las marcas de tiempo.

Funciones puras: reciben arrays y devuelven números. No leen archivos ni sensores.
"""

import numpy as np


def analizar_intervalos(t_ns, f_nominal_hz, factor_hueco):
    """Estadística de los intervalos Δt entre marcas de tiempo consecutivas.

    t_ns: array (N,) int64, marcas de tiempo [ns].
    f_nominal_hz: frecuencia esperada [Hz].
    factor_hueco: un Δt mayor que factor_hueco · T_nominal cuenta como hueco.
    Devuelve: dict con las métricas (los tiempos en ms salvo que se indique).
    """
    T = 1.0 / f_nominal_hz                         # periodo nominal [s]
    dt = np.diff(t_ns).astype(np.float64) / 1e9    # intervalos [s]
    duracion = (t_ns[-1] - t_ns[0]) / 1e9

    # Muestras perdidas: un Δt de unos k periodos implica k - 1 muestras ausentes
    perdidas_por_intervalo = np.maximum(np.round(dt / T) - 1, 0)
    perdidas = int(np.sum(perdidas_por_intervalo))

    # Jitter: desviación típica de Δt, solo con los intervalos normales
    # (si no, una sola pérdida lo inflaría y taparía cómo es el resto)
    normales = dt[dt <= factor_hueco * T]

    return {
        "n": len(t_ns),
        "duracion_s": duracion,
        "frecuencia_hz": (len(t_ns) - 1) / duracion,
        "dt_medio_ms": dt.mean() * 1e3,
        "jitter_ms": normales.std() * 1e3,
        "dt_min_ms": dt.min() * 1e3,
        "dt_max_ms": dt.max() * 1e3,
        "huecos": int(np.sum(dt > factor_hueco * T)),
        "perdidas": perdidas,
        "perdidas_pct": 100.0 * perdidas / (len(t_ns) + perdidas),
        "no_crecientes": int(np.sum(dt <= 0)),
        "dt_s": dt,
    }
    

def analizar_latencia(t_sensor_ns, t_llegada_ns):
    """Latencia de entrega: cuánto tarda cada imagen en llegar al programa.

    Si cámara y programa usan el mismo reloj, sale positiva y de decenas de
    ms. Valores negativos o enormes indican relojes distintos.
    Devuelve: dict con media_ms, min_ms y max_ms.
    """
    latencia_ms = (t_llegada_ns - t_sensor_ns).astype(np.float64) / 1e6
    return {"media_ms": latencia_ms.mean(),
            "min_ms": latencia_ms.min(),
            "max_ms": latencia_ms.max()}


def variacion_pct(valores):
    """(máximo - mínimo) / media, en %. 0 = todos los valores iguales."""
    v = np.asarray(valores, dtype=np.float64)
    return 100.0 * (v.max() - v.min()) / v.mean()