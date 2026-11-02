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


def analizar_cobertura(t_imu_ns, t_cam_ns):
    """¿Cubre la IMU todo el tramo de imágenes?

    Devuelve: dict con margen_ini_s (cuánto antes de la primera imagen empieza
    la IMU) y margen_fin_s (cuánto después de la última acaba). Ambos deben ser > 0.
    """
    return {"margen_ini_s": (t_cam_ns[0] - t_imu_ns[0]) / 1e9,
            "margen_fin_s": (t_imu_ns[-1] - t_cam_ns[-1]) / 1e9}


def analizar_reposo(t_ns, w, a, t_reposo_s):
    """Estadística del tramo inicial de IMU, que debe estar quieto.

    t_ns: array (N,) int64. w: array (N, 3) [rad/s]. a: array (N, 3) [m/s²].
    t_reposo_s: duración del tramo inicial que se analiza.
    Devuelve: dict con:
        n: muestras del tramo
        a_media: (3,) media del acelerómetro [m/s²]; apunta hacia arriba
        norma_a: |a_media| [m/s²]; debería rondar 9,8
        eje_arriba: texto como "+z", el eje más alineado con la vertical
        sesgo_gir_dps: (3,) media del giroscopio [°/s] = sesgo inicial
        ruido_gir_dps: (3,) desviación típica del giroscopio [°/s]
        ruido_acc: (3,) desviación típica del acelerómetro [m/s²]
    """
    en_reposo = (t_ns - t_ns[0]) / 1e9 < t_reposo_s
    a_r = a[en_reposo]
    w_r = np.degrees(w[en_reposo])

    a_media = a_r.mean(axis=0)
    eje = int(np.argmax(np.abs(a_media)))
    signo = "+" if a_media[eje] > 0 else "-"

    return {
        "n": int(en_reposo.sum()),
        "a_media": a_media,
        "norma_a": float(np.linalg.norm(a_media)),
        "eje_arriba": signo + "xyz"[eje],
        "sesgo_gir_dps": w_r.mean(axis=0),
        "ruido_gir_dps": w_r.std(axis=0),
        "ruido_acc": a_r.std(axis=0),
    }