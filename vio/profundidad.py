"""
vio/profundidad.py — Etapa 25: la profundidad de TODOS los píxeles de un keyframe, en metros.

El VIO da la profundidad de unas decenas de puntos por keyframe (etapa 24). Para reconstruir
una habitación hacen falta todos los píxeles. Se juntan dos fuentes que se complementan:

    - Depth Anything V2, una red que mira UNA imagen y dice, de cada píxel, si está más cerca
      o más lejos que los demás. Lo dice para todos los píxeles, pero sin unidades: su salida
      es una "profundidad inversa relativa" d (mayor = más cerca), que no son metros.
    - Los puntos del VIO: pocos, pero en metros.

La relación entre las dos es una recta: 1/z = a·d + b. Con los puntos del VIO se buscan a y
b (ajustar_a_metros) y con ellos se pasa a metros el mapa entero de la red (a_metros).

La red necesita PyTorch y la librería transformers, así que es para el PC. Como en la etapa
21, los imports van DENTRO de la clase: el resto del módulo (el ajuste, la evaluación)
funciona y se prueba también en la Pi.
"""

import time

import numpy as np

from vio import medidas_mapa
from vio.emparejador import elegir_dispositivo


# =====================================================================
# Paso 25.2 — La red, detrás de una clase sencilla
# =====================================================================
class EstimadorProfundidad:
    """Depth Anything V2 con entrada y salida en NumPy.

    Uso:
        estimador = EstimadorProfundidad(cargar_config("profundidad"))
        d = estimador.estimar(gris)["inversa"]      # (alto, ancho): mayor = más cerca
    """

    def __init__(self, cfg, procesador=None, red=None):
        """cfg: config/profundidad.yaml.

        procesador, red: solo para los tests, que pasan dos objetos falsos. Normalmente se
        dejan en None y se crean aquí (la primera vez se descargan los pesos, unos 100 MB:
        hace falta internet).
        """
        import torch                                   # aquí dentro: solo en el PC

        self.torch = torch
        self.dispositivo = elegir_dispositivo(cfg["dispositivo"], torch.cuda.is_available())
        if procesador is None or red is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            # El procesador prepara la imagen como la quiere la red (tamaño, brillo...)
            procesador = AutoImageProcessor.from_pretrained(cfg["modelo"])
            red = AutoModelForDepthEstimation.from_pretrained(cfg["modelo"])
            red = red.eval().to(self.dispositivo)      # modo "usar", y a la tarjeta o a la CPU
        self.procesador = procesador
        self.red = red

    def estimar(self, gris):
        """Depth Anything V2 sobre una imagen.

        gris: (alto, ancho) uint8.
        Devuelve: dict con
            inversa    (alto, ancho) float32: profundidad inversa relativa (mayor = más cerca,
                       sin unidades), al tamaño de la imagen original
            tiempo_ms  lo que ha tardado
        """
        if gris.ndim != 2 or gris.dtype != np.uint8:
            raise ValueError("estimar() espera una imagen en gris de 8 bits (alto, ancho)")
        t0 = time.perf_counter()
        torch = self.torch
        # La red se entrenó con fotos en color: se le da el gris repetido en los tres canales
        color = np.stack([gris, gris, gris], axis=-1)                   # (alto, ancho, 3)
        entrada = self.procesador(images=color, return_tensors="pt")["pixel_values"]
        with torch.no_grad():                                           # sin lo necesario para entrenar
            salida = self.red(pixel_values=entrada.to(self.dispositivo)).predicted_depth
        # salida: (1, alto', ancho'), al tamaño con el que ha trabajado la red, que no es el
        # de la imagen (el procesador la cambia de tamaño). Se lleva al original interpolando
        salida = torch.nn.functional.interpolate(salida[:, None], size=gris.shape,
                                                 mode="bicubic", align_corners=False)
        inversa = salida[0, 0].cpu().numpy().astype(np.float32)
        return {"inversa": inversa, "tiempo_ms": 1e3 * (time.perf_counter() - t0)}


# =====================================================================
# Paso 25.3 — De la profundidad relativa a metros, con los puntos del VIO
# =====================================================================
def error_relativo(d, z, a, b, inversa_min):
    """Cuánto se aparta, de la profundidad z del VIO, la que sale de la recta 1/z = a·d + b.

    d, z: (N,). Devuelve: (N,) |z_recta - z| / z; infinito donde la recta da una profundidad
    inversa menor que inversa_min (un punto absurdamente lejano, o detrás de la cámara).
    """
    inversa = a * d + b
    error = np.full(len(d), np.inf)
    bien = inversa >= inversa_min
    error[bien] = np.abs(1.0 / inversa[bien] - z[bien]) / z[bien]
    return error


def ajustar_recta(d, z, cfg_ajuste, rng):
    """La recta 1/z = a·d + b, con RANSAC (una recta sale de 2 puntos) y, al final, mínimos
    cuadrados con los puntos que encajan. Devuelve: (a, b), o None si no sale una fiable."""
    n = len(d)
    umbral = cfg_ajuste["umbral_relativo"]
    inversa_min = cfg_ajuste["inversa_min"]
    inversa_vio = 1.0 / z

    # 1. RANSAC: probar rectas que pasan por 2 puntos y quedarse con la que más puntos cumplen
    mejor = None
    for _ in range(cfg_ajuste["iteraciones"]):
        i, j = rng.choice(n, 2, replace=False)
        if abs(d[i] - d[j]) < 1e-9:
            continue                                   # los dos con la misma d: no definen una recta
        a = (inversa_vio[i] - inversa_vio[j]) / (d[i] - d[j])
        b = inversa_vio[i] - a * d[i]
        if a <= 0:
            continue                                   # al revés que la red (mayor d = más cerca)
        encajan = error_relativo(d, z, a, b, inversa_min) < umbral
        if mejor is None or encajan.sum() > mejor.sum():
            mejor = encajan
    if mejor is None or mejor.sum() < 2:
        return None

    # 2. Mínimos cuadrados con los puntos que encajan: la recta que mejor pasa entre todos ellos.
    #    Cada punto da una ecuación a·d_k + b = 1/z_k; lstsq resuelve el sistema entero
    A = np.column_stack([d[mejor], np.ones(int(mejor.sum()))])
    (a, b), _, _, _ = np.linalg.lstsq(A, inversa_vio[mejor], rcond=None)
    if a <= 0:
        return None
    return float(a), float(b)


def ajustar_escala(d, z):
    """Solo la escala: 1/z = a·d, sin b. Devuelve: (a, 0.0), o None si la red da valores nulos.

    Con una sola incógnita basta un punto; se toma la mediana de lo que diría cada uno, que
    aguanta que una parte de los puntos esté mal.
    """
    util = d > 1e-9
    if not util.any():
        return None
    return float(np.median(1.0 / (z[util] * d[util]))), 0.0


def ajustar_a_metros(d, z, cfg_ajuste, rng):
    """Busca la relación 1/z = a·d + b que une la salida de la red con los metros del VIO.

    d: (N,) valor de la red en el píxel de cada punto del VIO.
    z: (N,) profundidad de cada punto según el VIO [m].
    cfg_ajuste: sección "ajuste" de config/profundidad.yaml. rng: np.random.default_rng.
    Una recta tiene dos incógnitas, y para separarlas hacen falta puntos a distancias
    distintas. Si están casi todos a la misma (la cámara mira de frente a una pared), o si
    la recta no sale fiable, se ajusta solo la escala (b = 0).
    Devuelve: None si nada encaja, o dict con a, b, solo_escala (bool), encajan (N,) bool y
    error_mediano (el error relativo mediano de los puntos que encajan).
    """
    if len(d) < cfg_ajuste["puntos_min"]:
        return None
    # 1. ¿Hay puntos a distancias lo bastante distintas? Se mira en 1/z, que es donde se ajusta
    p10, mediana, p90 = np.percentile(1.0 / z, [10, 50, 90])
    recta = None
    if (p90 - p10) / mediana >= cfg_ajuste["rango_min"]:
        recta = ajustar_recta(d, z, cfg_ajuste, rng)
    solo_escala = recta is None
    if solo_escala:
        recta = ajustar_escala(d, z)
    if recta is None:
        return None

    # 2. Sea cual sea, vale solo si encajan bastantes puntos
    a, b = recta
    error = error_relativo(d, z, a, b, cfg_ajuste["inversa_min"])
    encajan = error < cfg_ajuste["umbral_relativo"]
    if encajan.mean() < cfg_ajuste["encajan_min"]:
        return None
    return {"a": a, "b": b, "solo_escala": solo_escala, "encajan": encajan,
            "error_mediano": float(np.median(error[encajan]))}


def a_metros(inversa, a, b, inversa_min):
    """El mapa de la red, en metros: z = 1 / (a·d + b).

    inversa: (alto, ancho) salida de la red. Devuelve: (alto, ancho) float32 [m], con NaN
    donde a·d + b queda por debajo de inversa_min (demasiado lejos para fiarse).
    """
    inversa_m = a * inversa.astype(np.float64) + b
    z = np.full(inversa.shape, np.nan, dtype=np.float32)
    bien = inversa_m >= inversa_min
    z[bien] = (1.0 / inversa_m[bien]).astype(np.float32)
    return z


# =====================================================================
# Paso 25.4 — ¿Cuánto se parece a la verdad? (solo en el simulador)
# =====================================================================
def evaluar_profundidad(z, z_verdad, umbral_delta):
    """Compara un mapa de profundidad con el verdadero, píxel a píxel.

    z, z_verdad: (alto, ancho) [m], con NaN donde no hay dato.
    Devuelve: dict con
        n                    píxeles comparados (con dato en los dos)
        cubierto             fracción de los píxeles con verdad para los que hay estimación
        error_rel_mediano    mediana de |z - z_verdad| / z_verdad
        error_rel_medio      su media (en los artículos, "AbsRel")
        delta                fracción de píxeles con z/z_verdad entre 1/umbral_delta y umbral_delta
    """
    con_verdad = np.isfinite(z_verdad) & (z_verdad > 0)
    ambos = con_verdad & np.isfinite(z)
    n = int(ambos.sum())
    if n == 0:
        return {"n": 0, "cubierto": 0.0, "error_rel_mediano": np.nan, "error_rel_medio": np.nan,
                "delta": np.nan}
    razon = z[ambos].astype(np.float64) / z_verdad[ambos]
    error = np.abs(razon - 1.0)
    return {"n": n, "cubierto": n / int(con_verdad.sum()),
            "error_rel_mediano": float(np.median(error)), "error_rel_medio": float(error.mean()),
            "delta": float(np.mean(np.maximum(razon, 1.0 / razon) < umbral_delta))}


# =====================================================================
# Paso 25.5 — Las piezas juntas: la profundidad de un keyframe
# =====================================================================
def profundidad_de_keyframe(kf, gris, estimador, cfg, rng):
    """La profundidad en metros de todos los píxeles de un keyframe de mapeo.

    kf: dict de protocolo.leer_keyframe (trae los puntos del VIO: xy, z, K, dist).
    gris: su imagen (alto, ancho) uint8. estimador: algo con estimar(gris), como
    EstimadorProfundidad. cfg: config/profundidad.yaml.
    Devuelve: dict con
        z          (alto, ancho) float32 [m], NaN sin dato; None si no ha habido ajuste
        ajuste     lo que devuelve ajustar_a_metros (None si no ha habido)
        inversa    la salida de la red, sin tocar
        n_puntos   puntos del VIO que se han podido usar
        tiempo_ms  lo que ha tardado la red
    """
    red = estimador.estimar(gris)
    # El valor de la red en el píxel de cada punto del VIO
    d = medidas_mapa.muestrear(red["inversa"], medidas_mapa.pixeles_de_keyframe(kf))
    dentro = np.isfinite(d)                             # los que caen dentro de la imagen
    ajuste = ajustar_a_metros(d[dentro], kf["z"][dentro], cfg["ajuste"], rng)
    z = None
    if ajuste is not None:
        z = a_metros(red["inversa"], ajuste["a"], ajuste["b"], cfg["ajuste"]["inversa_min"])
    return {"z": z, "ajuste": ajuste, "inversa": red["inversa"], "n_puntos": int(dentro.sum()),
            "tiempo_ms": red["tiempo_ms"]}
