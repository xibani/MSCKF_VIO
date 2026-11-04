"""
vio/vista.py — Lo que ve el VIO, para mostrarlo en el visor: la imagen con sus pistas.

Cada pista activa del front-end se dibuja con su "cola" (sus últimas posiciones), de rojo
(recién detectada) a verde (seguida muchas imágenes). Arriba, una línea de estado.
La imagen se comprime en JPEG para que quepa en un solo paquete UDP.
"""

import unicodedata
import cv2
import numpy as np


# Las fuentes de OpenCV solo tienen ASCII: estos símbolos se escriben con letras
SUSTITUCIONES_ASCII = {"σ": "sigma", "°": "deg", "²": "2", "¿": "", "¡": "", "→": "->"}
MAX_CARACTERES = 60           # lo que cabe en una línea de 640 px con la letra de dibujar_vista


def a_ascii(texto):
    """Texto que OpenCV puede dibujar: sin tildes ni símbolos, y como mucho MAX_CARACTERES."""
    for simbolo, letras in SUSTITUCIONES_ASCII.items():
        texto = texto.replace(simbolo, letras)
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return texto[:MAX_CARACTERES]


def color_por_longitud(longitud, longitud_larga):
    """Color BGR de una pista: rojo si es nueva, verde si ya es larga."""
    fraccion = min(longitud / longitud_larga, 1.0)
    return (0, int(255 * fraccion), int(255 * (1 - fraccion)))


def dibujar_vista(gris, frontend, lineas_texto, cfg_vista):
    """Imagen en color con las pistas activas y unas líneas de texto encima.

    gris: (alto, ancho) uint8. frontend: FrontEnd de la etapa 8, o None (sin pistas aún).
    lineas_texto: lista de str. cfg_vista: sección "vista" de config/vio.yaml.
    Devuelve: (alto, ancho, 3) uint8 en BGR (el orden de colores de OpenCV).
    """
    vista = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

    # 1. Cada pista: su cola y un punto en su posición actual
    if frontend is not None:
        for pista in frontend.activas.values():
            cola = np.array(pista.uv[-cfg_vista["cola_pista"]:], dtype=np.int32)
            color = color_por_longitud(pista.longitud(), cfg_vista["longitud_larga"])
            if len(cola) > 1:
                cv2.polylines(vista, [cola.reshape(-1, 1, 2)], False, color, 1)
            cv2.circle(vista, (int(cola[-1][0]), int(cola[-1][1])), 3, color, -1)

    # 2. Texto blanco sobre un rectángulo negro, para que se lea sobre cualquier imagen.
    #    (Un contorno negro con trazo grueso no vale: en algunas versiones de OpenCV el trazo
    #    grueso separa más las letras y el contorno no coincide con el texto)
    for k, linea in enumerate(lineas_texto):
        y = 20 + 22 * k                                   # línea base del texto
        linea = a_ascii(linea)
        (ancho, alto), bajo_linea = cv2.getTextSize(linea, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vista, (4, y - alto - 4), (8 + ancho + 4, y + bajo_linea), (0, 0, 0), -1)
        cv2.putText(vista, linea, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vista


def lineas_estado(pipeline, resultado):
    """Las líneas de texto que acompañan a la imagen.

    Mientras no hay pose, dicen por qué: cuántas muestras de IMU en reposo lleva la
    inicialización y por qué falló el último intento, o cuántas imágenes se descartan
    por llegar con una marca de tiempo anterior al filtro.
    """
    if pipeline.filtro is None:
        lineas = ["INICIALIZANDO: deja la placa QUIETA",
                  f"IMU: {len(pipeline.reposo)} muestras en reposo, intento {pipeline.intentos + 1}"]
        if pipeline.problemas:
            lineas.append("fallo: " + pipeline.problemas[0])
        return lineas
    if resultado is None:
        return ["VIO EN MARCHA (esperando imagen)",
                f"imagenes descartadas por tiempo: {pipeline.descartadas}"]
    p = resultado["p_w"]
    fe, est = resultado["frontend"], resultado["msckf"]
    estado = "QUIETO (ZUPT)" if est["quieto"] else "en movimiento"
    return [f"p = ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}) m   {estado}",
            f"puntos {fe['activos']}  nuevos {fe['nuevos']}  pistas usadas {est['usadas']}"
            f"  {resultado['tiempos_ms']['total']:.0f} ms"]


def codificar_jpeg(imagen, calidad, max_bytes):
    """JPEG de la imagen que no pase de max_bytes, bajando la calidad si hace falta.

    Devuelve: bytes, o None si ni con la calidad mínima cabe.
    """
    while calidad >= 20:
        ok, datos = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, int(calidad)])
        if ok and len(datos) <= max_bytes:
            return datos.tobytes()
        calidad -= 10
    return None