"""
vio/vista.py — Lo que ve el VIO, para mostrarlo en el visor: la imagen con sus pistas.

Cada pista activa del front-end se dibuja con su "cola" (sus últimas posiciones), de rojo
(recién detectada) a verde (seguida muchas imágenes). Arriba, una línea de estado.
La imagen se comprime en JPEG para que quepa en un solo paquete UDP.
"""

import cv2
import numpy as np


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

    # 2. Texto con fondo oscuro, para que se lea sobre cualquier imagen
    for k, linea in enumerate(lineas_texto):
        y = 20 + 22 * k
        cv2.putText(vista, linea, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vista, linea, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vista


def lineas_estado(pipeline, resultado):
    """Las líneas de texto que acompañan a la imagen."""
    if pipeline.filtro is None:
        return ["INICIALIZANDO: deja la placa QUIETA"]
    if resultado is None:
        return ["VIO EN MARCHA (esperando imagen)"]
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