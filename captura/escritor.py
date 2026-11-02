"""
captura/escritor.py — Guarda imágenes PNG en un hilo aparte.

El bucle de la cámara solo mete imágenes en una cola (instantáneo); un hilo
las va sacando y escribiendo a disco. Si el disco se atasca un momento, la
cola absorbe el retraso y no se pierden imágenes.
"""

import queue
import threading

import cv2


class EscritorPNG:
    """Cola de imágenes + hilo que las escribe a disco."""

    def __init__(self, cola_max, compresion):
        """cola_max: imágenes que caben en espera. compresion: nivel PNG 0-9."""
        self.cola = queue.Queue(maxsize=cola_max)
        self.opciones = [cv2.IMWRITE_PNG_COMPRESSION, compresion]
        self.hilo = threading.Thread(target=self._trabajar, daemon=True)
        self.iniciado = False
        self.escritas = 0
        self.errores = 0

    def iniciar(self):
        """Arranca el hilo de escritura."""
        self.hilo.start()
        self.iniciado = True

    def poner(self, ruta, imagen):
        """Encola una imagen sin esperar.

        Devuelve: True si se encoló; False si la cola estaba llena (imagen perdida).
        """
        try:
            self.cola.put_nowait((ruta, imagen))
            return True
        except queue.Full:
            return False

    def pendientes(self):
        """Imágenes que esperan a ser escritas."""
        return self.cola.qsize()

    def cerrar(self):
        """Espera a que se escriban todas las imágenes pendientes y para el hilo."""
        if not self.iniciado:
            return
        self.cola.put(None)          # señal de fin: va detrás de todas las pendientes
        self.hilo.join()             # esperar a que el hilo termine
        self.iniciado = False

    def _trabajar(self):
        """Bucle del hilo: saca imágenes de la cola y las escribe."""
        while True:
            elemento = self.cola.get()          # espera hasta que haya algo
            if elemento is None:
                break
            ruta, imagen = elemento
            if cv2.imwrite(str(ruta), imagen, self.opciones):
                self.escritas += 1
            else:
                self.errores += 1