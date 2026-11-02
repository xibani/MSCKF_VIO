"""tests/test_escritor.py — Tests de captura/escritor.py."""

import cv2
import numpy as np

from captura.escritor import EscritorPNG


def imagen_de_prueba(valor):
    """Imagen 480x640 con un degradado y un valor distinto en la esquina."""
    img = np.tile(np.arange(640, dtype=np.uint8), (480, 1))   # degradado horizontal
    img[0, 0] = valor
    return img


def test_escribe_todas_las_imagenes(tmp_path):
    escritor = EscritorPNG(cola_max=10, compresion=1)
    escritor.iniciar()
    originales = {}
    for i in range(5):
        ruta = tmp_path / f"{i}.png"
        originales[ruta] = imagen_de_prueba(i)
        assert escritor.poner(ruta, originales[ruta])
    escritor.cerrar()                          # espera a que se escriban todas

    assert escritor.escritas == 5
    assert escritor.errores == 0
    for ruta, original in originales.items():
        leida = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(leida, original)  # PNG: ni un píxel cambiado


def test_cola_llena_devuelve_false(tmp_path):
    escritor = EscritorPNG(cola_max=2, compresion=1)   # sin iniciar: nadie vacía la cola
    img = imagen_de_prueba(0)
    assert escritor.poner(tmp_path / "a.png", img)
    assert escritor.poner(tmp_path / "b.png", img)
    assert not escritor.poner(tmp_path / "c.png", img)  # no cabe
    assert escritor.pendientes() == 2
    escritor.cerrar()                          # sin iniciar no hace nada (y no se cuelga)


def test_error_de_escritura_se_cuenta(tmp_path):
    escritor = EscritorPNG(cola_max=10, compresion=1)
    escritor.iniciar()
    escritor.poner(tmp_path / "no_existe" / "x.png", imagen_de_prueba(0))   # carpeta inexistente
    escritor.cerrar()
    assert escritor.escritas == 0
    assert escritor.errores == 1