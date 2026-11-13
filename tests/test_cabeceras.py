"""tests/test_cabeceras.py — Cada módulo empieza diciendo qué archivo es (etapa 7.4).

Todos los .py del proyecto empiezan su docstring con su propia ruta, p. ej.
"herramientas/grabar.py — ...". Si un archivo se sobrescribe por error con el contenido
de otro (pasó con herramientas/grabar.py, que acabó siendo una copia de analizar_allan.py),
su cabecera nombra al otro y este test lo detecta.
"""

import ast

from vio.config import RAIZ_PROYECTO

CARPETAS = ["apps", "captura", "eval", "herramientas", "vio"]


def test_cada_modulo_se_nombra_a_si_mismo():
    mal = []
    for carpeta in CARPETAS:
        for ruta in sorted((RAIZ_PROYECTO / carpeta).glob("*.py")):
            if ruta.name == "__init__.py":
                continue
            doc = ast.get_docstring(ast.parse(ruta.read_text(encoding="utf-8"))) or ""
            primera = doc.strip().splitlines()[0] if doc.strip() else "(sin docstring)"
            if not primera.startswith(f"{carpeta}/{ruta.name}"):
                mal.append(f"{carpeta}/{ruta.name} empieza por: {primera[:60]}")
    assert not mal, "módulos cuya cabecera no es la suya:\n" + "\n".join(mal)