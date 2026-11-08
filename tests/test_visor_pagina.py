"""tests/test_visor_pagina.py — La página del visor web funciona de verdad (etapa 16).

Los demás tests miran que la página CONTIENE ciertos trozos, pero no que su JavaScript se pueda
ejecutar ni que el flujo de eventos llegue. Un fallo de tecleo en el JavaScript (una llave de
menos) para TODO el script: la página se queda en "CONECTANDO…" y esos tests ni se enteran.
Estos lo detectan sin navegador y dicen la línea de apps/visor_web.py donde mirar.
"""

import http.client
import json
import re
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import pytest

from apps import visor_web
from vio.config import RAIZ_PROYECTO

PAREJA = {")": "(", "]": "[", "}": "{"}


def javascript_y_linea():
    """El JavaScript de la página y la línea de apps/visor_web.py donde empieza (tras <script>)."""
    html = visor_web.PAGINA_HTML
    js = html[html.index("<script>") + len("<script>") : html.index("</script>")]
    fuente = (
        (RAIZ_PROYECTO / "apps" / "visor_web.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    linea_script = next(
        k for k, texto in enumerate(fuente, start=1) if texto.strip() == "<script>"
    )
    return js, linea_script


def primer_desequilibrio(js):
    """Primer paréntesis, corchete o llave sin pareja, saltando textos ("…", '…') y comentarios.

    Devuelve: None si todo cuadra, o (línea dentro del script, mensaje). En el mensaje, "+N" es
    la línea N del script. Ojo: la falta se NOTA más abajo de donde está; lo útil suele ser la
    línea del último paréntesis o llave abierto, que el mensaje también da.
    """
    pila = []  # (carácter abierto, línea)
    i, linea = 0, 1
    while i < len(js):
        c = js[i]
        if js.startswith("//", i):  # comentario hasta el final de la línea
            fin = js.find("\n", i)
            i = len(js) if fin < 0 else fin
            continue
        if js.startswith("/*", i):  # comentario de bloque
            fin = js.find("*/", i + 2)
            linea += js.count("\n", i, fin)
            i = fin + 2
            continue
        if c in "\"'":  # texto: hasta la comilla que lo cierra
            j = i + 1
            while j < len(js) and js[j] != c:
                if js[j] == "\\":
                    j += 1
                elif js[j] == "\n":
                    return linea, f"texto que empieza con {c} sin cerrar"
                j += 1
            i = j + 1
            continue
        if c == "\n":
            linea += 1
        elif c in "([{":
            pila.append((c, linea))
        elif c in ")]}":
            if not pila or pila[-1][0] != PAREJA[c]:
                abierto = (
                    f" (el último abierto es '{pila[-1][0]}' de la línea +{pila[-1][1]})"
                    if pila
                    else ""
                )
                return linea, f"'{c}' que no cierra nada{abierto}"
            pila.pop()
        i += 1
    if pila:
        return pila[-1][1], f"'{pila[-1][0]}' que no se cierra nunca"
    return None


def test_javascript_sin_parentesis_ni_llaves_sueltos():
    js, linea_script = javascript_y_linea()
    problema = primer_desequilibrio(js)
    if problema is not None:
        linea, mensaje = problema
        # "+N" (línea N del script) -> línea de apps/visor_web.py
        mensaje = re.sub(
            r"\+(\d+)", lambda g: str(linea_script + int(g.group(1)) - 1), mensaje
        )
        pytest.fail(
            f"JavaScript de la página: {mensaje}; se nota en la línea "
            f"{linea_script + linea - 1} de apps/visor_web.py"
        )


IDS = [
    "estado",
    "texto_pose",
    "video",
    "lienzo_xy",
    "lienzo_xz",
    "lienzo_ori",
    "lienzo_3d",
    "boton_prueba",
    "m_recorrido",
    "m_cierre",
    "m_pct",
    "m_alejamiento",
    "red_modo",
    "boton_punto",
    "red_lista",
    "boton_cliente",
    "red_aviso",
]  # etapa 18


def test_javascript_ids_existen_en_la_pagina():
    # getElementById("x") con una x que no está en el HTML devuelve null, y lo siguiente
    # (x.textContent, x.addEventListener...) rompe el script
    js, _ = javascript_y_linea()
    definidos = set(re.findall(r'id="([^"]+)"', visor_web.PAGINA_HTML))
    usados = set(re.findall(r'getElementById\("([^"]+)"\)', js))
    assert (
        not usados - definidos
    ), f"ids que usa el JavaScript y no están en el HTML: {sorted(usados - definidos)}"
    # Los que se usan a través de una variable (ponerCifra("m_pct", ...)): cada uno en los dos sitios
    for nombre in IDS:
        assert nombre in definidos, f'falta id="{nombre}" en el HTML'
        assert (
            '"' + nombre + '"' in js
        ), f'el JavaScript no usa "{nombre}" (¿mal escrito?)'


# Lo que se puede llamar sin definirlo en la página: palabras de JavaScript que van seguidas
# de "(" y funciones que trae el navegador
PERMITIDAS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "function",
    "return",
    "typeof",
    "EventSource",
    "Map",
    "fetch",
    "requestAnimationFrame",
    "setInterval",
    "setTimeout",
    "parseInt",
    "parseFloat",
    "isFinite",
    "Number",
    "String",
    "Boolean",
    "Array",
    "Object",
}


def test_javascript_funciones_que_existen():
    # Llamar a una función que no existe (actualizaPrueba en vez de actualizarPrueba) NO es un
    # error de sintaxis: el script arranca y se rompe al llegar a esa línea (ReferenceError).
    # Si es dentro de onmessage, pasa con cada mensaje y la página nunca se actualiza.
    js, linea_script = javascript_y_linea()
    sin_textos = re.sub(
        r"/\*.*?\*/", lambda g: "\n" * g.group(0).count("\n"), js, flags=re.S
    )
    sin_textos = re.sub(r"//[^\n]*", "", sin_textos)  # fuera comentarios
    sin_textos = re.sub(
        r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'', '""', sin_textos
    )  # y textos
    llamadas = set(
        re.findall(r"(?<![\w.$])([A-Za-z_$][\w$]*)\s*\(", sin_textos)
    )  # nombre( sin "." delante
    definidas = set(
        re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", sin_textos)
    ) | set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", sin_textos))
    # Los parámetros de las funciones también se pueden llamar
    for parametros in re.findall(
        r"function\s*[A-Za-z_$]*[\w$]*\s*\(([^)]*)\)", sin_textos
    ):
        definidas |= {p.strip() for p in parametros.split(",") if p.strip()}
    for nombre in sorted(llamadas - definidas - PERMITIDAS):
        linea = (
            sin_textos[
                : re.search(
                    r"(?<![\w.$])" + re.escape(nombre) + r"\s*\(", sin_textos
                ).start()
            ].count("\n")
            + 1
        )
        pytest.fail(
            f"el JavaScript llama a {nombre}(), que no está definida (¿mal escrita?), "
            f"en la línea {linea_script + linea - 1} de apps/visor_web.py"
        )


@pytest.mark.skipif(
    shutil.which("node") is None,
    reason="sin Node.js: no se puede comprobar la sintaxis completa",
)
def test_javascript_sintaxis_con_node(tmp_path):
    # Si hay Node.js (en el PC suele haberlo), comprueba la sintaxis entera, no solo las parejas
    js, linea_script = javascript_y_linea()
    ruta = tmp_path / "pagina.js"
    ruta.write_text(js, encoding="utf-8")
    r = subprocess.run(["node", "--check", str(ruta)], capture_output=True, text=True)
    assert (
        r.returncode == 0
    ), f"(línea del script + {linea_script - 1} = línea de visor_web.py)\n{r.stderr}"


@pytest.fixture
def visor():
    estado = visor_web.EstadoCompartido()
    anterior = visor_web.ManejadorHTTP.estado
    visor_web.ManejadorHTTP.estado = estado
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), visor_web.ManejadorHTTP)
    servidor.daemon_threads = True
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield estado, servidor.server_address[1]
    servidor.shutdown()
    servidor.server_close()
    visor_web.ManejadorHTTP.estado = anterior


def primer_mensaje(puerto):
    """Lee el primer mensaje del flujo de eventos (/eventos), como el navegador."""
    conexion = http.client.HTTPConnection("127.0.0.1", puerto, timeout=3.0)
    conexion.request("GET", "/eventos")
    respuesta = conexion.getresponse()
    assert (
        respuesta.status == 200
        and respuesta.getheader("Content-Type") == "text/event-stream"
    )
    linea = respuesta.fp.readline().decode("utf-8")
    conexion.close()
    assert linea.startswith("data: "), f"el flujo de eventos no manda datos: {linea!r}"
    return json.loads(linea[len("data: ") :])


def test_eventos_sin_datos(visor):
    # Recién arrancado, sin pi_vio: tiene que mandar mensajes igual (la página dice "ESPERANDO")
    _, puerto = visor
    m = primer_mensaje(puerto)
    assert m["sin_datos_s"] is None and m["prueba"] is None and m["nuevos"] == []


def test_eventos_con_trayectoria(visor):
    # Con trayectoria: el mensaje lleva las cifras de la prueba (y se puede pasar a JSON)
    estado, puerto = visor
    with estado.cerrojo:
        for k in range(200):
            visor_web.anadir_punto(estado, [0.01 * k, 0.0, 0.0])
        estado.version, estado.t_ns, estado.hora_ultimo = 2, 10**9, None
    m = primer_mensaje(puerto)
    assert len(m["nuevos"]) == 200
    assert (
        m["prueba"]["recorrido_m"] == pytest.approx(1.99)
        and m["prueba"]["nivel"] == "malo"
    )
