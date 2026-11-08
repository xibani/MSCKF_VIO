"""tests/test_red_wifi.py — Cambiar la Wi-Fi desde el visor web (etapa 18).

Sin tocar la Wi-Fi de verdad:
    - apps/red_wifi.py se prueba con un nmcli y un sudo FALSOS delante en el PATH: el nmcli falso
      contesta lo que diga un escenario (un JSON) y apunta cada orden que recibe;
    - los endpoints /red, /red/punto y /red/cliente del visor se prueban con un módulo de red
      falso (sin nmcli), para ver qué responden y qué cambio lanzan.
"""

import http.client
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer

import pytest

from apps import red_wifi, visor_web

solo_linux = pytest.mark.skipif(
    sys.platform == "win32", reason="nmcli y sudo falsos: scripts de Linux"
)

# Lo que contesta nmcli en la Pi conectada a la red "Casa" (orden -> [código, salida])
CONEXIONES = (
    "Casa:802-11-wireless\n"
    "preconfigured:802-11-wireless\n"
    "VIO-Pi-punto:802-11-wireless\n"
    "Wired connection 1:802-3-ethernet\n"
    "lo:loopback\n"
)
ESCENARIO_CLIENTE = {
    "-t -f NAME,TYPE,DEVICE connection show --active": [
        0,
        "Casa:802-11-wireless:wlan0\nlo:loopback:lo\n",
    ],
    "-t -g IP4.ADDRESS device show wlan0": [0, "192.168.1.57/24\n"],
    "-t -f NAME,TYPE connection show": [0, CONEXIONES],
    "-t -f NAME connection show": [
        0,
        "Casa\npreconfigured\nVIO-Pi-punto\nWired connection 1\nlo\n",
    ],
    "-t -g 802-11-wireless.ssid connection show Casa": [0, "MiCasa\n"],
    "-t -g 802-11-wireless.ssid connection show preconfigured": [
        0,
        "Movil\\:5G\n",
    ],  # SSID con ':'
    "-t -f DEVICE,STATE device status": [
        0,
        "wlan0:connected\neth0:unavailable\nlo:connected (externally)\n",
    ],
    "--version": [0, "nmcli tool, version 1.52.0\n"],
}


# =====================================================================
# nmcli y sudo falsos
# =====================================================================
class NmcliFalso:
    def __init__(self, carpeta, monkeypatch):
        self.carpeta = carpeta
        self.registro = carpeta / "nmcli.log"
        self.escenario = carpeta / "escenario.json"
        self.monkeypatch = monkeypatch
        nmcli = carpeta / "nmcli"
        nmcli.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "orden = ' '.join(sys.argv[1:])\n"
            "with open(os.environ['NMCLI_REGISTRO'], 'a') as f:\n"
            "    f.write(json.dumps({'orden': orden, 'root': os.environ.get('SUDO_FALSO') == '1'}) + '\\n')\n"
            "with open(os.environ['NMCLI_ESCENARIO']) as f:\n"
            "    escenario = json.load(f)\n"
            "codigo, salida = escenario.get(orden, [10, 'Error: orden no prevista\\n'])\n"
            "(sys.stdout if codigo == 0 else sys.stderr).write(salida)\n"
            "sys.exit(codigo)\n",
            encoding="utf-8",
        )
        sudo = carpeta / "sudo"
        sudo.write_text(
            "#!/usr/bin/env bash\n"
            'if [ -n "${SUDO_FALLA:-}" ]; then echo "sudo: a password is required" >&2; exit 1; fi\n'
            '[ "$1" = "-n" ] && shift\n'
            'SUDO_FALSO=1 exec "$@"\n',
            encoding="utf-8",
        )
        for script in (nmcli, sudo):
            script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{carpeta}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("NMCLI_REGISTRO", str(self.registro))
        monkeypatch.setenv("NMCLI_ESCENARIO", str(self.escenario))
        monkeypatch.setattr(
            red_wifi, "RUTA_ULTIMAS_IP", carpeta / "datos" / "red_ultimas_ip.json"
        )
        self.preparar(ESCENARIO_CLIENTE)

    def preparar(self, escenario, sudo_falla=False):
        self.escenario.write_text(json.dumps(escenario), encoding="utf-8")
        if sudo_falla:
            self.monkeypatch.setenv("SUDO_FALLA", "1")
        else:
            self.monkeypatch.delenv("SUDO_FALLA", raising=False)

    def ordenes(self, solo_root=False):
        if not self.registro.exists():
            return []
        lineas = [
            json.loads(l)
            for l in self.registro.read_text(encoding="utf-8").splitlines()
        ]
        return [l["orden"] for l in lineas if l["root"] or not solo_root]


@pytest.fixture
def falso(tmp_path, monkeypatch):
    return NmcliFalso(tmp_path, monkeypatch)


def escenario_con(respuestas):
    """ESCENARIO_CLIENTE con algunas respuestas cambiadas o añadidas."""
    escenario = dict(ESCENARIO_CLIENTE)
    escenario.update(respuestas)
    return escenario


# =====================================================================
# Lectura de la salida de nmcli
# =====================================================================
def test_campos_terse_con_escapes():
    assert red_wifi.campos_terse("Casa:802-11-wireless:wlan0") == [
        "Casa",
        "802-11-wireless",
        "wlan0",
    ]
    assert red_wifi.campos_terse("Movil\\:5G:802-11-wireless") == [
        "Movil:5G",
        "802-11-wireless",
    ]
    assert red_wifi.campos_terse("barra\\\\:x") == ["barra\\", "x"]
    assert red_wifi.campos_terse("") == [""]


def test_sin_escapes_no_parte_el_valor():
    assert red_wifi.sin_escapes("Movil\\:5G") == "Movil:5G"
    assert red_wifi.sin_escapes("a:b") == "a:b"  # un ':' sin escapar se queda


@solo_linux
def test_redes_guardadas_sin_el_punto_ni_cable(falso):
    assert red_wifi.redes_guardadas() == ["Casa", "preconfigured"]


# =====================================================================
# Estado
# =====================================================================
@solo_linux
def test_estado_cliente(falso):
    e = red_wifi.estado()
    assert e["modo"] == "cliente" and e["conexion"] == "Casa" and e["ssid"] == "MiCasa"
    assert e["ip"] == "192.168.1.57" and e["url"] == "http://192.168.1.57:8000"
    assert e["url_mdns"].endswith(".local:8000")
    assert [r["ssid"] for r in e["redes"]] == ["MiCasa", "Movil:5G"]
    # La IP de esta red queda apuntada para cuando se vuelva a ella desde el punto de acceso
    assert red_wifi.leer_ultimas_ip() == {"Casa": "192.168.1.57"}
    assert (
        e["redes"][0]["ultima_ip"] == "192.168.1.57"
        and e["redes"][1]["ultima_ip"] is None
    )
    assert falso.ordenes(solo_root=True) == []  # consultar no necesita root


@solo_linux
def test_estado_con_varias_ip_toma_la_primera(falso):
    falso.preparar(
        escenario_con(
            {
                "-t -g IP4.ADDRESS device show wlan0": [
                    0,
                    "192.168.1.57/24 | 10.0.0.2/8\n",
                ]
            }
        )
    )
    assert red_wifi.estado()["ip"] == "192.168.1.57"


@solo_linux
def test_estado_punto(falso):
    falso.preparar(
        escenario_con(
            {
                "-t -f NAME,TYPE,DEVICE connection show --active": [
                    0,
                    "VIO-Pi-punto:802-11-wireless:wlan0\n",
                ],
                "-t -g IP4.ADDRESS device show wlan0": [0, "10.42.0.1/24\n"],
            }
        )
    )
    e = red_wifi.estado()
    assert (
        e["modo"] == "punto"
        and e["ssid"] == "VIO-Pi"
        and e["url"] == "http://10.42.0.1:8000"
    )
    assert red_wifi.leer_ultimas_ip() == {}  # la IP del punto no se apunta como "red"


@solo_linux
def test_estado_desconectado(falso):
    falso.preparar(
        escenario_con(
            {
                "-t -f NAME,TYPE,DEVICE connection show --active": [
                    0,
                    "lo:loopback:lo\n",
                ],
                "-t -g IP4.ADDRESS device show wlan0": [0, ""],
            }
        )
    )
    e = red_wifi.estado()
    assert e["modo"] == "desconectado" and e["ip"] is None and e["url"] is None
    assert len(e["redes"]) == 2


@solo_linux
def test_estado_con_nmcli_fallando(falso):
    falso.preparar({})  # todo contesta con error
    e = red_wifi.estado()
    assert e["modo"] == "desconectado" and e["redes"] == []
    assert not red_wifi.existe_punto() and not red_wifi.conectada()


def test_sin_nmcli_no_revienta(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # carpeta vacía: no hay nmcli
    codigo, _ = red_wifi.nmcli("--version")
    assert codigo != 0
    assert red_wifi.estado()["modo"] == "desconectado"


def test_ultimas_ip_con_fichero_roto(tmp_path):
    ruta = tmp_path / "ultimas.json"
    ruta.write_text("{esto no es json", encoding="utf-8")
    assert red_wifi.leer_ultimas_ip(ruta) == {}
    red_wifi.guardar_ultima_ip("Casa", "192.168.1.57", ruta)
    red_wifi.guardar_ultima_ip("Movil", "172.20.10.3", ruta)
    assert red_wifi.leer_ultimas_ip(ruta) == {
        "Casa": "192.168.1.57",
        "Movil": "172.20.10.3",
    }


@solo_linux
def test_existe_punto(falso):
    assert red_wifi.existe_punto()
    falso.preparar(escenario_con({"-t -f NAME connection show": [0, "Casa\nlo\n"]}))
    assert not red_wifi.existe_punto()


@solo_linux
def test_puede_cambiar_depende_de_sudo(falso):
    assert red_wifi.puede_cambiar()
    falso.preparar(ESCENARIO_CLIENTE, sudo_falla=True)
    assert not red_wifi.puede_cambiar()


# =====================================================================
# Cambios de modo (y su vuelta atrás si fallan)
# =====================================================================
UP_PUNTO = "--wait 30 connection up VIO-Pi-punto"


@solo_linux
def test_cambiar_a_punto(falso):
    falso.preparar(escenario_con({UP_PUNTO: [0, "Conexión activada\n"]}))
    assert red_wifi.cambiar_a_punto(retraso_s=0)
    assert falso.ordenes(solo_root=True) == [
        UP_PUNTO
    ]  # con root, y nada más (no olvida redes)


@solo_linux
def test_cambiar_a_punto_que_falla_vuelve_a_la_red(falso):
    falso.preparar(
        escenario_con(
            {
                UP_PUNTO: [4, "Error: fallo al activar\n"],
                "device connect wlan0": [0, ""],
            }
        )
    )
    assert not red_wifi.cambiar_a_punto(retraso_s=0)
    assert falso.ordenes(solo_root=True) == [UP_PUNTO, "device connect wlan0"]


@solo_linux
def test_cambiar_a_red(falso):
    falso.preparar(escenario_con({"--wait 30 connection up Casa": [0, ""]}))
    assert red_wifi.cambiar_a_red("Casa", retraso_s=0)
    assert falso.ordenes(solo_root=True) == ["--wait 30 connection up Casa"]


@solo_linux
def test_cambiar_a_red_que_falla_vuelve_al_punto(falso):
    # Contraseña cambiada, red fuera de alcance...: en 30 s nmcli se rinde y la Pi vuelve a VIO-Pi
    falso.preparar(
        escenario_con(
            {
                "--wait 30 connection up Casa": [4, "Error: tiempo agotado\n"],
                UP_PUNTO: [0, ""],
            }
        )
    )
    assert not red_wifi.cambiar_a_red("Casa", retraso_s=0)
    assert falso.ordenes(solo_root=True) == ["--wait 30 connection up Casa", UP_PUNTO]


@solo_linux
def test_cambiar_a_red_con_nombre_raro_va_como_un_solo_argumento(falso):
    # El nombre va a nmcli como UN argumento (lista, sin shell): espacios y ';' no hacen nada raro
    nombre = "Red de casa; reboot"
    falso.preparar(escenario_con({f"--wait 30 connection up {nombre}": [0, ""]}))
    assert red_wifi.cambiar_a_red(nombre, retraso_s=0)


# =====================================================================
# Endpoints del visor web (con un módulo de red falso: sin nmcli)
# =====================================================================
class RedFalsa:
    """Imita a apps.red_wifi: mismas constantes y funciones, sin tocar nada."""

    CONEXION_PUNTO, SSID_PUNTO, IP_PUNTO = "VIO-Pi-punto", "VIO-Pi", "10.42.0.1"
    PUERTO_WEB, ESPERA_CONEXION_S = 8000, 30

    def __init__(self):
        self.permiso = True
        self.punto = True
        self.resultado = True
        self.cambios = []  # ("punto",) o ("red", conexion)
        self.terminado = threading.Event()
        self.soltar = (
            threading.Event()
        )  # el cambio no termina hasta que el test lo diga
        self.soltar.set()

    def estado(self):
        return {
            "modo": "cliente",
            "conexion": "Casa",
            "ssid": "MiCasa",
            "ip": "192.168.1.57",
            "url": "http://192.168.1.57:8000",
            "url_mdns": "http://vio-pi.local:8000",
            "redes": [
                {"conexion": "Casa", "ssid": "MiCasa", "ultima_ip": "192.168.1.57"},
                {"conexion": "Movil", "ssid": "Movil", "ultima_ip": None},
            ],
        }

    def existe_punto(self):
        return self.punto

    def puede_cambiar(self):
        return self.permiso

    def _cambio(self, *datos):
        self.soltar.wait(5)
        self.cambios.append(datos)
        self.terminado.set()
        return self.resultado

    def cambiar_a_punto(self):
        return self._cambio("punto")

    def cambiar_a_red(self, conexion):
        return self._cambio("red", conexion)


@pytest.fixture
def visor(monkeypatch):
    red = RedFalsa()
    monkeypatch.setattr(visor_web.ManejadorHTTP, "red", red)
    monkeypatch.setattr(visor_web.ManejadorHTTP, "cambio_red", visor_web.CambioRed())
    monkeypatch.setattr(visor_web.ManejadorHTTP, "estado", visor_web.EstadoCompartido())
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), visor_web.ManejadorHTTP)
    servidor.daemon_threads = True
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    yield red, servidor.server_address[1]
    servidor.shutdown()
    servidor.server_close()


def peticion(puerto, metodo, ruta, cuerpo=None):
    conexion = http.client.HTTPConnection("127.0.0.1", puerto, timeout=5)
    datos = (
        None
        if cuerpo is None
        else (cuerpo if isinstance(cuerpo, bytes) else json.dumps(cuerpo).encode())
    )
    conexion.request(
        metodo, ruta, body=datos, headers={"Content-Type": "application/json"}
    )
    r = conexion.getresponse()
    respuesta = json.loads(r.read().decode("utf-8"))
    conexion.close()
    return r.status, respuesta


def test_get_red(visor):
    _, puerto = visor
    codigo, e = peticion(puerto, "GET", "/red")
    assert codigo == 200
    assert e["modo"] == "cliente" and len(e["redes"]) == 2
    assert e["puede_cambiar"] and e["punto_creado"]
    assert e["cambiando"] is None and e["ultimo_cambio"] is None


def test_post_punto(visor):
    red, puerto = visor
    codigo, r = peticion(puerto, "POST", "/red/punto", {})
    assert codigo == 200 and r["ok"]
    assert r["enlaces"] == ["http://10.42.0.1:8000"] and "VIO-Pi" in r["mensaje"]
    assert red.terminado.wait(5) and red.cambios == [("punto",)]
    # Cuando termina, /red lo cuenta (la página lo enseña si falló)
    for _ in range(50):
        _, e = peticion(puerto, "GET", "/red")
        if e["ultimo_cambio"] is not None:
            break
    assert e["cambiando"] is None
    assert (
        e["ultimo_cambio"]["ok"]
        and e["ultimo_cambio"]["descripcion"] == "punto de acceso VIO-Pi"
    )


def test_post_cliente(visor):
    red, puerto = visor
    codigo, r = peticion(puerto, "POST", "/red/cliente", {"conexion": "Casa"})
    assert codigo == 200 and r["ok"]
    # Última IP conocida de esa red y, por si el router da otra, el nombre .local
    assert r["enlaces"] == ["http://192.168.1.57:8000", "http://vio-pi.local:8000"]
    assert red.terminado.wait(5) and red.cambios == [("red", "Casa")]


def test_post_cliente_sin_ip_conocida(visor):
    red, puerto = visor
    codigo, r = peticion(puerto, "POST", "/red/cliente", {"conexion": "Movil"})
    assert codigo == 200 and r["enlaces"] == ["http://vio-pi.local:8000"]


def test_post_cliente_que_falla_queda_apuntado(visor):
    red, puerto = visor
    red.resultado = False
    peticion(puerto, "POST", "/red/cliente", {"conexion": "Movil"})
    assert red.terminado.wait(5)
    for _ in range(50):
        _, e = peticion(puerto, "GET", "/red")
        if e["ultimo_cambio"] is not None:
            break
    assert (
        e["ultimo_cambio"]["ok"] is False
        and e["ultimo_cambio"]["descripcion"] == "red Movil"
    )


@pytest.mark.parametrize(
    "cuerpo", [{"conexion": "Vecino"}, {"otra": "cosa"}, b"esto no es json"]
)
def test_post_cliente_rechaza_redes_no_guardadas_y_basura(visor, cuerpo):
    red, puerto = visor
    codigo, r = peticion(puerto, "POST", "/red/cliente", cuerpo)
    assert codigo == 400 and not r["ok"]
    assert red.cambios == [] and not red.terminado.wait(0.3)


def test_un_cambio_a_la_vez(visor):
    red, puerto = visor
    red.soltar.clear()  # el primer cambio se queda "en curso"
    assert peticion(puerto, "POST", "/red/punto", {})[0] == 200
    codigo, r = peticion(puerto, "POST", "/red/cliente", {"conexion": "Casa"})
    assert codigo == 409 and "en curso" in r["mensaje"]
    _, e = peticion(puerto, "GET", "/red")
    assert e["cambiando"] == "punto de acceso VIO-Pi"
    red.soltar.set()
    assert red.terminado.wait(5) and red.cambios == [("punto",)]


def test_sin_permiso_no_cambia(visor):
    red, puerto = visor
    red.permiso = False
    codigo, r = peticion(puerto, "POST", "/red/punto", {})
    assert codigo == 403 and "red.sh permitir" in r["mensaje"]
    assert red.cambios == []


def test_sin_punto_creado_no_cambia(visor):
    red, puerto = visor
    red.punto = False
    codigo, r = peticion(puerto, "POST", "/red/cliente", {"conexion": "Casa"})
    assert codigo == 409 and "red.sh crear" in r["mensaje"]
    assert red.cambios == []
