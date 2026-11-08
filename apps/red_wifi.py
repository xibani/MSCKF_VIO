"""
apps/red_wifi.py — Cambiar la Wi-Fi de la Pi desde el visor web (etapa 18).

La Wi-Fi de la Pi hace UNA cosa a la vez:
    modo CLIENTE: conectada a una red guardada (con Internet: git, apt)
    modo PUNTO:   emite su propia red VIO-Pi (la tablet se conecta a la Pi, sin Internet)
Este módulo pregunta el estado a NetworkManager (nmcli) y cambia de modo. Las redes guardadas
NO se olvidan al pasar a punto de acceso: siguen en NetworkManager para volver a ellas.

Seguridad ante un cambio que sale mal (si no, la Pi se quedaría sin red y sin forma de llegar a ella):
    - si no consigue conectarse a la red elegida en ESPERA_CONEXION_S, vuelve a emitir VIO-Pi;
    - si no consigue montar el punto de acceso, vuelve a conectarse a una red conocida.

Solo usa la biblioteca estándar (como visor_web). Para consultar basta nmcli; para CAMBIAR
de red hace falta ser root: se usa "sudo -n nmcli" (-n: si sudo pidiera contraseña, falla en
el acto en vez de quedarse esperando). Ver sistema/red.sh permitir.
"""

import json
import re
import socket
import subprocess
import time
from pathlib import Path

# Copia de los valores de sistema/red.sh (etapa 17): la conexión del punto de acceso
CONEXION_PUNTO = "VIO-Pi-punto"
SSID_PUNTO = "VIO-Pi"
IP_PUNTO = "10.42.0.1"
INTERFAZ = "wlan0"
TIPO_WIFI = "802-11-wireless"

ESPERA_CONEXION_S = (
    30  # máximo para conectarse a una red antes de volver al punto de acceso
)
RETRASO_CAMBIO_S = 1.5  # antes de cambiar: que la respuesta HTTP llegue a la tablet
TIEMPO_CONSULTA_S = 10  # máximo para una consulta a nmcli
PUERTO_WEB = 8000

# Última IP que tuvo la Pi en cada red (para decir en la tablet a qué dirección ir)
RUTA_ULTIMAS_IP = (
    Path(__file__).resolve().parent.parent / "datos" / "red_ultimas_ip.json"
)


# =====================================================================
# Hablar con nmcli
# =====================================================================
def nmcli(*argumentos, root=False, tiempo_s=TIEMPO_CONSULTA_S):
    """Ejecuta nmcli. root=True: con "sudo -n" (para cambiar de red). Devuelve: (código, salida)."""
    orden = (["sudo", "-n"] if root else []) + ["nmcli", *argumentos]
    try:
        r = subprocess.run(orden, capture_output=True, text=True, timeout=tiempo_s)
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, str(error)
    return r.returncode, r.stdout if r.returncode == 0 else (r.stderr or r.stdout)


def campos_terse(linea):
    """Separa una línea de "nmcli -t" en campos. nmcli escribe ':' como '\\:' y '\\' como '\\\\'."""
    campos, actual, i = [], "", 0
    while i < len(linea):
        if linea[i] == "\\" and i + 1 < len(linea):
            actual += linea[i + 1]  # carácter escapado: tal cual
            i += 2
        elif linea[i] == ":":
            campos.append(actual)
            actual = ""
            i += 1
        else:
            actual += linea[i]
            i += 1
    campos.append(actual)
    return campos


def sin_escapes(texto):
    """Un valor suelto de "nmcli -g" (un solo campo): solo se quitan los escapes, sin partirlo."""
    return re.sub(r"\\(.)", r"\1", texto)


def ssid_de(conexion):
    """SSID de una conexión Wi-Fi guardada (el nombre de la conexión puede ser otro, p. ej.
    "preconfigured" si la configuró Raspberry Pi Imager)."""
    codigo, salida = nmcli(
        "-t", "-g", "802-11-wireless.ssid", "connection", "show", conexion
    )
    return sin_escapes(salida.strip()) if codigo == 0 and salida.strip() else conexion


# =====================================================================
# Últimas IP (en cada red el router suele dar la misma)
# =====================================================================
def leer_ultimas_ip(ruta=None):
    ruta = RUTA_ULTIMAS_IP if ruta is None else ruta
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def guardar_ultima_ip(conexion, ip, ruta=None):
    ruta = Path(RUTA_ULTIMAS_IP if ruta is None else ruta)
    ultimas = leer_ultimas_ip(ruta)
    if ultimas.get(conexion) == ip:
        return
    ultimas[conexion] = ip
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(ultimas, indent=2), encoding="utf-8")
    except OSError:
        pass  # sin poder guardarlo, solo se pierde la pista


# =====================================================================
# Estado
# =====================================================================
def redes_guardadas():
    """Conexiones Wi-Fi guardadas a las que se puede volver (sin el punto de acceso).

    Devuelve: lista de nombres de conexión de NetworkManager.
    """
    codigo, salida = nmcli("-t", "-f", "NAME,TYPE", "connection", "show")
    if codigo != 0:
        return []
    redes = []
    for linea in salida.splitlines():
        campos = campos_terse(linea)
        if len(campos) >= 2 and campos[1] == TIPO_WIFI and campos[0] != CONEXION_PUNTO:
            redes.append(campos[0])
    return redes


def existe_punto():
    """¿Está creada la conexión del punto de acceso? (bash sistema/red.sh crear)"""
    codigo, salida = nmcli("-t", "-f", "NAME", "connection", "show")
    return codigo == 0 and CONEXION_PUNTO in [
        campos_terse(l)[0] for l in salida.splitlines()
    ]


def estado(ruta_ultimas=None):
    """Modo actual de la Wi-Fi y redes disponibles.

    Devuelve: dict con modo ("punto", "cliente" o "desconectado"), conexion, ssid, ip, url,
    url_mdns y redes (lista de dicts con conexion, ssid y ultima_ip).
    """
    activa = None
    codigo, salida = nmcli(
        "-t", "-f", "NAME,TYPE,DEVICE", "connection", "show", "--active"
    )
    if codigo == 0:
        for linea in salida.splitlines():
            campos = campos_terse(linea)
            if len(campos) >= 3 and campos[2] == INTERFAZ:
                activa = campos[0]

    ip = None
    codigo, salida = nmcli("-t", "-g", "IP4.ADDRESS", "device", "show", INTERFAZ)
    if codigo == 0 and salida.strip():
        # "192.168.1.57/24" o, con varias, "192.168.1.57/24 | 10.0.0.2/8": la primera, sin la máscara
        ip = sin_escapes(salida.strip()).split(" | ")[0].split("/")[0] or None

    if activa == CONEXION_PUNTO:
        modo, ssid, ip = "punto", SSID_PUNTO, IP_PUNTO
    elif activa is not None:
        modo, ssid = "cliente", ssid_de(activa)
        if ip:
            guardar_ultima_ip(activa, ip, ruta_ultimas)
    else:
        modo, ssid = "desconectado", None

    ultimas = leer_ultimas_ip(ruta_ultimas)
    redes = [
        {"conexion": c, "ssid": ssid_de(c), "ultima_ip": ultimas.get(c)}
        for c in redes_guardadas()
    ]
    return {
        "modo": modo,
        "conexion": activa,
        "ssid": ssid,
        "ip": ip,
        "url": f"http://{ip}:{PUERTO_WEB}" if ip else None,
        "url_mdns": f"http://{socket.gethostname()}.local:{PUERTO_WEB}",
        "redes": redes,
    }


def conectada():
    """¿Está wlan0 conectada (a una red o emitiendo como punto de acceso)?"""
    codigo, salida = nmcli("-t", "-f", "DEVICE,STATE", "device", "status")
    return codigo == 0 and any(
        campos_terse(l)[:2] == [INTERFAZ, "connected"] for l in salida.splitlines()
    )


# =====================================================================
# Cambios (se llaman en un hilo aparte, DESPUÉS de responder a la tablet)
# =====================================================================
def cambiar_a_punto(retraso_s=RETRASO_CAMBIO_S):
    """Deja la red actual (sin olvidarla) y emite VIO-Pi. Si falla, vuelve a una red conocida.

    Devuelve: True si quedó como punto de acceso.
    """
    time.sleep(retraso_s)
    codigo, _ = nmcli(
        "--wait",
        str(ESPERA_CONEXION_S),
        "connection",
        "up",
        CONEXION_PUNTO,
        root=True,
        tiempo_s=ESPERA_CONEXION_S + 10,
    )
    if codigo == 0:
        return True
    nmcli(
        "device", "connect", INTERFAZ, root=True, tiempo_s=ESPERA_CONEXION_S + 10
    )  # vuelta atrás
    return False


def cambiar_a_red(conexion, retraso_s=RETRASO_CAMBIO_S):
    """Se conecta a una red guardada. Si no lo consigue, vuelve a emitir VIO-Pi.

    Devuelve: True si quedó conectada a esa red.
    """
    time.sleep(retraso_s)
    codigo, _ = nmcli(
        "--wait",
        str(ESPERA_CONEXION_S),
        "connection",
        "up",
        conexion,
        root=True,
        tiempo_s=ESPERA_CONEXION_S + 10,
    )
    if codigo == 0 and conectada():
        return True
    nmcli(
        "--wait",
        str(ESPERA_CONEXION_S),
        "connection",
        "up",
        CONEXION_PUNTO,
        root=True,
        tiempo_s=ESPERA_CONEXION_S + 10,
    )  # vuelta atrás
    return False


def puede_cambiar():
    """¿Puede este usuario cambiar de red sin contraseña? (sudo -n nmcli)"""
    codigo, _ = nmcli("--version", root=True)
    return codigo == 0


if __name__ == "__main__":
    # Para mirar lo que verá la tablet:  python -m apps.red_wifi
    print(json.dumps(estado(), indent=2, ensure_ascii=False))
    print("Punto de acceso creado:", existe_punto())
    print("Puede cambiar de red sin contraseña (sudo -n nmcli):", puede_cambiar())
