"""tests/test_servicios.py — Etapa 14: arranque automático con systemd.

Se comprueba lo que se puede comprobar sin instalar nada:
    - que SIGTERM (lo que envía "systemctl stop") se trata como Ctrl+C;
    - que pi_vio, parado con SIGTERM, cierra bien y deja el CSV completo;
    - que el script genera los servicios con las rutas del proyecto.
Solo en Linux (en la Pi): en Windows no hay SIGTERM ni systemd.
"""

import copy
import os
import re
import signal
import subprocess
import sys

import numpy as np
import pytest

from vio.config import RAIZ_PROYECTO, cargar_config

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="systemd y SIGTERM: solo en la Pi (Linux)"
)


@pytest.mark.parametrize("modulo", ["apps.pi_vio", "apps.visor_web"])
def test_sigterm_es_como_ctrl_c(modulo):
    import importlib

    app = importlib.import_module(modulo)
    anterior = signal.getsignal(signal.SIGTERM)
    try:
        app.terminar_con_sigterm()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)  # este mismo proceso se envía SIGTERM
            for _ in range(1000):  # el manejador salta entre dos instrucciones
                pass
    finally:
        signal.signal(signal.SIGTERM, anterior)


def test_pi_vio_parado_con_sigterm_cierra_bien(tmp_path):
    from herramientas.grabar_simulacion import guardar_grabacion_sintetica
    from vio import modelo_camara as mc

    # 1. Una grabación sintética (como en test_pipeline), larga para pararla a la mitad
    cfg_vio = copy.deepcopy(cargar_config("simulador")["vio"])
    cfg_vio["trayectoria"]["duracion_s"] = 10.0
    K, _, _ = mc.cargar_intrinsecos()
    guardar_grabacion_sintetica(
        tmp_path,
        cfg_vio,
        K,
        (640, 480),
        cargar_config("camara_imu"),
        np.random.default_rng(5),
    )

    # 2. pi_vio en otro proceso; al ver "VIO EN MARCHA", SIGTERM (lo que hace systemctl stop)
    entorno = dict(os.environ, PYTHONUNBUFFERED="1")
    proceso = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "apps.pi_vio",
            "--grabacion",
            str(tmp_path),
            "--sin-red",
            "--sin-calibracion-acc",
        ],
        cwd=RAIZ_PROYECTO,
        env=entorno,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    lineas = []
    for linea in proceso.stdout:
        lineas.append(linea)
        if "VIO EN MARCHA" in linea:
            proceso.send_signal(signal.SIGTERM)
            break
    salida = "".join(lineas) + proceso.stdout.read()
    codigo = proceso.wait(timeout=30)

    # 3. Sale por el camino de Ctrl+C: código 0, "Terminado" y el CSV cerrado con sus líneas
    assert codigo == 0, salida
    assert "Terminado" in salida and "Traceback" not in salida
    with open(tmp_path / "vio_trayectoria.csv", encoding="utf-8") as f:
        filas = f.read().splitlines()
    assert filas[0].startswith("#t_ns") and len(filas) >= 2
    assert all(
        len(fila.split(",")) == 13 for fila in filas[1:]
    )  # ninguna línea a medias


def test_servicios_generados():
    resultado = subprocess.run(
        ["bash", str(RAIZ_PROYECTO / "sistema" / "instalar_servicios.sh"), "ver"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    texto = resultado.stdout
    python = RAIZ_PROYECTO / ".venv-rpi" / "bin" / "python"
    assert "@" not in texto and "\r" not in texto  # sin marcas ni finales de Windows
    assert texto.count(f"WorkingDirectory={RAIZ_PROYECTO}\n") == 2
    assert f"ExecStart={python} -m apps.visor_web\n" in texto
    assert f"ExecStart={python} -m apps.pi_vio --ip 127.0.0.1\n" in texto
    assert texto.count("User=") == 2 and "KillMode=mixed" in texto


# =====================================================================
# Etapa 17 — La Pi como punto de acceso Wi-Fi (sistema/red.sh)
# =====================================================================
def test_scripts_de_red_sin_errores_de_sintaxis():
    for nombre in ("red.sh", "red_arranque.sh"):
        r = subprocess.run(
            ["bash", "-n", str(RAIZ_PROYECTO / "sistema" / nombre)],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{nombre}: {r.stderr}"


def test_servicio_de_red_generado():
    r = subprocess.run(
        ["bash", str(RAIZ_PROYECTO / "sistema" / "red.sh"), "ver"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert "@" not in r.stdout and "\r" not in r.stdout  # plantilla rellena
    assert (
        f"ExecStart=/bin/bash {RAIZ_PROYECTO}/sistema/red_arranque.sh 40 VIO-Pi-punto\n"
        in r.stdout
    )
    assert "Type=oneshot" in r.stdout and "After=NetworkManager.service" in r.stdout


def test_regla_sudo_solo_para_nmcli():
    # Etapa 18: "permitir" deja al usuario ejecutar nmcli como root sin contraseña, y nada más
    r = subprocess.run(
        ["bash", str(RAIZ_PROYECTO / "sistema" / "red.sh"), "ver"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    regla = r.stdout.split("(bash sistema/red.sh permitir) ===\n")[1].strip()
    import pwd  # solo existe en Linux (aquí ya lo es)

    usuario = pwd.getpwuid(os.getuid()).pw_name  # el mismo que da "id -un"
    assert re.fullmatch(
        re.escape(usuario) + r" ALL=\(root\) NOPASSWD: /\S*/nmcli", regla
    ), regla


def arranque_con_nmcli_falso(tmp_path, estado_wlan0, espera_s):
    """Ejecuta red_arranque.sh con un nmcli de mentira delante en el PATH.

    El nmcli falso contesta "wlan0:<estado_wlan0>" a "device status" y apunta en un registro
    todas las órdenes que recibe. Devuelve: (salida del script, órdenes que recibió nmcli).
    """
    registro = tmp_path / "nmcli.log"
    falso = tmp_path / "nmcli"
    falso.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{registro}"\n'
        f'if [ "$*" = "-t -f DEVICE,STATE device status" ]; then echo "wlan0:{estado_wlan0}"; fi\n',
        encoding="utf-8",
    )
    falso.chmod(0o755)
    entorno = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}")
    r = subprocess.run(
        [
            "bash",
            str(RAIZ_PROYECTO / "sistema" / "red_arranque.sh"),
            str(espera_s),
            "VIO-Pi-punto",
        ],
        capture_output=True,
        text=True,
        env=entorno,
        timeout=30,
    )
    ordenes = (
        registro.read_text(encoding="utf-8").splitlines() if registro.exists() else []
    )
    return r.stdout, ordenes


def test_arranque_con_wifi_conocida_no_monta_el_punto(tmp_path):
    salida, ordenes = arranque_con_nmcli_falso(tmp_path, "connected", espera_s=10)
    assert "modo cliente" in salida
    assert "connection up VIO-Pi-punto" not in ordenes  # no toca nada


def test_arranque_sin_wifi_conocida_monta_el_punto(tmp_path):
    salida, ordenes = arranque_con_nmcli_falso(tmp_path, "disconnected", espera_s=2)
    assert "punto de acceso" in salida
    assert ordenes[-1] == "connection up VIO-Pi-punto"  # al final, el punto de acceso
