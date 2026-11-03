"""tests/test_servicios.py — Etapa 14: arranque automático con systemd.

Se comprueba lo que se puede comprobar sin instalar nada:
    - que SIGTERM (lo que envía "systemctl stop") se trata como Ctrl+C;
    - que pi_vio, parado con SIGTERM, cierra bien y deja el CSV completo;
    - que el script genera los servicios con las rutas del proyecto.
Solo en Linux (en la Pi): en Windows no hay SIGTERM ni systemd.
"""

import copy
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from vio.config import RAIZ_PROYECTO, cargar_config

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="systemd y SIGTERM: solo en la Pi (Linux)")


@pytest.mark.parametrize("modulo", ["apps.pi_vio", "apps.visor_web"])
def test_sigterm_es_como_ctrl_c(modulo):
    import importlib
    app = importlib.import_module(modulo)
    anterior = signal.getsignal(signal.SIGTERM)
    try:
        app.terminar_con_sigterm()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)          # este mismo proceso se envía SIGTERM
            for _ in range(1000):                         # el manejador salta entre dos instrucciones
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
    guardar_grabacion_sintetica(tmp_path, cfg_vio, K, (640, 480), cargar_config("camara_imu"),
                                np.random.default_rng(5))

    # 2. pi_vio en otro proceso; al ver "VIO EN MARCHA", SIGTERM (lo que hace systemctl stop)
    entorno = dict(os.environ, PYTHONUNBUFFERED="1")
    proceso = subprocess.Popen([sys.executable, "-m", "apps.pi_vio", "--grabacion", str(tmp_path), "--sin-red"],
                               cwd=RAIZ_PROYECTO, env=entorno, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8")
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
    assert all(len(fila.split(",")) == 13 for fila in filas[1:])   # ninguna línea a medias


def test_servicios_generados():
    resultado = subprocess.run(["bash", str(RAIZ_PROYECTO / "sistema" / "instalar_servicios.sh"), "ver"],
                               capture_output=True, text=True, encoding="utf-8", check=True)
    texto = resultado.stdout
    python = RAIZ_PROYECTO / ".venv-rpi" / "bin" / "python"
    assert "@" not in texto and "\r" not in texto                           # sin marcas ni finales de Windows
    assert texto.count(f"WorkingDirectory={RAIZ_PROYECTO}\n") == 2
    assert f"ExecStart={python} -m apps.visor_web\n" in texto
    assert f"ExecStart={python} -m apps.pi_vio --ip 127.0.0.1\n" in texto
    assert texto.count("User=") == 2 and "KillMode=mixed" in texto