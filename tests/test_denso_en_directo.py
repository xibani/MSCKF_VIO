"""tests/test_denso_en_directo.py — Etapa 30: la nube densa, keyframe a keyframe.

Ninguno necesita PyTorch: la profundidad es la verdadera de la sesión simulada (la fixture
de la etapa 24), que es lo que hace apps/pc_denso con --con-verdad.
"""

import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from apps import pc_denso
from herramientas import medir_mapa
from herramientas import nube_densa
from vio import densificar as den
from vio import denso_en_directo as dd
from vio import protocolo
from vio.config import RAIZ_PROYECTO, cargar_config
from vio.reconstruccion import leer_ply

CFG = cargar_config("densificar")
CFG_DIRECTO = cargar_config("directo")


def ordenados(p):
    """Los mismos puntos, en un orden que no depende de cómo se hayan calculado."""
    return p[np.lexsort((p[:, 2], p[:, 1], p[:, 0]))]


@pytest.fixture(scope="module")
def sesion(sesion_con_textura):
    """La sesión de la etapa 24: carpeta, paquetes del mapa, vistas con la profundidad
    verdadera y la nube que da con ellas la etapa 26."""
    carpeta = sesion_con_textura["carpeta"]
    mapa = medir_mapa.leer_mapa(carpeta / "mapa.paquetes")
    vistas = nube_densa.cargar_vistas(mapa.keyframes, carpeta, True, sesion_con_textura["t_d_ns"])
    R, t = nube_densa.transformacion_de_la_sesion(carpeta)
    return {"carpeta": carpeta, "paquetes": list(protocolo.leer_archivo_paquetes(carpeta / "mapa.paquetes")),
            "vistas": vistas, "de_una_vez": den.densificar(vistas, CFG), "R": R, "t": t,
            "dimensiones": cargar_config("simulador")["vio"]["mundo"]["habitacion_m"]}


def copia(vista):
    """Una vista nueva con los mismos datos (el densificador en directo vacía las que ya no usa)."""
    return {"kf": vista["kf"], "z": vista["z"], "gris": vista["gris"]}


# =====================================================================
# Paso 30.1 — Una nube a la que se le van sumando puntos
# =====================================================================
def test_nube_incremental_igual_que_de_una_vez():
    generador = np.random.default_rng(0)
    p = generador.uniform(-1.0, 1.0, (5000, 3))
    gris = generador.integers(0, 256, 5000).astype(np.uint8)
    nube = dd.NubeIncremental(0.10)
    assert len(nube) == 0 and len(nube.puntos(1)[0]) == 0
    for inicio in range(0, 5000, 700):                   # en 8 tandas
        nube.anadir(p[inicio:inicio + 700], gris[inicio:inicio + 700])
    nube.anadir(p[:0], gris[:0])                         # una tanda vacía no cambia nada
    for puntos_min in (1, 2, 3):
        q, brillo, cuentas = nube.puntos(puntos_min)
        q_ref, brillo_ref, cuentas_ref = den.fundir_en_voxeles(p, gris, 0.10, puntos_min)
        assert len(q) == len(q_ref) and ordenados(q) == pytest.approx(ordenados(q_ref), abs=1e-9)
        assert sorted(cuentas) == sorted(cuentas_ref) and brillo.dtype == np.uint8
        assert abs(int(brillo.astype(int).sum()) - int(brillo_ref.astype(int).sum())) <= len(q)    # redondeos
    assert len(nube) == len(den.fundir_en_voxeles(p, gris, 0.10, 1)[0])


# =====================================================================
# Paso 30.2 — Los keyframes, de uno en uno
# =====================================================================
def test_con_una_ventana_que_lo_abarca_todo_sale_lo_mismo_que_en_la_etapa_26(sesion):
    densificador = dd.DensificadorEnDirecto(CFG, {"ventana": 1000, "retraso": 1000})
    for vista in sesion["vistas"]:
        assert densificador.nueva_vista(copia(vista)) == 0       # con ese retraso, ninguna entra todavía
    assert len(densificador.resultado()["p_w"]) == 0
    densificador.terminar()
    r, referencia = densificador.resultado(), sesion["de_una_vez"]
    assert len(r["p_w"]) == len(referencia["p_w"])
    assert ordenados(r["p_w"]) == pytest.approx(ordenados(referencia["p_w"]), abs=1e-9)
    assert r["por_vista"] == referencia["por_vista"]


def test_con_la_ventana_de_verdad(sesion):
    densificador = dd.DensificadorEnDirecto(CFG, CFG_DIRECTO)
    retraso, ventana = CFG_DIRECTO["retraso"], CFG_DIRECTO["ventana"]
    entradas = [densificador.nueva_vista(copia(vista)) for vista in sesion["vistas"]]
    # Las primeras esperan; después entra una por cada una que llega
    assert entradas[:retraso] == [0] * retraso and entradas[retraso:] == [1] * (len(entradas) - retraso)
    assert densificador.n_cerradas == len(sesion["vistas"]) - retraso
    a_medias = len(densificador.resultado()["p_w"])
    densificador.terminar()
    r = densificador.resultado()
    assert densificador.n_cerradas == len(sesion["vistas"]) == len(r["por_vista"]) and len(r["p_w"]) > a_medias
    # La nube es casi la misma que con todos los keyframes a la vez, e igual de buena
    referencia = sesion["de_una_vez"]
    assert len(r["p_w"]) == pytest.approx(len(referencia["p_w"]), rel=0.1)
    evaluaciones = [den.evaluar_nube(nube["p_w"] @ sesion["R"].T + sesion["t"], sesion["dimensiones"], CFG["evaluacion"])
                    for nube in (r, referencia)]
    assert evaluaciones[0]["error_mediano"] < evaluaciones[1]["error_mediano"] + 0.005
    assert evaluaciones[0]["cubierto"] > 0.9 * evaluaciones[1]["cubierto"]
    # La memoria no crece: de las vistas viejas solo queda el keyframe, y de las que ya han
    # entrado no se guardan los puntos
    assert densificador.vistas[0]["z"] is None and densificador.vistas[0]["kf"] is sesion["vistas"][0]["kf"]
    assert densificador.vistas[-1]["z"] is not None and all(p is None for p in densificador.pendientes)
    assert sum(v["z"] is not None for v in densificador.vistas) <= ventana + 1


# =====================================================================
# Paso 30.4 — La sesión: paquetes de la Pi -> nube densa
# =====================================================================
def test_sesion_densa_con_los_paquetes_de_un_archivo(sesion, tmp_path, capsys):
    profundidad_de = pc_denso.profundidad_verdadera(sesion["carpeta"], 0)
    densa = pc_denso.SesionDensa(tmp_path, None, profundidad_de, CFG, CFG_DIRECTO)
    for datos in sesion["paquetes"]:
        densa.procesar(datos)
    n = len(sesion["vistas"])
    assert len(densa.camaras) == n and densa.sin_profundidad == 0
    assert f"keyframes {n:3d}" in densa.resumen() and "ms por keyframe" in densa.resumen()
    assert "keyframes   5" in capsys.readouterr().out            # ha ido avisando por el camino
    assert (tmp_path / "nube_densa_directo.png").exists()        # y dibujando
    densa.cerrar()
    assert densa.guardadas == [(tmp_path, len(densa.denso.resultado()["p_w"]))]
    p, colores = leer_ply(tmp_path / "nube_densa_directo.ply")
    assert len(p) > 5000 and (colores[:, 0] == colores[:, 1]).all()
    assert (tmp_path / "keyframes.csv").exists() and (tmp_path / "puntos_3d.ply").exists()     # lo de pc_mapa, también
    # Frente a la habitación de verdad, como la de la etapa 26
    linea = pc_denso.comparar_con_la_verdad(densa, sesion["carpeta"])
    assert linea.startswith("Frente a la habitación de verdad") and pc_denso.comparar_con_la_verdad(densa, None) is None


def test_sesion_densa_sin_profundidad_y_con_prueba_nueva(sesion, tmp_path):
    # Una "red" que no consigue poner en metros uno de cada tres keyframes
    verdadera = pc_denso.profundidad_verdadera(sesion["carpeta"], 0)
    llamadas = []

    def profundidad_de(kf, gris):
        llamadas.append(kf["indice"])
        return None if len(llamadas) % 3 == 0 else verdadera(kf, gris)

    densa = pc_denso.SesionDensa(tmp_path, None, profundidad_de, CFG, CFG_DIRECTO)
    keyframes = [d for d in sesion["paquetes"] if d[0] == protocolo.VERSION_KEYFRAME]
    for datos in keyframes[:6]:
        densa.procesar(datos)
    assert densa.sin_profundidad == 2 and len(densa.camaras) == 4
    # "Nueva prueba" en el visor: la nube de la primera se guarda y empieza otra, en otra carpeta
    assert densa.procesar(protocolo.empaquetar_prueba(123, 2)) == "prueba"
    assert densa.sin_profundidad == 0 and len(densa.camaras) == 0 and densa.mapa.n == 2
    for datos in keyframes[6:]:
        densa.procesar(datos)
    densa.cerrar()
    assert [carpeta for carpeta, _ in densa.guardadas] == [tmp_path, tmp_path / "prueba_2"]
    assert (tmp_path / "nube_densa_directo.ply").exists() and (tmp_path / "prueba_2" / "nube_densa_directo.ply").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT a otro proceso: solo en Linux")
def test_pc_denso_en_directo(sesion, tmp_path):
    # pc_denso escuchando; le llegan los paquetes por UDP; Ctrl+C (SIGINT) -> guarda la nube
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as libre:
        libre.bind(("127.0.0.1", 0))
        puerto = libre.getsockname()[1]
    proceso = subprocess.Popen(
        [sys.executable, "-m", "apps.pc_denso", "--puerto", str(puerto), "--salida", str(tmp_path),
         "--con-verdad", "--grabacion", str(sesion["carpeta"]), "--desfase-ns", "0"],
        cwd=RAIZ_PROYECTO, env=dict(os.environ, PYTHONUNBUFFERED="1"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    lineas = queue.Queue()
    threading.Thread(target=lambda: [lineas.put(linea) for linea in proceso.stdout], daemon=True).start()

    def esperar(texto, segundos):
        """Lee la salida del proceso hasta que aparece `texto`. Devuelve todo lo leído."""
        leido, limite = "", time.monotonic() + segundos
        while texto not in leido:
            leido += lineas.get(timeout=max(limite - time.monotonic(), 0.1))
        return leido

    try:
        esperar("Escuchando UDP", 60)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as emisor:
            for datos in sesion["paquetes"]:
                emisor.sendto(datos, ("127.0.0.1", puerto))
                time.sleep(0.01)
        esperar("keyframes  10", 120)                    # ha recibido y procesado 10 keyframes
        proceso.send_signal(signal.SIGINT)
        salida = esperar("Nube densa de", 60)
        proceso.wait(timeout=60)
    finally:
        if proceso.poll() is None:
            proceso.kill()
    assert "Terminado" in salida and proceso.returncode == 0
    assert len(leer_ply(tmp_path / "nube_densa_directo.ply")[0]) > 5000
    assert (tmp_path / "nube_densa_directo.png").exists() and (tmp_path / "mapa.paquetes").exists()
