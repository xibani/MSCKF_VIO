"""tests/test_grafo.py — Etapa 29: grafo de poses.

Ninguno necesita PyTorch: pasan también en la Pi. Dos grupos:
  - Con poses hechas a mano: una cámara que da dos vueltas a un círculo. Las medidas se
    sacan de las poses verdaderas, así que se sabe adónde tiene que llegar el grafo.
  - Con la sesión simulada: los bucles de verdad de la etapa 28 (ORB) y poses estropeadas.
"""

import copy

import numpy as np
import pytest

from herramientas import buscar_bucles
from herramientas import cerrar_bucles
from herramientas import medir_mapa
from herramientas import nube_densa
from herramientas import ver_planos
from vio import bucles as bu
from vio import densificar as den
from vio import grafo as gr
from vio import so3
from vio.config import cargar_config
from vio.emparejador_orb import EmparejadorOrb

CFG = cargar_config("grafo")
ABAJO = np.array([0.0, 0.0, -1.0])


def poses_en_circulo(n=40, por_vuelta=20, radio=1.5):
    """n poses de una cámara que recorre un círculo mirando hacia fuera, subiendo y bajando
    un poco: la pose k + por_vuelta está en el mismo sitio (en planta) que la k."""
    poses = []
    for k in range(n):
        a = 2.0 * np.pi * k / por_vuelta
        adelante = np.array([np.cos(a), np.sin(a), 0.0])
        R = np.column_stack([np.cross(ABAJO, adelante), ABAJO, adelante])
        poses.append((R, np.array([radio * np.cos(a), radio * np.sin(a), 1.5 + 0.2 * np.sin(0.7 * k)])))
    return poses


def bucles_exactos(poses, por_vuelta=20):
    """Un bucle entre cada pose de la segunda vuelta y la de una vuelta antes, con la pose
    relativa verdadera."""
    bucles = []
    for j in range(por_vuelta, len(poses)):
        R, p = bu.pose_relativa(*poses[j - por_vuelta], *poses[j])
        bucles.append({"i": j - por_vuelta, "j": j, "R": R, "p": p})
    return bucles


def error_medio(poses, verdad):
    """Distancia media entre las posiciones de dos listas de poses [m]."""
    return float(np.mean([np.linalg.norm(p - p_ver) for (_, p), (_, p_ver) in zip(poses, verdad)]))


# =====================================================================
# Pasos 29.1 y 29.2 — Aristas, residuo y jacobianos
# =====================================================================
def test_aristas_y_residuo():
    poses = poses_en_circulo()
    odometria = gr.aristas_de_odometria(poses, CFG)
    assert len(odometria) == 39 and [(a["i"], a["j"]) for a in odometria[:2]] == [(0, 1), (1, 2)]
    assert not odometria[0]["bucle"] and odometria[0]["sigma_m"] == CFG["odometria"]["sigma_m"]
    assert odometria[0]["sigma_rad"] == pytest.approx(np.radians(CFG["odometria"]["sigma_grados"]))
    bucles = gr.aristas_de_bucles(bucles_exactos(poses), CFG)
    assert len(bucles) == 20 and bucles[0]["bucle"] and (bucles[0]["i"], bucles[0]["j"]) == (0, 20)
    # Con las poses de las que salieron las medidas, todos los residuos son cero
    for a in odometria + bucles:
        assert gr.residuo(poses[a["i"]], poses[a["j"]], a) == pytest.approx(np.zeros(6), abs=1e-9)
    assert gr.coste(poses, odometria + bucles, CFG) == pytest.approx(0.0, abs=1e-12)
    # Moviendo el segundo nudo 5 cm y girándolo 2°: el residuo, en sigmas
    a = odometria[3]
    R_j, p_j = poses[4]
    movida = (R_j @ so3.exp_so3(np.radians([0.0, 2.0, 0.0])), p_j + [0.03, 0.0, 0.04])
    r = gr.residuo(poses[3], movida, a)
    assert np.linalg.norm(r[0:3]) == pytest.approx(2.0 / CFG["odometria"]["sigma_grados"])
    assert np.linalg.norm(r[3:6]) == pytest.approx(0.05 / CFG["odometria"]["sigma_m"])


def test_jacobianos_frente_a_diferencias_finitas():
    # Los jacobianos dicen cuánto cambia el residuo al mover un poco cada nudo. Se comprueba
    # moviéndolos de verdad una cantidad muy pequeña, de uno en uno de los 6 números
    poses = poses_en_circulo()
    generador = np.random.default_rng(0)
    for a in (gr.aristas_de_odometria(poses, CFG)[5], gr.aristas_de_bucles(bucles_exactos(poses), CFG)[7]):
        # Poses algo apartadas de la medida (1 cm y 0,3°), para que el residuo no sea cero
        pose_i = (poses[a["i"]][0] @ so3.exp_so3(generador.normal(0.0, 0.005, 3)), poses[a["i"]][1] + generador.normal(0.0, 0.01, 3))
        pose_j = (poses[a["j"]][0] @ so3.exp_so3(generador.normal(0.0, 0.005, 3)), poses[a["j"]][1] + generador.normal(0.0, 0.01, 3))
        J_i, J_j = gr.jacobianos(pose_i, pose_j, a)
        r = gr.residuo(pose_i, pose_j, a)
        paso = 1e-6
        for columna in range(6):
            d = np.zeros(6)
            d[columna] = paso
            movida_i = (pose_i[0] @ so3.exp_so3(d[0:3]), pose_i[1] + d[3:6])
            movida_j = (pose_j[0] @ so3.exp_so3(d[0:3]), pose_j[1] + d[3:6])
            numerico_i = (gr.residuo(movida_i, pose_j, a) - r) / paso
            numerico_j = (gr.residuo(pose_i, movida_j, a) - r) / paso
            # Los bloques de posición son exactos; los de giro, aproximados (aquí, al 1 %)
            escala = max(np.abs(J_i[:, columna]).max(), 1.0)
            assert J_i[:, columna] == pytest.approx(numerico_i, abs=0.01 * escala)
            assert J_j[:, columna] == pytest.approx(numerico_j, abs=0.01 * escala)


def test_peso_de_huber():
    arista_bucle = {"bucle": True}
    arista_odometria = {"bucle": False}
    k = CFG["bucle"]["huber"]
    assert gr.peso_de_huber(np.full(6, 0.1), arista_bucle, CFG) == 1.0
    grande = np.array([4.0 * k, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert gr.peso_de_huber(grande, arista_bucle, CFG) == pytest.approx(0.25)
    assert gr.peso_de_huber(grande, arista_odometria, CFG) == 1.0


# =====================================================================
# Paso 29.3 — Gauss-Newton
# =====================================================================
def test_optimizar_vuelve_a_las_poses_verdaderas():
    # Todas las medidas son exactas (salen de las poses verdaderas); se empieza con las
    # poses movidas al azar (5 cm y 2°): tiene que volver a las verdaderas
    verdad = poses_en_circulo()
    aristas = gr.aristas_de_odometria(verdad, CFG) + gr.aristas_de_bucles(bucles_exactos(verdad), CFG)
    generador = np.random.default_rng(1)
    inicio = [verdad[0]] + [(R @ so3.exp_so3(generador.normal(0.0, 0.035, 3)), p + generador.normal(0.0, 0.05, 3))
                            for R, p in verdad[1:]]
    assert error_medio(inicio, verdad) > 0.05
    poses, historia = gr.optimizar(inicio, aristas, CFG)
    assert historia[-1] < 1e-9 * historia[0] and all(b < a for a, b in zip(historia[:-1], historia[1:]))
    for (R, p), (R_ver, p_ver) in zip(poses, verdad):
        assert p == pytest.approx(p_ver, abs=1e-6) and so3.angulo_entre(R, R_ver) < 1e-6
    # El primer nudo es el ancla: no se ha movido
    assert (poses[0][1] == verdad[0][1]).all() and (poses[0][0] == verdad[0][0]).all()


def test_los_bucles_corrigen_la_deriva():
    verdad = poses_en_circulo()
    bucles = bucles_exactos(verdad)
    con_deriva = gr.anadir_deriva(verdad, 0.5, 1.02)     # medio grado de más y un 2 % largo en cada paso
    aristas = gr.aristas_de_odometria(con_deriva, CFG) + gr.aristas_de_bucles(bucles, CFG)
    antes = gr.residuos_de_bucles(con_deriva, aristas)
    poses, historia = gr.optimizar(con_deriva, aristas, CFG)
    despues = gr.residuos_de_bucles(poses, aristas)
    assert antes.shape == (20, 2) and np.median(antes[:, 0]) > 0.15 and np.median(antes[:, 1]) > 5.0
    # Los keyframes pasan a cumplir los bucles, y el error frente a la verdad baja mucho
    assert np.median(despues[:, 0]) < 0.02 and np.median(despues[:, 1]) < 0.5
    assert error_medio(poses, verdad) < 0.35 * error_medio(con_deriva, verdad)
    assert historia[-1] < 0.2 * historia[0]
    # Sin bucles no hay nada que corregir: la odometría se cumple sola y las poses no cambian
    solo_odometria = gr.aristas_de_odometria(con_deriva, CFG)
    iguales, historia = gr.optimizar(con_deriva, solo_odometria, CFG)
    assert historia[-1] < 1e-20 and error_medio(iguales, con_deriva) < 1e-9


def test_un_bucle_falso_no_lo_estropea_todo():
    verdad = poses_en_circulo()
    bucles = bucles_exactos(verdad)
    con_deriva = gr.anadir_deriva(verdad, 0.5, 1.02)
    falso = {"i": 3, "j": 30, "R": np.eye(3), "p": np.array([0.0, 0.0, 0.3])}      # no tiene nada que ver
    aristas = gr.aristas_de_odometria(con_deriva, CFG) + gr.aristas_de_bucles(bucles + [falso], CFG)
    sin_el_falso = gr.optimizar(con_deriva, aristas[:-1], CFG)[0]
    con_huber = gr.optimizar(con_deriva, aristas, CFG)[0]
    cfg_sin_huber = copy.deepcopy(CFG)
    cfg_sin_huber["bucle"]["huber"] = 1e9                # mínimos cuadrados de siempre
    sin_huber = gr.optimizar(con_deriva, aristas, cfg_sin_huber)[0]
    # Con Huber, el resultado es casi el mismo que si el bucle falso no estuviera
    assert error_medio(con_huber, sin_el_falso) < 0.03
    assert error_medio(sin_huber, sin_el_falso) > 2.0 * error_medio(con_huber, sin_el_falso)


# =====================================================================
# Pasos 29.4 y 29.5 — Deriva de prueba y medidas
# =====================================================================
def test_anadir_deriva():
    verdad = poses_en_circulo()
    con_deriva = gr.anadir_deriva(verdad, 0.5, 1.02)
    assert len(con_deriva) == len(verdad) and con_deriva[0][1] == pytest.approx(verdad[0][1])
    # Cada paso mide un 2 % más, y el rumbo se va medio grado más en cada uno
    for k in (0, 10, 38):
        largo = np.linalg.norm(verdad[k + 1][1] - verdad[k][1])
        assert np.linalg.norm(con_deriva[k + 1][1] - con_deriva[k][1]) == pytest.approx(1.02 * largo)
    assert np.degrees(so3.angulo_entre(con_deriva[20][0], verdad[20][0])) == pytest.approx(10.0, abs=0.01)
    # Sin deriva, las mismas poses
    iguales = gr.anadir_deriva(verdad, 0.0, 1.0)
    assert error_medio(iguales, verdad) < 1e-12
    # errores_de_poses: en el mismo mundo (R = identidad, t = 0), la distancia y el ángulo
    errores = gr.errores_de_poses(con_deriva, verdad, np.eye(3), np.zeros(3))
    assert errores.shape == (40, 2) and errores[0] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert errores[20, 1] == pytest.approx(10.0, abs=0.01) and errores[:, 0].max() > 0.3


# =====================================================================
# Con la sesión simulada: bucles de verdad (ORB) y poses estropeadas
# =====================================================================
@pytest.fixture(scope="module")
def sesion(sesion_con_textura):
    """Las vistas de la sesión de la etapa 24, sus poses verdaderas y sus bucles (como en
    tests/test_bucles.py: entre cada keyframe y los uno o dos anteriores).

    cfg: el grafo con MÁS confianza en los bucles que en la odometría. Aquí los "bucles" unen
    keyframes vecinos y compiten con la odometría del mismo tramo, que es la estropeada: para
    que la corrijan tienen que mandar ellos. (En una sesión real es al revés: el bucle une
    keyframes lejanos, y gana porque la odometría acumula el error de muchos pasos.)
    """
    carpeta = sesion_con_textura["carpeta"]
    mapa = medir_mapa.leer_mapa(carpeta / "mapa.paquetes")
    vistas = nube_densa.cargar_vistas(mapa.keyframes, carpeta, True, sesion_con_textura["t_d_ns"])
    s = sesion_con_textura["s"]
    cfg_ext = {"R_i_c": s["R_c_i"].T, "p_i_c_m": -s["R_c_i"].T @ s["t_c_i"]}
    cfg_bucles = cargar_config("bucles")
    cfg_bucles["candidatos"].update(separacion_min=0, distancia_max_m=1.5, angulo_max_grados=40.0)
    bucles, _ = bu.buscar_bucles(vistas, EmparejadorOrb(cfg_bucles["orb"]), cfg_bucles)
    R, t = nube_densa.transformacion_de_la_sesion(carpeta)
    cfg = copy.deepcopy(CFG)
    cfg["bucle"].update(sigma_m=0.003, sigma_grados=0.05, huber=1e9)
    return {"vistas": vistas, "bucles": bucles, "R": R, "t": t, "cfg": cfg,
            "poses_verdad": buscar_bucles.verdad_de_la_sesion(carpeta, vistas, cfg_ext),
            "dimensiones": cargar_config("simulador")["vio"]["mundo"]["habitacion_m"]}


def test_el_grafo_corrige_la_sesion_estropeada(sesion):
    vistas = sesion["vistas"]
    antes = gr.anadir_deriva([(v["kf"]["R_w_c"], v["kf"]["p_w_c"]) for v in vistas], 1.0, 1.05)
    cfg = sesion["cfg"]
    aristas = gr.aristas_de_odometria(antes, cfg) + gr.aristas_de_bucles(sesion["bucles"], cfg)
    despues, historia = gr.optimizar(antes, aristas, cfg)
    errores_antes = gr.errores_de_poses(antes, sesion["poses_verdad"], sesion["R"], sesion["t"])
    errores_despues = gr.errores_de_poses(despues, sesion["poses_verdad"], sesion["R"], sesion["t"])
    # 1° y un 5 % por paso, en 10 pasos: unos 10° y más de 10 cm al final. Los bucles lo recortan
    assert errores_antes[:, 0].max() > 0.10 and errores_antes[:, 1].max() > 8.0
    assert errores_despues[:, 0].max() < 0.6 * errores_antes[:, 0].max()
    assert errores_despues[:, 1].max() < 0.5 * errores_antes[:, 1].max()
    assert historia[-1] < historia[0]
    # Y la nube densa rehecha con las poses corregidas se parece más a la habitación
    cfg_densificar = cargar_config("densificar")
    nubes = []
    for poses in (antes, despues):
        nube = den.densificar(cerrar_bucles.vistas_con_poses(vistas, poses), cfg_densificar)
        nubes.append(den.evaluar_nube(nube["p_w"] @ sesion["R"].T + sesion["t"], sesion["dimensiones"],
                                      cfg_densificar["evaluacion"]))
    assert nubes[1]["error_mediano"] < 0.7 * nubes[0]["error_mediano"]


def test_la_herramienta_escribe_los_resultados(sesion, tmp_path):
    vistas = sesion["vistas"]
    antes = gr.anadir_deriva([(v["kf"]["R_w_c"], v["kf"]["p_w_c"]) for v in vistas], 1.0, 1.05)
    cfg = sesion["cfg"]
    aristas = gr.aristas_de_odometria(antes, cfg) + gr.aristas_de_bucles(sesion["bucles"], cfg)
    despues, historia = gr.optimizar(antes, aristas, cfg)
    movido = cerrar_bucles.cambios_de_poses(antes, despues)
    assert movido.shape == (len(vistas), 2) and movido[0] == pytest.approx([0.0, 0.0], abs=1e-9) and movido[:, 0].max() > 0.02
    # Las vistas con otras poses: cambia la pose y nada más
    nuevas = cerrar_bucles.vistas_con_poses(vistas, despues)
    assert nuevas[5]["kf"]["p_w_c"] == pytest.approx(despues[5][1]) and nuevas[5]["z"] is vistas[5]["z"]
    assert nuevas[5]["kf"]["indice"] == vistas[5]["kf"]["indice"]
    assert vistas[5]["kf"]["p_w_c"] != pytest.approx(despues[5][1], abs=1e-4)        # la original no se ha tocado
    # El CSV
    cerrar_bucles.guardar_keyframes(tmp_path / "kf.csv", vistas, despues, movido)
    lineas = (tmp_path / "kf.csv").read_text(encoding="utf-8").splitlines()
    assert lineas[0] == cerrar_bucles.CABECERA_CSV.strip() and len(lineas) == len(vistas) + 1
    campos = lineas[6].split(",")
    assert len(campos) == 16 and int(campos[0]) == vistas[5]["kf"]["indice"]
    assert [float(c) for c in campos[2:5]] == pytest.approx(despues[5][1], abs=1e-8)
    # El informe, con verdad y sin ella
    errores_antes = gr.errores_de_poses(antes, sesion["poses_verdad"], sesion["R"], sesion["t"])
    errores_despues = gr.errores_de_poses(despues, sesion["poses_verdad"], sesion["R"], sesion["t"])
    evaluacion = {"n": 1000, "error_mediano": 0.05, "error_p90": 0.1, "cubierto": 0.2}
    texto = cerrar_bucles.informe(len(antes), aristas, historia, gr.residuos_de_bucles(antes, aristas),
                                  gr.residuos_de_bucles(despues, aristas), movido, errores_antes, errores_despues,
                                  evaluacion, evaluacion)
    assert f"{len(sesion['bucles'])} de bucle" in texto and "Error de las poses frente a la verdad" in texto
    assert "Nube densa frente a la habitación" in texto
    corto = cerrar_bucles.informe(len(antes), aristas, historia, gr.residuos_de_bucles(antes, aristas),
                                  gr.residuos_de_bucles(despues, aristas), movido, None, None, None, None)
    assert "frente a la verdad" not in corto and "Lo que se han movido" in corto
    # La figura
    verdad_en_vio = (np.array([p for _, p in sesion["poses_verdad"]]) - sesion["t"]) @ sesion["R"]
    cerrar_bucles.dibujar(tmp_path / "grafo.png", antes, despues, aristas, verdad_en_vio)
    assert (tmp_path / "grafo.png").stat().st_size > 10000
    # ver_planos: el apellido de los resultados sale del nombre de la nube
    assert ver_planos.apellido_de_nube("nube_densa.ply") == ""
    assert ver_planos.apellido_de_nube("nube_densa_verdad.ply") == "_verdad"
    assert ver_planos.apellido_de_nube("nube_densa_corregida.ply") == "_corregida"
    assert ver_planos.apellido_de_nube("otra.ply") == "_otra"
