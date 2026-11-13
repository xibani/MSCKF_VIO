"""tests/test_rechazo_giro.py — Etapa 23: rechazo geométrico con el giro del giroscopio.

Todo sin redes (pasa también en la Pi): la función rechazo_con_giro con dos vistas
sintéticas, el giro de la cámara que calcula el pipeline, y el front-end y el VIO completo
con el emparejador de mentira de la etapa 22.
"""

import copy

import numpy as np

from test_frontend_lightglue import EmparejadorDeMentira, identidad_de_cada_pista, vistas_de_ventana
from vio import frontend as fe
from vio import frontend_lightglue as fel
from vio import modelo_camara as mc
from vio import simulador as sim
from vio import so3
from vio.config import cargar_config
from vio.pipeline import PipelineVio, giro_camara

CFG = cargar_config("lightglue")
CFG_GIRO = CFG["giro"]
CFG_SIM = cargar_config("simulador")["vio"]
FOCAL = 643.0
K = mc.matriz_K(FOCAL, FOCAL, 320.0, 240.0)
SIN_DIST = np.zeros(5)
DIST = np.array([0.168, -0.59, -0.0073, 0.0004, 0.955])
TAMANO = (640, 480)
PASO_NS = 50_000_000
GIRO_Y_DESPLAZAMIENTO = (so3.exp_so3([0.02, -0.05, 0.01]), np.array([0.3, 0.05, 0.0]))   # ~3° y 30 cm
SOLO_GIRO = (so3.exp_so3([0.0, np.radians(15.0), 0.0]), np.zeros(3))                     # 15° sin moverse
QUIETA = (np.eye(3), np.zeros(3))


def dos_vistas(n, n_malos, movimiento, semilla=0, ruido_px=0.5):
    """Puntos 3D vistos desde dos posiciones de cámara, con ruido y n_malos parejas falsas.

    movimiento: (R, t) con p_actual = R·p_anterior + t.
    Devuelve: (xy_ant (n, 2), xy_act (n, 2), malos (n_malos,)) en coordenadas normalizadas.
    """
    R, t = movimiento
    rng = np.random.default_rng(semilla)
    P = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), rng.uniform(3, 8, n)])
    P2 = P @ R.T + t
    xy_ant = P[:, :2] / P[:, 2:3]
    xy_act = P2[:, :2] / P2[:, 2:3] + rng.normal(0.0, ruido_px / FOCAL, (n, 2))
    malos = rng.choice(n, n_malos, replace=False)
    xy_act[malos] += rng.uniform(-0.3, 0.3, (n_malos, 2))          # hasta unos 190 px de error
    return xy_ant, xy_act, malos


def con_giro(xy_ant, xy_act, R):
    return fel.rechazo_con_giro(xy_ant, xy_act, R, CFG_GIRO, FOCAL, np.random.default_rng(0))


def con_esencial(xy_ant, xy_act):
    desplazamiento = np.linalg.norm(xy_act - xy_ant, axis=1) * FOCAL
    return fe.rechazo_geometrico(xy_ant, xy_act, desplazamiento, CFG["geometria"], FOCAL)


# =====================================================================
# Paso 23.2
# =====================================================================
def test_encuentra_las_parejas_falsas():
    # El caso normal, el del test de la etapa 8: los dos métodos hacen lo mismo
    xy_ant, xy_act, malos = dos_vistas(200, 40, GIRO_Y_DESPLAZAMIENTO)
    encajan = con_giro(xy_ant, xy_act, GIRO_Y_DESPLAZAMIENTO[0])
    buenos = np.setdiff1d(np.arange(200), malos)
    assert encajan.dtype == bool and encajan.shape == (200,)
    assert encajan[buenos].mean() > 0.98                          # casi ningún bueno rechazado
    assert (~encajan[malos]).mean() > 0.9                         # casi todos los malos fuera


def test_camara_quieta_o_que_solo_gira():
    # Donde la matriz esencial no puede: sin desplazamiento no hay rectas epipolares
    for movimiento in (QUIETA, SOLO_GIRO):
        xy_ant, xy_act, malos = dos_vistas(150, 30, movimiento)
        encajan = con_giro(xy_ant, xy_act, movimiento[0])
        buenos = np.setdiff1d(np.arange(150), malos)
        assert encajan[buenos].mean() > 0.98 and not encajan[malos].any()
    # Con la cámara quieta el rechazo de la etapa 8 ni lo intenta: lo da todo por bueno
    xy_ant, xy_act, malos = dos_vistas(150, 30, QUIETA)
    assert con_esencial(xy_ant, xy_act).all()


def test_con_pocas_parejas():
    # 6 parejas, una falsa: a la matriz esencial le hacen falta 8 para empezar
    falsas_vistas = 0
    for semilla in range(10):
        xy_ant, xy_act, malos = dos_vistas(6, 1, GIRO_Y_DESPLAZAMIENTO, semilla)
        encajan = con_giro(xy_ant, xy_act, GIRO_Y_DESPLAZAMIENTO[0])
        buenos = np.setdiff1d(np.arange(6), malos)
        assert encajan[buenos].all()
        falsas_vistas += int(not encajan[malos[0]])
        assert con_esencial(xy_ant, xy_act).all()                 # etapa 8: sin opinión
    assert falsas_vistas >= 8                                     # alguna cae sobre su recta


def test_con_muchas_parejas_falsas():
    # Tres de cada cuatro parejas falsas: con 2 parejas por prueba es fácil dar con 2 buenas
    xy_ant, xy_act, malos = dos_vistas(200, 150, GIRO_Y_DESPLAZAMIENTO)
    encajan = con_giro(xy_ant, xy_act, GIRO_Y_DESPLAZAMIENTO[0])
    buenos = np.setdiff1d(np.arange(200), malos)
    assert encajan[buenos].mean() > 0.95 and (~encajan[malos]).mean() > 0.9


def test_si_el_giro_esta_mal_se_pierden_parejas_buenas():
    # El punto débil del método: depende de que el giro que se le da sea el verdadero.
    # (Con datos reales: de la extrínseca cámara-IMU y del desfase de tiempo, etapa 7.4)
    xy_ant, xy_act, malos = dos_vistas(200, 40, GIRO_Y_DESPLAZAMIENTO)
    buenos = np.setdiff1d(np.arange(200), malos)
    R = GIRO_Y_DESPLAZAMIENTO[0]

    def con_error(grados):
        return R @ so3.exp_so3(np.radians(grados) * np.array([0.6, 0.8, 0.0]))
    assert con_giro(xy_ant, xy_act, con_error(0.1))[buenos].mean() > 0.97      # 0,1°: como si nada
    assert con_giro(xy_ant, xy_act, con_error(1.0))[buenos].mean() < 0.8       # 1°: se pierden muchas


def test_casos_limite():
    sin_nada = con_giro(np.zeros((0, 2)), np.zeros((0, 2)), np.eye(3))
    assert sin_nada.shape == (0,) and sin_nada.dtype == bool
    # Una sola pareja: no se puede buscar traslación; vale si el giro solo la explica
    una = np.array([[0.1, 0.05]])
    assert con_giro(una, una, np.eye(3)).tolist() == [True]
    assert con_giro(una, una + 0.1, np.eye(3)).tolist() == [False]              # 64 px sin motivo


# =====================================================================
# Paso 23.3 — El giro de la cámara, en el pipeline
# =====================================================================
def test_giro_camara_con_la_verdad_del_simulador():
    cfg = copy.deepcopy(CFG_SIM)
    cfg["trayectoria"]["duracion_s"] = 2.0
    s = sim.generar_secuencia(cfg, K, TAMANO, np.random.default_rng(0))
    R_w_i = s["verdad"]["R_w_i"][s["camaras"]["indice_imu"]]      # la IMU en el instante de cada imagen
    for j in (1, 10, 25):
        R = giro_camara(R_w_i[j - 1], R_w_i[j], s["R_c_i"])
        # Lo mismo, con las poses verdaderas de la cámara: de mundo a cámara j, deshaciendo j-1
        assert np.allclose(R, s["camaras"]["R_c_w"][j] @ s["camaras"]["R_c_w"][j - 1].T, atol=1e-12)
    # Y lo que significa: un punto muy lejano (solo le afecta el giro) se ve donde dice R
    j = 10
    lejano_w = np.array([1.0e6, 2.0e5, 3.0e5])
    d_ant = s["camaras"]["R_c_w"][j - 1] @ lejano_w
    d_act = s["camaras"]["R_c_w"][j] @ lejano_w
    R = giro_camara(R_w_i[j - 1], R_w_i[j], s["R_c_i"])
    assert np.allclose(R @ d_ant / np.linalg.norm(d_ant), d_act / np.linalg.norm(d_act), atol=1e-9)


# =====================================================================
# Paso 23.4 — El front-end con el giro
# =====================================================================
def frontend_con_parejas_cruzadas(metodo):
    """Cámara quieta, 5 imágenes normales y, antes de la sexta, 12 pistas con la pareja cruzada."""
    vistas, puntuacion = vistas_de_ventana(8, 0)
    falso = EmparejadorDeMentira(vistas, puntuacion)
    cfg = copy.deepcopy(CFG)
    cfg["geometria"]["metodo"] = metodo
    frontend = fel.FrontEndLightGlue(K, SIN_DIST, cfg, falso)
    gris = np.zeros((480, 640), np.uint8)
    for k in range(5):
        frontend.procesar(gris, k * PASO_NS, np.eye(3))
    seguidos = identidad_de_cada_pista(frontend).tolist()
    for p, q in zip(seguidos[:6], seguidos[6:12]):
        falso.cambiadas[p], falso.cambiadas[q] = q, p
    return frontend, gris


def test_frontend_usa_el_giro_si_se_lo_dan():
    assert fel.FrontEndLightGlue.usa_giro and not getattr(fe.FrontEnd, "usa_giro", False)
    # Con el giro (aquí, ninguno: la cámara está quieta) las 12 parejas cruzadas se ven
    frontend, gris = frontend_con_parejas_cruzadas("giro")
    est = frontend.procesar(gris, 5 * PASO_NS, np.eye(3))["estadisticas"]
    assert est["con_giro"] == 1 and est["rechazados_ransac"] == 12
    # Sin giro (None) se usa la matriz esencial, que con la cámara quieta no descarta nada
    frontend, gris = frontend_con_parejas_cruzadas("giro")
    est = frontend.procesar(gris, 5 * PASO_NS)["estadisticas"]
    assert est["con_giro"] == 0 and est["rechazados_ransac"] == 0
    # Y si la configuración pide "esencial", el giro que llegue no se usa
    frontend, gris = frontend_con_parejas_cruzadas("esencial")
    est = frontend.procesar(gris, 5 * PASO_NS, np.eye(3))["estadisticas"]
    assert est["con_giro"] == 0 and est["rechazados_ransac"] == 0


def test_vio_con_el_rechazo_del_giroscopio():
    # El VIO completo en el caso difícil: una imagen de cada 6 (3,3 por segundo), puntos en
    # píxeles enteros y una de cada cinco parejas falsa. El giro lo saca el pipeline de la IMU
    from herramientas.comparar_trayectoria import alinear

    cada = 6
    cfgs = {n: cargar_config(n) for n in ("vio", "frontend", "msckf", "triangulacion")}
    ext = cargar_config("camara_imu")
    cfg_vio = copy.deepcopy(CFG_SIM)
    cfg_vio["trayectoria"]["duracion_s"] = 10.0
    cfg_vio["camara"].update(R_i_c=ext["R_i_c"], p_i_c_m=ext["p_i_c_m"], ruido_px=0.0)
    graba = cfg_vio["grabacion_sintetica"]

    def pose(t):
        return sim.pose_con_reposo(t, cfg_vio["trayectoria"], graba["reposo_s"], graba["rampa_s"])
    s = sim.generar_secuencia(cfg_vio, K, TAMANO, np.random.default_rng(3), pose=pose, dist=DIST)

    rng = np.random.default_rng(4)
    vistas = []
    for j in range(len(s["camaras"]["t_ns"])):
        en_imagen = s["obs"]["imagen"] == j
        vistas.append((s["obs"]["id"][en_imagen], np.round(s["obs"]["uv"][en_imagen])))
    falso = EmparejadorDeMentira(vistas, rng.uniform(0.1, 1.0, len(s["puntos"])))

    def emparejar_con_fallos(a, b):
        indices, confianza, tiempo_ms = EmparejadorDeMentira.emparejar(falso, a, b)
        cuales = rng.choice(len(indices), len(indices) // 5, replace=False)
        indices[cuales, 1] = indices[rng.permutation(cuales), 1]
        return indices, confianza, tiempo_ms
    falso.emparejar = emparejar_con_fallos

    cfg = copy.deepcopy(CFG)
    cfg["geometria"]["metodo"] = "giro"
    frontend = fel.FrontEndLightGlue(K, DIST, cfg, falso)
    pipeline = PipelineVio(K, DIST, s["R_c_i"], s["t_c_i"], ext["t_d_ns"], cfgs, frontend=frontend)
    gris = np.zeros((480, 640), np.uint8)
    resultados, pistas = [], []
    i = 0
    for j, t_cam in enumerate(s["camaras"]["t_ns"]):
        while i < len(s["imu"]["t_ns"]) and s["imu"]["t_ns"][i] <= t_cam:
            pipeline.nueva_imu(int(s["imu"]["t_ns"][i]), s["imu"]["w"][i], s["imu"]["a"][i])
            i += 1
        if j % cada != 0:
            continue                                   # imagen saltada: la IMU sí se ha usado
        falso.siguiente = j
        r = pipeline.nueva_imagen(int(t_cam), gris)
        if r is not None:
            resultados.append(r)
            pistas += r["pistas_terminadas"]

    assert len(resultados) >= 25                                  # 8 s con pose, a 3,3 por segundo
    # En todas las imágenes menos la primera, el rechazo ha usado el giro
    assert sum(r["frontend"]["con_giro"] for r in resultados) == len(resultados) - 1
    t_est = np.array([r["t_ns"] for r in resultados])
    p_est = np.array([r["p_w"] for r in resultados])
    R_est = np.array([r["R_w_i"] for r in resultados])
    p_ver, _ = alinear(t_est, p_est, R_est, s["verdad"]["t_ns"], s["verdad"]["p_w"], s["verdad"]["R_w_i"])
    error = np.linalg.norm(p_est - p_ver, axis=1)
    assert error.max() < 0.15 and error[-1] < 0.1                 # los límites de siempre
    assert sum(r["frontend"]["rechazados_ransac"] for r in resultados) > 200
    assert sum(r["msckf"]["usadas"] for r in resultados) > 100
    assert len(pistas) > 0 and all(p.longitud() >= 1 for p in pistas)
