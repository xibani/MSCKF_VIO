"""tests/test_planos.py — Etapa 27: los planos de la nube densa.

Ninguno necesita PyTorch: pasan también en la Pi. Dos grupos:
  - Con nubes SINTÉTICAS: puntos repartidos a mano sobre las caras de una caja, con ruido y
    puntos sueltos. Ahí se sabe qué planos tienen que salir.
  - Con la nube densa de la sesión simulada (la fixture de la etapa 24).
"""

import numpy as np
import pytest

from herramientas import medir_mapa
from herramientas import nube_densa
from herramientas import ver_planos
from vio import densificar as den
from vio import planos as pl
from vio.config import cargar_config
from vio.reconstruccion import leer_ply

CFG = cargar_config("planos")
CFG_DENSIFICAR = cargar_config("densificar")
DIMENSIONES = (5.0, 4.0, 2.5)                            # una caja distinta de la del simulador
DENTRO = np.array([0.3, -0.2, 1.4])                      # por donde andan las cámaras


def rng():
    return np.random.default_rng(CFG["ransac"]["semilla"])


def puntos_de_cara(cara, n, generador, ruido=0.0):
    """n puntos al azar sobre una cara de la caja DIMENSIONES (numeradas como en vio/escena.py),
    desplazados `ruido` [m] (desviación típica) en perpendicular a la cara."""
    lx, ly, lz = DIMENSIONES
    p = np.column_stack([generador.uniform(-lx / 2, lx / 2, n), generador.uniform(-ly / 2, ly / 2, n),
                         generador.uniform(0.0, lz, n)])
    eje = cara // 2
    valor = [-lx / 2, lx / 2, -ly / 2, ly / 2, 0.0, lz][cara]
    p[:, eje] = valor + generador.normal(0.0, ruido, n) if ruido > 0 else valor
    return p


def nube_de_caja(ruido=0.01, sueltos=300, semilla=0):
    """Una habitación de mentira: 6 caras con puntos (más en las paredes grandes) y unos
    cuantos puntos sueltos por el interior."""
    generador = np.random.default_rng(semilla)
    por_cara = (3000, 3000, 4000, 4000, 2500, 2500)
    p = [puntos_de_cara(cara, n, generador, ruido) for cara, n in enumerate(por_cara)]
    lx, ly, lz = DIMENSIONES
    p.append(np.column_stack([generador.uniform(-lx / 2, lx / 2, sueltos), generador.uniform(-ly / 2, ly / 2, sueltos),
                              generador.uniform(0.0, lz, sueltos)]))
    return np.vstack(p)


# =====================================================================
# Paso 27.1 — Un plano
# =====================================================================
def test_plano_por_tres_puntos():
    # Tres puntos del plano z = 2
    n, d = pl.plano_por_tres_puntos(np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    assert n == pytest.approx([0.0, 0.0, 1.0]) and d == pytest.approx(2.0)
    # Un plano inclinado: x + y + z = 3 -> n = (1, 1, 1)/√3, d = 3/√3
    n, d = pl.plano_por_tres_puntos(np.array([3.0, 0.0, 0.0]), np.array([0.0, 3.0, 0.0]), np.array([0.0, 0.0, 3.0]))
    assert n == pytest.approx(np.ones(3) / np.sqrt(3)) and d == pytest.approx(np.sqrt(3))
    assert pl.distancia_al_plano(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]), n, d) == \
        pytest.approx([0.0, -np.sqrt(3), np.sqrt(3)])
    # Tres puntos en línea no definen un plano
    assert pl.plano_por_tres_puntos(np.zeros(3), np.ones(3), 2 * np.ones(3)) is None


def test_ajustar_plano():
    # 2000 puntos de la pared x = 2,5 con 2 cm de ruido: el plano sale mucho mejor que cada punto
    p = puntos_de_cara(1, 2000, np.random.default_rng(0), ruido=0.02)
    n, d, ejes = pl.ajustar_plano(p)
    if n[0] < 0:                                         # n y -n son el mismo plano
        n, d = -n, -d
    assert np.degrees(np.arccos(n[0])) < 0.1 and d == pytest.approx(2.5, abs=0.002)
    assert np.linalg.norm(n) == pytest.approx(1.0)
    # Los ejes están dentro del plano y son perpendiculares entre sí
    assert ejes @ n == pytest.approx([0.0, 0.0], abs=1e-9) and ejes[0] @ ejes[1] == pytest.approx(0.0, abs=1e-9)


# =====================================================================
# Pasos 27.2 y 27.3 — RANSAC, un plano tras otro
# =====================================================================
def test_mejor_plano_con_puntos_sueltos():
    generador = np.random.default_rng(1)
    pared = puntos_de_cara(3, 1500, generador, ruido=0.01)
    sueltos = generador.uniform(-2.0, 2.0, (1500, 3))    # la mitad de los puntos no son de ningún plano
    n, d, encajan = pl.mejor_plano(np.vstack([pared, sueltos]), CFG["ransac"], rng())
    assert abs(n[1]) > 0.999 and abs(d) == pytest.approx(2.0, abs=0.01)
    assert encajan[:1500].mean() > 0.99 and encajan[1500:].mean() < 0.1
    assert pl.mejor_plano(pared[:2], CFG["ransac"], rng()) is None


def test_extraer_planos_de_una_caja():
    p = nube_de_caja()
    grupos = pl.extraer_planos(p, CFG["ransac"], rng())
    assert len(grupos) == 6
    assert sorted(len(g) for g in grupos) == pytest.approx([2500, 2500, 3000, 3000, 4000, 4000], rel=0.1)
    # Ningún punto está en dos planos, y casi todos los de las caras están en alguno
    todos = np.concatenate(grupos)
    assert len(np.unique(todos)) == len(todos) and len(todos) > 0.97 * 19000
    # Con puntos_min muy alto no sale ninguno
    assert pl.extraer_planos(p, dict(CFG["ransac"], puntos_min=10000), rng()) == []


# =====================================================================
# Paso 27.4 — Juntar los planos que son la misma superficie
# =====================================================================
def test_fundir_planos_de_una_pared_gruesa():
    # La pared x+ con 6 cm de grosor y un umbral de RANSAC de 3 cm: sale partida en láminas
    generador = np.random.default_rng(2)
    gruesa = puntos_de_cara(1, 6000, generador, ruido=0.06)
    enfrente = puntos_de_cara(0, 3000, generador, ruido=0.005)
    al_lado = puntos_de_cara(3, 3000, generador, ruido=0.005)
    p = np.vstack([gruesa, enfrente, al_lado])
    grupos = pl.extraer_planos(p, dict(CFG["ransac"], umbral_m=0.03), rng())
    assert len(grupos) >= 5                              # la gruesa, en 3 láminas o más
    fundidos = pl.fundir_planos(p, grupos, DENTRO, CFG["fusion"])
    assert len(fundidos) == 3
    # La pared gruesa vuelve a ser una, con casi todos sus puntos. Las otras dos siguen
    # aparte (la de al lado se queda con unos pocos puntos de la gruesa, los de la esquina)
    assert (fundidos[0] < 6000).all() and len(fundidos[0]) > 0.9 * 6000
    assert sorted(len(g) for g in fundidos[1:]) == pytest.approx([3000, 3000], abs=100)


def test_son_la_misma_superficie():
    generador = np.random.default_rng(3)
    pared = puntos_de_cara(1, 1000, generador)
    assert pl.son_la_misma_superficie(pared, pared + [0.10, 0.0, 0.0], DENTRO, CFG["fusion"])       # a 10 cm
    assert not pl.son_la_misma_superficie(pared, pared - [0.40, 0.0, 0.0], DENTRO, CFG["fusion"])   # a 40 cm
    assert not pl.son_la_misma_superficie(pared, puntos_de_cara(0, 1000, generador), DENTRO, CFG["fusion"])   # enfrente
    assert not pl.son_la_misma_superficie(pared, puntos_de_cara(3, 1000, generador), DENTRO, CFG["fusion"])   # en esquina


# =====================================================================
# Paso 27.5 — Describir cada plano
# =====================================================================
def test_tipo_de_plano():
    tolerancia = CFG["tipos"]["tolerancia_grados"]
    assert pl.tipo_de_plano(np.array([0.0, 0.0, 1.0]), tolerancia) == "horizontal_arriba"
    assert pl.tipo_de_plano(np.array([0.0, 0.0, -1.0]), tolerancia) == "horizontal_abajo"
    assert pl.tipo_de_plano(np.array([0.6, -0.8, 0.0]), tolerancia) == "pared"
    inclinada = np.array([np.cos(np.radians(5.0)), 0.0, np.sin(np.radians(5.0))])        # pared 5° caída
    assert pl.tipo_de_plano(inclinada, tolerancia) == "pared"
    rampa = np.array([np.sin(np.radians(30.0)), 0.0, np.cos(np.radians(30.0))])
    assert pl.tipo_de_plano(rampa, tolerancia) == "inclinado"


def test_describir_plano_de_una_pared():
    # La pared y = +2: 5 m de ancho y 2,5 de alto
    p = puntos_de_cara(3, 20000, np.random.default_rng(4), ruido=0.01)
    q = pl.describir_plano(p, DENTRO, CFG)
    assert q["tipo"] == "pared" and q["puntos"] == 20000
    assert q["n"] == pytest.approx([0.0, -1.0, 0.0], abs=0.01)           # hacia dentro: hacia y negativa
    assert q["d"] == pytest.approx(-2.0, abs=0.005)                       # n·p = d: -y = -2
    assert q["centro"] == pytest.approx([0.0, 2.0, 1.25], abs=0.05)
    assert abs(q["ejes"][0][2]) < 1e-9 and q["ejes"][1][2] > 0.999        # uno horizontal y otro hacia arriba
    a_min, a_max, b_min, b_max = q["extremos"]
    assert a_max - a_min == pytest.approx(5.0, abs=0.15) and b_max - b_min == pytest.approx(2.5, abs=0.1)
    assert q["superficie"] == pytest.approx(12.5, rel=0.06) and q["grosor"] == pytest.approx(0.01, rel=0.1)


def test_suelo_techo_y_mesa():
    generador = np.random.default_rng(5)
    suelo = puntos_de_cara(4, 2000, generador)
    techo = puntos_de_cara(5, 2000, generador)
    mesa = puntos_de_cara(4, 800, generador) * [0.2, 0.2, 1.0] + [1.0, 1.0, 0.75]        # 1 x 0,8 m a 75 cm
    pared = puntos_de_cara(0, 2000, generador)
    p = np.vstack([suelo, techo, mesa, pared])
    grupos = [np.arange(0, 2000), np.arange(2000, 4000), np.arange(4000, 4800), np.arange(4800, 6800)]
    planos = pl.describir_planos(p, grupos, DENTRO, CFG)
    assert [q["tipo"] for q in planos] == ["suelo", "techo", "horizontal_arriba", "pared"]
    assert planos[2]["centro"][2] == pytest.approx(0.75) and (planos[3]["indices"] == grupos[3]).all()


# =====================================================================
# Paso 27.6 — Las medidas de la habitación
# =====================================================================
def test_buscar_planos_y_medir_la_caja():
    p = nube_de_caja()
    planos = pl.buscar_planos(p, DENTRO, CFG, rng())
    assert sorted(q["tipo"] for q in planos) == ["pared", "pared", "pared", "pared", "suelo", "techo"]
    medidas = pl.medidas_de_habitacion(planos, CFG["tipos"]["tolerancia_grados"])
    assert medidas["altura"] == pytest.approx(2.5, abs=0.01)
    assert sorted(s for _, _, s in medidas["opuestas"]) == pytest.approx([4.0, 5.0], abs=0.01)
    assert len(medidas["esquinas"]) == 4
    assert [angulo for _, _, angulo in medidas["esquinas"]] == pytest.approx([90.0] * 4, abs=0.3)
    # Sin techo no hay altura; con una sola pared no hay ni parejas ni esquinas
    sin_techo = [q for q in planos if q["tipo"] != "techo"]
    assert pl.medidas_de_habitacion(sin_techo, 10.0)["altura"] is None
    una = [q for q in planos if q["tipo"] == "pared"][:1]
    assert pl.medidas_de_habitacion(una, 10.0) == {"altura": None, "opuestas": [], "esquinas": []}


# =====================================================================
# Paso 27.7 — Comparar con la habitación de verdad
# =====================================================================
def test_comparar_con_la_verdad():
    assert [float(n @ n) for n, _ in pl.caras_de_la_habitacion(DIMENSIONES)] == [1.0] * 6
    p = nube_de_caja()
    # La misma caja vista desde otro mundo: girada 30° y desplazada. p_verdad = R·p + t
    angulo = np.radians(30.0)
    R = np.array([[np.cos(angulo), -np.sin(angulo), 0.0], [np.sin(angulo), np.cos(angulo), 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.5, -1.0, 0.2])
    p_otro = (p - t) @ R                                 # p_otro = Rᵀ·(p - t)
    planos = pl.buscar_planos(p_otro, R.T @ (DENTRO - t), CFG, rng())
    comparaciones = pl.comparar_con_la_verdad(planos, R, t, DIMENSIONES)
    assert sorted(c["cara"] for c in comparaciones) == [0, 1, 2, 3, 4, 5]
    assert max(c["error_angulo"] for c in comparaciones) < 0.3
    assert max(c["error_m"] for c in comparaciones) < 0.01


# =====================================================================
# Con la nube densa de la sesión simulada
# =====================================================================
@pytest.fixture(scope="module")
def nube_de_sesion(sesion_con_textura):
    """La nube densa de la sesión de la etapa 24, hecha con la profundidad verdadera."""
    carpeta = sesion_con_textura["carpeta"]
    mapa = medir_mapa.leer_mapa(carpeta / "mapa.paquetes")
    vistas = nube_densa.cargar_vistas(mapa.keyframes, carpeta, True, sesion_con_textura["t_d_ns"])
    nube = den.densificar(vistas, CFG_DENSIFICAR)
    camaras = np.array([kf["p_w_c"] for kf in mapa.keyframes])
    R, t = nube_densa.transformacion_de_la_sesion(carpeta)
    return {"p": nube["p_w"], "camaras": camaras, "R": R, "t": t,
            "dimensiones": cargar_config("simulador")["vio"]["mundo"]["habitacion_m"]}


def test_planos_de_la_sesion(nube_de_sesion):
    s = nube_de_sesion
    planos = pl.buscar_planos(s["p"], s["camaras"].mean(axis=0), CFG, rng())
    comparaciones = pl.comparar_con_la_verdad(planos, s["R"], s["t"], s["dimensiones"])
    # En 10 s la cámara ve dos paredes (y un trozo de suelo): al menos esas dos salen, y bien
    paredes = [(q, c) for q, c in zip(planos, comparaciones) if q["tipo"] == "pared"]
    assert len(paredes) >= 2
    for q, c in zip(planos, comparaciones):
        assert c["error_angulo"] < 2.0 and c["error_m"] < 0.05
    assert len({c["cara"] for _, c in paredes}) == len(paredes)          # cada pared, una cara distinta
    # Son dos paredes en esquina
    medidas = pl.medidas_de_habitacion(planos, CFG["tipos"]["tolerancia_grados"])
    assert len(medidas["esquinas"]) >= 1 and abs(medidas["esquinas"][0][2] - 90.0) < 2.0
    assert sum(q["puntos"] for q in planos) > 0.9 * len(s["p"])


def test_la_herramienta_escribe_los_planos(nube_de_sesion, tmp_path):
    s = nube_de_sesion
    planos = pl.buscar_planos(s["p"], s["camaras"].mean(axis=0), CFG, rng())
    medidas = pl.medidas_de_habitacion(planos, CFG["tipos"]["tolerancia_grados"])
    comparaciones = pl.comparar_con_la_verdad(planos, s["R"], s["t"], s["dimensiones"])
    texto = ver_planos.informe(planos, medidas, len(s["p"]), comparaciones, s["dimensiones"])
    assert "cara de verdad" in texto and f"{len(planos)} planos con" in texto and "Esquinas entre paredes" in texto
    assert "cara de verdad" not in ver_planos.informe(planos, medidas, len(s["p"]), None, None)
    # El CSV: una fila por plano con 22 campos
    ver_planos.guardar_csv(tmp_path / "planos.csv", planos)
    lineas = (tmp_path / "planos.csv").read_text(encoding="utf-8").splitlines()
    assert lineas[0] == ver_planos.CABECERA_CSV.strip() and len(lineas) == len(planos) + 1
    campos = lineas[1].split(",")
    assert len(campos) == 22 and campos[1] == planos[0]["tipo"] and int(campos[21]) == planos[0]["puntos"]
    assert [float(c) for c in campos[2:5]] == pytest.approx(planos[0]["n"], abs=1e-5)
    # Los colores y la figura
    colores = ver_planos.colores_de_planos(len(planos))
    assert colores.shape == (len(planos), 3) and len({tuple(c) for c in colores}) == len(planos)
    ver_planos.guardar_ply(tmp_path / "planos.ply", s["p"], np.tile(colores[0], (len(s["p"]), 1)))
    assert len(leer_ply(tmp_path / "planos.ply")[0]) == len(s["p"])
    ver_planos.dibujar(tmp_path / "planos.png", s["p"], planos, s["camaras"])
    assert (tmp_path / "planos.png").stat().st_size > 10000
