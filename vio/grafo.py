"""
vio/grafo.py — Etapa 29: grafo de poses.

Después de la etapa 28 hay dos clases de medidas sobre dónde están los keyframes:
    - Odometría: el paso de cada keyframe al siguiente, según el VIO. Muy preciso en cada
      paso, pero los errores se suman al encadenarlos (la deriva).
    - Bucles: la pose de un keyframe vista desde otro muy anterior. Menos precisa, pero no
      arrastra la deriva del camino intermedio.
No se pueden cumplir todas a la vez: si el VIO ha derivado, los bucles le contradicen. El
"grafo de poses" busca las poses de todos los keyframes que mejor cumplen todas las medidas
juntas, cada una según lo que se confía en ella. Los nudos del grafo son las poses; las
aristas, las medidas entre dos nudos.

    1. Aristas: de la odometría y de los bucles.
    2. El residuo de una arista: cuánto se aparta la pose relativa de dos nudos de la medida.
    3. Gauss-Newton (con amortiguación): mover todas las poses un poco para reducir la suma
       de los cuadrados de los residuos, y repetir.

Una pose es una pareja (R, p): R_w_c (3, 3) y p_w_c (3,). Las poses relativas, como en la
etapa 28: la de j en i es (R_i_j, p_i_j) = (R_iᵀ·R_j, R_iᵀ·(p_j - p_i)).
"""

import numpy as np

from vio import so3
from vio.bucles import pose_relativa


# =====================================================================
# Paso 29.1 — Las aristas
# =====================================================================
def crear_arista(i, j, R, p, cfg_tipo, es_bucle):
    """Una arista: la medida (R, p) de la pose del nudo j vista desde el nudo i.

    cfg_tipo: sección "odometria" o "bucle" de config/grafo.yaml (de ahí salen las sigmas).
    Devuelve: dict con i, j, R, p, sigma_m, sigma_rad y bucle (bool).
    """
    return {"i": int(i), "j": int(j), "R": np.array(R, dtype=float), "p": np.array(p, dtype=float),
            "sigma_m": float(cfg_tipo["sigma_m"]), "sigma_rad": float(np.radians(cfg_tipo["sigma_grados"])),
            "bucle": bool(es_bucle)}


def aristas_de_odometria(poses, cfg):
    """Una arista entre cada keyframe y el siguiente, con el paso que dan las poses del VIO."""
    aristas = []
    for k in range(len(poses) - 1):
        R, p = pose_relativa(*poses[k], *poses[k + 1])
        aristas.append(crear_arista(k, k + 1, R, p, cfg["odometria"], False))
    return aristas


def aristas_de_bucles(bucles, cfg):
    """Una arista por bucle de la etapa 28 (dict con i, j, R y p)."""
    return [crear_arista(b["i"], b["j"], b["R"], b["p"], cfg["bucle"], True) for b in bucles]


# =====================================================================
# Paso 29.2 — El residuo de una arista y cómo cambia al mover las poses
# =====================================================================
def residuo(pose_i, pose_j, arista):
    """Cuánto se aparta la pose relativa de dos nudos de lo que mide la arista.

    Seis números, ya divididos por sus sigmas (así todos valen "cuántas sigmas"):
        los 3 primeros, de orientación: el vector de giro que queda entre la orientación
            relativa medida y la que tienen los nudos,  log(R_medᵀ·R_iᵀ·R_j) / sigma_rad
        los 3 últimos, de posición:  (R_iᵀ·(p_j - p_i) - p_med) / sigma_m
    Si los nudos cumplen la medida exactamente, los seis son cero.
    """
    R_i, p_i = pose_i
    R_j, p_j = pose_j
    r_giro = so3.log_so3(arista["R"].T @ R_i.T @ R_j)
    r_posicion = R_i.T @ (p_j - p_i) - arista["p"]
    return np.concatenate([r_giro / arista["sigma_rad"], r_posicion / arista["sigma_m"]])


def jacobianos(pose_i, pose_j, arista):
    """Cómo cambia el residuo al mover un poco cada uno de los dos nudos.

    "Mover un poco" un nudo son 6 números (dg, dp): un giro pequeño dg aplicado en sus
    propios ejes (R <- R·exp(dg)) y un desplazamiento dp en el mundo (p <- p + dp).
    Devuelve: (J_i, J_j), dos matrices (6, 6): residuo nuevo ≈ residuo + J_i·(dg_i, dp_i) + J_j·(dg_j, dp_j).

    De dónde sale cada bloque (para giros pequeños, exp(dg) ≈ I + hat(dg)):
      - Posición respecto a dp_j: R_iᵀ. Respecto a dp_i: -R_iᵀ.
      - Posición respecto a dg_i: al girar i, R_iᵀ pasa a ser exp(-dg_i)·R_iᵀ, y el vector
        v = R_iᵀ·(p_j - p_i) pasa a ser v - dg_i × v = v + hat(v)·dg_i.
      - Giro respecto a dg_j: la identidad (el giro de j se añade tal cual al final).
      - Giro respecto a dg_i: -R_jᵀ·R_i (el giro de i, visto desde los ejes de j, y restando).
    Los dos del giro son aproximados: exactos cuando el residuo de giro es pequeño. No
    importa, porque el residuo sí se calcula exacto en cada vuelta.
    """
    R_i, p_i = pose_i
    R_j, p_j = pose_j
    J_i = np.zeros((6, 6))
    J_j = np.zeros((6, 6))
    J_i[0:3, 0:3] = -R_j.T @ R_i
    J_j[0:3, 0:3] = np.eye(3)
    J_i[3:6, 0:3] = so3.hat(R_i.T @ (p_j - p_i))
    J_i[3:6, 3:6] = -R_i.T
    J_j[3:6, 3:6] = R_i.T
    # Las mismas sigmas que el residuo
    for J in (J_i, J_j):
        J[0:3] /= arista["sigma_rad"]
        J[3:6] /= arista["sigma_m"]
    return J_i, J_j


def peso_de_huber(r, arista, cfg):
    """Cuánto cuenta una arista, de 0 a 1 (la odometría, siempre 1).

    Un bucle falso mide una pose que no tiene nada que ver, y con mínimos cuadrados normales
    arrastraría a todos los nudos. Con el peso de Huber, un bucle cuyo residuo pasa de
    `huber` sigmas cuenta menos cuanto más se aparta: peso = huber / |residuo|.
    """
    if not arista["bucle"]:
        return 1.0
    tamano = float(np.linalg.norm(r))
    return 1.0 if tamano <= cfg["bucle"]["huber"] else cfg["bucle"]["huber"] / tamano


# =====================================================================
# Paso 29.3 — Gauss-Newton con amortiguación
# =====================================================================
def coste(poses, aristas, cfg):
    """Lo que se quiere hacer mínimo: la suma, sobre todas las aristas, de |residuo|²/2
    (para un bucle más allá del umbral de Huber, huber·(|residuo| - huber/2): crece en
    línea recta en vez de al cuadrado)."""
    total = 0.0
    for a in aristas:
        tamano = float(np.linalg.norm(residuo(poses[a["i"]], poses[a["j"]], a)))
        k = cfg["bucle"]["huber"]
        total += 0.5 * tamano ** 2 if (not a["bucle"] or tamano <= k) else k * (tamano - 0.5 * k)
    return total


def ecuaciones_normales(poses, aristas, cfg):
    """El sistema H·d = b de un paso de Gauss-Newton.

    d junta la corrección de todos los nudos: 6 números por nudo (dg, dp). Cada arista
    aporta a H y a b en las filas y columnas de sus dos nudos:
        H += Jᵀ·J        b += -Jᵀ·r        (con J = [J_i, J_j], y todo por el peso de Huber)
    Devuelve: (H (6N, 6N), b (6N,)).
    """
    n = len(poses)
    H = np.zeros((6 * n, 6 * n))
    b = np.zeros(6 * n)
    for a in aristas:
        i, j = a["i"], a["j"]
        r = residuo(poses[i], poses[j], a)
        J_i, J_j = jacobianos(poses[i], poses[j], a)
        w = peso_de_huber(r, a, cfg)
        filas_i, filas_j = slice(6 * i, 6 * i + 6), slice(6 * j, 6 * j + 6)
        H[filas_i, filas_i] += w * J_i.T @ J_i
        H[filas_j, filas_j] += w * J_j.T @ J_j
        H[filas_i, filas_j] += w * J_i.T @ J_j
        H[filas_j, filas_i] += w * J_j.T @ J_i
        b[filas_i] -= w * J_i.T @ r
        b[filas_j] -= w * J_j.T @ r
    return H, b


def aplicar_correccion(poses, d):
    """Las poses movidas: a cada nudo k, su trozo de d (dg, dp): R <- R·exp(dg), p <- p + dp."""
    nuevas = []
    for k, (R, p) in enumerate(poses):
        dg, dp = d[6 * k:6 * k + 3], d[6 * k + 3:6 * k + 6]
        nuevas.append((R @ so3.exp_so3(dg), p + dp))
    return nuevas


def optimizar(poses, aristas, cfg):
    """Las poses que mejor cumplen todas las aristas.

    En cada vuelta se resuelve (H + amortiguacion·diagonal(H))·d = b. La amortiguación es
    la de Levenberg-Marquardt: con ella grande, el paso es corto y prudente; con ella
    pequeña, es el de Gauss-Newton. Si el paso baja el coste, se acepta y se baja la
    amortiguación; si lo sube, se rechaza y se sube.
    El primer nudo no se mueve (es el ancla): sus 6 incógnitas se quitan del sistema. Sin
    eso, mover todas las poses juntas no cambiaría ningún residuo y no habría una solución
    única.
    poses: lista de (R, p). Devuelve: (poses nuevas, historia): historia es la lista de
    costes, empezando por el inicial.
    """
    amortiguacion = cfg["optimizacion"]["amortiguacion"]
    actual = coste(poses, aristas, cfg)
    historia = [actual]
    for _ in range(cfg["optimizacion"]["iteraciones_max"]):
        H, b = ecuaciones_normales(poses, aristas, cfg)
        H, b = H[6:, 6:], b[6:]                          # sin las 6 incógnitas del primer nudo
        d = np.zeros(6 * len(poses))
        d[6:] = np.linalg.solve(H + amortiguacion * np.diag(np.diag(H)), b)
        candidatas = aplicar_correccion(poses, d)
        nuevo = coste(candidatas, aristas, cfg)
        if nuevo < actual:
            poses, actual = candidatas, nuevo
            historia.append(actual)
            amortiguacion = max(amortiguacion / 3.0, 1e-9)
            if np.abs(d).max() < cfg["optimizacion"]["paso_min"]:
                break
        else:
            amortiguacion *= 5.0
            if amortiguacion > 1e9:
                break
    return poses, historia


# =====================================================================
# Paso 29.4 — Para probar: poses estropeadas a propósito
# =====================================================================
def anadir_deriva(poses, giro_grados, escala):
    """Las mismas poses con una deriva inventada, para ver qué arregla el grafo.

    El VIO del simulador casi no deriva. Aquí se rehace la cadena de pasos entre keyframes
    estropeando cada uno: se le añade un giro de giro_grados alrededor del eje y de la
    cámara (el vertical: es deriva de rumbo) y su desplazamiento se multiplica por escala.
    El primer keyframe no cambia; el error crece con cada paso, como una deriva de verdad.
    Devuelve: lista de (R, p).
    """
    giro = so3.exp_so3(np.array([0.0, np.radians(giro_grados), 0.0]))
    nuevas = [poses[0]]
    for k in range(len(poses) - 1):
        R_paso, p_paso = pose_relativa(*poses[k], *poses[k + 1])
        R, p = nuevas[-1]
        nuevas.append((R @ R_paso @ giro, p + R @ (escala * p_paso)))
    return nuevas


# =====================================================================
# Paso 29.5 — Medir el resultado
# =====================================================================
def residuos_de_bucles(poses, aristas):
    """De cada arista de bucle, cuánto se apartan los nudos de su medida, sin dividir por
    las sigmas. Devuelve: (M, 2): [metros, grados] por bucle."""
    filas = []
    for a in aristas:
        if a["bucle"]:
            r = residuo(poses[a["i"]], poses[a["j"]], a)
            filas.append([np.linalg.norm(r[3:6]) * a["sigma_m"], np.degrees(np.linalg.norm(r[0:3]) * a["sigma_rad"])])
    return np.array(filas).reshape(-1, 2)


def errores_de_poses(poses, poses_verdad, R, t):
    """El error de cada pose frente a la verdadera (solo en el simulador).

    poses: en el mundo del VIO. poses_verdad: en el del simulador (de
    bucles.poses_verdaderas_de_camara). R, t: de densificar.transformacion_a_la_verdad.
    Devuelve: (N, 2): [metros, grados] por keyframe.
    """
    filas = []
    for (R_k, p_k), (R_ver, p_ver) in zip(poses, poses_verdad):
        filas.append([np.linalg.norm(R @ p_k + t - p_ver), np.degrees(so3.angulo_entre(R @ R_k, R_ver))])
    return np.array(filas).reshape(-1, 2)
