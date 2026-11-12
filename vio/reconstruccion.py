"""
vio/reconstruccion.py — Etapa 17: el mapa en el PC, construido poco a poco con los paquetes
que envía la Pi (vio/protocolo.py, versiones 4 y 5).

Los paquetes pueden llegar por red (apps/pc_mapa, en directo) o leerse de un archivo
mapa.paquetes (una sesión ya grabada): el Reconstructor no sabe de dónde vienen.

    versión 5 (puntos)   -> MapaDisperso: un punto por id, media de sus triangulaciones
    versión 4 (keyframe) -> lista de keyframes: pose, intrínsecos, profundidades y JPEG
                            (la etapa 17.4 los densificará con Depth Anything)

Mundo: el del VIO, con z hacia arriba (la gravedad es -z) y el origen donde arrancó el filtro.
"""

import struct
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from vio import protocolo


# =====================================================================
# Mapa disperso
# =====================================================================
class MapaDisperso:
    """Un punto por identificador.

    El mismo punto físico puede llegar varias veces con el mismo id: el MSCKF corrige una
    pista cuando ocupa toda la ventana (15 imágenes) y la vuelve a empezar si se sigue viendo.
    Cada llegada es una triangulación independiente; se guarda la media.
    """

    def __init__(self):
        self.suma = {}  # id -> suma de las posiciones (3,)
        self.cuenta = {}  # id -> número de triangulaciones

    def anadir(self, ids, p_w):
        """ids (N,), p_w (N, 3) [m]."""
        for id_punto, p in zip(ids, p_w):
            id_punto = int(id_punto)
            if id_punto in self.suma:
                self.suma[id_punto] = self.suma[id_punto] + p
                self.cuenta[id_punto] += 1
            else:
                self.suma[id_punto] = np.array(p, dtype=float)
                self.cuenta[id_punto] = 1

    def __len__(self):
        return len(self.suma)

    def puntos(self):
        """Devuelve: (ids (N,) int64, p_w (N, 3) media de cada id, cuentas (N,)), por id."""
        ids = np.array(sorted(self.suma), dtype=np.int64)
        p_w = np.array([self.suma[i] / self.cuenta[i] for i in ids]).reshape(-1, 3)
        cuentas = np.array([self.cuenta[i] for i in ids], dtype=np.int64)
        return ids, p_w, cuentas


def puntos_keyframe_en_mundo(kf):
    """Los puntos de un keyframe (dict de protocolo.leer_keyframe), llevados al mundo.

    En la cámara, un punto con coordenadas normalizadas (x, y) y profundidad z está en
    p_c = z·(x, y, 1). En el mundo: p_w = R_w_c·p_c + p_w_c.
    Devuelve: (N, 3).
    """
    p_c = kf["z"][:, None] * np.column_stack([kf["xy"], np.ones(len(kf["z"]))])
    return p_c @ kf["R_w_c"].T + kf["p_w_c"]


# =====================================================================
# Reconstructor: recibe paquetes
# =====================================================================
class Reconstructor:
    """Acumula el mapa paquete a paquete."""

    def __init__(self):
        self.disperso = MapaDisperso()
        self.keyframes = []  # dicts de protocolo.leer_keyframe, en orden de llegada
        self.recuentos = Counter()  # "puntos", "keyframe", "prueba", "malos"

    def procesar(self, datos):
        """Un paquete (bytes). Devuelve su tipo: "puntos", "keyframe", "prueba" o "malos".

        "prueba" (etapa 19) no cambia este mapa: quien lo recibe (MapaPorPruebas) decide si
        hay que empezar otro.

        Un paquete que no se entiende (de otra versión, cortado...) se cuenta y se ignora:
        en directo, por UDP, puede llegar cualquier cosa al puerto.
        """
        tipo = "malos"
        try:
            if len(datos) > 0 and datos[0] == protocolo.VERSION_PUNTOS:
                _, ids, p_w = protocolo.leer_puntos(datos)
                self.disperso.anadir(ids, p_w)
                tipo = "puntos"
            elif len(datos) > 0 and datos[0] == protocolo.VERSION_KEYFRAME:
                self.keyframes.append(protocolo.leer_keyframe(datos))
                tipo = "keyframe"
            elif len(datos) > 0 and datos[0] == protocolo.VERSION_PRUEBA:
                protocolo.leer_prueba(datos)  # solo se comprueba que está entero
                tipo = "prueba"
        except (ValueError, struct.error):
            tipo = "malos"
        self.recuentos[tipo] += 1
        return tipo

    def resumen(self):
        """Una línea con el estado, para imprimir."""
        return (
            f"keyframes {len(self.keyframes):4d} | puntos {len(self.disperso):6d}"
            f" | paquetes de puntos {self.recuentos['puntos']:6d} | malos {self.recuentos['malos']}"
        )

    # ---------------- Guardar ----------------
    def guardar(self, carpeta):
        """Escribe en `carpeta` el mapa disperso (PLY), los keyframes y una figura.

        Devuelve: dict con las rutas escritas.
        """
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        rutas = {}
        _, p_w, _ = self.disperso.puntos()
        if len(p_w) > 0:
            rutas["ply"] = carpeta / "puntos_3d.ply"
            guardar_ply(rutas["ply"], p_w, colores_por_altura(p_w[:, 2]))
        if self.keyframes:
            rutas["keyframes"] = carpeta / "keyframes.csv"
            guardar_keyframes(carpeta, self.keyframes)
        if len(p_w) > 0 or self.keyframes:
            rutas["figura"] = carpeta / "mapa.png"
            dibujar_mapa(rutas["figura"], p_w, self.keyframes)
        return rutas


# =====================================================================
# Etapa 19 — Un mapa por prueba
# =====================================================================
def carpeta_de_prueba(salida, n):
    """La primera prueba se guarda en `salida`; las siguientes, en salida/prueba_2, _3..."""
    salida = Path(salida)
    return salida if n == 1 else salida / f"prueba_{n}"


class MapaPorPruebas:
    """Un Reconstructor por cada prueba del VIO.

    Cada vez que el VIO se reinicia ("Nueva prueba" en el visor, o pi_vio relanzado) empieza
    un mundo distinto: otro origen, otro rumbo, y los ids de los puntos vuelven a empezar en 0.
    Si todo fuera al mismo Reconstructor, MapaDisperso haría la media de puntos que no tienen
    nada que ver solo porque comparten id. Por eso, al llegar un aviso de prueba nueva
    (versión 6) se guarda el mapa que había y se empieza otro, en otra carpeta.

    salida: carpeta de la primera prueba.
    nombre_copia: si no es None, cada paquete recibido se copia a ese archivo dentro de la
    carpeta de su prueba (en directo: para poder repetirlo luego con --archivo).
    """

    def __init__(self, salida, nombre_copia=None):
        self.salida = Path(salida)
        self.nombre_copia = nombre_copia
        self.ultimo_aviso = None  # bytes del último paquete de prueba visto
        self.guardadas = []  # (número de prueba, carpeta, rutas escritas) de las ya cerradas
        self._empezar(1)

    def _empezar(self, n):
        self.n = n
        self.reconstructor = Reconstructor()
        self.carpeta = carpeta_de_prueba(self.salida, n)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.copia = None
        if self.nombre_copia is not None:
            self.copia = open(self.carpeta / self.nombre_copia, "wb")

    def tiene_datos(self):
        """¿Ha llegado algún punto o keyframe a la prueba en curso?"""
        r = self.reconstructor.recuentos
        return r["puntos"] + r["keyframe"] > 0

    def procesar(self, datos):
        """Un paquete (bytes). Devuelve su tipo, como Reconstructor.procesar."""
        tipo = self.reconstructor.procesar(datos)
        if tipo != "prueba":
            if self.copia is not None:
                protocolo.escribir_paquete(self.copia, datos)
            return tipo
        # Aviso de prueba nueva. El mismo aviso llega repetido (REPETICIONES_PRUEBA): solo
        # cuenta la primera vez. Y si aún no había llegado nada, la prueba en curso ya sirve.
        if datos != self.ultimo_aviso:
            self.ultimo_aviso = datos
            if self.tiene_datos():
                self.cerrar()
                self._empezar(self.n + 1)
        return tipo

    def vaciar_copia(self):
        """Lleva al disco lo copiado hasta ahora (por si se va la luz)."""
        if self.copia is not None:
            self.copia.flush()

    def cerrar(self):
        """Guarda el mapa de la prueba en curso. Devuelve: dict con las rutas escritas."""
        if self.copia is not None:
            self.copia.close()
            self.copia = None
        rutas = self.reconstructor.guardar(self.carpeta)
        self.guardadas.append((self.n, self.carpeta, rutas))
        return rutas

    def resumen(self):
        return f"prueba {self.n} | " + self.reconstructor.resumen()


# =====================================================================
# Archivos de salida
# =====================================================================
def colores_por_altura(z):
    """Color RGB (N, 3) uint8 de cada punto según su altura (escala viridis, entre los
    percentiles 2 y 98 para que unos pocos puntos extremos no la aplasten)."""
    from matplotlib import colormaps

    if len(z) == 0:
        return np.zeros((0, 3), np.uint8)
    bajo, alto = np.percentile(z, [2, 98])
    t = np.clip((z - bajo) / max(alto - bajo, 1e-9), 0.0, 1.0)
    return (colormaps["viridis"](t)[:, :3] * 255).astype(np.uint8)


def guardar_ply(ruta, p_w, colores):
    """Nube de puntos en PLY binario: x y z (float32) y rojo verde azul (uint8) por punto.

    Es el formato que abren CloudCompare, MeshLab, Open3D o Blender. La cabecera es texto y
    dice cómo leer lo que viene detrás; los datos van en binario (little-endian).
    """
    n = len(p_w)
    tipo = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    datos = np.zeros(n, dtype=tipo)
    datos["x"], datos["y"], datos["z"] = p_w[:, 0], p_w[:, 1], p_w[:, 2]
    datos["red"], datos["green"], datos["blue"] = (
        colores[:, 0],
        colores[:, 1],
        colores[:, 2],
    )
    cabecera = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(ruta, "wb") as f:
        f.write(cabecera.encode("ascii"))
        f.write(datos.tobytes())


def leer_ply(ruta):
    """Inversa de guardar_ply. Devuelve: (p_w (N, 3), colores (N, 3) uint8)."""
    with open(ruta, "rb") as f:
        contenido = f.read()
    fin = contenido.index(b"end_header\n") + len(b"end_header\n")
    tipo = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    datos = np.frombuffer(contenido[fin:], dtype=tipo)
    p_w = np.column_stack([datos["x"], datos["y"], datos["z"]]).astype(float)
    colores = np.column_stack([datos["red"], datos["green"], datos["blue"]])
    return p_w, colores


CABECERA_KEYFRAMES = (
    "#indice,t_ns,p_x,p_y,p_z,r00,r01,r02,r10,r11,r12,r20,r21,r22,n_puntos,"
    "fx,fy,cx,cy,k1,k2,p1,p2,k3,ancho,alto,imagen\n"
)


def guardar_keyframes(carpeta, keyframes):
    """keyframes.csv (pose R_w_c, p_w_c e intrínsecos de cada uno) y keyframes/<indice>.jpg.

    El JPEG se guarda tal como llegó: sin volver a comprimirlo, no pierde más calidad.
    """
    carpeta_img = Path(carpeta) / "keyframes"
    carpeta_img.mkdir(parents=True, exist_ok=True)
    with open(Path(carpeta) / "keyframes.csv", "w", encoding="utf-8") as f:
        f.write(CABECERA_KEYFRAMES)
        for kf in keyframes:
            nombre = f"{kf['indice']:06d}.jpg"
            (carpeta_img / nombre).write_bytes(kf["jpeg"])
            K = kf["K"]
            numeros = (
                list(kf["p_w_c"])
                + list(kf["R_w_c"].ravel())
                + [len(kf["z"])]
                + [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
                + list(kf["dist"])
            )
            f.write(
                f"{kf['indice']},{kf['t_ns']},"
                + ",".join(f"{x:.9g}" for x in numeros)
                + f",{kf['tamano'][0]},{kf['tamano'][1]},keyframes/{nombre}\n"
            )


def decodificar_gris(kf):
    """La imagen de un keyframe (dict de protocolo.leer_keyframe), en gris (alto, ancho) uint8."""
    return cv2.imdecode(np.frombuffer(kf["jpeg"], np.uint8), cv2.IMREAD_GRAYSCALE)


def dibujar_mapa(ruta, p_w, keyframes):
    """Figura con dos vistas del mapa disperso: planta (x, y) y alzado (x, z), con las
    posiciones de los keyframes y hacia dónde mira cada uno."""
    import matplotlib

    matplotlib.use("Agg")  # sin ventana: solo se guarda el PNG
    import matplotlib.pyplot as plt

    fig, (planta, alzado) = plt.subplots(1, 2, figsize=(13, 6))
    colores = colores_por_altura(p_w[:, 2]) / 255.0 if len(p_w) else None
    for ax, (a, b), titulo in (
        (planta, (0, 1), "Planta (x, y)"),
        (alzado, (0, 2), "Alzado (x, z)"),
    ):
        if len(p_w):
            ax.scatter(p_w[:, a], p_w[:, b], s=2, c=colores, label=f"{len(p_w)} puntos")
        if keyframes:
            c = np.array([kf["p_w_c"] for kf in keyframes])
            ax.plot(
                c[:, a],
                c[:, b],
                "r.-",
                lw=0.8,
                ms=5,
                label=f"{len(keyframes)} keyframes",
            )
            # Eje óptico (z de la cámara = tercera columna de R_w_c), 0,3 m
            eje = np.array([kf["R_w_c"][:, 2] for kf in keyframes]) * 0.3
            ax.quiver(
                c[:, a],
                c[:, b],
                eje[:, a],
                eje[:, b],
                color="r",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.003,
            )
        ax.set_title(titulo)
        ax.set_xlabel("xyz"[a] + " [m]")
        ax.set_ylabel("xyz"[b] + " [m]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
