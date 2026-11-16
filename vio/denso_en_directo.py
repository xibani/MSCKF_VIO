"""
vio/denso_en_directo.py — Etapa 30: la nube densa, keyframe a keyframe.

La etapa 26 hace la nube cuando ya están todos los keyframes: compara cada uno con todos
los demás y lo funde todo al final. En directo los keyframes van llegando y la nube tiene
que crecer con ellos, sin rehacerla entera cada vez. Dos cambios:
    - La nube se guarda como un diccionario de cubos al que se le van sumando puntos.
    - Cada keyframe se comprueba solo con sus vecinos en el tiempo (una ventana), y entra
      en la nube con un pequeño retraso, cuando ya han llegado algunos de los siguientes.
Con una ventana que abarque todos los keyframes, el resultado es el mismo que el de la
etapa 26.
"""

import numpy as np

from vio import densificar as den


# =====================================================================
# Paso 30.1 — Una nube a la que se le van sumando puntos
# =====================================================================
class NubeIncremental:
    """Los cubos (vóxeles) ocupados, en un diccionario.

    Clave: los tres enteros del cubo. Valor: array de 5 números: la suma de las x, de las y,
    de las z y de los grises de sus puntos, y cuántos son. Con la suma y la cuenta se saca
    la media cuando haga falta, y sumar puntos nuevos no obliga a recordar los anteriores.
    """

    def __init__(self, lado_m):
        self.lado_m = lado_m
        self.cubos = {}

    def __len__(self):
        return len(self.cubos)

    def anadir(self, p_w, gris):
        """Suma unos puntos a la nube. p_w: (N, 3). gris: (N,)."""
        if len(p_w) == 0:
            return
        claves = np.floor(p_w / self.lado_m).astype(np.int64)
        # Primero se juntan los puntos nuevos que caen en el mismo cubo (como en la etapa 26),
        # y después se suma cada grupo a su cubo del diccionario
        unicas, de_que_cubo, cuentas = np.unique(claves, axis=0, return_inverse=True, return_counts=True)
        de_que_cubo = de_que_cubo.ravel()
        sumas = np.zeros((len(unicas), 5))
        for columna in range(3):
            sumas[:, columna] = np.bincount(de_que_cubo, weights=p_w[:, columna], minlength=len(unicas))
        sumas[:, 3] = np.bincount(de_que_cubo, weights=gris.astype(float), minlength=len(unicas))
        sumas[:, 4] = cuentas
        for clave, suma in zip(map(tuple, unicas), sumas):
            if clave in self.cubos:
                self.cubos[clave] += suma
            else:
                self.cubos[clave] = suma.copy()

    def puntos(self, puntos_min):
        """La nube tal como está. Devuelve: (p (M, 3), gris (M,) uint8, cuentas (M,) int), de
        los cubos con puntos_min puntos o más."""
        if not self.cubos:
            return np.zeros((0, 3)), np.zeros(0, np.uint8), np.zeros(0, int)
        sumas = np.array(list(self.cubos.values()))
        sumas = sumas[sumas[:, 4] >= puntos_min]
        cuentas = sumas[:, 4]
        return (sumas[:, 0:3] / cuentas[:, None], np.round(sumas[:, 3] / cuentas).astype(np.uint8),
                cuentas.astype(int))


# =====================================================================
# Paso 30.2 — Los keyframes, de uno en uno
# =====================================================================
class DensificadorEnDirecto:
    """Hace la nube densa según llegan las vistas.

    Uso:
        densificador = DensificadorEnDirecto(cfg_densificar, cfg_directo)
        densificador.nueva_vista(vista)      # cada vez que llega un keyframe con su profundidad
        ...
        densificador.terminar()              # al acabar: mete en la nube los que estaban esperando
        resultado = densificador.resultado() # en cualquier momento: la nube tal como está
    """

    def __init__(self, cfg, cfg_directo):
        """cfg: config/densificar.yaml. cfg_directo: config/directo.yaml."""
        self.cfg = cfg
        self.ventana = cfg_directo["ventana"]
        self.retraso = cfg_directo["retraso"]
        self.vistas = []                  # todas las recibidas (a las viejas se les quita la profundidad)
        self.pendientes = []              # de cada vista, su nube_de_vista hasta que entra en la nube
        self.n_cerradas = 0               # cuántas vistas han entrado ya en la nube
        self.nube = NubeIncremental(cfg["voxel"]["lado_m"])
        self.por_vista = []

    def nueva_vista(self, vista):
        """Llega un keyframe con su profundidad (dict con kf, z y gris).

        Se calculan sus puntos y se queda esperando. Entran en la nube las vistas que ya
        tienen `retraso` vistas posteriores. Devuelve: cuántas vistas han entrado ahora.
        """
        self.vistas.append(vista)
        self.pendientes.append(den.nube_de_vista(vista, self.cfg))
        antes = self.n_cerradas
        while len(self.vistas) - 1 - self.n_cerradas >= self.retraso:
            self._cerrar(self.n_cerradas)
        return self.n_cerradas - antes

    def terminar(self):
        """No van a llegar más vistas: entran en la nube todas las que esperaban."""
        while self.n_cerradas < len(self.vistas):
            self._cerrar(self.n_cerradas)

    def _cerrar(self, k):
        """La vista k entra en la nube: sus puntos se comprueban desde sus vecinas (las
        `ventana` anteriores y las que hayan llegado después, hasta `retraso`), con la misma
        regla de la etapa 26, y los que quedan se suman a la nube."""
        nube = self.pendientes[k]
        confirmaciones = np.zeros(len(nube["p_w"]), dtype=int)
        contradicciones = np.zeros(len(nube["p_w"]), dtype=int)
        primera = max(k - self.ventana, 0)
        ultima = min(k + self.retraso, len(self.vistas) - 1)
        for j in range(primera, ultima + 1):
            if j == k:
                continue
            confirma, contradice = den.comprobar_desde_otra(nube["p_w"], self.vistas[j],
                                                            self.cfg["coherencia"]["umbral_relativo"])
            confirmaciones += confirma
            contradicciones += contradice
        queda = (confirmaciones >= self.cfg["coherencia"]["vistas_min"]) & (confirmaciones > contradicciones)
        self.nube.anadir(nube["p_w"][queda], nube["gris"][queda])
        self.por_vista.append({"indice": self.vistas[k]["kf"]["indice"], "candidatos": nube["candidatos"],
                               "sin_dato": nube["sin_dato"], "en_borde": nube["en_borde"],
                               "sin_confirmar": int((~queda).sum()), "usados": int(queda.sum())})
        # Lo que ya no hace falta se suelta, para que la memoria no crezca sin fin: los puntos
        # de esta vista, y la profundidad de la que acaba de salir de la ventana de la siguiente
        self.pendientes[k] = None
        if k - self.ventana >= 0:
            self.vistas[k - self.ventana] = {"kf": self.vistas[k - self.ventana]["kf"], "z": None, "gris": None}
        self.n_cerradas += 1

    def resultado(self):
        """La nube tal como está. Devuelve: dict como el de densificar.densificar (p_w, gris,
        cuentas y por_vista), con las vistas que han entrado hasta ahora."""
        p_w, gris, cuentas = self.nube.puntos(self.cfg["voxel"]["puntos_min"])
        return {"p_w": p_w, "gris": gris, "cuentas": cuentas, "por_vista": list(self.por_vista)}
