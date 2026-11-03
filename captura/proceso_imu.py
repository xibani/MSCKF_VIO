"""
captura/proceso_imu.py — La IMU en un proceso independiente.

El proceso principal lo arranca con multiprocessing y se comunica con él
mediante dos eventos (listo / parar) y contadores compartidos.
    ejecutar:      graba las muestras en un CSV (etapa 3)
    ejecutar_cola: las entrega por una cola al proceso principal (etapa 13, VIO en vivo)
"""

import os
import queue
import signal

from captura import euroc, mpu6050


def bucle_imu(bus, cfg, ajustes, archivo, evento_parar, n_muestras, n_errores):
    """Lee muestras y las escribe en `archivo` hasta que se active evento_parar.

    archivo: archivo de texto abierto, con la cabecera EuRoC ya escrita.
    evento_parar: objeto con is_set() (multiprocessing.Event).
    n_muestras, n_errores: objetos con atributo .value (multiprocessing.Value);
        se actualizan para que el proceso principal pueda mostrarlos.
    """
    contador = {"errores_i2c": 0}
    for t_ns, w, a, temp in mpu6050.leer_continuo(bus, cfg, ajustes, contador):
        archivo.write(euroc.linea_imu(t_ns, w, a))
        n_muestras.value += 1
        n_errores.value = contador["errores_i2c"]
        if evento_parar.is_set():
            break


def ejecutar(carpeta, cfg, evento_listo, evento_parar, n_muestras, n_errores):
    """Punto de entrada del proceso de la IMU.

    carpeta: carpeta de la grabación (debe existir mav0/imu0/).
    evento_listo: se activa cuando la IMU ya está grabando.
    """
    from smbus2 import SMBus

    # 1. Ignorar Ctrl+C: es el proceso principal quien decide cuándo parar,
    #    para que la IMU siga grabando hasta después de la última imagen
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # 2. Reservarse un núcleo de la CPU (menos interrupciones = marcas más regulares)
    try:
        os.sched_setaffinity(0, {cfg["nucleo_cpu"]})
    except OSError:
        print(f"Aviso: no se pudo fijar la IMU al núcleo {cfg['nucleo_cpu']}")

    # 3. Configurar y grabar
    with SMBus(cfg["bus_i2c"]) as bus, \
         open(euroc.ruta_imu(carpeta), "w", encoding="utf-8") as archivo:
        ajustes = mpu6050.configurar(bus, cfg)
        archivo.write(euroc.CABECERA_IMU)
        evento_listo.set()
        bucle_imu(bus, cfg, ajustes, archivo, evento_parar, n_muestras, n_errores)




# =====================================================================
# Etapa 13 — Las muestras por una cola, para el VIO en vivo
# =====================================================================
def bucle_cola(muestras, cola, evento_parar, n_perdidas, n_errores, contador):
    """Pasa las muestras del generador a la cola hasta que se active evento_parar.

    muestras: generador de (t_ns, w, a, temp), como mpu6050.leer_continuo.
    cola: multiprocessing.Queue (o queue.Queue en los tests).
    n_perdidas, n_errores: objetos con .value. contador: el dict de errores de leer_continuo.
    Si la cola está llena (el proceso principal no da abasto), la muestra se pierde y se
    cuenta: es mejor perder una muestra que bloquear la lectura de la IMU.
    """
    for t_ns, w, a, _ in muestras:
        try:
            cola.put_nowait((t_ns, w, a))
        except queue.Full:
            n_perdidas.value += 1
        n_errores.value = contador["errores_i2c"]
        if evento_parar.is_set():
            break


def ejecutar_cola(cfg, cola, evento_listo, evento_parar, n_perdidas, n_errores):
    """Punto de entrada del proceso de la IMU del VIO en vivo (como ejecutar, sin CSV)."""
    from smbus2 import SMBus

    signal.signal(signal.SIGINT, signal.SIG_IGN)             # para lo decide el principal
    try:
        os.sched_setaffinity(0, {cfg["nucleo_cpu"]})
    except OSError:
        print(f"Aviso: no se pudo fijar la IMU al núcleo {cfg['nucleo_cpu']}")

    with SMBus(cfg["bus_i2c"]) as bus:
        ajustes = mpu6050.configurar(bus, cfg)
        contador = {"errores_i2c": 0}
        evento_listo.set()
        bucle_cola(mpu6050.leer_continuo(bus, cfg, ajustes, contador), cola, evento_parar,
                   n_perdidas, n_errores, contador)