"""
herramientas/probar_camara.py — Configura la cámara, fija la exposición y guarda una imagen.

Uso, desde ~/svo-pi:
    python -m herramientas.probar_camara
"""

import cv2

from captura import camara
from vio.config import RAIZ_PROYECTO, cargar_config

N_COMPROBACION = 10     # imágenes para comprobar que la exposición no cambia


def main():
    cfg = cargar_config("camara")
    cfg_proyecto = cargar_config("proyecto")

    picam2 = camara.abrir(cfg)
    try:
        # --- 1. Qué modo del sensor se ha elegido y qué límites tiene ---
        print("Modo del sensor:", picam2.camera_configuration()["sensor"])
        print("Límites ExposureTime [us]:", picam2.camera_controls["ExposureTime"])
        print("Límites AnalogueGain:     ", picam2.camera_controls["AnalogueGain"])

        # --- 2. Arrancar, dejar que converja la exposición y fijarla ---
        picam2.start()
        print(f"Ajustando exposición ({cfg['t_ajuste_ae_s'] + cfg['t_aplicar_s']:.1f} s)...")
        fijada = camara.fijar_exposicion(picam2, cfg)
        print("Controles enviados:", fijada["controles"])
        print(f"Aplicado: exposición {fijada['exposicion_us']} us, "
              f"ganancia {fijada['ganancia']:.2f}, "
              f"brillo relativo {fijada['brillo_relativo']:.2f}")

        # --- 3. Comprobar que la exposición ya no cambia ---
        print(f"\n{N_COMPROBACION} imágenes seguidas:")
        for i in range(N_COMPROBACION):
            gris, t_sensor, t_llegada, md = camara.capturar(picam2, cfg)
            print(f"  {i}: exposición {md['ExposureTime']} us | ganancia {md['AnalogueGain']:.2f}"
                  f" | latencia {(t_llegada - t_sensor) / 1e6:5.1f} ms | brillo medio {gris.mean():5.1f}")

        # --- 4. Guardar la última imagen ---
        print("\nScalerCrop (zona del sensor usada):", md["ScalerCrop"])
        print("Forma de la imagen:", gris.shape, gris.dtype)
        ruta = RAIZ_PROYECTO / cfg_proyecto["carpeta_datos"] / "prueba_camara.png"
        cv2.imwrite(str(ruta), gris)
        print(f"Imagen guardada en {ruta}")
    finally:
        picam2.stop()
        picam2.close()


if __name__ == "__main__":
    main()