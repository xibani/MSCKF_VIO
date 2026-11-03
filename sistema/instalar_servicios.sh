#!/usr/bin/env bash
# sistema/instalar_servicios.sh — Arranque automático del VIO al encender la Pi (etapa 14).
#
# Instala dos servicios de systemd (el programa que arranca todo en Linux):
#   vio-visor-web  ->  python -m apps.visor_web               (página de la tablet, puerto 8000)
#   vio-pi         ->  python -m apps.pi_vio --ip 127.0.0.1   (cámara + IMU + MSCKF)
#
# Uso, desde la raíz del proyecto y SIN sudo (el script lo pide cuando hace falta):
#   bash sistema/instalar_servicios.sh ver        # enseña los servicios que instalaría
#   bash sistema/instalar_servicios.sh instalar   # los instala, los activa y los lanza ya
#   bash sistema/instalar_servicios.sh estado     # ¿están en marcha? y sus últimas líneas
#   bash sistema/instalar_servicios.sh quitar     # los para y deja de lanzarlos al arrancar

set -euo pipefail          # parar en el primer error (y con variables sin definir)

SERVICIOS="vio-visor-web vio-pi"
PROYECTO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # la carpeta padre de sistema/
USUARIO="$(id -un)"
PYTHON="$PROYECTO/.venv-rpi/bin/python"                       # el python del venv (no hace falta activarlo)

generar() {
    # Rellena la plantilla sistema/<servicio>.service con el usuario y las rutas de esta Pi
    # (y quita los \r de los finales de línea de Windows, si los hubiera: systemd no los quiere)
    sed -e 's/\r$//' -e "s|@USUARIO@|$USUARIO|g" -e "s|@PROYECTO@|$PROYECTO|g" -e "s|@PYTHON@|$PYTHON|g" \
        "$PROYECTO/sistema/$1.service"
}

case "${1:-}" in
    ver)
        for s in $SERVICIOS; do
            echo "=== /etc/systemd/system/$s.service ==="
            generar "$s"
            echo
        done
        ;;

    instalar)
        # 1. Comprobaciones: sin sudo (si no, los servicios irían como root), el venv,
        #    sus librerías y los permisos del usuario
        if [ "$(id -u)" -eq 0 ]; then
            echo "Ejecútalo SIN sudo:  bash sistema/instalar_servicios.sh instalar"
            exit 1
        fi
        if [ ! -x "$PYTHON" ]; then
            echo "ERROR: no encuentro $PYTHON (¿el venv se llama .venv-rpi?)"
            exit 1
        fi
        if ! "$PYTHON" -c "import picamera2, smbus2" 2>/dev/null; then
            echo "ERROR: el venv no puede importar picamera2 o smbus2"
            exit 1
        fi
        for grupo in video i2c; do
            if ! id -nG "$USUARIO" | grep -qw "$grupo"; then
                echo "AVISO: $USUARIO no está en el grupo $grupo (sin él no podrá abrir la cámara o la IMU)"
            fi
        done

        # 2. Nada lanzado a mano: la cámara y los puertos solo los puede usar un programa
        sudo systemctl stop $SERVICIOS 2>/dev/null || true          # por si se reinstala
        if pgrep -f "apps\.(pi_vio|visor_web)" >/dev/null; then
            echo "ERROR: hay un pi_vio o visor_web lanzado a mano. Ciérralo (Ctrl+C) y repite."
            exit 1
        fi

        # 3. Copiar los servicios, activarlos al arrancar (enable) y lanzarlos ya (--now)
        for s in $SERVICIOS; do
            generar "$s" | sudo tee "/etc/systemd/system/$s.service" >/dev/null
        done
        sudo systemctl daemon-reload
        sudo systemctl enable --now $SERVICIOS

        IP="$(hostname -I | awk '{print $1}')"
        echo
        echo "Instalados: se lanzarán solos cada vez que se encienda la Pi."
        echo "En la tablet:  http://${IP:-<IP de la Pi>}:8000   (o http://$(hostname).local:8000)"
        echo "Deja la placa QUIETA unos segundos al encenderla (inicialización)."
        echo "Ver el VIO en directo:  journalctl -u vio-pi -f      (Ctrl+C para dejar de mirar)"
        ;;

    estado)
        systemctl --no-pager --lines=5 status $SERVICIOS || true   # devuelve error si alguno está parado
        ;;

    quitar)
        sudo systemctl disable --now $SERVICIOS 2>/dev/null || true
        for s in $SERVICIOS; do
            sudo rm -f "/etc/systemd/system/$s.service"
        done
        sudo systemctl daemon-reload
        echo "Quitados: ya no se lanzan al arrancar. Se pueden lanzar a mano como antes."
        ;;

    *)
        echo "Uso: bash sistema/instalar_servicios.sh {ver|instalar|estado|quitar}"
        exit 1
        ;;
esac