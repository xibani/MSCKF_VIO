#!/usr/bin/env bash
# sistema/red_arranque.sh — Al encender: Wi-Fi conocida o, si no aparece, punto de acceso propio.
#
# Lo lanza el servicio vio-red (sistema/red.sh instalar), como root:
#     red_arranque.sh <segundos de espera> <nombre de la conexión del punto de acceso>
# NetworkManager se conecta solo a las Wi-Fi conocidas que estén a la vista. Si pasado ese
# tiempo wlan0 sigue sin conectar, no hay ninguna cerca: se levanta el punto de acceso.

set -u

ESPERA_S="$1"
CONEXION="$2"
PASO_S=2                                    # cada cuánto se mira si ya está conectada

for ((s = 0; s < ESPERA_S; s += PASO_S)); do
    if nmcli -t -f DEVICE,STATE device status | grep -q '^wlan0:connected$'; then
        echo "Conectada a una Wi-Fi conocida: modo cliente"
        exit 0
    fi
    sleep "$PASO_S"
done

echo "Ninguna Wi-Fi conocida en ${ESPERA_S} s: punto de acceso ($CONEXION)"
nmcli connection up "$CONEXION"