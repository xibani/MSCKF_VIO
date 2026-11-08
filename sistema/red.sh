#!/usr/bin/env bash
# sistema/red.sh — La Pi como punto de acceso Wi-Fi propio, para la tablet (etapa 17).
#
# La Wi-Fi de la Pi hace UNA cosa a la vez: o se conecta a una red con Internet (modo cliente:
# git, apt) o crea la suya (punto de acceso: la tablet se conecta a la Pi, sin Internet).
# Con "instalar", al encender decide sola: Wi-Fi conocida si la encuentra; si no, punto de acceso.
#
# Uso, desde la raíz del proyecto y SIN sudo (el script lo pide cuando hace falta):
#   bash sistema/red.sh crear       # una vez: crea el punto de acceso (pide la contraseña)
#   bash sistema/red.sh instalar    # al encender: Wi-Fi conocida o, si no hay, punto de acceso
#   bash sistema/red.sh punto       # pasar YA a punto de acceso (se corta el SSH que vaya por Wi-Fi)
#   bash sistema/red.sh cliente     # volver YA a la Wi-Fi con Internet (se corta el SSH por el punto)
#   bash sistema/red.sh estado      # en qué modo está ahora
#   bash sistema/red.sh ver         # el servicio que instalaría
#   bash sistema/red.sh quitar      # deja de decidir al encender (se queda siempre en cliente)
#   bash sistema/red.sh permitir    # etapa 18: el visor web puede cambiar de red (sudo nmcli sin contraseña)
#                                   # (para deshacerlo: sudo rm /etc/sudoers.d/vio-red)

set -euo pipefail          # parar en el primer error (y con variables sin definir)

PROYECTO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # la carpeta padre de sistema/
CONEXION="VIO-Pi-punto"    # nombre de la conexión dentro de NetworkManager
SSID="VIO-Pi"              # nombre de la red que verá la tablet
IP_PUNTO="10.42.0.1/24"    # la Pi dentro de su red: en la tablet, http://10.42.0.1:8000
CANAL=6                    # 2,4 GHz, canal 6: lo ve cualquier tablet
ESPERA_CLIENTE_S=40        # al encender: segundos para encontrar una Wi-Fi conocida
SERVICIO="vio-red"
SUDOERS="/etc/sudoers.d/vio-red"   # etapa 18: permiso del visor web para cambiar de red

generar() {
    # Rellena la plantilla sistema/vio-red.service con las rutas y valores de este script
    sed -e 's/\r$//' -e "s|@PROYECTO@|$PROYECTO|g" -e "s|@ESPERA@|$ESPERA_CLIENTE_S|g" \
        -e "s|@CONEXION@|$CONEXION|g" "$PROYECTO/sistema/$SERVICIO.service"
}

regla_sudo() {
    # Este usuario (el del servicio vio-visor-web) puede ejecutar nmcli como root sin contraseña.
    # Solo nmcli: no da permiso para nada más
    echo "$(id -un) ALL=(root) NOPASSWD: $(command -v nmcli || echo /usr/bin/nmcli)"
}

existe_punto() {
    nmcli -t -f NAME connection show | grep -qx "$CONEXION"
}

en_segundo_plano() {
    # Cambiar de red corta la conexión SSH por la que quizá se ha lanzado esta orden: nmcli
    # va aparte (nohup) para que el cambio termine aunque se corte la sesión
    sudo -v                                       # pide la contraseña ahora, no en segundo plano
    nohup sudo nmcli "$@" >/dev/null 2>&1 &
}

case "${1:-}" in
    crear)
        read -r -s -p "Contraseña para la red $SSID (8 caracteres o más): " CLAVE
        echo
        if [ "${#CLAVE}" -lt 8 ]; then
            echo "ERROR: WPA2 necesita una contraseña de 8 caracteres o más"
            exit 1
        fi
        sudo nmcli connection delete "$CONEXION" >/dev/null 2>&1 || true    # por si se recrea
        # Modo AP en 2,4 GHz; "shared": NetworkManager reparte direcciones (DHCP) a la tablet.
        # WPA2 con CCMP y sin PMF: lo que mejor aceptan las tablets con el chip Wi-Fi de la Pi
        sudo nmcli connection add type wifi ifname wlan0 con-name "$CONEXION" autoconnect no \
            ssid "$SSID" 802-11-wireless.mode ap 802-11-wireless.band bg \
            802-11-wireless.channel "$CANAL" \
            ipv4.method shared ipv4.addresses "$IP_PUNTO" ipv6.method disabled \
            wifi-sec.key-mgmt wpa-psk wifi-sec.proto rsn wifi-sec.pairwise ccmp \
            wifi-sec.group ccmp wifi-sec.pmf disable wifi-sec.psk "$CLAVE"
        echo "Punto de acceso creado (red $SSID). Pruébalo con: bash sistema/red.sh punto"
        ;;

    instalar)
        if ! existe_punto; then
            echo "ERROR: primero crea el punto de acceso: bash sistema/red.sh crear"
            exit 1
        fi
        generar | sudo tee "/etc/systemd/system/$SERVICIO.service" >/dev/null
        sudo systemctl daemon-reload
        sudo systemctl enable "$SERVICIO"         # sin --now: actúa en el próximo encendido
        echo "Instalado. Al encender: Wi-Fi conocida si la encuentra en $ESPERA_CLIENTE_S s;"
        echo "si no, red $SSID y en la tablet http://${IP_PUNTO%/*}:8000"
        ;;

    punto)
        if ! existe_punto; then
            echo "ERROR: primero crea el punto de acceso: bash sistema/red.sh crear"
            exit 1
        fi
        echo "Pasando a punto de acceso: conecta la tablet a la red $SSID y abre http://${IP_PUNTO%/*}:8000"
        echo "(si estás por SSH a través de la Wi-Fi, la sesión se cortará ahora)"
        en_segundo_plano connection up "$CONEXION"
        ;;

    cliente)
        echo "Volviendo a la Wi-Fi con Internet (si estás por SSH a través del punto, se cortará)"
        en_segundo_plano connection down "$CONEXION"   # NetworkManager se conecta solo a la conocida
        ;;

    estado)
        if nmcli -t -f NAME connection show --active | grep -qx "$CONEXION"; then
            echo "Modo PUNTO DE ACCESO: red $SSID, en la tablet http://${IP_PUNTO%/*}:8000"
        else
            echo "Modo CLIENTE: conexiones activas:"
            nmcli -t -f NAME,DEVICE connection show --active | sed 's/^/  /'
            echo "IP: $(hostname -I)"
        fi
        if systemctl is-enabled "$SERVICIO" >/dev/null 2>&1; then
            echo "Al encender: decide solo (servicio $SERVICIO instalado)"
        else
            echo "Al encender: siempre cliente (servicio $SERVICIO no instalado)"
        fi
        ;;

    ver)
        echo "=== /etc/systemd/system/$SERVICIO.service ==="
        generar
        echo "=== $SUDOERS (bash sistema/red.sh permitir) ==="
        regla_sudo
        ;;

    permitir)
        TEMPORAL="$(mktemp)"
        regla_sudo > "$TEMPORAL"
        # visudo -c comprueba la regla ANTES de instalarla: una regla mal escrita en
        # /etc/sudoers.d/ puede dejar sudo sin funcionar
        sudo visudo -cf "$TEMPORAL" >/dev/null
        sudo install -m 0440 -o root -g root "$TEMPORAL" "$SUDOERS"
        rm -f "$TEMPORAL"
        echo "Hecho: $(regla_sudo)"
        echo "Ya se puede cambiar de red desde el panel 'Red Wi-Fi' del visor web"
        ;;

    quitar)
        sudo systemctl disable "$SERVICIO" 2>/dev/null || true
        sudo rm -f "/etc/systemd/system/$SERVICIO.service"
        sudo systemctl daemon-reload
        echo "Quitado: al encender se queda en la Wi-Fi conocida (el punto de acceso sigue creado)"
        ;;

    *)
        echo "Uso: bash sistema/red.sh {crear|instalar|punto|cliente|estado|ver|quitar|permitir}"
        exit 1
        ;;
esac