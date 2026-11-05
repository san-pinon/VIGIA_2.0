#!/bin/bash
# Reset the Canon's USB port on multicam: disable it, wait, enable it again. This
# is like a replug. It can't switch the camera itself off and on.
#
# Runs as root through sudo, called by the orchestrator. Install a root-owned copy:
#   sudo install -m 755 -o root -g root scripts/canon_usb_reset.sh \
#       /usr/local/sbin/vigia-canon-usb-reset
#
# Usage: vigia-canon-usb-reset [fallback-port]
# The Canon's own port is used when it's on the bus. When it has dropped off, the
# fallback port (e.g. 1-4) is used, unless another device is there: the Optris
# must never be touched.
#
# Exit: 0 Canon back; 2 no usable port; 3 another device on that port;
#       4 kernel control missing; 5 write failed; 6 reset done, Canon not back.

set -u

SYS=/sys/bus/usb/devices
VENDOR=04a9
PRODUCT=32d9
fallback=${1:-}

is_canon() {
    [ "$(cat "$1/idVendor" 2>/dev/null)" = "$VENDOR" ] &&
        [ "$(cat "$1/idProduct" 2>/dev/null)" = "$PRODUCT" ]
}

find_canon() {
    local d
    for d in "$SYS"/*; do
        if is_canon "$d"; then
            echo "${d##*/}"
            return 0
        fi
    done
    return 1
}

port=$(find_canon) || port=$fallback
if ! [[ "$port" =~ ^[0-9]+-[0-9]+(\.[0-9]+)*$ ]]; then
    echo "Canon not on the bus and no valid fallback port: '$port'" >&2
    exit 2
fi
if [ -e "$SYS/$port/idVendor" ] && ! is_canon "$SYS/$port"; then
    echo "port $port holds $(cat "$SYS/$port/idVendor"):$(cat "$SYS/$port/idProduct"), not the Canon: left alone" >&2
    exit 3
fi

bus=${port%%-*}
path=${port#*-}
if [[ "$path" == *.* ]]; then
    parent="$bus-${path%.*}"
    control="$SYS/$parent:1.0/$parent-port${path##*.}/disable"
else
    control="$SYS/$bus-0:1.0/usb$bus-port$path/disable"
fi
if [ ! -w "$control" ]; then
    echo "no control file: $control" >&2
    exit 4
fi

echo "resetting port $port"
echo 1 >"$control" || exit 5
sleep 3
echo 0 >"$control" || exit 5

for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    if back=$(find_canon); then
        echo "Canon back on port $back"
        exit 0
    fi
done
echo "reset done, Canon not back on the bus" >&2
exit 6
