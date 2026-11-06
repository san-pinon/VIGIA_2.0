#!/bin/bash
# Reset the Canon's USB connection on multicam: the kernel resets its port and
# detects it again. This can free a camera that's on the bus but stuck. It can't
# switch the camera off and on, and it can't help when the camera has dropped off
# the bus: multicam's root hub has no power switching (lsusb), and its kernel
# (5.4) has no per-port disable.
#
# Runs as root through sudo, called by the orchestrator. Install a root-owned copy:
#   sudo install -m 755 -o root -g root scripts/canon_usb_reset.sh \
#       /usr/local/sbin/vigia-canon-usb-reset
#
# It only ever acts on the device with the Canon's IDs, so never on the Optris.
#
# Exit: 0 Canon back; 2 Canon not on the bus; 5 reset failed; 6 reset done, Canon
# not back.

set -u

SYS=/sys/bus/usb/devices
USBRESET=/usr/bin/usbreset
VENDOR=04a9
PRODUCT=32d9

find_canon() {
    local d
    for d in "$SYS"/*; do
        if [ "$(cat "$d/idVendor" 2>/dev/null)" = "$VENDOR" ] &&
            [ "$(cat "$d/idProduct" 2>/dev/null)" = "$PRODUCT" ]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

if ! device=$(find_canon); then
    echo "Canon not on the bus: nothing to reset" >&2
    exit 2
fi

address=$(printf "%03d/%03d" "$(cat "$device/busnum")" "$(cat "$device/devnum")")
echo "resetting ${device##*/} ($address)"
"$USBRESET" "$address" || exit 5

for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    if back=$(find_canon); then
        echo "Canon back on port ${back##*/}"
        exit 0
    fi
done
echo "reset done, Canon not back on the bus" >&2
exit 6
