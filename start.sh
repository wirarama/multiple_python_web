#!/usr/bin/env bash
# =============================================================================
#  start.sh - menjalankan semuanya dengan satu langkah (Linux).
#
#     bash start.sh                     setup otomatis bila perlu, jalankan layanan,
#                                       buka browser (bila ada desktop)
#     bash start.sh --boot              SERVER: pasang layanan systemd agar portal &
#                                       Data Mining UI menyala otomatis setiap boot
#                                       (setelah ini tidak perlu SSH untuk menyalakan)
#     bash start.sh --no-boot           matikan autostart systemd
#     bash start.sh --install-launcher  DESKTOP: buat ikon "Start/Stop Portal" di menu
#                                       aplikasi & Desktop (cukup klik dua kali)
#
#  Menghentikan: bash stop.sh   (atau ikon "Stop Portal")
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
chmod +x "$ROOT"/*.sh 2>/dev/null || true
if [ -f config.env ]; then set -a; . ./config.env; set +a; fi
PORTAL_PORT="${PORTAL_PORT:-8000}"
UNITS="mpw-portal mpw-datamining"

has_gui() { [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; }

fail() {
  echo "GAGAL: $*" >&2
  if ! [ -t 1 ] && has_gui; then   # dijalankan dari ikon: tampilkan pesan grafis
    command -v notify-send >/dev/null 2>&1 && notify-send "Portal Simulator dan ML" "Gagal: $*"
    command -v zenity >/dev/null 2>&1 && zenity --error --text="Gagal: $*\nLihat folder logs/ di $ROOT" 2>/dev/null
  fi
  [ -t 0 ] && [ -z "${SSH_CONNECTION:-}" ] && read -rp "Tekan Enter untuk menutup ..." _
  exit 1
}

sudo_cmd() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi; }

ensure_setup() {
  if [ ! -x .venv/bin/python ]; then
    echo "Instalasi pertama kali: menjalankan setup.sh (beberapa menit) ..."
    bash setup.sh || fail "setup.sh gagal"
  fi
  [ -f config.env ] || cp config.env.example config.env
}

# ------------------------------------------------------- autostart systemd
install_boot() {
  command -v systemctl >/dev/null 2>&1 || fail "systemd tidak tersedia di server ini"
  ensure_setup
  local user; user="$(id -un)"
  echo "Memasang layanan systemd ($UNITS) untuk user $user ..."
  bash stop.sh >/dev/null 2>&1 || true            # hindari bentrok port dengan run.sh
  for svc in portal datamining; do
    local desc="Portal gabungan Flask (SDN, Smart Home, RL-FL, PySpark)"
    [ "$svc" = datamining ] && desc="Data Mining UI (Streamlit)"
    sudo_cmd tee "/etc/systemd/system/mpw-$svc.service" >/dev/null <<EOF
[Unit]
Description=$desc
After=network-online.target
Wants=network-online.target

[Service]
User=$user
WorkingDirectory=$ROOT
ExecStart=/bin/bash $ROOT/run.sh fg $svc
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  done
  sudo_cmd systemctl daemon-reload
  if grep -q '^ENABLE_DATAMINING=0' config.env 2>/dev/null; then
    sudo_cmd systemctl enable --now mpw-portal || fail "systemctl enable gagal"
  else
    sudo_cmd systemctl enable --now $UNITS || fail "systemctl enable gagal"
  fi
  sleep 3
  systemctl --no-pager --lines=0 status $UNITS 2>/dev/null | grep -E '●|Active:' || true
  echo
  echo "Selesai. Layanan kini menyala otomatis setiap server boot."
  echo "Portal: http://<ip-server>:$PORTAL_PORT/"
  echo "Matikan autostart: bash start.sh --no-boot"
}

remove_boot() {
  echo "Mematikan autostart systemd ..."
  sudo_cmd systemctl disable --now $UNITS 2>/dev/null || true
  for u in $UNITS; do sudo_cmd rm -f "/etc/systemd/system/$u.service"; done
  sudo_cmd systemctl daemon-reload
  echo "Selesai. Jalankan manual dengan: bash start.sh"
}

# -------------------------------------------------- ikon desktop (Linux GUI)
install_launcher() {
  local apps="$HOME/.local/share/applications"
  mkdir -p "$apps"
  write_desktop() {  # $1=file $2=nama $3=script $4=ikon
    cat > "$1" <<EOF
[Desktop Entry]
Type=Application
Name=$2
Comment=Portal Simulator & Machine Learning ($ROOT)
Exec=bash "$ROOT/$3"
Path=$ROOT
Icon=$4
Terminal=true
Categories=Education;Development;
EOF
    chmod +x "$1"
  }
  write_desktop "$apps/mpw-start.desktop" "Start Portal" start.sh applications-internet
  write_desktop "$apps/mpw-stop.desktop" "Stop Portal" stop.sh process-stop
  local desk where="menu aplikasi"
  desk="$(command -v xdg-user-dir >/dev/null 2>&1 && xdg-user-dir DESKTOP || echo "$HOME/Desktop")"
  if [ -d "$desk" ]; then
    where="menu aplikasi dan $desk"
    cp "$apps/mpw-start.desktop" "$apps/mpw-stop.desktop" "$desk/"
    chmod +x "$desk"/mpw-*.desktop
    command -v gio >/dev/null 2>&1 && for f in "$desk"/mpw-*.desktop; do gio set "$f" metadata::trusted true 2>/dev/null || true; done
  fi
  echo "Ikon \"Start Portal\" dan \"Stop Portal\" dibuat di $where."
}

# --------------------------------------------------------------- utama
case "${1:-}" in
  --boot) install_boot; exit 0 ;;
  --no-boot) remove_boot; exit 0 ;;
  --install-launcher) install_launcher; exit 0 ;;
  -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "Opsi tidak dikenal: $1 (lihat: bash start.sh --help)" >&2; exit 1 ;;
esac

ensure_setup
if systemctl is-active --quiet mpw-portal 2>/dev/null; then
  echo "Layanan sudah dikelola systemd (autostart) dan sedang berjalan."
else
  bash run.sh start || fail "layanan tidak bisa dijalankan (lihat: bash run.sh logs)"
fi

URL="http://localhost:$PORTAL_PORT/"
if has_gui && [ -z "${SSH_CONNECTION:-}" ] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 &
  echo "Membuka $URL di browser ..."
  sleep 3     # beri waktu sebelum jendela terminal (dari ikon) tertutup
else
  echo "Portal: http://<ip-server>:$PORTAL_PORT/"
fi
