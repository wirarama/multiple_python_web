#!/usr/bin/env bash
# =============================================================================
#  stop.sh - menghentikan layanan portal dan Data Mining UI (Linux).
#
#     bash stop.sh                  hentikan semua layanan
#     bash stop.sh portal           hentikan portal saja
#     bash stop.sh datamining       hentikan Data Mining UI saja
#
#  Langkah: (1) hentikan lewat PID di run/ (sama dengan "bash run.sh stop"),
#  (2) bila port masih dipakai oleh proses gunicorn/streamlit milik paket ini
#  (mis. file PID hilang atau dijalankan dengan "run.sh fg"), proses itu ikut dihentikan.
#  Proses lain yang kebetulan memakai port yang sama TIDAK disentuh.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN="$ROOT/run"

if [ -f config.env ]; then set -a; . ./config.env; set +a; fi
PORTAL_PORT="${PORTAL_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

TARGET="${1:-all}"
case "$TARGET" in
  all|portal|datamining) ;;
  -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "Layanan tidak dikenal: $TARGET (pilihan: portal, datamining)" >&2; exit 1 ;;
esac

# PID yang LISTEN di port tertentu
pids_on_port() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnpH "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2
  elif command -v lsof >/dev/null 2>&1; then
    lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null
  elif command -v fuser >/dev/null 2>&1; then
    fuser "$port"/tcp 2>/dev/null | tr ' ' '\n'
  fi | sort -u | grep -E '^[0-9]+$' || true
}

# hentikan via PID file
stop_pidfile() {
  local svc="$1" f="$RUN/$1.pid"
  [ -f "$f" ] || return 0
  local pid; pid="$(cat "$f" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Menghentikan $svc (pid $pid) ..."
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$f"
}

# hentikan sisa proses milik paket ini yang masih memegang port
stop_port() {
  local svc="$1" port="$2" pattern="$3" found=0
  for pid in $(pids_on_port "$port"); do
    local cmd; cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
    if echo "$cmd" | grep -Eq "$pattern"; then
      found=1
      echo "Menghentikan sisa proses $svc di port $port (pid $pid)"
      kill -TERM "$pid" 2>/dev/null || true
    elif [ -n "$cmd" ]; then
      echo "  ! port $port dipakai proses lain (pid $pid: ${cmd:0:80}) - tidak dihentikan"
    fi
  done
  if [ "$found" = 1 ]; then
    sleep 1.5
    for pid in $(pids_on_port "$port"); do
      tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -Eq "$pattern" && kill -KILL "$pid" 2>/dev/null || true
    done
  fi
}

stop_service() {
  local svc="$1"
  # layanan yang dipasang dengan "start.sh --boot" dikelola systemd
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "mpw-$svc" 2>/dev/null; then
    echo "Menghentikan layanan systemd mpw-$svc (menyala lagi saat boot; matikan permanen: bash start.sh --no-boot)"
    if [ "$(id -u)" = 0 ]; then systemctl stop "mpw-$svc"; else sudo systemctl stop "mpw-$svc"; fi
  fi
  stop_pidfile "$svc"
  case "$svc" in
    portal)     stop_port portal "$PORTAL_PORT" 'gunicorn|wsgi(\.py|:application)|serve_waitress' ;;
    datamining) stop_port datamining "$STREAMLIT_PORT" 'streamlit' ;;
  esac
  local port; [ "$svc" = portal ] && port="$PORTAL_PORT" || port="$STREAMLIT_PORT"
  if [ -z "$(pids_on_port "$port")" ]; then
    echo "$svc berhenti (port $port bebas)"
  fi
}

if [ "$TARGET" = all ]; then
  stop_service portal
  stop_service datamining
else
  stop_service "$TARGET"
fi

# dijalankan dari ikon desktop: beri waktu membaca sebelum jendela tertutup
if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && [ -z "${SSH_CONNECTION:-}" ] && [ -t 1 ]; then sleep 2; fi
