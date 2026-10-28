#!/usr/bin/env bash
# =============================================================================
#  run.sh - menjalankan / menghentikan layanan di latar belakang (aman ditinggal
#  logout dari SSH).
#
#     bash run.sh start   [portal|datamining]   jalankan (default: semua)
#     bash run.sh stop    [portal|datamining]   hentikan
#     bash run.sh restart [portal|datamining]
#     bash run.sh status                        cek proses & health check
#     bash run.sh logs    [portal|datamining]   tail -f log
#     bash run.sh fg      portal|datamining     jalan di foreground (untuk systemd/debug)
#
#  Tanpa Data Mining UI (Streamlit tidak dijalankan, port 8501 tidak dibuka/disentuh):
#     bash run.sh start --no-datamining         sekali jalan (juga untuk restart/status)
#     bash run.sh datamining off                permanen (ENABLE_DATAMINING=0 di config.env)
#     bash run.sh datamining on                 aktifkan kembali
#
#  Port & opsi diatur di config.env (dibuat oleh setup.sh).
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv"
VPY="$VENV/bin/python"
LOG="$ROOT/logs"
RUN="$ROOT/run"
mkdir -p "$LOG" "$RUN"

# ------------------------------------------------------------ argumen
NO_DM=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --no-datamining|--web-only) NO_DM=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# ------------------------------------------------------------ konfigurasi
if [ -f config.env ]; then set -a; . ./config.env; set +a; fi
[ "$NO_DM" = 1 ] && ENABLE_DATAMINING=0
HOST="${HOST:-0.0.0.0}"
PORTAL_PORT="${PORTAL_PORT:-8000}"
PORTAL_THREADS="${PORTAL_THREADS:-8}"
PORTAL_TIMEOUT="${PORTAL_TIMEOUT:-300}"
ENABLE_DATAMINING="${ENABLE_DATAMINING:-1}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
STREAMLIT_BASE_PATH="${STREAMLIT_BASE_PATH:-datamining}"
export HOST PORTAL_PORT STREAMLIT_PORT STREAMLIT_BASE_PATH ENABLE_DATAMINING MPLBACKEND=Agg
for v in SMARTHOME_DB MPW_APPS DATAMINING_URL; do   # variabel kosong jangan diekspor
  if [ -z "${!v:-}" ]; then unset "$v"; fi
done

[ -x "$VPY" ] || { echo "Virtual environment belum ada. Jalankan dulu: bash setup.sh" >&2; exit 1; }

# ---------------------------------------------------------------- utilitas
pidfile() { echo "$RUN/$1.pid"; }
is_running() {
  local f; f="$(pidfile "$1")"
  [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null
}
http_ok() {  # $1 = url
  "$VPY" -c "import sys,urllib.request
try: sys.exit(0 if urllib.request.urlopen(sys.argv[1], timeout=2).status == 200 else 1)
except Exception: sys.exit(1)" "$1"
}
wait_http() {  # $1 = nama, $2 = url
  for _ in $(seq 1 40); do http_ok "$2" && return 0; sleep 0.5; done
  echo "   (belum merespons $2 - cek: bash run.sh logs $1)"; return 1
}

cmd_portal() {   # perintah gunicorn (exec bila $1=fg)
  local extra=()
  [ "${1:-}" = "fg" ] || extra=(--daemon --pid "$(pidfile portal)")
  exec_or_run "${1:-}" "$VENV/bin/gunicorn" --chdir "$ROOT" \
    --workers 1 --worker-class gthread --threads "$PORTAL_THREADS" \
    --bind "$HOST:$PORTAL_PORT" --timeout "$PORTAL_TIMEOUT" \
    --access-logfile "$LOG/portal-access.log" --error-logfile "$LOG/portal.log" \
    --capture-output "${extra[@]}" wsgi:application
}
cmd_datamining() {
  cd "$ROOT/apps/datamining"   # agar .streamlit/config.toml terbaca
  exec_or_run "${1:-}" "$VPY" -m streamlit run app.py \
    --server.address "$HOST" --server.port "$STREAMLIT_PORT" \
    --server.baseUrlPath "$STREAMLIT_BASE_PATH" --server.headless true \
    --browser.gatherUsageStats false
}
exec_or_run() {
  local mode="$1"; shift
  if [ "$mode" = "fg" ]; then exec "$@"; else "$@"; fi
}

start_one() {
  case "$1" in
    portal)
      if is_running portal; then echo "portal sudah berjalan (pid $(cat "$(pidfile portal)"))"; return; fi
      echo "Menjalankan portal di $HOST:$PORTAL_PORT ..."
      cmd_portal
      wait_http portal "http://127.0.0.1:$PORTAL_PORT/healthz" && echo "   portal OK  -> http://<ip-server>:$PORTAL_PORT/"
      ;;
    datamining)
      if [ "$ENABLE_DATAMINING" != 1 ]; then
        echo "datamining tidak dijalankan (Data Mining UI dinonaktifkan; port $STREAMLIT_PORT tidak dipakai)"
        return
      fi
      if is_running datamining; then echo "datamining sudah berjalan (pid $(cat "$(pidfile datamining)"))"; return; fi
      echo "Menjalankan Data Mining UI (Streamlit) di $HOST:$STREAMLIT_PORT/$STREAMLIT_BASE_PATH ..."
      ( nohup setsid bash "$ROOT/run.sh" fg datamining >>"$LOG/datamining.log" 2>&1 & echo $! >"$(pidfile datamining)" )
      wait_http datamining "http://127.0.0.1:$STREAMLIT_PORT/$STREAMLIT_BASE_PATH/_stcore/health" \
        && echo "   datamining OK -> http://<ip-server>:$STREAMLIT_PORT/$STREAMLIT_BASE_PATH/"
      ;;
    *) echo "Layanan tidak dikenal: $1" >&2; exit 1 ;;
  esac
}

stop_one() {
  local f; f="$(pidfile "$1")"
  if ! is_running "$1"; then echo "$1 tidak berjalan"; rm -f "$f"; return; fi
  local pid; pid="$(cat "$f")"
  echo "Menghentikan $1 (pid $pid) ..."
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true   # grup proses (setsid)
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$pid" 2>/dev/null && { kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; }
  rm -f "$f"
}

status_one() {
  if [ "$1" = datamining ] && [ "$ENABLE_DATAMINING" != 1 ] && ! is_running datamining; then
    printf '%-11s dinonaktifkan (Streamlit tidak dijalankan)\n' "$1"; return
  fi
  if is_running "$1"; then printf '%-11s berjalan (pid %s)' "$1" "$(cat "$(pidfile "$1")")"
  else printf '%-11s berhenti' "$1"; fi
  case "$1" in
    portal) http_ok "http://127.0.0.1:$PORTAL_PORT/healthz" && echo "  | http :$PORTAL_PORT OK" || echo "  | http :$PORTAL_PORT -" ;;
    datamining) http_ok "http://127.0.0.1:$STREAMLIT_PORT/$STREAMLIT_BASE_PATH/_stcore/health" \
                  && echo "  | http :$STREAMLIT_PORT OK" || echo "  | http :$STREAMLIT_PORT -" ;;
  esac
}

targets() {
  if [ -n "${1:-}" ]; then echo "$1"
  elif [ "$ENABLE_DATAMINING" = 1 ]; then echo "portal datamining"
  else echo "portal"; fi
}

# Mode tanpa Data Mining: pastikan Streamlit yang masih berjalan ikut dimatikan
drop_datamining() {
  if [ "$ENABLE_DATAMINING" != 1 ] && is_running datamining; then
    echo "Data Mining UI dinonaktifkan - menghentikan Streamlit yang masih berjalan"
    stop_one datamining
  fi
}

set_cfg() {  # set_cfg KEY VALUE di config.env
  [ -f config.env ] || cp config.env.example config.env
  sed -i "/^$1=/d" config.env
  printf '%s=%s\n' "$1" "$2" >> config.env
}

# ------------------------------------------------------------------ perintah
ACTION="${1:-}"; shift || true
case "$ACTION" in
  start)
    drop_datamining
    if [ "$NO_DM" = 1 ] && [ -z "${1:-}" ] && is_running portal; then
      echo "Catatan: portal sudah berjalan; pakai 'bash run.sh restart --no-datamining' agar kartu Data Mining disembunyikan."
    fi
    for s in $(targets "${1:-}"); do start_one "$s"; done ;;
  stop)    for s in $(targets "${1:-}"); do stop_one "$s"; done ;;
  restart) drop_datamining; for s in $(targets "${1:-}"); do stop_one "$s"; start_one "$s"; done ;;
  status)  for s in portal datamining; do status_one "$s"; done ;;
  logs)
    case "${1:-}" in
      portal) tail -n 100 -f "$LOG/portal.log" ;;
      datamining) tail -n 100 -f "$LOG/datamining.log" ;;
      *) if [ "$ENABLE_DATAMINING" = 1 ]; then tail -n 50 -f "$LOG/portal.log" "$LOG/datamining.log"
         else tail -n 100 -f "$LOG/portal.log"; fi ;;
    esac ;;
  datamining)
    case "${1:-}" in
      off)
        set_cfg ENABLE_DATAMINING 0
        echo "Data Mining UI dinonaktifkan permanen (ENABLE_DATAMINING=0 di config.env)."
        ENABLE_DATAMINING=0; export ENABLE_DATAMINING
        drop_datamining
        if is_running portal; then stop_one portal; start_one portal; fi ;;
      on)
        if grep -q '^DATAMINING_NOTE=' config.env 2>/dev/null; then
          echo "Tidak bisa diaktifkan: $(grep '^DATAMINING_NOTE=' config.env | cut -d= -f2- | tr -d '"')" >&2; exit 1
        fi
        set_cfg ENABLE_DATAMINING 1
        echo "Data Mining UI diaktifkan (ENABLE_DATAMINING=1 di config.env)."
        ENABLE_DATAMINING=1; export ENABLE_DATAMINING
        "$VPY" -c "import streamlit" 2>/dev/null || { echo "Paket Streamlit belum terpasang - jalankan: bash setup.sh" >&2; exit 1; }
        if is_running portal; then stop_one portal; start_one portal; fi
        start_one datamining ;;
      *) echo "pakai: bash run.sh datamining on|off" >&2; exit 1 ;;
    esac ;;
  fg)
    case "${1:-}" in
      portal) cmd_portal fg ;;
      datamining) cmd_datamining fg ;;
      *) echo "pakai: bash run.sh fg portal|datamining" >&2; exit 1 ;;
    esac ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
