#!/usr/bin/env bash
# =============================================================================
#  setup.sh - membuat virtual environment Python (.venv) dan memasang semua
#  dependensi untuk portal gabungan + Data Mining UI (Streamlit).
#
#  Pemakaian (di server Linux, lewat SSH):
#     git clone https://github.com/<akun>/multiple_python_web.git
#     cd multiple_python_web
#     bash setup.sh                 # setup baru
#     bash run.sh start             # jalankan semua layanan
#
#  Opsi:
#     --python /usr/bin/python3.11  pilih interpreter Python tertentu
#     --recreate                    hapus .venv lama lalu buat ulang dari nol
#     --sync-git                    tarik ulang script dari repo GitHub sumber dulu
#     --no-datamining               lewati paket Streamlit/Data Mining (lebih ringan)
#     --legacy-cpu                  paksa mode CPU lama (lihat constraints-legacy-cpu.txt)
#     --no-cpu-check                lewati pemeriksaan CPU x86-64-v2
#     -h | --help                   tampilkan bantuan
#
#  Folder .venv/ tercantum di .gitignore sehingga tidak ikut ter-commit/clone.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv"
PY_REQ_MIN="3.9"

PYTHON_BIN="${PYTHON:-}"
RECREATE=0
SYNC_GIT=0
WITH_DM=1
LEGACY_CPU=0
CPU_CHECK=1
MISSING=""

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m OK\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m !!\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31mERR\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --recreate) RECREATE=1; shift ;;
    --sync-git) SYNC_GIT=1; shift ;;
    --no-datamining) WITH_DM=0; shift ;;
    --legacy-cpu) LEGACY_CPU=1; shift ;;
    --no-cpu-check) CPU_CHECK=0; shift ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Opsi tidak dikenal: $1 (lihat: bash setup.sh --help)" ;;
  esac
done

# ------------------------------------------------------------- 0. cek CPU
# Wheel NumPy >= 2.4, pandas >= 3.1 dan pyarrow butuh CPU x86-64-v2 (SSE4.2, POPCNT, ...).
# VM dengan tipe CPU lama (mis. "kvm64"/"qemu64" di Proxmox/KVM) tidak memilikinya.
CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
if [ "$CPU_CHECK" = 1 ] && [ "$LEGACY_CPU" = 0 ] && [ "$(uname -m)" = x86_64 ] && [ -r /proc/cpuinfo ]; then
  FLAGS=" $(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2) "
  for f in cx16 lahf_lm popcnt sse4_1 sse4_2 ssse3; do
    case "$FLAGS" in *" $f "*) ;; *) MISSING="$MISSING $f" ;; esac
  done
  [ -n "$MISSING" ] && LEGACY_CPU=1
fi
if [ "$LEGACY_CPU" = 1 ]; then
  warn "CPU server tidak mendukung x86-64-v2 (tidak ada:${MISSING:- -})"
  warn "  Model CPU: ${CPU_MODEL:-?}"
  warn "  -> MODE KOMPATIBEL: numpy<2.4, pandas<3.1, tanpa pyarrow/Streamlit."
  warn "     Data Mining UI dinonaktifkan; 4 aplikasi Flask tetap berjalan."
  warn "  Solusi permanen: minta pengelola VM mengganti tipe CPU menjadi 'host'"
  warn "  (atau 'x86-64-v2-AES' di Proxmox), lalu jalankan: bash setup.sh --recreate"
  WITH_DM=0
  PIP_CONSTRAINT_ARGS=(-c constraints-legacy-cpu.txt)
else
  PIP_CONSTRAINT_ARGS=()
fi

# ---------------------------------------------------------------- 1. Python
py_ok() {  # versi >= minimum?
  "$1" -c "import sys; sys.exit(0 if sys.version_info >= tuple(map(int,'$PY_REQ_MIN'.split('.'))) else 1)" 2>/dev/null
}
if [ -z "$PYTHON_BIN" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$c" >/dev/null 2>&1 && py_ok "$(command -v "$c")"; then
      PYTHON_BIN="$(command -v "$c")"; break
    fi
  done
fi
[ -n "$PYTHON_BIN" ] || die "Python >= $PY_REQ_MIN tidak ditemukan. Pasang dulu, mis.: sudo apt install python3 python3-venv"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Interpreter tidak ditemukan: $PYTHON_BIN"
py_ok "$PYTHON_BIN" || die "$PYTHON_BIN terlalu lama; butuh Python >= $PY_REQ_MIN"
PY_VER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
info "Memakai $PYTHON_BIN (Python $PY_VER)"

if ! "$PYTHON_BIN" -c "import venv, ensurepip" >/dev/null 2>&1; then
  die "Modul venv/ensurepip belum ada. Debian/Ubuntu: sudo apt install python${PY_VER}-venv  (atau python3-venv)
     RHEL/Rocky/Fedora: sudo dnf install python3 (venv sudah termasuk)"
fi

# ------------------------------------------------------------ 2. sync (opsional)
if [ "$SYNC_GIT" = 1 ]; then
  command -v git >/dev/null 2>&1 || die "git belum terpasang (sudo apt install git)"
  info "Sinkronisasi script dari GitHub sumber"
  "$PYTHON_BIN" sync.py --source git
fi
for d in sdn smarthome rlfl pyspark datamining; do
  [ -f "apps/$d/app.py" ] || warn "apps/$d/app.py tidak ada - jalankan: python3 sync.py --source git"
done

# ------------------------------------------------------------------ 3. venv
if [ -d "$VENV" ] && [ "$RECREATE" = 1 ]; then
  info "Menghapus .venv lama (--recreate)"
  rm -rf "$VENV"
fi
if [ -d "$VENV" ] && ! "$VENV/bin/python" -c "import sys" >/dev/null 2>&1; then
  warn ".venv rusak / dibuat di mesin lain - dibuat ulang"
  rm -rf "$VENV"
fi
if [ ! -d "$VENV" ]; then
  info "Membuat virtual environment di .venv"
  "$PYTHON_BIN" -m venv "$VENV"
else
  info ".venv sudah ada - memperbarui paket (pakai --recreate untuk setup bersih)"
fi
VPY="$VENV/bin/python"

info "Memperbarui pip"
"$VPY" -m pip install --upgrade --quiet pip setuptools wheel

if [ "$WITH_DM" = 1 ]; then REQ="requirements.txt"; else REQ="requirements-web.txt"; fi
info "Memasang dependensi dari $REQ (bisa beberapa menit)"
"$VPY" -m pip install --quiet "${PIP_CONSTRAINT_ARGS[@]}" -r "$REQ"
if [ "$LEGACY_CPU" = 1 ]; then
  # sisa instalasi lama: pyarrow membuat pandas crash di CPU ini
  for pkg in pyarrow streamlit; do
    if "$VPY" -m pip show -q "$pkg" >/dev/null 2>&1; then
      info "Menghapus $pkg (tidak didukung CPU ini)"
      "$VPY" -m pip uninstall -y -q "$pkg"
    fi
  done
fi
ok "Dependensi terpasang"

# Uji impor tiap paket di proses terpisah (crash "Illegal instruction" tidak menghentikan setup)
check_imports() {
  local bad=""
  for m in "$@"; do
    if ! MPLBACKEND=Agg "$VPY" -c "import $m" >/dev/null 2>&1; then
      bad="$bad $m"
      warn "  gagal mengimpor: $m"
      MPLBACKEND=Agg "$VPY" -c "import $m" 2>&1 | tail -2 | sed 's/^/        /' >&2 || true
    fi
  done
  [ -z "$bad" ]
}
info "Uji impor paket"
if ! check_imports numpy pandas scipy matplotlib networkx openpyxl flask_sqlalchemy; then
  if [ "$LEGACY_CPU" = 0 ]; then
    die "Paket inti gagal dimuat. Bila pesannya 'Illegal instruction' atau 'X86_V2', jalankan: bash setup.sh --legacy-cpu"
  fi
  die "Paket inti gagal dimuat walau dalam mode kompatibel - kirimkan pesan di atas ke pengembang."
fi
ok "Paket inti dapat dimuat"

# --------------------------------------------------------------- 4. konfigurasi
mkdir -p logs run
chmod +x "$ROOT"/*.sh 2>/dev/null || true
# izin eksekusi di atas jangan dianggap perubahan oleh git (agar "git pull" tidak bentrok)
[ -d "$ROOT/.git" ] && git -C "$ROOT" config core.fileMode false 2>/dev/null || true   # file dari Windows tidak membawa izin eksekusi
if [ ! -f config.env ]; then
  cp config.env.example config.env
  ok "config.env dibuat dari config.env.example (ubah port di sini bila perlu)"
fi
set_cfg() {  # set_cfg KEY VALUE  (tambah atau ganti baris di config.env)
  if grep -q "^$1=" config.env; then
    sed -i "/^$1=/d" config.env
  fi
  case "$2" in
    *" "*) printf '%s="%s"\n' "$1" "$2" >> config.env ;;   # nilai berspasi wajib dikutip (config.env di-source)
    *)     printf '%s=%s\n' "$1" "$2" >> config.env ;;
  esac
}
if [ "$LEGACY_CPU" = 1 ]; then
  set_cfg ENABLE_DATAMINING 0
  set_cfg DATAMINING_NOTE "CPU server tidak mendukung x86-64-v2 (SSE4.2) yang dibutuhkan Streamlit/pyarrow. Minta pengelola VM mengganti tipe CPU menjadi host, lalu jalankan: bash setup.sh --recreate"
elif [ "$WITH_DM" = 0 ]; then
  set_cfg ENABLE_DATAMINING 0
elif grep -q '^DATAMINING_NOTE="CPU' config.env; then
  # CPU sudah diperbaiki: aktifkan kembali Data Mining UI
  set_cfg ENABLE_DATAMINING 1
  sed -i '/^DATAMINING_NOTE=/d' config.env
  ok "CPU kini mendukung x86-64-v2 - Data Mining UI diaktifkan kembali"
fi

# ----------------------------------------------------------------- 5. uji cepat
info "Uji impor aplikasi Flask"
MPLBACKEND=Agg "$VPY" - <<'PYEOF'
import sys
from portal.gateway import build, FLASK_APPS
gw = build()
loaded = [p for p in gw.apps]
print("   dimuat :", ", ".join(loaded) or "-")
if len(loaded) != len(FLASK_APPS):
    sys.exit("   Ada aplikasi yang gagal dimuat (lihat pesan GAGAL di atas)")
PYEOF
ok "Semua aplikasi Flask dapat dimuat"
if [ "$WITH_DM" = 1 ]; then
  check_imports pyarrow streamlit sklearn statsmodels mlxtend \
    || die "Paket Data Mining UI gagal dimuat. Coba: bash setup.sh --no-datamining (atau --legacy-cpu)"
  ok "Paket Data Mining UI siap"
fi

cat <<EOF

Setup selesai.
  Jalankan     : bash run.sh start
  Status       : bash run.sh status
  Hentikan     : bash run.sh stop
  Log          : bash run.sh logs
Portal (default) di http://<ip-server>:8000/  - port diatur di config.env
EOF
if [ "$LEGACY_CPU" = 1 ]; then
  echo
  warn "Mode kompatibel CPU lama aktif: Data Mining UI dinonaktifkan (lihat pesan di atas)."
fi
