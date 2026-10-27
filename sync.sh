#!/usr/bin/env bash
# Sinkronisasi script dari repo sumber. Argumen diteruskan ke sync.py, contoh:
#   bash sync.sh                 (dari folder saudara ../software_defined_network, dst.)
#   bash sync.sh --source git    (clone dari GitHub - cocok di server)
#   bash sync.sh --only sdn --dry-run
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="python3"; [ -x .venv/bin/python ] && PY=".venv/bin/python"
"$PY" sync.py "$@"
