"""
Pengendali layanan untuk Windows (dipanggil oleh run.bat).

    python -m portal.winctl start   [portal|datamining]
    python -m portal.winctl stop    [portal|datamining]
    python -m portal.winctl restart [portal|datamining]
    python -m portal.winctl status
    python -m portal.winctl logs    [portal|datamining]
    python -m portal.winctl fg      portal|datamining
    python -m portal.winctl open    [portal|datamining]   buka di browser

Proses dijalankan di latar belakang tanpa jendela konsol; PID disimpan di run\\,
log di logs\\. Hanya memakai pustaka standar (selain waitress/streamlit yang dijalankan).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs"
RUN = ROOT / "run"
IS_WIN = os.name == "nt"
SERVICES = ("portal", "datamining")
EMPTY_UNSET = ("SMARTHOME_DB", "MPW_APPS", "DATAMINING_URL")


# ------------------------------------------------------------- konfigurasi
def load_config():
    """Baca config.env (KEY=VALUE) ke os.environ; nilai kosong = tidak diset."""
    f = ROOT / "config.env"
    if f.is_file():
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k, v)
            elif k in EMPTY_UNSET:
                os.environ.pop(k, None)
    defaults = {"HOST": "0.0.0.0", "PORTAL_PORT": "8000", "PORTAL_THREADS": "8",
                "PORTAL_TIMEOUT": "300", "ENABLE_DATAMINING": "1", "STREAMLIT_PORT": "8501",
                "STREAMLIT_BASE_PATH": "datamining", "MPLBACKEND": "Agg"}
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def cfg(k):
    return os.environ[k]


# ---------------------------------------------------------------- utilitas
def health_url(svc):
    if svc == "portal":
        return f"http://127.0.0.1:{cfg('PORTAL_PORT')}/healthz"
    base = cfg("STREAMLIT_BASE_PATH").strip("/")
    return f"http://127.0.0.1:{cfg('STREAMLIT_PORT')}/{base + '/' if base else ''}_stcore/health"


def public_url(svc):
    if svc == "portal":
        return f"http://localhost:{cfg('PORTAL_PORT')}/"
    base = cfg("STREAMLIT_BASE_PATH").strip("/")
    return f"http://localhost:{cfg('STREAMLIT_PORT')}/{base + '/' if base else ''}"


def http_ok(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def pidfile(svc):
    return RUN / f"{svc}.pid"


def read_pid(svc):
    try:
        return int(pidfile(svc).read_text().strip())
    except Exception:
        return None


def pid_alive(pid):
    if not pid:
        return False
    if IS_WIN:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, errors="ignore").stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def command(svc):
    py = sys.executable
    if svc == "portal":
        if IS_WIN:
            return [py, "-m", "portal.serve_waitress"], ROOT
        return [py, "wsgi.py"], ROOT
    return [py, "-m", "streamlit", "run", "app.py",
            "--server.address", cfg("HOST"), "--server.port", cfg("STREAMLIT_PORT"),
            "--server.baseUrlPath", cfg("STREAMLIT_BASE_PATH"), "--server.headless", "true",
            "--browser.gatherUsageStats", "false"], ROOT / "apps" / "datamining"


# ---------------------------------------------------------------- perintah
def start(svc):
    if svc == "datamining" and cfg("ENABLE_DATAMINING") != "1":
        print("datamining dinonaktifkan (ENABLE_DATAMINING=0 di config.env)")
        return True
    pid = read_pid(svc)
    if pid_alive(pid):
        print(f"{svc} sudah berjalan (pid {pid})")
        return True
    if http_ok(health_url(svc), 1):
        print(f"{svc}: port sudah dipakai proses lain - hentikan dulu proses tersebut")
        return False
    LOG.mkdir(exist_ok=True)
    RUN.mkdir(exist_ok=True)
    cmd, cwd = command(svc)
    logf = open(LOG / f"{svc}.log", "ab")
    logf.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n".encode())
    logf.flush()
    kw = {}
    if IS_WIN:
        # konsol tersembunyi sendiri: tanpa jendela dan tetap jalan bila jendela run.bat ditutup
        # (DETACHED_PROCESS tidak dipakai karena launcher venv akan membuka konsol baru)
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, env=env, close_fds=True, **kw)
    pidfile(svc).write_text(str(p.pid))
    print(f"Menjalankan {svc} (pid {p.pid}) ...", flush=True)
    for _ in range(120):                       # tunggu sampai 60 detik
        if http_ok(health_url(svc), 1):
            print(f"   {svc} OK -> {public_url(svc)}")
            return True
        if p.poll() is not None:
            print(f"   {svc} BERHENTI (kode {p.returncode}). Lihat logs\\{svc}.log")
            pidfile(svc).unlink(missing_ok=True)
            return False
        time.sleep(0.5)
    print(f"   {svc} belum merespons - lihat logs\\{svc}.log")
    return False


def stop(svc):
    pid = read_pid(svc)
    if not pid_alive(pid):
        print(f"{svc} tidak berjalan")
        pidfile(svc).unlink(missing_ok=True)
        return True
    print(f"Menghentikan {svc} (pid {pid}) ...")
    if IS_WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(20):
        if not pid_alive(pid):
            break
        time.sleep(0.5)
    pidfile(svc).unlink(missing_ok=True)
    return True


def status():
    for svc in SERVICES:
        pid = read_pid(svc)
        run = f"berjalan (pid {pid})" if pid_alive(pid) else "berhenti"
        port = cfg("PORTAL_PORT") if svc == "portal" else cfg("STREAMLIT_PORT")
        ok = "OK" if http_ok(health_url(svc)) else "-"
        print(f"{svc:<11} {run:<22} | http :{port} {ok}")


def logs(svc):
    files = [LOG / f"{s}.log" for s in ([svc] if svc else SERVICES)]
    pos = {}
    for f in files:
        if f.exists():
            data = f.read_bytes()[-6000:].decode("utf-8", "replace")
            print(f"--- {f.name} ---\n{data}")
            pos[f] = f.stat().st_size
    print("(mengikuti log - tekan Ctrl+C untuk keluar)")
    try:
        while True:
            for f in files:
                if f.exists() and f.stat().st_size > pos.get(f, 0):
                    with open(f, "rb") as fh:
                        fh.seek(pos.get(f, 0))
                        sys.stdout.write(fh.read().decode("utf-8", "replace"))
                        sys.stdout.flush()
                    pos[f] = f.stat().st_size
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def fg(svc):
    cmd, cwd = command(svc)
    os.chdir(cwd)
    sys.exit(subprocess.call(cmd))


def main(argv):
    load_config()
    if not argv or argv[0] not in ("start", "stop", "restart", "status", "logs", "fg", "open"):
        print(__doc__)
        return 1
    action, target = argv[0], (argv[1] if len(argv) > 1 else None)
    if target and target not in SERVICES:
        print(f"Layanan tidak dikenal: {target} (pilihan: portal, datamining)")
        return 1
    targets = [target] if target else list(SERVICES)
    if action == "status":
        status()
    elif action == "open":
        import webbrowser
        webbrowser.open(public_url(target or "portal"))
    elif action == "logs":
        logs(target)
    elif action == "fg":
        if not target:
            print("pakai: run.bat fg portal|datamining")
            return 1
        fg(target)
    else:
        ok = True
        for s in targets:
            if action in ("stop", "restart"):
                stop(s)
            if action in ("start", "restart"):
                ok = start(s) and ok
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
