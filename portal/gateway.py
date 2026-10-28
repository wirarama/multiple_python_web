"""
Gateway WSGI: menggabungkan beberapa aplikasi Flask di bawah satu server.

    /                -> portal (daftar aplikasi)
    /sdn/...         -> apps/sdn        (SDN Simulator)
    /smarthome/...   -> apps/smarthome  (Smart Home Simulator)
    /rlfl/...        -> apps/rlfl       (RL & Federated Learning SDN)
    /pyspark/...     -> apps/pyspark    (PySpark ML Script Generator)
    /csbot/...       -> apps/csbot      (CS Bot, Decision Tree SLM)

Kode aplikasi sumber TIDAK diubah. Aplikasi-aplikasi itu memakai URL absolut
seperti fetch('/api/state') atau href="/tutorial". Permintaan seperti itu
datang tanpa prefix, sehingga gateway menebak aplikasi asalnya dari header
Referer (halaman yang memanggil), lalu cookie 'mpw_app' sebagai cadangan:
  - GET/HEAD  -> dialihkan (307) ke URL berprefix, misalnya /api/state -> /smarthome/api/state
  - POST/PUT/DELETE/PATCH -> diteruskan langsung ke aplikasi yang sesuai
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = ROOT / "apps"

# kunci -> (prefix URL, folder di apps/, judul)
FLASK_APPS = {
    "sdn": ("/sdn", "sdn", "SDN Simulator"),
    "smarthome": ("/smarthome", "smarthome", "Smart Home Simulator"),
    "rlfl": ("/rlfl", "rlfl", "RL & Federated Learning SDN"),
    "pyspark": ("/pyspark", "pyspark", "PySpark ML Script Generator"),
    "csbot": ("/csbot", "csbot", "CS Bot (Decision Tree SLM)"),
}

COOKIE = "mpw_app"
PORTAL_PATHS = ("/portal", "/portal/")


# ------------------------------------------------------------------ loader
def _top_level_modules(folder: Path):
    names = set()
    for p in folder.iterdir():
        if p.suffix == ".py" and p.stem != "app":
            names.add(p.stem)
        elif p.is_dir() and (p / "__init__.py").exists():
            names.add(p.name)
    return names


def load_flask_app(key: str, folder: Path, owners: dict):
    """Impor apps/<folder>/app.py sebagai modul unik 'mpw_<key>' dan kembalikan objek Flask."""
    app_py = folder / "app.py"
    if not app_py.is_file():
        raise FileNotFoundError(f"{app_py} tidak ada - jalankan: python sync.py")
    # Data awal (data_seed/, ikut git) disalin ke data/ (runtime, di .gitignore) bila belum ada,
    # sehingga perubahan data di server tidak bentrok dengan git pull.
    seed, data = folder / "data_seed", folder / "data"
    if seed.is_dir() and not data.exists():
        import shutil
        shutil.copytree(seed, data)
        print(f"[gateway] {key}: data/ dibuat dari data_seed/", file=sys.stderr)
    # Modul lokal aplikasi (models, core, ...) tidak boleh bernama sama antar aplikasi
    for name in _top_level_modules(folder):
        if name in owners and owners[name] != key:
            raise RuntimeError(
                f"Nama modul '{name}' dipakai oleh '{owners[name]}' dan '{key}'. "
                "Aplikasi ini tidak bisa digabung dalam satu proses tanpa mengganti nama modul.")
        owners[name] = key
    sys.path.insert(0, str(folder))
    mod_name = f"mpw_{key}"
    spec = importlib.util.spec_from_file_location(mod_name, app_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    prev_cwd = os.getcwd()
    os.chdir(folder)            # jaga-jaga bila aplikasi memakai path relatif saat impor
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(prev_cwd)
    app = getattr(mod, "app", None)
    if app is None and hasattr(mod, "create_app"):
        app = mod.create_app()
    if app is None:
        raise RuntimeError(f"{app_py} tidak punya objek 'app' atau fungsi create_app()")
    return app


# ----------------------------------------------------------------- gateway
class Gateway:
    def __init__(self, portal, apps: dict, portal_badge: bool = True):
        self.portal = portal
        self.apps = apps                      # prefix -> wsgi app
        self.prefixes = sorted(apps, key=len, reverse=True)
        self.portal_badge = portal_badge

    # -- helpers
    def _prefix_of(self, path: str):
        for p in self.prefixes:
            if path == p or path.startswith(p + "/"):
                return p
        return None

    def _guess_prefix(self, environ, use_cookie=True):
        ref = environ.get("HTTP_REFERER")
        if ref:
            r = urlsplit(ref)
            host = environ.get("HTTP_HOST", "")
            if not r.netloc or r.netloc == host:
                p = self._prefix_of(r.path or "/")
                if p:
                    return p
                if r.path in ("/",) + PORTAL_PATHS:
                    return None      # dipanggil dari portal: jangan tebak dari cookie
        if not use_cookie:
            return None
        cookies = environ.get("HTTP_COOKIE", "")
        for part in cookies.split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE and v in self.apps:
                return v
        return None

    @staticmethod
    def _redirect(start_response, location, code="307 Temporary Redirect"):
        start_response(code, [("Location", location), ("Content-Length", "0"),
                              ("Cache-Control", "no-store")])
        return [b""]

    def _call_app(self, prefix, environ, start_response, path_info):
        environ = dict(environ)
        environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "").rstrip("/") + prefix
        environ["PATH_INFO"] = path_info or "/"
        app = self.apps[prefix]
        captured = {}

        def sr(status, headers, exc_info=None):
            captured["status"], captured["headers"], captured["exc"] = status, headers, exc_info
            return lambda data: captured.setdefault("early", []).append(data)

        body = app(environ, sr)
        headers = list(captured["headers"])
        ctype = next((v for k, v in headers if k.lower() == "content-type"), "")
        encoded = any(k.lower() == "content-encoding" for k, _ in headers)
        is_html = (ctype.startswith("text/html") and captured["status"].startswith("200")
                   and not encoded and environ.get("REQUEST_METHOD") != "HEAD")
        if not is_html:
            def passthrough():
                try:
                    for chunk in captured.get("early", []):
                        yield chunk
                    for chunk in body:
                        yield chunk
                finally:
                    if hasattr(body, "close"):
                        body.close()
            start_response(captured["status"], headers, captured["exc"])
            return passthrough()

        # Halaman HTML: catat aplikasi aktif (cookie) dan sisipkan tombol kembali ke portal
        try:
            data = b"".join(captured.get("early", [])) + b"".join(body)
        finally:
            if hasattr(body, "close"):
                body.close()
        if self.portal_badge:
            data = _inject_badge(data)
        headers = [(k, v) for k, v in headers if k.lower() != "content-length"]
        headers.append(("Content-Length", str(len(data))))
        headers.append(("Set-Cookie", f"{COOKIE}={prefix}; Path=/; SameSite=Lax"))
        start_response(captured["status"], headers, captured["exc"])
        return [data]

    # -- WSGI entry
    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO") or "/"
        qs = environ.get("QUERY_STRING", "")
        method = environ.get("REQUEST_METHOD", "GET").upper()

        # 1) URL berprefix: /sdn/..., /smarthome/...
        prefix = self._prefix_of(path)
        if prefix:
            if path == prefix:
                return self._redirect(start_response, prefix + "/" + ("?" + qs if qs else ""),
                                      "308 Permanent Redirect")
            return self._call_app(prefix, environ, start_response, path[len(prefix):])

        # 2) Halaman portal dan health check
        if path in PORTAL_PATHS or path.startswith("/portal/") or path == "/healthz":
            return self.portal(environ, start_response)

        # 3) URL absolut dari dalam aplikasi (tanpa prefix) -> tebak dari Referer/cookie
        # (untuk "/" hanya Referer: mengetik alamat root tetap membuka portal)
        guess = self._guess_prefix(environ, use_cookie=(path != "/"))
        if guess:
            if method in ("GET", "HEAD"):
                loc = guess + quote(path) + ("?" + qs if qs else "")
                return self._redirect(start_response, loc)
            return self._call_app(guess, environ, start_response, path)

        # 4) Root -> portal; selainnya 404 dari portal
        return self.portal(environ, start_response)


BADGE = (b'<a href="/portal" title="Kembali ke portal" style="position:fixed;left:0;top:50%;'
         b'transform:translateY(-50%);z-index:2147483000;writing-mode:vertical-rl;'
         b'font:600 11px/1 system-ui,-apple-system,Segoe UI,sans-serif;letter-spacing:.04em;'
         b'padding:10px 5px;border-radius:0 8px 8px 0;background:#1d2733;color:#fff;'
         b'text-decoration:none;box-shadow:0 2px 8px rgba(0,0,0,.25);opacity:.75">&#8962; Portal</a>')


def _inject_badge(html: bytes) -> bytes:
    idx = html.lower().rfind(b"</body>")
    if idx == -1:
        return html
    return html[:idx] + BADGE + html[idx:]


# ------------------------------------------------------------------- build
def build():
    from portal.portal_app import create_portal

    owners = {}
    apps, loaded, failed = {}, [], {}
    enabled = os.environ.get("MPW_APPS")  # mis. "sdn,pyspark" untuk membatasi
    enabled = {k.strip() for k in enabled.split(",")} if enabled else None
    # Muat NumPy sekali di awal: bila gagal (mis. CPU tanpa x86-64-v2), semua aplikasi
    # mendapat pesan asli yang jelas, bukan "cannot load module more than once per process".
    numpy_error = None
    try:
        import numpy  # noqa: F401
    except Exception as e:
        numpy_error = (f"NumPy tidak dapat dimuat: {type(e).__name__}: {e} "
                       "-- jalankan ulang 'bash setup.sh' (mendeteksi CPU lama otomatis) "
                       "atau 'bash setup.sh --legacy-cpu'.")
        print(f"[gateway] {numpy_error}", file=sys.stderr)
    for key, (prefix, folder, title) in FLASK_APPS.items():
        if enabled is not None and key not in enabled:
            continue
        if numpy_error:
            failed[key] = numpy_error
            continue
        try:
            apps[prefix] = load_flask_app(key, APPS_DIR / folder, owners)
            loaded.append(key)
        except Exception as e:  # satu aplikasi gagal tidak menjatuhkan yang lain
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"[gateway] GAGAL memuat {key}: {failed[key]}", file=sys.stderr)
    portal = create_portal(loaded, failed)
    badge = os.environ.get("MPW_PORTAL_BADGE", "1") != "0"
    return Gateway(portal, apps, portal_badge=badge)
