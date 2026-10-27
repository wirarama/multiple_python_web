"""Titik masuk WSGI untuk gunicorn:  gunicorn wsgi:application"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")   # matplotlib tanpa layar (server)

from portal.gateway import build  # noqa: E402

application = build()
app = application

if __name__ == "__main__":
    # Mode pengembangan tanpa gunicorn:  python wsgi.py
    from werkzeug.serving import run_simple
    port = int(os.environ.get("PORTAL_PORT", "8000"))
    run_simple(os.environ.get("HOST", "127.0.0.1"), port, application, threaded=True, use_reloader=False)
