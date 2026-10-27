"""
Menjalankan portal gabungan dengan waitress (dipakai di Windows, karena gunicorn
tidak mendukung Windows).  Jalankan dari folder repo:

    .venv\\Scripts\\python -m portal.serve_waitress
"""
import os

from waitress import serve

from portal.winctl import load_config, ROOT


def main():
    load_config()
    os.chdir(ROOT)
    from wsgi import application   # memuat semua aplikasi Flask

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORTAL_PORT", "8000"))
    threads = int(os.environ.get("PORTAL_THREADS", "8"))
    print(f"[waitress] portal di http://{host}:{port}/ (threads={threads})", flush=True)
    # 1 proses + banyak thread: state simulasi aplikasi tersimpan di memori proses
    serve(application, host=host, port=port, threads=threads,
          channel_timeout=int(os.environ.get("PORTAL_TIMEOUT", "300")),
          max_request_body_size=200 * 1024 * 1024)


if __name__ == "__main__":
    main()
