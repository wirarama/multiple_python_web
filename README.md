# multiple_python_web — Portal Gabungan Simulator & ML

Satu paket untuk menjalankan lima aplikasi web Python di server Linux:

| Aplikasi | Sumber | URL | Cara jalan |
|---|---|---|---|
| SDN Simulator | `software_defined_network` | `/sdn/` | digabung (Flask) |
| Smart Home Simulator | `smarthome` | `/smarthome/` | digabung (Flask) |
| RL & Federated Learning SDN | `reinforcement_and_federated_network` | `/rlfl/` | digabung (Flask) |
| PySpark ML Script Generator | `pysparkML` | `/pyspark/` | digabung (Flask) |
| Data Mining UI | `dataminingui` | `:8501/datamining/` | **terpisah** (Streamlit) |

Empat aplikasi Flask berjalan dalam **satu proses gunicorn** di port 8000, dengan
portal di `/`. Data Mining UI berbasis Streamlit (server Tornado + WebSocket) sehingga
tidak bisa digabung ke dalam WSGI; ia dijalankan sebagai proses kedua dan ditautkan
dari portal. Dengan nginx (`deploy/nginx.conf.example`) keduanya bisa tampil di satu
domain/port 80.

## Instalasi di server (via SSH)

```bash
git clone https://github.com/<akun>/multiple_python_web.git
cd multiple_python_web
bash setup.sh          # membuat .venv baru + memasang semua dependensi + uji impor
bash run.sh start      # menjalankan portal (8000) dan Data Mining UI (8501)
bash run.sh status
```

Buka `http://<ip-server>:8000/`. Pastikan port 8000 dan 8501 dibuka di firewall
(mis. `sudo ufw allow 8000,8501/tcp`) atau gunakan nginx.

Kebutuhan server: Python ≥ 3.9 dengan modul venv (Debian/Ubuntu:
`sudo apt install python3-venv`), serta `git`.

Opsi `setup.sh`:

| Opsi | Fungsi |
|---|---|
| `--python /usr/bin/python3.11` | pilih interpreter |
| `--recreate` | hapus `.venv` lalu buat ulang dari nol |
| `--sync-git` | tarik ulang script dari repo GitHub sumber sebelum instalasi |
| `--no-datamining` | lewati Streamlit & paket data mining (lebih ringan) |

`.venv/`, `config.env`, `logs/`, `run/`, database SQLite, dan file unggahan tercantum
di `.gitignore`, jadi tidak ikut ter-commit maupun ter-clone. Setiap server selalu
membuat virtual environment sendiri.

## Instalasi di Windows (.bat)

Butuh Python ≥ 3.9 dari python.org (centang **Add python.exe to PATH**). Di Command
Prompt, dari folder repo:

```bat
setup.bat            :: membuat .venv + memasang dependensi + uji impor
run.bat start        :: portal (8000) + Data Mining UI (8501), berjalan tanpa jendela
run.bat open         :: buka portal di browser
run.bat status
run.bat stop
```

| Perintah | Fungsi |
|---|---|
| `setup.bat --recreate` | buat ulang `.venv` dari nol |
| `setup.bat --sync` | salin ulang script dari folder sumber sebelum instalasi |
| `setup.bat --no-datamining` | tanpa Streamlit |
| `run.bat restart [portal\|datamining]` | mulai ulang |
| `run.bat logs [portal\|datamining]` | ikuti log (Ctrl+C keluar) |
| `run.bat fg portal` | jalan di jendela ini (debug) |
| `run.bat autostart-on` / `autostart-off` | jalankan otomatis saat login (Task Scheduler) |

Di Windows portal dilayani oleh **waitress** (gunicorn tidak mendukung Windows); paket
yang tepat dipilih otomatis lewat penanda platform di `requirements-web.txt`. Logika
start/stop ada di `portal/winctl.py` (PID di `run\`, log di `logs\`). Saat pertama kali
dijalankan, Windows Firewall mungkin meminta izin. Bila hanya dipakai di komputer
sendiri, isi `HOST=127.0.0.1` di `config.env`.

## Mengelola layanan

```bash
bash run.sh start | stop | restart [portal|datamining]
bash run.sh status
bash run.sh logs [portal|datamining]
```

Port dan opsi lain ada di `config.env` (dibuat dari `config.env.example`). Proses
berjalan di latar belakang dan tetap hidup setelah keluar dari SSH. Untuk start otomatis
saat boot, lihat `deploy/mpw-portal.service` dan `deploy/mpw-datamining.service`.

> Portal sengaja memakai **1 worker gunicorn + 8 thread** karena aplikasi menyimpan
> state simulasi di memori proses. Jangan menaikkan jumlah worker.

## Sinkronisasi script dari sumber

`apps/` berisi salinan file yang diperlukan saja (kode, template, static, data
skenario). Dokumen paper/HKI, cache, `Claude outputs`, contoh paket, dan test tidak
disalin. Aturannya ada di `sync_config.json` (`include` / `exclude` / `preserve`).

```bash
# Di Windows (folder sumber berdampingan di ...\GitHub\)
sync.bat                         # atau: python sync.py
python sync.py --dry-run         # lihat perubahan saja
python sync.py --only sdn,rlfl   # sebagian aplikasi

# Di server (clone dari GitHub)
bash sync.sh --source git
bash sync.sh --source git --git-base git@github.com:wirarama/   # repo private via SSH
bash run.sh restart
```

File yang sudah dihapus di sumber juga dihapus di `apps/`, kecuali data runtime yang
tercantum di `preserve`. Bila sebuah aplikasi menambah paket baru di
`requirements.txt`-nya, `sync.py` memberi peringatan; tambahkan paket itu ke
`requirements-web.txt` atau `requirements-datamining.txt`, lalu jalankan `bash setup.sh`.

Alur kerja yang disarankan: ubah kode di repo sumber → `sync.bat` di Windows → commit &
push `multiple_python_web` → di server `git pull && bash setup.sh && bash run.sh restart`.

## Cara penggabungan bekerja

`portal/gateway.py` memuat `apps/<nama>/app.py` masing-masing sebagai modul terpisah dan
memasangnya di bawah prefix URL. Kode aplikasi sumber **tidak diubah**. Aplikasi-aplikasi
itu memakai URL absolut (`fetch('/api/state')`, `href="/tutorial"`), sehingga permintaan
tanpa prefix ditebak asal aplikasinya dari header `Referer` (cadangan: cookie
`mpw_app`). GET dialihkan ke URL berprefix, sedangkan POST/PUT/DELETE diteruskan
langsung. Gateway juga menyisipkan tombol kecil **⌂ Portal** di pojok kiri bawah setiap
aplikasi (matikan dengan `MPW_PORTAL_BADGE=0`).

Modul lokal antar aplikasi tidak boleh bernama sama (misalnya dua aplikasi sama-sama punya
`models.py`). Gateway memeriksa hal ini saat start dan menampilkan pesan jelas bila terjadi.

## Struktur

```
apps/                 salinan aplikasi (dikelola sync.py)
  sdn/ smarthome/ rlfl/ pyspark/ datamining/
portal/               gateway WSGI + halaman portal
wsgi.py               titik masuk gunicorn
setup.sh              membuat .venv & memasang dependensi
run.sh                start/stop/status/logs
sync.py / sync.sh / sync.bat, sync_config.json
requirements.txt      = requirements-web.txt + requirements-datamining.txt
config.env.example    contoh konfigurasi port
deploy/               contoh nginx & systemd
```
