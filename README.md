# multiple_python_web — Portal Gabungan Simulator & ML

Satu paket untuk menjalankan enam aplikasi web Python di server Linux:

| Aplikasi | Sumber | URL | Cara jalan |
|---|---|---|---|
| SDN Simulator | `software_defined_network` | `/sdn/` | digabung (Flask) |
| Smart Home Simulator | `smarthome` | `/smarthome/` | digabung (Flask) |
| RL & Federated Learning SDN | `reinforcement_and_federated_network` | `/rlfl/` | digabung (Flask) |
| PySpark ML Script Generator | `pysparkML` | `/pyspark/` | digabung (Flask) |
| CS Bot (Decision Tree SLM) | `csbot` | `/csbot/` | digabung (Flask) |
| Data Mining UI | `dataminingui` | `:8501/datamining/` | **terpisah** (Streamlit) |

Lima aplikasi Flask berjalan dalam **satu proses gunicorn** di port 8000, dengan
portal di `/`. Data Mining UI berbasis Streamlit (server Tornado + WebSocket) sehingga
tidak bisa digabung ke dalam WSGI; ia dijalankan sebagai proses kedua dan ditautkan
dari portal. Dengan nginx (`deploy/nginx.conf.example`) keduanya bisa tampil di satu
domain/port 80.

## Cara paling mudah (tanpa mengetik perintah)

**Windows** — klik dua kali di File Explorer:

| File | Fungsi |
|---|---|
| `start.bat` | setup otomatis bila belum pernah, jalankan layanan, buka portal di browser; jendela tertutup sendiri |
| `start-silent.vbs` | sama, tanpa jendela sama sekali (muncul pesan hanya bila gagal) |
| `stop.bat` / `stop-silent.vbs` | hentikan semua layanan |

Agar menyala otomatis setiap login Windows: `run.bat autostart-on` (sekali saja).

**Linux server** — cukup sekali lewat SSH, setelah itu layanan menyala otomatis setiap
server boot (systemd), tanpa perlu SSH lagi:

```bash
bash start.sh --boot       # pasang & nyalakan layanan systemd (butuh sudo)
bash start.sh --no-boot    # matikan autostart
```

**Linux desktop** — `bash start.sh --install-launcher` membuat ikon **Start Portal** dan
**Stop Portal** di menu aplikasi dan Desktop; selanjutnya cukup klik ikon.
`bash start.sh` sendiri = setup otomatis bila perlu + jalankan + buka browser.

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
run.bat stop         :: atau: stop.bat
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

## Server dengan CPU lama / VM "kvm64"

Gejala: `NumPy was built with baseline optimizations: (X86_V2) but your machine doesn't
support: (X86_V2)`, lalu `cannot load module more than once per process` di aplikasi lain.

Penyebab: wheel NumPy ≥ 2.4, pandas ≥ 3.1, dan pyarrow butuh CPU **x86-64-v2**
(SSE4.2, POPCNT, SSSE3). VM dengan tipe CPU lama (mis. `kvm64`/`qemu64` di Proxmox/KVM)
tidak menyediakannya, meskipun prosesor fisiknya modern. Cek di server:

```bash
grep -m1 'model name' /proc/cpuinfo          # "Common KVM processor" / "QEMU Virtual CPU" = tipe CPU lama
grep -cw sse4_2 /proc/cpuinfo                 # 0 = tidak didukung
```

`setup.sh` mendeteksi ini otomatis dan beralih ke **mode kompatibel**
(`constraints-legacy-cpu.txt`: `numpy<2.4`, `pandas<3.1`, tanpa pyarrow). Empat aplikasi Flask
berjalan normal; **Data Mining UI dinonaktifkan** karena Streamlit wajib memakai pyarrow, dan
pyarrow selalu crash di CPU seperti ini. Paksa manual: `bash setup.sh --legacy-cpu`.

Solusi permanen: minta pengelola VM mengganti tipe CPU menjadi **host** (atau
**x86-64-v2-AES** di Proxmox: *Hardware → Processors → Type*; di VMware aktifkan EVC minimal
Nehalem). Setelah VM di-restart: `bash setup.sh --recreate && bash run.sh restart`.
Data Mining UI aktif kembali otomatis.

## Mengelola layanan

### Tanpa Data Mining UI

Streamlit tidak dijalankan sama sekali dan port 8501 tidak dibuka; kartu Data Mining
disembunyikan dari portal.

```bash
bash run.sh start --no-datamining      # sekali jalan (berlaku juga untuk restart)
bash run.sh restart --no-datamining    # bila portal sudah berjalan
bash start.sh --no-datamining          # versi satu langkah
bash run.sh datamining off             # permanen (ENABLE_DATAMINING=0 di config.env)
bash run.sh datamining on              # aktifkan kembali
```

Di Windows: `run.bat start --no-datamining` dan `run.bat datamining off|on`. Untuk server yang
tidak butuh Data Mining sama sekali, `bash setup.sh --no-datamining` juga melewati pemasangan
Streamlit beserta paketnya.

### Perintah lain

```bash
bash stop.sh [portal|datamining]          # Linux   - hentikan layanan
stop.bat [portal|datamining]              # Windows - hentikan layanan
```

`stop.sh` / `stop.bat` menghentikan layanan lewat file PID, lalu juga membersihkan
proses portal/Streamlit milik paket ini yang tertinggal tapi masih memegang port
(misalnya file PID hilang atau layanan dijalankan dengan `fg`). Proses lain yang
kebetulan memakai port yang sama tidak dihentikan, hanya dilaporkan.

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

Data awal csbot (`data/` di repo sumber: knowledge base, SLM, model RNN) disalin ke
`apps/csbot/data_seed/` (ikut git). Saat pertama dijalankan, gateway menyalinnya ke
`apps/csbot/data/`, yang dipakai dan diubah aplikasi. Folder `data/` itu ada di `.gitignore`,
jadi perubahan knowledge base di server tidak bentrok dengan `git pull`. Untuk mengulang dari
data awal, hapus `apps/csbot/data/` lalu `bash run.sh restart`.

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
  sdn/ smarthome/ rlfl/ pyspark/ csbot/ datamining/
portal/               gateway WSGI + halaman portal
wsgi.py               titik masuk gunicorn
setup.sh              membuat .venv & memasang dependensi
run.sh                start/stop/status/logs
sync.py / sync.sh / sync.bat, sync_config.json
requirements.txt      = requirements-web.txt + requirements-datamining.txt
config.env.example    contoh konfigurasi port
deploy/               contoh nginx & systemd
```
