# Simulator Smart Home & Instalasi Listrik

Aplikasi web berbasis **Python Flask** untuk mensimulasikan rumah pintar beserta
instalasi listriknya: denah rumah, kamar, sumber listrik, grup MCB, saklar/sensor,
perangkat elektronik, aturan otomasi (rule), penghuni, waktu, dan cuaca.

Antarmuka dwibahasa (Indonesia / English) — tombol **ID/EN** di kanan atas.

---

## Menjalankan

```bash
pip install -r requirements.txt
python app.py
```

Buka <http://127.0.0.1:5000>. Basis data SQLite dibuat otomatis di
`instance/smarthome.db` dan langsung diisi **data contoh** berupa rumah 10 ruang.

Lokasi basis data dapat dipindahkan lewat variabel lingkungan:

```bash
SMARTHOME_DB=/path/lain/smarthome.db python app.py
```

---

## Struktur berkas

| Berkas | Isi |
|---|---|
| `app.py` | Aplikasi Flask: halaman + REST API |
| `models.py` | Model SQLAlchemy (House, Room, Circuit, Component, SwitchLink, Rule, Occupant, log) |
| `simulation.py` | Mesin simulasi: lingkungan, sensor, rule, logika seri/paralel, perhitungan listrik |
| `seed.py` | Data contoh + penataan otomatis komponen di denah |
| `transfer.py` | Ekspor dan impor konfigurasi rumah dalam format JSON (pemetaan ulang seluruh id) |
| `scenarios/` | Enam berkas kasus siap muat + `make_scenarios.py` untuk membangun ulang |
| `templates/` | Halaman: simulator, komponen, rule, laporan, studi kasus, panduan |
| `static/js/` | `floorplan.js` (denah SVG), `simulator.js`, `components.js`, `rules.js`, `charts.js`, `i18n.js`, `icons.js`, `api.js` |
| `static/css/style.css` | Gaya antarmuka (mode terang & gelap) |

---

## Fitur

### Denah rumah
- Kanvas SVG: kamar dapat digeser dan diubah ukurannya (**Edit Denah**).
- **Tambah Kamar** dan **Tambah Komponen** (pilih jenis → klik posisi di denah).
- **Rapikan** menyusun ulang komponen menjadi grid di dalam tiap kamar.
- Lampu yang menyala memberi efek cahaya; tiap kamar menampilkan suhu, lux, kelembaban.
- Lapisan opsional: **kabel daya** (warna per grup MCB) dan **jalur kendali** (saklar → perangkat).
- Klik dua kali komponen untuk membalik statusnya; seret ikon penghuni untuk memindahkannya.

### Komponen
- **Sumber listrik**: PLN, genset, panel surya, baterai — masing-masing dengan kapasitas (W).
- **Saklar & sensor**: tombol, saklar, PIR (gerak), LDR (cahaya), infrared/remote,
  sensor suhu, kelembaban, hujan, timer, asap, sensor pintu.
- **Perangkat**: lampu, TV, AC, komputer, mesin cuci, kipas, kulkas, pompa air,
  microwave, rice cooker, water heater, router, CCTV, charger, setrika, dispenser,
  exhaust fan, dan lain-lain (daya bawaan otomatis terisi, dapat diubah).

### Instalasi listrik
- Perangkat dikelompokkan ke **grup instalasi (MCB)**; tiap grup disuplai satu sumber.
- Arus dihitung `I = P / (V × cos φ)`; energi `kWh` dan biaya (Rp) diakumulasi tiap langkah.
- **Proteksi**: grup *trip* bila arus melebihi rating MCB, MCB utama *trip* bila total
  melebihi batas; peringatan bila beban melampaui kapasitas sumber.

### Saklar seri & paralel
Satu perangkat boleh dikendalikan banyak saklar sekaligus:

| Mode | Logika | Arti |
|---|---|---|
| **Paralel** | OR | menyala bila salah satu saklar aktif |
| **Seri** | AND | menyala hanya bila semua saklar aktif |

Tiap hubungan dapat dibalik logikanya (**NC / normally closed**).

### Rule otomasi
Format `JIKA (syarat) MAKA (aksi)` dengan logika **AND/OR** dan **prioritas**
(prioritas lebih besar dieksekusi terakhir sehingga keputusannya menang).

- Syarat: periode waktu, jam, cuaca, suhu/cahaya luar, suhu/cahaya/kelembaban kamar,
  jumlah orang di kamar, total orang di rumah, status sensor, status perangkat,
  total daya, energi.
- Aksi: nyalakan / matikan / balik status perangkat, tekan saklar, putus grup,
  matikan semua perangkat.
- Tombol **Uji sekarang** menampilkan nilai aktual tiap syarat.

### Manusia, waktu, cuaca
- **Penghuni** memiliki jadwal per jam (kamar tujuan atau "di luar rumah"),
  memicu sensor PIR, dan menambah beban panas ruangan.
- **Waktu**: pagi (05–10), siang (10–15), sore (15–18), malam. Slider jam, tombol periode,
  dan pengaturan menit per langkah simulasi.
- **Cuaca**: cerah, berawan, hujan, badai, kabut — memengaruhi cahaya luar
  (kurva matahari), suhu luar (siklus harian tropis), dan kelembaban.

### Model fisika ruangan
```
lux_kamar   = lux_luar × faktor_jendela + Σ (daya_lampu × 85 / luas)
suhu_target = suhu_luar + panas_perangkat/110 + 0,45 × jumlah_penghuni
              (dibatasi set point AC, dikurangi 1,2 °C bila kipas menyala)
suhu_kamar += (suhu_target − suhu_kamar) × (1 − e^(−Δt/0,75))     # τ ≈ 45 menit
```

### Laporan
Grafik daya dan energi (SVG, mode terang & gelap), konsumsi per kamar dan per grup,
tabel perangkat, serta unduhan **CSV**.

### Ekspor & impor JSON
Seluruh isi rumah — properti rumah, kamar, grup instalasi, komponen, pengkabelan saklar,
aturan otomasi, dan penghuni — dapat disimpan ke satu berkas JSON dan dimuat kembali.
Tombolnya tersedia pada panel **Kontrol Simulasi** di halaman Simulator dan pada halaman
**Studi Kasus**. Impor mengganti seluruh isi rumah dan memetakan ulang setiap id, termasuk
acuan kamar/komponen di dalam syarat dan aksi rule serta jadwal penghuni.

```jsonc
{
  "format": "smarthome-sim",   // penanda berkas, diperiksa saat impor
  "version": 1,
  "meta":   { "title": "...", "description": "...", "case": { ... } },
  "house":  { "voltage": 220, "tariff": 1444.7, "sim_minute": 1140, "weather": "cerah", ... },
  "rooms":  [ { "id": 1, "name": "Teras", "x": 30, "y": 30, ... } ],
  "components": [ { "id": 1, "kind": "source", "subtype": "pln", ... } ],
  "circuits":   [ { "id": 1, "name": "Grup 1 - Penerangan", "source_id": 1, "mcb_a": 4 } ],
  "links":      [ { "switch_id": 30, "device_id": 5, "inverted": false } ],
  "rules":      [ { "name": "...", "conditions": [...], "actions": [...] } ],
  "occupants":  [ { "name": "Ayah", "room_id": 2, "schedule": { "7": "away" } } ]
}
```

### Studi kasus
Halaman **Studi Kasus** memuat enam kasus siap pakai yang berbeda sumber listrik dan waktunya,
masing-masing dilengkapi ringkasan, tujuan pembelajaran, langkah pengamatan, pertanyaan pemandu,
hasil yang diharapkan (angka hasil pengukuran), dan catatan.

| Kode | Kasus | Waktu | Sumber |
|---|---|---|---|
| K1 | Jaringan PLN pada malam hari | 19.00, cerah | PLN untuk keempat grup |
| K2 | Pemadaman PLN dan suplai genset | 20.00, cerah | Genset untuk Grup 1 & 4; Grup 2 & 3 ikut padam |
| K3 | Panel surya pada siang hari | 12.00, cerah | Panel surya 800 Wp + PLN, dua sumber sekaligus |
| K4 | Baterai cadangan dini hari | 02.00, cerah | Baterai 600 W untuk seluruh grup |
| K5 | Beban puncak pagi dan pemutusan MCB | 05.30, cerah | PLN; Grup 3 terputus pada 12,83 A |
| K6 | Hujan sore hari dan reaksi sensor luar | 16.00, hujan | PLN; sensor hujan dan LDR aktif |

Berkas kasus dapat dibangun ulang setelah data contoh diubah:

```bash
python scenarios/make_scenarios.py
```

Skrip tersebut menyusun tiap kasus dari data contoh, menjalankan simulasinya, lalu menuliskan
angka hasil pengukuran ke dalam berkas kasus sehingga bagian "hasil yang diharapkan" selalu
sesuai dengan perilaku program.

---

## REST API ringkas

| Metode | Endpoint | Fungsi |
|---|---|---|
| GET | `/api/state` | Seluruh keadaan rumah (kamar, komponen, grup, rule, penghuni, log, sampel) |
| GET | `/api/catalog` | Daftar jenis komponen & daya bawaan |
| POST | `/api/sim/tick` | Maju `steps` langkah simulasi |
| POST | `/api/sim/recompute` | Hitung ulang tanpa memajukan waktu |
| POST | `/api/sim/env` | Ubah jam / periode / cuaca / koreksi suhu |
| POST | `/api/sim/reset-trip`, `/api/sim/reset-energy` | Reset proteksi / energi |
| POST | `/api/sim/reset-demo`, `/api/sim/clear` | Muat data contoh / kosongkan rumah |
| POST | `/api/layout` | Rapikan posisi komponen |
| GET | `/api/export` | Unduh seluruh konfigurasi rumah sebagai JSON |
| POST | `/api/import` | Ganti seluruh isi rumah dari dokumen JSON |
| GET | `/api/scenarios` | Daftar kasus beserta penjelasannya |
| POST | `/api/scenarios/<slug>` | Muat sebuah kasus |
| GET | `/api/scenarios/<slug>/file` | Unduh berkas kasus |
| CRUD | `/api/rooms`, `/api/circuits`, `/api/components`, `/api/links`, `/api/rules`, `/api/occupants` | Kelola objek |
| POST | `/api/components/<id>/toggle` | Ubah status manual komponen |
| POST | `/api/rules/<id>/test` | Uji syarat rule pada kondisi saat ini |

---

## Catatan

- Simulasi dijalankan per **langkah** (bawaan 10 menit simulasi per langkah).
  Tombol **Jalan** memanggil satu langkah secara berkala sesuai pengaturan kecepatan.
- Seluruh perhitungan bersifat deterministik kecuali pilihan cuaca acak harian.
- Aplikasi ditujukan untuk pembelajaran dan demonstrasi, bukan perancangan
  instalasi listrik sebenarnya.
