# FedNet Edu — Simulator Federated Learning untuk Optimasi Jaringan

Aplikasi pendidikan berbasis **Flask + NumPy + NetworkX** untuk mempelajari *federated learning*
(FL) dan *federated reinforcement learning* pada masalah optimasi jalur jaringan. Tidak hanya
menampilkan akurasi model, aplikasi memvisualisasikan **jalur yang dioptimasi oleh tiap model
lokal, model global, model terpusat, dan oracle**, lengkap dengan evolusi jalur global per ronde.

## Fitur
- **Simulator kondisi jaringan** (gaya proyek `software_defined_network`): bangkitkan topologi
  (jumlah node, k tetangga, luas wilayah, jumlah klien, seed), bagi node ke region klien (k-means),
  tambah/hapus node & link, ubah bandwidth/loss/bobot trafik, pindahkan node antar region,
  seret node pada peta, import/export CSV dan konfigurasi JSON.
- **Profil trafik tiap klien** (intensitas, burstiness, jam puncak, lokalitas tujuan) → data
  pengukuran lokal **non-IID**, ditampilkan dengan histogram per klien.
- **Mode 1 – Federated Learning (supervised)**: MLP/linear numpy memprediksi delay link;
  agregasi FedAvg / FedProx / FedMedian / Trimmed-Mean; fraksi klien, straggler, Differential
  Privacy (clip + noise), standarisasi federated. Delay prediksi → Dijkstra → jalur.
- **Mode 2 – Federated Q-Routing (RL)**: agen Q-routing per region, agregasi tabel-Q berbobot
  kunjungan atau FedAvg.
- **Tab Reinforcement Learning – Reward per Langkah**: agen Q-learning tunggal source→target
  dengan fungsi reward terurai (+ tiba, ± kemajuan, − delay, − hop, − loop, − kongesti,
  − timeout) yang bobotnya dapat diatur; grafik reward plus/minus per episode; pemutar
  routing langkah-demi-langkah (peta, nilai Q pilihan, rincian reward, perhitungan update Q)
  untuk kebijakan akhir, jalur delay minimum, atau episode pelatihan mana pun; peta kebijakan.
- **Pembanding**: Global (federated), Lokal-saja tiap klien, Terpusat (batas atas), Oracle.
- **Metrik**: MAE, RMSE, R², akurasi (± toleransi), MAE per tingkat beban, gap jalur vs optimal,
  % jalur optimal, keberhasilan jalur, cakupan state, biaya komunikasi, log per ronde.
- **Visualisasi jalur** lokal & global (small multiples, slider/animasi ronde, pewarnaan delay link).
- **Unduh per file / ZIP**: `data/*.csv` (nodes, links, clients, measurements, snapshot),
  `config/fl_config.json`, script standalone `01_data_jaringan.py`, `02_federated_learning.py`,
  `03_visualisasi.py`, `04_reinforcement_learning.py`, `run_all.py`, hasil CSV/JSON, dan gambar PNG (matplotlib).

Script standalone berisi salinan engine yang sama (dirakit otomatis dari folder `core/`),
sehingga dengan seed yang sama hasilnya **identik** dengan aplikasi web.

## Menjalankan
```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Buka `http://127.0.0.1:5000`. Panduan konsep & praktikum: `http://127.0.0.1:5000/tutorial`.

## Contoh paket script
Folder `contoh_paket_script/` berisi contoh hasil unduhan ZIP (topologi bawaan 24 node, 4 klien):
jalankan `cd contoh_paket_script && python run_all.py` tanpa membuka aplikasi web.

## Struktur
```
app.py                     # Flask app + REST API
core/
  network.py               # topologi, region, simulasi trafik, CSV I/O
  routing.py               # tabel next-hop, ekstraksi jalur, metrik jalur
  federated.py             # FL supervised (MLP numpy, FedAvg/FedProx/FedMedian/Trimmed, DP)
  qrouting.py              # Federated Q-Routing
  rl_reward.py             # RL routing dengan reward plus/minus per langkah
  rl_viz.py                # gambar matplotlib untuk RL
  results_io.py            # penulisan hasil CSV/JSON
  viz.py                   # gambar matplotlib
  exporter.py              # perakit script standalone + ZIP
  script_mains/            # bagian "main" script 01/02/03
templates/index.html, templates/tutorial.html
static/app.js, static/style.css
```

## REST API ringkas
| Method | Endpoint | Fungsi |
|---|---|---|
| GET | `/api/state` | topologi, konfigurasi, statistik data |
| POST | `/api/topology/generate` · `/api/topology/partition` | bangkitkan topologi / bagi ulang region |
| POST/DELETE | `/api/node/*`, `/api/link` | edit node & link |
| POST | `/api/clients` | ubah profil trafik klien |
| POST | `/api/import/csv` · `/api/config/import` | impor CSV / JSON |
| POST | `/api/data/generate` | simulasi data trafik lokal |
| POST | `/api/run/supervised` · `/api/run/qrouting` | jalankan eksperimen |
| POST | `/api/rl/run` | latih agen RL (reward per langkah) |
| GET | `/api/figure/<mode>/<kind>.png` | gambar matplotlib |
| GET | `/api/files` · `/api/download/<file>` · `/api/download.zip` | unduhan |

Developed by Wirarama Wedashwara 2026 — Universitas Mataram.
