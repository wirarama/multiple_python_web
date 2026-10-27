# Simulator Node & Evaluasi Algoritma Optimasi Routing SDN

Proyek Flask untuk mensimulasikan topologi node jaringan, mengatur parameter tiap
algoritma, dan membandingkan 10 algoritma optimasi routing pada jaringan Software
Defined Network (SDN), berbasis **Flask**, **JSON** (data topologi & pertukaran data
API), dan **NetworkX** (representasi graf).

## Fitur simulator
- **Simulasi jumlah node & luas wilayah** — bangkitkan topologi acak dengan
  mengatur jumlah node dan luas wilayah (lebar x tinggi, meter); node disebar
  acak pada area tsb dan dihubungkan ke k tetangga terdekat, delay tiap link
  diturunkan dari jarak antar node (`topology_generator.py`).
- **Editor topologi manual** — tambah/hapus node dan link (delay, bandwidth,
  packet loss dasar) langsung dari UI, atau reset ke topologi bawaan
  (`data/topology.json`).
- **Parameter input tiap algoritma** — tiap algoritma metaheuristik (ACO, GA, PSO,
  SA, Tabu Search, Q-Learning) punya formulir parameter sendiri (mis. jumlah
  generasi pada Genetic Algorithm, jumlah episode pada Q-Learning, dst.).
- **Luaran proses optimasi** — tiap hasil algoritma menyertakan info luaran
  proses (bukan sekadar echo parameter), misalnya generasi ke berapa GA
  menemukan solusi terbaik, iterasi konvergen ACO/PSO/SA/Tabu Search, atau
  episode pertama Q-Learning berhasil mencapai target — dapat dilihat lewat
  tombol "Lihat" pada tiap baris tabel hasil.
- **Tabel hasil + QoS** — tabel jalur/struktur, Delay, Hop Count, Throughput
  (simulasi), Packet Loss (simulasi), skor QoS (0-100), dan kelas QoS
  (Excellent/Good/Fair/Poor) per algoritma (`qos.py`).
- **Visualisasi** — grafik topologi (klik baris hasil untuk menyorot jalur) dan
  grafik batang perbandingan tiap metrik antar algoritma.
- **Simulasi Deteksi Intrusi (IDS)** — tab terpisah: bangkitkan topologi
  server & client (proporsi client umum/normal vs penyerang dapat diatur),
  centang jenis intrusion yang ingin disimulasikan & dideteksi beserta
  potensi/intensitasnya, lalu bangkitkan traffic acak dan lihat hasil deteksi
  (ringkasan per jenis, confusion matrix, skor, grafik, visualisasi jaringan)
  — lihat `intrusion_detection.py`.

## Algoritma yang tersedia
1. Dijkstra
2. A*
3. Ant Colony Optimization (ACO)
4. Genetic Algorithm (GA)
5. Particle Swarm Optimization (PSO)
6. Simulated Annealing (SA)
7. Tabu Search
8. Q-Learning (Reinforcement Learning sederhana)
9. Greedy Routing
10. Minimum Spanning Tree (MST)

## Metrik evaluasi
- **Delay** (ms) — total bobot delay sepanjang jalur/pohon.
- **Hop Count** — jumlah lompatan (edge) sepanjang jalur/pohon.
- **Throughput (simulasi)** (Mbps) — diestimasi dari bandwidth link bottleneck,
  jumlah hop, dan faktor kongesti yang dapat diatur.
- **Packet Loss (simulasi)** (%) — diestimasi dari probabilitas loss dasar tiap
  link (gabungan sepanjang jalur) ditambah kontribusi kongesti.

> Throughput dan Packet Loss diberi label "(simulasi)" karena bukan hasil
> pengukuran trafik nyata, melainkan estimasi berbasis atribut link pada
> topologi (lihat `metrics.py`).

## Struktur proyek
```
sdn_eval/
├── app.py                  # Flask app + REST API (termasuk editor & generator topologi)
├── graph_utils.py          # load topologi JSON -> networkx.Graph
├── topology_generator.py    # bangkitkan topologi acak dari jumlah node & luas wilayah
├── metrics.py               # perhitungan Delay/Hop/Throughput/Packet Loss
├── qos.py                   # skor & klasifikasi QoS
├── intrusion_detection.py   # simulasi topologi server/client + traffic & deteksi intrusi (IDS)
├── algorithms/
│   ├── __init__.py          # registry algoritma routing
│   ├── dijkstra.py
│   ├── astar.py
│   ├── aco.py
│   ├── genetic.py
│   ├── pso.py
│   ├── simulated_annealing.py
│   ├── tabu_search.py
│   ├── qlearning.py
│   ├── greedy.py
│   ├── mst.py
│   └── load_balancing.py    # algoritma & simulasi cluster load balancing
├── data/
│   └── topology.json        # topologi contoh (10 switch, mesh)
├── templates/
│   └── index.html           # UI perbandingan (3 tab: routing, load balancing, IDS)
├── static/
│   ├── style.css
│   └── app.js
└── requirements.txt
```

## Instalasi & menjalankan
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Buka `http://127.0.0.1:5000` di browser.

## REST API
| Method | Endpoint                    | Deskripsi                                                          |
|--------|------------------------------|--------------------------------------------------------------------|
| GET    | `/api/topology`              | Ambil topologi aktif (node, edge, posisi layout) dalam JSON        |
| POST   | `/api/topology`              | Ganti seluruh topologi aktif (`{"nodes":[...], "edges":[...]}`)    |
| POST   | `/api/topology/reset`        | Muat ulang topologi bawaan (`data/topology.json`)                  |
| POST   | `/api/topology/generate`     | Bangkitkan topologi acak dari `num_nodes`, `area_width`, `area_height`, `k_neighbors`, `seed` |
| GET    | `/api/topology/status`       | Info ringkas mode topologi aktif (default/generated), luas area, jumlah node/link |
| POST   | `/api/topology/node`         | Tambah node (`{"id": "S11"}`)                                      |
| DELETE | `/api/topology/node/<id>`    | Hapus node (beserta link yang terhubung)                           |
| POST   | `/api/topology/edge`         | Tambah/ubah link (`{"source","target","delay","bandwidth","loss"}`)|
| DELETE | `/api/topology/edge`         | Hapus link (`{"source","target"}`)                                 |
| GET    | `/api/algorithms`            | Daftar algoritma + skema parameter yang dapat diatur               |
| POST   | `/api/evaluate`              | Jalankan satu algoritma dan hitung metrik + QoS                    |
| POST   | `/api/compare`               | Jalankan beberapa/semua algoritma sekaligus, tiap algoritma bisa diberi `params` sendiri |

### Contoh `POST /api/evaluate`
```json
{
  "algorithm": "dijkstra",
  "source": "S1",
  "target": "S10",
  "congestion": 0.2,
  "seed": 42,
  "params": {}
}
```

### Contoh `POST /api/compare`
```json
{
  "source": "S1",
  "target": "S10",
  "congestion": 0.2,
  "seed": 42,
  "algorithms": ["dijkstra", "astar", "aco", "genetic", "pso",
                 "simulated_annealing", "tabu_search", "qlearning",
                 "greedy", "mst"]
}
```
Setiap algoritma (kecuali `mst`, yang khusus mengevaluasi pohon) menerima
parameter opsional lewat `params`, misalnya `{"n_iter": 100}` untuk ACO,
`{"generations": 200}` untuk GA, `{"episodes": 800}` untuk Q-Learning, dsb.
Lihat masing-masing modul pada folder `algorithms/` untuk daftar parameter.

### Endpoint Simulasi Deteksi Intrusi (IDS)
| Method | Endpoint                | Deskripsi                                                          |
|--------|--------------------------|----------------------------------------------------------------------|
| GET    | `/api/ids/attack-types`  | Daftar jenis intrusion (key, label, deskripsi, potensi default) untuk checkbox UI |
| GET    | `/api/ids/topology`      | Ambil topologi server/client (peran, posisi) aktif                 |
| POST   | `/api/ids/generate`      | Bangkitkan topologi baru dari `num_servers`, `num_clients`, `attacker_ratio` (0-1), `area_width`, `area_height`, `seed` |
| POST   | `/api/ids/simulate`      | Bangkitkan traffic acak & jalankan detector: `attacks` (daftar key jenis intrusion tercentang), `potensi` (dict key->0..1), `flows_per_client`, `seed` |

`POST /api/ids/simulate` mengembalikan `flows` (daftar flow beserta fitur,
label sebenarnya, hasil deteksi & confidence) dan `summary` (agregat per
label, confusion matrix TP/FN/FP/TN, precision/recall/F1/akurasi, serta skor
deteksi 0-100 & kelasnya). Bila `attacks` dikirim kosong, seluruh traffic
(termasuk dari client penyerang) dianggap normal -- tidak ada intrusion yang
disimulasikan.

## Mengganti topologi
Edit `data/topology.json` (format: daftar `nodes` dan `edges` dengan atribut
`delay`, `bandwidth`, `loss`), atau kirim topologi baru melalui `POST /api/topology`.

## Catatan
- Semua algoritma metaheuristik (ACO, GA, PSO, SA, Tabu Search) memakai `seed`
  agar hasil dapat direproduksi untuk keperluan evaluasi/laporan akademik.
- MST dievaluasi secara berbeda dari algoritma jalur lainnya karena hasilnya
  berupa pohon (menghubungkan seluruh node), bukan satu jalur source→target.
- Proyek ini melengkapi buku ajar "Pengelolaan Jaringan Berbasis Perangkat
  Lunak (SDN)" sebagai media praktikum evaluasi algoritma.
