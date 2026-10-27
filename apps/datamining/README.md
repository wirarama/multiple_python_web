# WebUI Data Mining & Statistik

Aplikasi web berbasis Python (Streamlit) untuk analisis data mining dan statistik:
**klasifikasi, regresi, klastering, association rule mining, dan time series (ARIMA/SARIMA)**.
Antarmuka dwibahasa (Indonesia / English).

---

## Fitur utama

| Fitur | Keterangan |
|---|---|
| Template data | Setiap proses punya template `.xlsx` (berisi sheet **Data** + **Petunjuk_Guide**) dan `.csv` yang bisa diunduh langsung dari aplikasi |
| Input data | Unggah `.xlsx` / `.xls` / `.xlsm` / `.csv` (pemisah `,` atau `;`, angka gaya Indonesia otomatis dikenali), pakai data contoh bawaan, atau **bangkitkan data acak** sesuai parameter Anda |
| Generator data acak | Parameter lengkap per proses (rentang & distribusi per kolom, density, persebaran label, jumlah klaster, aturan asosiasi tertanam, musiman & anomali deret waktu) — konfigurasinya bisa **diekspor/diimpor sebagai JSON** |
| Analisis statistik | Deskriptif lengkap, tabel frekuensi, korelasi (Pearson/Spearman/Kendall), uji normalitas (Shapiro-Wilk, K-S, D'Agostino), deteksi outlier (IQR & skor-z), multikolinearitas (VIF), uji beda (ANOVA, Kruskal-Wallis, Chi-Square), uji stasioneritas (ADF & KPSS) |
| Evaluasi model | Metrik lengkap per proses + perbandingan antar algoritma + validasi silang + diagnostik residual |
| Parameter algoritma | **Seluruh hiperparameter setiap algoritma** dapat diubah langsung dari antarmuka, lengkap dengan penjelasan dwibahasa per parameter, batas nilai yang aman, dan tombol kembali ke bawaan |
| Simpan & muat pengaturan | Seluruh konfigurasi proses — kolom, penskalaan, validasi silang, algoritma, **dan semua parameternya** — dapat diunduh sebagai satu berkas **JSON** lalu dimuat kembali untuk mereproduksi analisis yang sama persis |
| Fine tuning | Pencarian hiperparameter (Grid/Random Search) untuk klasifikasi & regresi, lengkap dengan bukti statistik **mengapa** kombinasi parameter itu yang terbaik |
| Interpretasi otomatis | Setiap proses menghasilkan paragraf interpretasi naratif siap salin ke laporan/skripsi |
| Visualisasi | 8–15 grafik per proses, palet warna tervalidasi *colorblind-safe*, 200 dpi |
| Visualisasi khusus algoritma | Grafik yang menjelaskan **cara kerja** algoritma terpilih: struktur pohon keputusan, contoh pohon dalam hutan, hyperplane & margin SVM, batas keputusan k-NN, jalur regularisasi Lasso/Ridge, sel Voronoi K-Means, elips kovarians GMM, kisi itemset Apriori, akar terbalik ARIMA, dan seterusnya |
| Ekspor | Semua tabel → satu file **Excel multi-sheet** (plus sheet `Informasi`); semua grafik → **PNG** satuan atau **ZIP** |

---

## Instalasi

Butuh **Python 3.9+**.

```bash
# 1. Masuk ke folder proyek
cd dataminingui

# 2. (disarankan) buat virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Pasang dependensi
pip install -r requirements.txt
```

## Menjalankan

```bash
streamlit run app.py
```

Aplikasi terbuka otomatis di `http://localhost:8501`.
Di Windows bisa juga klik dua kali **`run.bat`**; di macOS/Linux jalankan **`./run.sh`**.

---

## Alur pemakaian

1. **Pilih proses** di sidebar (klasifikasi / regresi / klastering / association rule / time series).
2. **Langkah 1 — Template**: baca tabel ketentuan, unduh template Excel atau CSV, isi dengan data Anda.
3. **Langkah 2 — Muat data**: unggah file, pilih *Gunakan data contoh* untuk mencoba, atau
   *Bangkitkan data acak* untuk membuat data sintetis sesuai parameter Anda (lihat bagian
   [Pembangkit data acak](#pembangkit-data-acak)). Opsi pembersihan (hapus duplikat, penanganan
   nilai kosong) ada di sidebar.
4. **Langkah 3 — Konfigurasi & Jalankan**: pilih kolom target/fitur dan algoritma. Expander
   *⚙ Parameter algoritma* menampilkan **seluruh hiperparameter** tiap algoritma yang dipilih dan
   dapat diubah bebas (lihat [Editor parameter algoritma](#editor-parameter-algoritma)); expander
   *💾 Simpan / muat pengaturan (JSON)* menyimpan atau memulihkan seluruh isian halaman ini (lihat
   [Simpan & muat pengaturan](#simpan--muat-pengaturan-json)). Khusus klasifikasi dan regresi,
   expander *🎛 Fine tuning hiperparameter* menyediakan pencarian parameter terbaik beserta analisis
   perbandingannya (lihat [Fine tuning hiperparameter](#fine-tuning-hiperparameter)). Lalu klik
   **Jalankan Analisis**.
5. **Hasil**: statistik → evaluasi model → visualisasi → **visualisasi khusus algoritma** →
   interpretasi → **Ekspor Hasil** (Excel + ZIP PNG).

---

## Pembangkit data acak

Pilih **Bangkitkan data acak** pada *Sumber data* di sidebar. Panel generator muncul di Langkah 2
dengan parameter yang berbeda untuk tiap proses. Setiap konfigurasi dapat diekspor ke berkas
`.json` dan diimpor kembali, sehingga satu eksperimen bisa diulang persis sama (seed + parameter
yang sama selalu menghasilkan data identik).

### Parameter bersama (klasifikasi, regresi, klastering)

Jumlah baris, seed acak, **nilai kosong global (%)** yang menentukan density data, persentase baris
duplikat, dan kolom ID opsional. Di bawahnya ada **tabel spesifikasi kolom** yang bisa diedit
langsung — tambah atau hapus baris sesuka hati, dan tiap kolom punya pengaturannya sendiri:

| Kolom tabel | Fungsi |
|---|---|
| Nama kolom | Nama kolom pada dataset hasil |
| Tipe | `numerik` atau `kategorik` |
| Distribusi | `normal`, `uniform`, `lognormal`, `eksponensial`, `integer`, `poisson` |
| Min / Max | **Rentang berbeda untuk setiap kolom** |
| Rata-rata / Std | Parameter sebaran (dipakai normal, lognormal, eksponensial, poisson) |
| Desimal | 0 = bilangan bulat |
| Kategori / Bobot kategori | Daftar kategori beserta proporsinya, mis. `Rendah, Sedang, Tinggi` dan `0.5, 0.3, 0.2` |
| % kosong | Density per kolom (di luar nilai kosong global) |
| % outlier | Persentase nilai yang sengaja dilempar ke luar rentang |
| Pengaruh target | Bobot kontribusi kolom terhadap target; 0 berarti kolom murni derau |

### Parameter khusus per proses

**Klasifikasi** — nama kolom target, daftar kelas beserta **persebaran label** (proporsi tiap kelas,
diedit lewat tabel), **keterpisahan kelas** 0,05–0,98 (mendekati 1 = kelas mudah dipisahkan model;
mendekati 0 = label nyaris acak), dan **label bising (%)** untuk mensimulasikan kesalahan anotasi.

**Regresi** — nama, rata-rata, simpangan baku, desimal, dan batas rentang target; **R² yang
diinginkan** (kekuatan hubungan fitur–target, terealisasi dengan galat < 0,06); **nonlinearitas**
(0 = linear murni, 1 = sangat melengkung sehingga model pohon unggul atas model linear);
**heteroskedastisitas** (ragam galat membesar seiring nilai prediktor); dan persentase target kosong.

**Klastering** — jumlah klaster, **proporsi tiap klaster**, **pemisahan antar klaster** (jarak antar
pusat relatif terhadap sebaran anggota), **kerapatan klaster** (seberapa merapat anggota ke pusatnya),
bentuk `bulat`/`lonjong`, persentase **titik noise** untuk menguji DBSCAN, serta opsi menyertakan
kolom `klaster_asli` sebagai kunci jawaban.

**Association Rule Mining** — jumlah transaksi, jumlah item unik, daftar item khusus,
**rata-rata & simpangan baku ukuran keranjang**, **kemiringan popularitas item** (0 = semua item
sama populer, tinggi = beberapa item mendominasi), **persentase item acak/bising**, serta
**aturan yang ditanam**: jumlah aturan, ukuran antecedent & consequent, support dan confidence
tiap aturan. Format keluaran dapat dipilih: panjang (`id_transaksi` + `item`), keranjang (kolom 0/1),
atau daftar item dipisah koma — lengkap dengan kolom tanggal dan jumlah opsional.

**Time Series** — jumlah periode, **frekuensi/agregasi** (harian, mingguan, bulanan, kuartalan,
tahunan), tanggal mulai, dan **catatan per periode** (lebih dari 1 menghasilkan data mentah
sub-periode yang harus diagregasi, untuk menguji fitur agregasi pada modul time series);
level dasar, tipe tren (`tidak ada`/`linear`/`eksponensial`/`logistik`) dan kekuatannya (boleh
negatif untuk tren menurun); **pola musiman** (periode, amplitudo, harmonik kedua, musiman kedua,
mode aditif/multiplikatif); derau acak; **tingkat anomali (%)**, besarnya dalam kelipatan simpangan
baku, dan tipenya (lonjakan/penurunan/keduanya); jumlah & besar **pergeseran level**
(structural break); persentase nilai kosong; serta nama kolom dan jumlah desimal.

### Format konfigurasi JSON

```json
{
  "schema_version": 1,
  "process": "classification",
  "seed": 42,
  "dibuat": "2026-08-26 10:15",
  "params": {
    "jumlah_baris": 500,
    "kelas": ["Kelas A", "Kelas B"],
    "bobot_kelas": "0.6, 0.4",
    "keterpisahan": 0.7,
    "kolom": [
      {"nama": "umur", "tipe": "numerik", "distribusi": "integer",
       "min": 17, "max": 65, "desimal": 0, "persen_kosong": 0, "pengaruh": 1.0}
    ]
  }
}
```

Berkas boleh ditulis manual atau diedit di luar aplikasi. Field yang hilang otomatis diisi nilai
baku, nilai yang tidak masuk akal dikoreksi, dan konfigurasi untuk proses yang berbeda ditolak
dengan pesan yang jelas.

Setelah data terbangkit, tersedia pratinjau tabel, grafik pratinjau, dan tombol unduh dataset ke
`.xlsx` (menyertakan sheet `Konfigurasi` berisi seluruh parameter) atau `.csv`. Data langsung
mengalir ke Langkah 3 untuk dianalisis seperti data unggahan biasa.

---

## Fine tuning hiperparameter

Tersedia untuk **klasifikasi** dan **regresi**, di dalam expander *Fine tuning hiperparameter*
pada Langkah 3. Fitur ini tidak berhenti pada "ini parameter terbaik" — seluruh keluarannya
dirancang untuk menjawab **mengapa** kombinasi itu yang menang, sehingga bisa dipertanggungjawabkan
dalam laporan atau sidang.

### Pengaturan

| Pengaturan | Pilihan |
|---|---|
| Algoritma yang di-tuning | Salah satu algoritma yang sedang diuji (8 untuk klasifikasi, 10 untuk regresi) |
| Metrik yang dioptimalkan | Klasifikasi: F1-macro, akurasi, akurasi seimbang, F1-weighted, precision/recall-macro, ROC AUC. Regresi: R², RMSE, MAE, MAPE |
| Metode pencarian | Grid Search (semua kombinasi) atau Random Search (sampel acak, untuk ruang besar) |
| Lipatan validasi silang | 3–10 |
| Nilai tiap parameter | Tabel yang bisa diedit: centang parameter yang ingin diuji, ubah daftar nilainya (pisahkan koma; `None`, `True`, `False` ditulis apa adanya). Tiap baris disertai penjelasan arti parameter |

Sebelum dijalankan, aplikasi menampilkan jumlah kombinasi dan total pelatihan model, serta
memperingatkan bila pencarian akan terlalu berat. Hasil pencarian disimpan sementara sehingga
tidak dihitung ulang setiap kali Anda berpindah tab.

### Enam tab hasil — bukti "mengapa"

**1. Peringkat konfigurasi.** Tabel seluruh kombinasi: skor CV, simpangan antar lipatan, skor latih,
**selisih latih–CV** (indikator overfitting), dan waktu latih. Dilengkapi bar chart konfigurasi
teratas dengan galat baku, serta grafik perkembangan pencarian.

**2. Kepentingan parameter.** Untuk setiap parameter dihitung **Eta²** — berapa persen variasi skor
antar kombinasi yang dijelaskan oleh parameter tersebut — beserta uji F dan p-value-nya. Inilah
jawaban kuantitatif atas "parameter mana yang benar-benar menentukan". Parameter ber-Eta² kecil
(&lt; 0,06) dilaporkan sebagai kandidat untuk dikunci pada nilai bawaan agar pencarian berikutnya
lebih ringkas.

**3. Kurva validasi.** Satu panel per parameter, menampilkan **skor latih vs skor validasi silang**
untuk setiap nilai. Jarak antara kedua garis adalah besarnya overfitting, sehingga terlihat langsung
mengapa nilai tertentu dipilih: di bawahnya model underfitting, di atasnya model mulai menghafal
data latih. Nilai terpilih ditandai dan tabel angkanya tersedia.

**4. Interaksi parameter.** Heatmap dua parameter terpenting untuk melihat apakah nilai optimal satu
parameter bergantung pada parameter lain (ada interaksi) atau keduanya bekerja mandiri.

**5. Uji signifikansi.** **Uji-t berpasangan antar lipatan** antara konfigurasi terbaik dan
pesaingnya — p ≥ 0,05 berarti keduanya *setara secara statistik*, bukan benar-benar kalah.
Ditambah boxplot sebaran skor antar lipatan, scatter latih-vs-validasi untuk deteksi overfitting,
dan **kaidah one-standard-error**: konfigurasi paling sederhana yang skornya masih dalam 1 galat
baku dari yang terbaik, lengkap dengan perbandingan waktu latihnya.

**6. Sebelum vs sesudah.** Perbandingan terhadap **parameter bawaan** pada lipatan CV yang sama
(uji-t berpasangan + Cohen's d), grafik berpasangan per lipatan, dan evaluasi pada **data uji**
yang belum pernah dilihat model — untuk memastikan perbaikan CV benar-benar terbawa, bukan hasil
menyesuaikan diri pada lipatan validasi.

Di bawah keenam tab, blok **"Mengapa konfigurasi ini yang terbaik"** merangkum semuanya menjadi
paragraf naratif siap salin: parameter apa yang paling menentukan dan seberapa besar, mengapa nilai
itu yang optimal dilihat dari kurva validasi, berapa konfigurasi lain yang setara secara statistik,
apakah ada alternatif yang lebih sederhana, dan apakah tuning benar-benar mengungguli parameter
bawaan. Seluruh tabel dan grafik tuning ikut terekspor ke Excel dan ZIP PNG bersama hasil analisis
lainnya.

---

## Editor parameter algoritma

Setiap algoritma pada kelima proses punya panel parameter sendiri di **Langkah 3**
(expander *⚙ Parameter algoritma*). Ketika beberapa algoritma dipilih sekaligus
(klasifikasi & regresi), masing-masing mendapat satu tab.

- Setiap isian punya **penjelasan dwibahasa** — arahkan kursor ke tanda tanya di sebelah nama parameter.
- Batas nilai sudah dijaga, sehingga nilai yang mustahil tidak bisa dimasukkan.
- Baris ringkasan di bawah tiap tab menyebutkan parameter mana yang berbeda dari bawaan;
  tombol **Bawaan** mengembalikan satu algoritma ke setelan aslinya.
- Nilai yang dipakai ikut tercatat pada sheet **Parameter Algoritma** di berkas Excel hasil ekspor,
  sehingga laporan Anda selalu memuat setelan yang benar-benar dijalankan.
- Parameter yang Anda atur juga menjadi **titik awal** fine tuning: hiperparameter yang tidak ikut
  dicari tetap memakai nilai pilihan Anda.

Jumlah parameter yang dapat diedit:

| Proses | Algoritma | Contoh parameter |
|---|---|---|
| Klasifikasi | 8 | `C`, `penalty`, `solver`, `max_depth`, `min_samples_leaf`, `ccp_alpha`, `n_estimators`, `max_features`, `bootstrap`, `oob_score`, `learning_rate`, `subsample`, `kernel`, `gamma`, `degree`, `n_neighbors`, `weights`, `metric`, `var_smoothing`, `class_weight`, … |
| Regresi | 10 | `alpha`, `l1_ratio`, `selection`, `positive`, `criterion`, `max_leaf_nodes`, `epsilon`, `coef0`, `loss`, `max_samples`, … |
| Klastering | 4 | `n_clusters`, `init`, `n_init`, `max_iter`, `tol`, `algorithm`, `linkage`, `metric`, `eps`, `min_samples`, `covariance_type`, `reg_covar`, `init_params`, … |
| Association | 2 | `min_support`, `metric`, `min_threshold`, `min_lift`, `max_len`, `low_memory` |
| Time series | 2 | `p`, `d`, `q`, `P`, `D`, `Q`, `trend`, `enforce_stationarity`, `enforce_invertibility`, `max_p`…`max_Q`, `ic` |

Nilai tak wajar dari berkas konfigurasi lama tidak pernah menggagalkan aplikasi: nilai di luar
rentang dijepit, pilihan yang tidak dikenal dikembalikan ke bawaan, dan kombinasi yang mustahil
diperbaiki otomatis (mis. `penalty="l1"` memaksa solver `liblinear`; `linkage="ward"` memaksa
metrik `euclidean`).

---

## Simpan & muat pengaturan (JSON)

Expander *💾 Simpan / muat pengaturan (JSON)* pada Langkah 3 menyimpan **seluruh isian halaman**
untuk proses yang sedang aktif:

- kolom target & fitur, proporsi data uji, penskalaan, lipatan validasi silang, random state;
- daftar algoritma yang diuji dan **seluruh parameter tiap algoritma**;
- setelan fine tuning, pilihan bidang visualisasi, dan setiap isian lain pada halaman itu.

Bentuk berkasnya:

```json
{
  "app": "dataminingui",
  "schema_version": 1,
  "process": "classification",
  "saved_at": "2026-09-04T13:20:11",
  "widgets": {
    "cls_target": "kelas",
    "cls_feats": ["suhu", "kelembaban", "ph"],
    "cls_ts": 0.25,
    "cls_scale": "standard",
    "cls_models": ["Decision Tree", "Random Forest"],
    "cls_p_decisiontree_max_depth": 5,
    "cls_p_decisiontree_criterion": "entropy",
    "cls_p_randomforest_n_estimators": 400
  },
  "config": { "...": "ringkasan konfigurasi hasil, untuk dibaca manusia" }
}
```

Bagian `widgets` adalah yang dipulihkan saat berkas dimuat; `config` hanya catatan agar isi berkas
mudah dibaca. Kunci parameter algoritma berpola `<proses>_p_<algoritma>_<parameter>`.

**Cara pakai**

1. Atur konfigurasi seperti yang Anda inginkan, lalu klik **⬇ Unduh pengaturan (.json)**.
2. Lain waktu (atau di komputer lain), pilih proses yang sama di sidebar, buka expander ini,
   dan unggah berkas tadi pada **Muat pengaturan dari berkas .json**. Halaman akan dimuat ulang
   dengan seluruh isian pulih seperti semula.
3. Tombol **↺ Kembalikan ke pengaturan bawaan** mengosongkan seluruh isian proses aktif.

**Aman untuk data yang berbeda.** Jika berkas memuat nama kolom yang tidak ada pada data yang
sedang dibuka, kolom itu diabaikan dan sisanya tetap dipulihkan — aplikasi tidak pernah berhenti
karena berkas yang tidak cocok. Berkas untuk proses lain ditolak dengan pesan yang menyebutkan
proses mana yang harus dipilih lebih dulu.

> Catatan: berkas ini berbeda dari konfigurasi **pembangkit data acak** (bagian
> [Format konfigurasi JSON](#format-konfigurasi-json)). Yang ini menyimpan *cara menganalisis*,
> yang itu menyimpan *cara membangkitkan data*.

---

## Visualisasi khusus per algoritma

Selain grafik evaluasi yang berlaku umum (matriks konfusi, ROC, residual, dan sebagainya), setiap
hasil analisis memuat satu bagian khusus yang menggambarkan **mekanisme internal** algoritma yang
dipilih, beserta catatan cara membacanya.

| Algoritma | Visualisasi |
|---|---|
| Decision Tree (klasifikasi & regresi) | Struktur pohon dengan kedalaman yang bisa diatur, tabel aturan jika-maka per daun, penurunan ketidakmurnian per fitur, jalur pemangkasan cost-complexity |
| Random Forest / Extra Trees | Beberapa pohon contoh berdampingan (memperlihatkan keragaman antar pohon), sebaran kedalaman & jumlah daun, kurva konvergensi skor terhadap jumlah pohon, boxplot kestabilan kepentingan fitur |
| Gradient Boosting | Skor latih vs uji per tahap boosting dengan penanda puncak, kurva penurunan kerugian, pohon lemah pertama |
| SVM | **Hyperplane** dengan pewarnaan fungsi keputusan, garis margin ±1, support vector yang dilingkari, sebaran jarak ke hyperplane, tabel support vector per kelas |
| SVR | Pita epsilon di sekitar garis prediksi beserta support vector di luar pita |
| k-NN | Lingkaran k tetangga terdekat pada titik contoh beserta hasil pemungutan suaranya, peta batas keputusan, kurva sensitivitas terhadap k |
| Logistic Regression | Koefisien bertanda, rasio odds per fitur, kurva logistik dengan ambang 0,5, peta probabilitas |
| Linear / Ridge / Lasso / Elastic Net | Koefisien bertanda + intersep, jalur regularisasi (koefisien vs alpha, skala log) dengan penanda alpha yang dipakai |
| Naive Bayes | Kurva Gauss yang dipelajari untuk tiap fitur pada tiap kelas, sebaran selisih log-likelihood sebagai ukuran keyakinan |
| K-Means | Sel Voronoi (wilayah milik tiap pusat), boxplot jarak anggota ke pusat |
| DBSCAN | Grafik k-distance untuk memilih `eps`, pemisahan titik inti / titik batas / noise, tabel komposisi titik |
| Agglomerative | Dendrogram, kurva jarak penggabungan per jumlah klaster dengan penanda lompatan terbesar |
| Gaussian Mixture | Elips kovarians 1σ/2σ/3σ per komponen beserta bobotnya, kurva AIC/BIC per jumlah komponen, sebaran keyakinan keanggotaan |
| Apriori / FP-Growth | Kisi (lattice) itemset bertingkat dengan garis induk-anak dan ukuran lingkaran = support, jumlah & sebaran support per tingkat, kuadran mutu aturan |
| ARIMA / SARIMA | Akar terbalik AR & MA pada lingkaran satuan, koefisien dengan selang kepercayaan & penanda signifikansi, respons impuls + dampak kumulatif, peta AIC/BIC antar kombinasi ordo |

**Bidang penggambaran.** Batas keputusan, hyperplane, dan sel Voronoi hanya bisa digambar pada
bidang dua dimensi. Tersedia dua pilihan: **PCA 2 komponen** (bawaan — seluruh fitur terwakili,
label sumbu menyebutkan persentase varians) atau **dua fitur asli** pilihan Anda. Model dilatih
ulang pada bidang yang dipilih agar batas yang digambar benar-benar batas model di bidang itu,
bukan proyeksi yang menyesatkan.

---

## Rincian tiap proses

### 1. Klasifikasi
- **Algoritma**: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVM (RBF), K-Nearest Neighbors, Naive Bayes.
- **Statistik**: distribusi & rasio ketidakseimbangan kelas, ANOVA + Kruskal-Wallis + Eta² per fitur numerik, Chi-Square + Cramér's V per fitur kategorik, ringkasan rata-rata fitur per kelas.
- **Evaluasi**: akurasi, akurasi seimbang, precision/recall/F1 (macro & weighted), ROC AUC, Cohen's Kappa, MCC, Log Loss, validasi silang berstratifikasi, laporan klasifikasi per kelas.
- **Grafik**: distribusi kelas, matriks konfusi (mentah & ternormalisasi), kurva ROC, kurva Precision-Recall, kepentingan fitur, perbandingan model, kurva pembelajaran.
- **Fine tuning**: 8 algoritma dengan 3–6 hiperparameter masing-masing (lihat bagian Fine tuning).

### 2. Regresi
- **Algoritma**: Linear, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVR (RBF), KNN.
- **Statistik**: ringkasan OLS lengkap (koefisien, galat baku, uji-t, p-value, IK 95%, R², Adjusted R², F-statistic, AIC/BIC, Durbin-Watson), korelasi fitur–target.
- **Evaluasi**: R², Adjusted R², MAE, MSE, RMSE, MAPE, R² validasi silang, uji asumsi residual (Shapiro-Wilk, Jarque-Bera, Durbin-Watson).
- **Grafik**: aktual vs prediksi, plot residual, histogram + Q-Q plot residual, galat per observasi, kepentingan fitur, perbandingan model (R² & RMSE).
- **Fine tuning**: 10 algoritma dengan 2–5 hiperparameter masing-masing (lihat bagian Fine tuning).

### 3. Klastering
- **Algoritma**: K-Means, Agglomerative (ward/complete/average/single), DBSCAN, Gaussian Mixture.
- **Penentuan k**: metode Elbow (inertia/WCSS), Silhouette, Davies-Bouldin, Calinski-Harabasz untuk rentang k yang dipilih.
- **Evaluasi**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, siluet per klaster, ANOVA antar klaster.
- **Profil**: rata-rata & simpangan baku tiap fitur per klaster, profil skor-z, ukuran klaster, data berlabel klaster.
- **Grafik**: elbow, metrik per k, plot siluet, sebaran PCA 2D + sentroid, sebaran dua fitur asli, heatmap profil klaster, ukuran klaster, dendrogram.

### 4. Association Rule Mining
- **Algoritma**: Apriori dan FP-Growth (mlxtend).
- **Format data yang didukung** (dideteksi otomatis):
  1. *Long* — kolom `id_transaksi` + kolom `item`, satu baris satu item **(disarankan)**
  2. *Basket* — satu baris satu transaksi, tiap item jadi kolom 0/1
  3. *Delimited* — satu kolom berisi daftar item dipisah koma/titik koma
- **Statistik**: jumlah transaksi & item unik, kepadatan matriks, ukuran keranjang, support tiap item, distribusi panjang itemset.
- **Evaluasi aturan**: support, confidence, lift, leverage, conviction, Zhang's metric, klasifikasi kekuatan aturan.
- **Grafik**: frekuensi item, distribusi ukuran keranjang, scatter support–confidence (warna & ukuran = lift), bar aturan teratas (lift & confidence), matriks antecedent × consequent, jaringan aturan.

### 5. Time Series (ARIMA / SARIMA)
- **Penentuan ordo**: manual `(p,d,q)(P,D,Q,m)` **atau** otomatis lewat grid search berbasis AIC.
- **Statistik**: deskriptif deret, autokorelasi lag-1, pertumbuhan total, uji ADF & KPSS pada level dan setelah differencing, saran nilai `d`.
- **Identifikasi**: plot ACF & PACF (sebelum & sesudah differencing), dekomposisi aditif/multiplikatif, kekuatan musiman, boxplot pola musiman.
- **Evaluasi**: AIC, BIC, HQIC, Log-Likelihood, signifikansi tiap parameter (uji-z), MAE, MSE, RMSE, MAPE, MASE, R² pada data uji, baseline naif.
- **Diagnostik residual**: Ljung-Box, Shapiro-Wilk, Jarque-Bera, ACF residual, Q-Q plot.
- **Peramalan**: n periode ke depan dengan interval keyakinan 95%, model dilatih ulang pada seluruh data.

---

## Struktur proyek

```
dataminingui/
├── app.py                  # Entry point Streamlit (navigasi + alur 3 langkah)
├── requirements.txt
├── run.bat / run.sh        # Skrip jalan cepat
├── .streamlit/config.toml  # Tema aplikasi
├── core/
│   ├── i18n.py             # Dwibahasa ID/EN
│   ├── params.py           # Registri hiperparameter semua algoritma + pembangun estimator
│   ├── paramui.py          # Panel editor parameter per algoritma
│   ├── settings.py         # Simpan/muat seluruh konfigurasi proses sebagai JSON
│   ├── algviz.py           # Visualisasi khusus per algoritma (pohon, hyperplane, Voronoi, ...)
│   ├── generator.py        # Engine pembangkit data acak + konfigurasi JSON
│   ├── genui.py            # Panel UI generator (form, editor kolom, ekspor/impor JSON)
│   ├── tuning.py           # Engine fine tuning: ruang parameter, pencarian, Eta², uji-t, one-SE
│   ├── tuneui.py           # Panel UI fine tuning (editor grid, enam tab hasil, narasi)
│   ├── theme.py            # Palet warna & gaya matplotlib
│   ├── data_io.py          # Baca CSV/Excel, deteksi tipe kolom & tanggal, pembersihan
│   ├── templates.py        # Generator template data + lembar petunjuk
│   ├── stats.py            # Semua fungsi statistik & uji asumsi
│   ├── viz.py              # Semua fungsi grafik
│   ├── mlutils.py          # Pipeline preprocessing & kepentingan fitur
│   ├── export.py           # Ekspor Excel multi-sheet, PNG, ZIP
│   └── ui.py               # Komponen UI bersama (tabel, grafik, blok statistik, tombol ekspor)
├── modules/
│   ├── classification.py
│   ├── regression.py
│   ├── clustering.py
│   ├── association.py
│   └── timeseries.py
└── tests/
    ├── test_core.py        # Uji fungsi inti (template, statistik, grafik, ekspor)
    ├── test_app.py         # Uji end-to-end kelima proses
    ├── test_variants.py    # Uji varian algoritma & bahasa
    ├── test_generator.py   # Uji kontrol parameter generator & roundtrip JSON
    ├── test_genui.py       # Uji panel generator + analisis atas data bangkitan
    ├── test_tuning.py      # Uji engine tuning: pemilihan konfigurasi, Eta², uji-t, one-SE
    ├── test_tuneui.py      # Uji panel tuning end-to-end di dalam aplikasi
    ├── test_params.py      # Uji registri parameter: pembangunan estimator & koreksi nilai liar
    ├── test_settings.py    # Uji simpan/muat JSON: pemulihan utuh & ketahanan pada data berbeda
    └── test_algviz.py      # Uji seluruh visualisasi khusus algoritma
```

## Menjalankan pengujian

```bash
python tests/test_core.py
python tests/test_app.py
python tests/test_variants.py
python tests/test_generator.py
python tests/test_genui.py
python tests/test_tuning.py
python tests/test_tuneui.py

# berbasis pytest
python -m pytest tests/test_params.py tests/test_settings.py tests/test_algviz.py -q
```

---

## Catatan teknis

- Template dan data contoh dibangkitkan dengan seed tetap, jadi hasilnya konsisten setiap kali diunduh.
- Kolom mirip-ID (nilai unik hampir sebanyak jumlah baris) dikecualikan otomatis dari daftar fitur.
- Tanggal ISO (`YYYY-MM-DD`) dan gaya Indonesia (`DD/MM/YYYY`) sama-sama dikenali; frekuensi deret (harian/mingguan/bulanan/kuartalan/tahunan) dideteksi otomatis.
- Seluruh isian konfigurasi memakai kunci `st.session_state` berawalan proses (`cls_`, `reg_`, `clu_`, `ar_`, `ts_`); itulah yang membuat satu berkas JSON cukup untuk memulihkan seisi halaman.
- Argumen estimator disaring terhadap tanda tangan scikit-learn yang terpasang, sehingga parameter yang tidak dikenal versi tertentu diabaikan alih-alih menggagalkan pelatihan.
- Untuk visualisasi batas keputusan, model disalin dan dilatih ulang pada bidang 2D; model utama yang dievaluasi tetap dilatih pada seluruh fitur.
- Palet warna grafik lolos uji keterbacaan *colorblind* (Delta E CVD ≥ 8 pada pasangan bersebelahan) dan setiap grafik ≥ 2 seri selalu punya legenda.
- Nilai kosong pada fitur ditangani di dalam pipeline (median untuk numerik, modus untuk kategorik) sehingga tidak ada kebocoran data antara data latih dan uji.
