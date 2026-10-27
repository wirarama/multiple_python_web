# PySpark MLlib Script Generator

Aplikasi web (Flask) untuk menyusun script **PySpark MLlib** secara interaktif,
mencakup empat jenis tugas machine learning:

- **Classification** — Logistic Regression, Decision Tree, Random Forest, GBT, Naive Bayes, Linear SVM
- **Regression** — Linear Regression, Decision Tree, Random Forest, GBT, Generalized Linear Regression
- **Clustering** — K-Means, Bisecting K-Means, Gaussian Mixture Model
- **Association Rule Mining** — FP-Growth

Untuk setiap task, Anda bisa memilih sumber data berupa **upload file CSV**
(kolom & tipe data terbaca otomatis) atau **data acak** yang parameternya
diatur langsung dari form (jumlah sampel, fitur, kelas/cluster, dll).

Hasil akhirnya adalah **satu file `.py` yang berdiri sendiri** (standalone),
lengkap dengan:

1. Inisialisasi `SparkSession`
2. Pemuatan data dari CSV **atau** kode pembuatan data acak (numpy + pandas → Spark DataFrame)
3. Praproses (VectorAssembler, dan untuk CSV: StringIndexer/OneHotEncoder otomatis untuk kolom kategorikal)
4. Definisi & pelatihan model sesuai algoritma dan hyperparameter yang dipilih
5. Evaluasi:
   - Classification → accuracy, F1-score, precision, recall, confusion matrix
   - Regression → RMSE, MAE, R²
   - Clustering → **Silhouette Score**, ukuran tiap cluster, cluster centers
   - Association Rule Mining → frequent itemsets, association rules (support/confidence/lift)

Script bisa langsung di-**copy** ke clipboard atau di-**download** sebagai file `.py`.

> Penting: aplikasi Flask ini sendiri **tidak menjalankan PySpark**. Ia hanya
> menghasilkan kode. Jalankan file `.py` hasil generate secara terpisah, di
> lingkungan yang sudah terpasang `pyspark` (dan `numpy`, `pandas` untuk mode
> data acak).

## Menjalankan aplikasi

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Lalu buka `http://localhost:5000` di browser.

## Menjalankan script hasil generate

Script hasil generate membutuhkan PySpark (dan numpy/pandas bila memakai mode
data acak):

```bash
pip install pyspark numpy pandas
python pyspark_classification_logistic_regression.py
# atau
spark-submit pyspark_classification_logistic_regression.py
```

Jika memilih sumber data **CSV**, pastikan file CSV yang sama (nama file
tercantum di variabel `CSV_PATH` pada script) diletakkan di direktori yang
sesuai, atau ubah `CSV_PATH` ke lokasi file Anda.

## Struktur proyek

```
pysparkML/
├── app.py                 # Flask backend (routes: /, /api/algorithms, /api/upload, /api/generate)
├── algo_config.py         # Definisi algoritma & hyperparameter per task
├── script_generator.py    # Penyusun script PySpark (string.Template based)
├── templates/
│   └── index.html         # UI utama (form dinamis)
├── static/
│   ├── style.css
│   └── app.js              # Logika interaktif: pilih task -> sumber data -> algoritma -> generate
├── uploads/                # Penyimpanan sementara file CSV yang di-upload
└── requirements.txt
```

## Menambah algoritma / task baru

Cukup tambahkan entri baru di `algo_config.py` (`ALGORITHMS[task][key]`) dengan
`label`, `class_name` (nama class PySpark ML), `import_stmt`, dan daftar
`params`. Frontend (`app.js`) dan backend (`app.py`) mengambil daftar ini
secara dinamis lewat endpoint `/api/algorithms/<task>`, sehingga tidak perlu
mengubah HTML/JS untuk menambah algoritma pada task yang sudah ada.

Blok pembuatan script (data loading, praproses, model, evaluasi) ada di
`script_generator.py`, disusun dengan `string.Template` (placeholder `$nama`)
agar kurung kurawal `{ }` pada kode PySpark hasil generate (f-string, dict,
dsb.) tidak perlu di-escape.
