# Laporan Proyek Machine Learning - Kenny Aldebaran Roberts

## Domain Proyek

Kemampuan untuk memprediksi harga rumah secara akurat memiliki peran yang signifikan dalam dunia ekonomi. Bagi individu, prediksi ini menginformasikan keputusan penting tentang pembelian, penjualan, dan investasi dalam real estat. Bagi lembaga keuangan, prediksi ini penting untuk menilai risiko dan mengelola portofolio. Secara tradisional, penilaian harga rumah sangat bergantung pada penilaian manual, analisis pasar komparatif, dan penilaian ahli. Meskipun metode ini memberikan wawasan yang berharga, metode ini dapat bersifat subjektif, memakan waktu, dan mungkin tidak sepenuhnya menangkap interaksi kompleks dari berbagai faktor yang memengaruhi nilai properti. Meningkatnya ketersediaan dataset rumah yang lengkap, ditambah dengan kemajuan dalam machine learning, dapat merevolusi proses ini. Proyek ini mengeksplorasi penggunaan algoritma regresi untuk membangun model prediktif untuk harga rumah. Analisis regresi sangat cocok untuk tugas ini karena memungkinkan kita mengukur hubungan antara variabel dependen berkelanjutan (harga rumah) dan serangkaian variabel independen (fitur properti). Tujuan proyek adalah untuk membuat model yang kuat dan akurat yang dapat secara efektif memprediksi harga rumah berdasarkan kumpulan data yang diberikan.

Referensi riset terkait: [Abdul-Rahman, S., & Mutalib, S. (2021). Advanced Machine Learning Algorithms for House Price Prediction: Case Study in Kuala Lumpur. International Journal of Advanced Computer Science and Applications, 12(12).](https://thesai.org/Downloads/Volume12No12/Paper_91-Advanced_Machine_Learning_Algorithms.pdf)

## Business Understanding

### Problem Statements

- Bagaimana hubungan harga rumah dengan fitur-fitur tertentu?
- Bagaimana mengetahui harga rumah dengan karakteristik atau fitur tertentu?

### Goals

- Mengetahui hubungan fitur-fitur dengan harga rumah.
- Membuat model machine learning yang dapat memprediksi harga rumah seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements
- Menggunakan pairplot dan heatmap untuk melihat hubungan fitur dengan harga rumah.
- Menggunakan tiga jenis algoritma regresi untuk menggunakan model yang mencapai performa yang terbaik.

## Data Understanding
Dataset yang digunakan pada proyek ini adalah [House Price Regression Dataset](https://www.kaggle.com/datasets/prokshitha/home-value-insights/data). Dataset ini merupakan dataset kumpulan data mengenai harga rumah dan fitur properti tersebut sebesar 1000 baris, dengan setiap baris mewakili sebuah rumah dan berbagai atribut yang memengaruhi harganya.

### Variabel-variabel pada dataset ini adalah sebagai berikut:

- Square_Footage: Ukuran rumah dalam kaki persegi. Rumah yang lebih besar biasanya memiliki harga yang lebih tinggi.

- Num_Bedrooms: Jumlah kamar tidur di rumah. Lebih banyak kamar tidur umumnya meningkatkan nilai rumah.

- Num_Bathrooms: Jumlah kamar mandi di rumah. Rumah dengan lebih banyak kamar mandi biasanya memiliki harga yang lebih tinggi.

- Year_Built: Tahun rumah dibangun. Rumah yang lebih tua mungkin memiliki harga yang lebih rendah karena keausan.

- Lot_Size: Ukuran tanah tempat rumah dibangun, diukur dalam hektar. Tanah yang lebih besar cenderung menambah nilai properti.

- Garage_Size: Jumlah mobil yang dapat muat di garasi. Rumah dengan garasi yang lebih besar biasanya lebih mahal.

- Neighborhood_Quality: Peringkat kualitas lingkungan pada skala 1-10, di mana 10 menunjukkan lingkungan yang berkualitas tinggi. Lingkungan yang lebih baik biasanya memiliki harga yang lebih tinggi.

- House_Price (Variabel Target): Harga rumah, yang merupakan variabel dependen yang ingin diprediksi.

Menggunakan df.info(), dapat dilihat informasi dataset sebagai berikut:

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 8 columns):
|   # | Column              |   Non-Null Count | Dtype   |
|----:|:--------------------|-----------------:|:--------|
|   0 | Square_Footage      |             1000 | int64   |
|   1 | Num_Bedrooms        |             1000 | int64   |
|   2 | Num_Bathrooms       |             1000 | int64   |
|   3 | Year_Built          |             1000 | int64   |
|   4 | Lot_Size            |             1000 | float64 |
|   5 | Garage_Size         |             1000 | int64   |
|   6 | Neighborhood_Quality|             1000 | int64   |
|   7 | House_Price         |             1000 | float64 |
dtypes: float64(2), int64(6)
memory usage: 62.6 KB

Dapat dilihat bahwa terdapat 1000 baris dari 8 kolom data numerik, dengan tidak ada data null atau hilang, dan jenis data berupa int64 (6 fitur) dan float64 (2 fitur).

Dengan df.describe(), dapat dilihat informasi statistik tiap kolom:

|                    |   Square_Footage |   Num_Bedrooms |   Num_Bathrooms |   Year_Built |   Lot_Size |   Garage_Size |   Neighborhood_Quality |   House_Price |
|:-------------------|-----------------:|---------------:|----------------:|-------------:|-----------:|--------------:|-----------------------:|--------------:|
| count              |      1000       |        1000   |          1000   |       1000   |   1000     |        1000   |                   1000 |    1.00000e+03 |
| mean               |      2815.42    |          2.99  |             1.973 |       1986.55  |      2.77809 |          1.022  |                      5.615 |    6.18861e+05 |
| std                |      1255.51    |          1.42756|             0.820332|         20.6329  |      1.2979  |          0.814973|                      2.88706|    2.53568e+05 |
| min                |       503       |          1     |             1     |       1950   |      0.506058|          0     |                      1     |    1.11627e+05 |
| 25%                |      1749.5     |          2     |             1     |       1969   |      1.66595 |          0     |                      3     |    4.01648e+05 |
| 50%                |      2862.5     |          3     |             2     |       1986   |      2.80974 |          1     |                      6     |    6.28267e+05 |
| 75%                |      3849.5     |          4     |             3     |       2004.25|      3.92332 |          2     |                      8     |    8.27141e+05 |
| max                |      4999       |          5     |             3     |       2022   |      4.9893  |          2     |                     10     |    1.10824e+06 |

Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:

- Count  adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

Karena sudah mengetahui tidak ada missing value dari df.info(), selanjutnya dataset dapat diperiksa untuk duplikat dan outlier.

Data duplikat dapat kita periksa dengan df.duplicated() seperti berikut:

duplicate_rows = df[df.duplicated()]
print("Duplicate Rows:")
print(duplicate_rows)

dengan hasil output:

Duplicate Rows:
Empty DataFrame
Columns: [Square_Footage, Num_Bedrooms, Num_Bathrooms, Year_Built, Lot_Size, Garage_Size, Neighborhood_Quality, House_Price]
Index: []

yang menandakan bahwa tidak ada baris data duplikat.

Berikutnya untuk mengecek outlier menggunakan metode IQR, digunakan kode berikut:

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
outliers

dengan output:

| Column              | Missing Values |
|:--------------------|---------------:|
| Square_Footage      |              0 |
| Num_Bedrooms        |              0 |
| Num_Bathrooms       |              0 |
| Year_Built          |              0 |
| Lot_Size            |              0 |
| Garage_Size         |              0 |
| Neighborhood_Quality|              0 |
| House_Price         |              0 |

yang menandakan bahwa tidak ada outlier pada data. Hal ini dapat dilihat pada boxplot di notebook juga:

![img](https://i.imgur.com/vdKVZHi.png)

**Univariate Analysis**
Menggunakan teknik analisis satu variabel untuk menganalisa data. Histogram data sebagai berikut:

![img](https://i.imgur.com/U3Joo9e.png)

Dapat dilihat bahwa dataset memiliki distribusi yang rata. Tiga fitur (Num_Bedrooms, Num_Bathrooms, Garage_Size) memiliki distribusi yang terlihat agak aneh karena rentang nilai diskrit mereka yang kecil, namun distribusi mereka juga rata.

**Multivariate Analysis**
Menggunakan teknik multivariate untuk menunjukkan hubungan antara dua atau lebih variabel pada data. Kali ini, kita tertarik pada hasil hubungan antara variabel target (harga rumah) dengan variabel lainnya.

![img](https://i.imgur.com/wcqLWBp.png)

Dapat dilihat bahwa Square_Footage memiliki korelasi positif yang kuat, dengan semua fitur lain memiliki korelasi yang lemah karena tidak membentuk pola yang positif atau negatif.

![img](https://i.imgur.com/dkND031.png)

Matriks korelasi di atas mengonfirmasi pengamatan kita dari pairplot, dengan House_Price memiliki korelasi sangat tinggi (0.99) dengan Square_Footage, dan korelasi lemah dengan semua fitur lain. Bagi fitur dengan korelasi sangat kecil (<=0.01), kita akan melakukan drop dengan kode berikut:

df = df.drop(['Num_Bedrooms', 'Num_Bathrooms', 'Neighborhood_Quality'], axis=1)

Dengan df.info(), kita dapat melihat dataframe yang sudah didrop fitur tersebut:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 5 columns):
|   # | Column          |   Non-Null Count | Dtype   |
|----:|:----------------|-----------------:|:--------|
|   0 | Square_Footage  |             1000 | int64   |
|   1 | Year_Built      |             1000 | int64   |
|   2 | Lot_Size        |             1000 | float64 |
|   3 | Garage_Size     |             1000 | int64   |
|   4 | House_Price     |             1000 | float64 |
dtypes: float64(2), int64(3)
memory usage: 39.2 KB

## Data Preparation
Pada proyek ini, akan digunakan teknik Train-Test Split dan Standardisasi Data.

**1. Train-Test Split**
Disini kita membagi data menjadi set training dan test, dengan rasio yang digunakan berupa 80/20, yang berarti 80 persen data digunakan untuk pelatihan, dan 20 untuk pengujian.

Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru.

**2. Standardisasi Data**
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standardisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan, yang pada proyek ini menggunakan teknik StandarScaler dari library Scikitlearn.

StandardScaler melakukan proses standardisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

Untuk menghindari kebocoran informasi pada data uji, kita hanya akan menerapkan standardisasi pada data latih. Kemudian, pada tahap evaluasi, kita akan melakukan standardisasi pada data uji. 

Pada proyek ini, tahap ini hanya digunakan bagi algoritma KNN, karena algoritma Random Forest dan AdaBoost tidak mempedulikan skala.

## Modeling
Model development adalah tahapan di mana kita menggunakan algoritma machine learning untuk menjawab problem statement dari tahap business understanding.

Pada tahap ini, kita akan mengembangkan model machine learning dengan tiga algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan adalah:

**1. K-Nearest Neighbor**
K-Nearest Neighbor (KNN) adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bisa digunakan untuk kasus klasifikasi dan regresi. Pada proyek ini, KNN akan digunakan untuk kasus regresi. 

Dalam regresi KNN, algoritma memprediksi nilai kontinu untuk titik data baru dengan menemukan 'K' titik data terdekat (tetangga) dalam set pelatihan. Prediksi tersebut kemudian merupakan rata-rata nilai target dari K tetangga ini. Pada dasarnya, KNN memperkirakan nilai titik data baru berdasarkan kemiripannya dengan titik data terdekat.

"Kedekatan" titik data ditentukan oleh metrik jarak. Metrik umum meliputi:
- Jarak Euclidean: Jarak garis lurus antara dua titik.
- Jarak Manhattan: Jumlah selisih absolut koordinat dua titik.
- Jarak Minkowski: Generalisasi jarak Euclidean dan Manhattan.

**Kekuatan**:
- Mudah dipahami dan diterapkan: KNN secara konseptual mudah dipahami dan dipraktikkan.
- Tidak ada fase pelatihan yang eksplisit: KNN tidak memerlukan langkah pelatihan terpisah. Model dibangun sesuai kebutuhan selama prediksi.
- Nonparametrik: KNN tidak mengasumsikan bentuk fungsional tertentu untuk hubungan antara fitur dan variabel target, sehingga cocok untuk data kompleks dan nonlinier.

**Kelemahan**:
- Mahal secara komputasi: Memprediksi nilai untuk titik data baru memerlukan penghitungan jaraknya ke semua titik data dalam set pelatihan, yang dapat sangat memakan waktu untuk set data besar.
- Sensitif terhadap fitur yang tidak relevan: KNN mempertimbangkan semua fitur secara setara saat menghitung jarak, yang dapat menyebabkan kinerja yang buruk jika data berisi banyak fitur yang tidak relevan.
- Memerlukan penskalaan fitur: Fitur dengan skala yang lebih besar dapat mendominasi kalkulasi jarak, yang mengarah pada prediksi yang bias. Oleh karena itu, sangat penting untuk menskalakan fitur sebelum menerapkan KNN.
- Menentukan nilai K yang optimal: Pilihan K sangat penting untuk kinerja KNN. K yang kecil dapat menyebabkan prediksi yang tidak jelas, sementara K yang besar dapat memperhalus pola penting dalam data. K yang optimal sering ditemukan melalui eksperimen dan validasi.
- Data yang tidak seimbang: KNN dapat bias terhadap kelas mayoritas dalam kumpulan data yang tidak seimbang.

Cara pelatihan model KNN sangatlah mudah, hanya perlu menggunakan kode berikut:

knn = KNeighborsRegressor()
knn.fit(X_train_scaled, y_train)

knn = KNeighborsRegressor(): Membuat objek model KNN yang disebut knn menggunakan kelas KNeighborsRegressor. Objek ini menggunakan parameter default untuk model.
knn.fit(X_train_scaled, y_train): Kode ini melatih model KNN menggunakan dua argumen:
- X_train_scaled: Data pelatihan yang berisi fitur (seperti luas persegi, tahun pembangunan, dll.) yang telah diskalakan menggunakan StandardScaler. Penskalaan penting untuk KNN karena sensitif terhadap skala fitur.
- y_train: Data pelatihan yang berisi variabel target, yaitu harga rumah. Model mempelajari hubungan antara fitur dan harga rumah dari data ini.

**2. Random Forest**
Algoritma random forest adalah salah satu algoritma supervised learning yang termasuk ke dalam kategori ensemble (group) learning. Model ensemble sendiri merupakan model prediksi yang terdiri dari beberapa model yang bekerja sama untuk menyelesaikan masalah. Pada model ensemble, setiap model harus membuat prediksi secara independen yang nanti digabungkan untuk membuat prediksi akhir. 

Random Forest Regressor menggunakan banyak decision tree untuk meningkatkan akurasi regresi dan mengurangi overfitting. Decision Tree adalah struktur seperti diagram alir di mana setiap node mewakili pengujian pada atribut (fitur), setiap cabang mewakili hasil pengujian, dan setiap node daun mewakili prediksi (dalam regresi, nilai kontinu). Decision Tree untuk regresi biasanya bertujuan untuk meminimalkan varians atau mean squared error (MSE) pada setiap pemisahan. Untuk membuat prediksi untuk titik data baru, setiap decision tree memprediksi suatu nilai secara independen. Random Forest Regressor kemudian merata-ratakan prediksi dari semua decision tree untuk mendapatkan prediksi akhir.

**Kekuatan**:
- Akurasi tinggi: Dengan merata-ratakan prediksi beberapa decision tree, Random Forest umumnya mencapai akurasi yang lebih tinggi daripada satu decision tree saja. 
- Kuat terhadap overfitting: Kombinasi banyak decision tree membantu mengurangi overfitting.
- Menangani dimensionalitas tinggi: Random Forest dapat menangani set data dengan sejumlah besar fitur secara efektif.
- Menyediakan estimasi kepentingan fitur: Random Forest dapat memperkirakan pentingnya setiap fitur dalam set data, yang dapat berguna untuk pemilihan fitur dan pemahaman data.
- Menangani hubungan non-linier: Random Forest dapat memodelkan hubungan non-linier yang kompleks antara fitur dan variabel target.
- Tidak sensitif terhadap outlier: Proses perataan mengurangi dampak outlier.
- Bekerja dengan fitur kategoris dan numerik: Random Forest dapat menangani kedua jenis fitur tanpa memerlukan preprocessing.
**Kelemahan**:
- Komputasi intensif: Melatih Random Forest dapat memakan banyak komputasi, terutama untuk dataset besar. 
- Kurang dapat diinterpretasikan dibandingkan Decision Tree: Meskipun Random Forest dapat memberikan estimasi pentingnya fitur, namun secara umum Random Forest kurang dapat diinterpretasikan dibandingkan satu Decision Tree saja.
• Dapat menjadi "black box": Karena kompleksitasnya, proses prediksi Random Forest dapat susah dipahami.
• Potensi bias: Jika kumpulan data asli memiliki data yang bias, model random forest juga akan bias.

Cara pelatihan model Random Forest sangatlah mudah, hanya perlu menggunakan kode berikut:

rf = RandomForestRegressor()
rf.fit(X_train, y_train) # No need for standardized data

rf = RandomForestRegressor(): Membuat objek model Random Forest yang disebut rf menggunakan kelas RandomForestRegressor dengan parameter default.
rf.fit(X_train, y_train): Kode ini melatih model Random Forest menggunakan dua argumen:
- X_train: Data pelatihan yang berisi fitur. Data pelatihan yang tidak diskalakan digunakan di sini karena model Random Forest tidak terpengaruh oleh penskalaan fitur.
- y_train: Data pelatihan yang berisi harga rumah.

**3. AdaBoost**
AdaBoost, singkatan dari Adaptive Boosting, adalah metode pembelajaran ensemble yang menggabungkan prediksi beberapa pembelajar lemah untuk menciptakan pembelajar yang kuat. Tidak seperti Random Forest, yang melatih model secara paralel, AdaBoost melatihnya secara berurutan atau iteratif, dengan setiap model mencoba memperbaiki kesalahan pendahulunya. 

Awalnya, semua kasus dalam data latih memiliki weight atau bobot yang sama. Algoritme AdaBoost melatih secara berulang serangkaian model dan mengevaluasi performa mereka. Bobot (kepentingan) diberikan kepada model berdasarkan performanya. Model yang lebih baik mempelajari data mendapatkan bobot yang lebih tinggi. Prediksi akhir diperoleh dengan mengambil rata-rata tertimbang dari prediksi individual, di mana bobotnya adalah bobot kepentingan model. Proses iteratif ini berlanjut sampai model mencapai akurasi yang diinginkan.

Kekuatan:
- Akurasi yang bagus: AdaBoost memiliki akurasi tinggi karena merupakan metode ensemble.
- Kepentingan fitur: AdaBoost dapat memberikan estimasi pentingnya fitur.
- Relatif mudah diimplementasikan: Algoritma inti AdaBoost relatif mudah diimplementasikan.

Kelemahan:
-  Sensitif terhadap noise dan outlier: AdaBoost mencoba menyesuaikan setiap titik data dengan tepat, yang membuatnya sensitif terhadap noise dan outlier. Outlier dapat memengaruhi pembobotan titik data secara tidak proporsional, yang membuat model yang kurang optimal.
- Komputasi intensif: Seperti metode boosting lainnya, AdaBoost menggunakan komputasi intensif, terutama untuk dataset besar. 
- Dapat overfitting: Jika pelatihan terlalu rumit, AdaBoost dapat melakukan overfitting pada data pelatihan.

Cara pelatihan model AdaBoost sangatlah mudah, hanya perlu menggunakan kode berikut:

ada = AdaBoostRegressor()
ada.fit(X_train, y_train) # No need for standardized data

ada = AdaBoostRegressor(): Membuat objek model AdaBoost yang disebut ada menggunakan kelas AdaBoostRegressor dengan parameter default.
ada.fit(X_train, y_train): Kode ini melatih model AdaBoost menggunakan dua argumen. 
- X_train: Data pelatihan yang berisi fitur. Mirip dengan Random Forest, metode ini menggunakan data pelatihan yang tidak diskalakan (X_train) karena model AdaBoost tidak terpengaruh oleh penskalaan fitur.
- y_train: Data pelatihan yang berisi harga rumah.


**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Dalam analisis regresi ketiga model, digunakan metrik khusus untuk mengevaluasi seberapa baik model kami memprediksi variabel target yang berupa harga rumah. Berikut ini penjelasan dari tiga metrik yang digunakan: Mean Squared Error (MSE), Mean Absolute Error (MAE), dan R-squared (R²).

**Mean Squared Error (MSE)**
MSE menghitung rata-rata perbedaan kuadrat antara nilai yang diprediksi dan nilai aktual.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Dimana:
- $$(n)$$ adalah jumlah titik data.
- $$(y_i)$$ adalah nilai aktual dari variabel target untuk titik data ke-(i).
- $$(\hat{y}_i)$$ adalah nilai prediksi dari variabel target untuk titik data ke-(i).

MSE mengukur besarnya rata-rata kesalahan kuadrat, yang membuat MSE sensitif terhadap outlier. Nilai MSE yang lebih rendah menunjukkan kinerja model yang lebih baik. MSE 0 berarti model memprediksi dengan sempurna.

**Mean Absolute Error (MAE)**
MAE menghitung rata-rata perbedaan absolut antara nilai yang diprediksi dan nilai aktual.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Dimana:
- $$(n)$$ adalah jumlah titik data.
- $$(y_i)$$ adalah nilai aktual dari variabel target untuk titik data ke-(i).
- $$(\hat{y}_i)$$ adalah nilai prediksi dari variabel target untuk titik data ke-(i).

MAE kurang sensitif terhadap outlier dibandingkan dengan MSE karena menggunakan nilai absolut daripada kesalahan kuadrat. Nilai MAE yang lebih rendah menunjukkan kinerja model yang lebih baik. MAE 0 berarti model memprediksi dengan sempurna.

**R-kuadrat (R²)**
R-kuadrat mengukur proporsi varians dalam variabel dependen yang dijelaskan oleh variabel independen dalam model. Ini menunjukkan seberapa baik model tersebut dalam memprediksi variabel target dibandingkan dengan hanya menggunakan rata-rata variabel target.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Dimana:
- $$(SS_{res})$$ adalah jumlah kuadrat residual: $$(\sum_{i=1}^{n} (y_i - \hat{y}_i)^2)$$
- $$(SS_{tot})$$ adalah jumlah kuadrat total: $$(\sum_{i=1}^{n} (y_i - \bar{y})^2)$$
- $$(\bar{y})$$ adalah rata-rata nilai aktual variabel target.

R² berkisar dari 0 hingga 1, dengan 1 menunjukkan bahwa model menjelaskan semua varians dalam variabel target, dan 0 menunjukkan bahwa model tidak menjelaskan varians apa pun dalam variabel target.

**Evaluasi Hasil Proyek**

Pada tahap ini, kita akan mengevaluasi performa ketiga model menggunakan metrik Mean Square Error (MSE), Mean Absolute Error (MAE), dan R2. Sebelum pengujian menggunakan test set, kita harus melakukan standardisasi dengan scaler yang sama seperti pada train set, agar data berskala sama (Hanya untuk KNN, karena RF dan AdaBoost tidak peduli skala).

|        | train_mse    | test_mse     | train_mae    | test_mae     | train_r2     | test_r2      |
| :------- | :------------- | :------------- | :------------- | :------------- | :------------- | :------------- |
| KNN      | 1379094432.787155 | 2285279964.636393 | 29263.346088   | 38271.001851   | 0.978511       | 0.964547       |
| RandomForest | 81912686.95175  | 476631737.055197 | 7057.040997    | 17970.394885   | 0.998724       | 0.992606       |
| Boosting   | 880410352.054793 | 948741444.830329 | 24261.081325   | 25087.878261   | 0.986281       | 0.985281       |

Evaluasi dilakukan dengan ketiga metrik pada set train dan test. Dapat dilihat bahwa ketiga model mengalami peningkatan MSE dan MAE pada test set dibandingkan dengan train set, yang merupakan hal normal. Performa Random Forest paling baik, namun performa AdaBoost paling konsisten pada train dan test set. Perubahan R2 dari train ke test sangat kecil bagi ketiga model, menandakan model tidak overfitting.

![img](https://i.imgur.com/u38Y1sE.png)

Dapat dilihat dari grafik diatas bahwa performa model Random Forest adalah yang paling baik, dengan MAE dan MSE yang paling kecil, dan R2 yang paling besar, baik pada train set dan test set. AdaBoost memiliki performa kedua paling baik, dan KNN menempati posisi terakhir, dengan performa paling buruk.

Selanjutnya, untuk memastikan lebih lanjut, digunakan kode pada notebook untuk menghitung proporsi akar MSE (atau disebut RMSE, agar skala sama dengan data asli) dan MAE terhadap mean, sebagai berikut:

//Calculate the mean of house prices
mean_house_price = np.mean(df['House_Price'])

//Get RMSE (Root Mean Square Error) values of test set
rmse_knn = np.sqrt(models.loc['test_mse', 'KNN'])
rmse_rf = np.sqrt(models.loc['test_mse', 'RandomForest'])
rmse_ada = np.sqrt(models.loc['test_mse', 'Boosting'])

//Calculate the proportion
proportion_knn = rmse_knn / mean_house_price
proportion_rf = rmse_rf / mean_house_price
proportion_ada = rmse_ada / mean_house_price

print(f"Proportion of KNN RMSE to mean of house price: {proportion_knn}")
print(f"Proportion of RF RMSE to mean of house price: {proportion_rf}")
print(f"Proportion of AB RMSE to mean of house price: {proportion_ada}")

yang mengeluarkan output berikut:
Proportion of KNN RMSE to mean of house price: 0.07724610288681014
Proportion of RF RMSE to mean of house price: 0.03527754468484738
Proportion of AB RMSE to mean of house price: 0.04977150904574513

Dapat dilihat bahwa proporsi RMSE sangat kecil jika dibandingkan dengan mean dari data harga rumah, yang menunjukkan bahwa akar dari selisih nilai prediksi dengan nilai sebenarnya yang dikuadrat sangat kecil dibandingkan mean harga rumah. Untuk MAE:

//Get MAE values of test set
mae_knn = models.loc['test_mae', 'KNN']
mae_rf = models.loc['test_mae', 'RandomForest']
mae_ada = models.loc['test_mae', 'Boosting']

//Calculate the proportion
proportion_knn = mae_knn / mean_house_price
proportion_rf = mae_rf / mean_house_price
proportion_ada = mae_ada / mean_house_price

print(f"Proportion of KNN MAE to mean of house price: {proportion_knn}")
print(f"Proportion of RF MAE to mean of house price: {proportion_rf}")
print(f"Proportion of AB MAE to mean of house price: {proportion_ada}")

dengan output:
Proportion of KNN MAE to mean of house price: 0.06184102843486997
Proportion of RF MAE to mean of house price: 0.02903785235066378
Proportion of AB MAE to mean of house price: 0.04053879224105463

Maka terlihat bahwa proporsi MAE terhadap mean juga sangat kecil, yang menunjukkan bahwa selisih nilai prediksi dengan nilai sebenarnya sangat kecil dibandingkan mean harga rumah.

Berikutnya ketiga model juga diuji menggunakan set test, memprediksi 5 nilai pertama dengan kode berikut:

//Create a DataFrame with the first 5 samples from the test set
comparison_df = pd.DataFrame({
    'Actual': y_test.iloc[:5],
    'KNN': y_pred_knn_test[:5],
    'RandomForest': y_pred_rf_test[:5],
    'AdaBoost': y_pred_ada_test[:5]
})

Dengan output dataframe sebagai berikut:

|   | Actual        | KNN             | RandomForest    | AdaBoost        |
|--:|--------------:|----------------:|----------------:|----------------:|
| 0 | 9.01000e+05 | 822241.955935 | 8.55440e+05   | 8.53697e+05   |
| 1 | 4.94537e+05 | 516527.861433 | 5.08420e+05   | 5.03458e+05   |
| 2 | 9.49404e+05 | 975953.974347 | 9.49884e+05   | 9.70672e+05   |
| 3 | 1.04039e+06 | 978179.212127 | 1.047002e+06  | 1.033746e+06  |
| 4 | 7.94010e+05 | 769437.013846 | 8.125074e+05  | 7.876537e+05  |

![img](https://i.imgur.com/U1lNrfB.png)

Tabel dan grafik diatas menunjukkan bahwa selisih memang kecil dengan nilai asli seperti ditandakan oleh proporsi RMSE dan MAE, dan tingginya nilai R2.

Mari mengevaluasi hasil proyek sesuai dengan statement awal:
### Problem Statements

- Bagaimana hubungan harga rumah dengan fitur-fitur tertentu?
- Bagaimana mengetahui harga rumah dengan karakteristik atau fitur tertentu?

### Goals

- Mengetahui hubungan fitur-fitur dengan harga rumah.
- Membuat model machine learning yang dapat memprediksi harga rumah seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements
- Menggunakan pairplot dan heatmap untuk melihat hubungan fitur dengan harga rumah.
- Menggunakan tiga jenis algoritma regresi untuk menggunakan model yang mencapai performa yang terbaik.


**1. Apakah kita mengetahui hubungan harga rumah dengan fitur-fitur tertentu seperti pada goals, dengan menggunakan pairplot dan heatmap untuk melihat hubungan fitur dengan harga rumah?**

Kita telah mengetahui hubungan antara harga rumah dengan fitur-fitur lainnya, yang terlihat pada bagian EDA, dengan pairplot dan matriks korelasi sebagai berikut:

![img](https://i.imgur.com/wcqLWBp.png)

![img](https://i.imgur.com/dkND031.png)

Dapat dilihat bahwa Square_Footage memiliki korelasi positif yang kuat, dengan semua fitur lain memiliki korelasi yang lemah karena tidak membentuk pola yang positif atau negatif. Matriks korelasi di atas mengonfirmasi pengamatan kita dari pairplot, dengan House_Price memiliki korelasi sangat tinggi (0.99) dengan Square_Footage, dan korelasi lemah dengan semua fitur lain.

**2. Apakah kita telah membuat cara mengetahui harga rumah dengan karakteristik atau fitur tertentu, dengan membuat 3 model machine learning regresi yang dapat memprediksi harga rumah seakurat mungkin berdasarkan fitur-fitur yang ada untuk diuji model mana memiliki performa terbaik sehingga terpilih sebagai model untuk prediksi harga rumah?**

Kita telah membuat 3 model regresi yang dapat memprediksi harga rumah dengan karakteristik atau fitur tertentu secara akurat, dengan model paling buruk performanya (KNN) memiliki proporsi RMSE dan MAE terhadap mean yang sangat kecil terhadap mean (0.077 dan 0.062), kurang dari 10% mean, dan nilai R² sebesar 0.964547 pada test set, menunjukkan bahwa KNN dan dua model lainnya yang memiliki performa lebih baik dari dia, memiliki selisih prediksi yang kecil dibandingkan nilai asli harga rumah, dan dapat menjelaskan hampir semua varians pada variabel target (harga rumah).

![img](https://i.imgur.com/u38Y1sE.png)

Hasil Prediksi 5 sampel data test set:

|   | Actual        | KNN             | RandomForest    | AdaBoost        |
|--:|--------------:|----------------:|----------------:|----------------:|
| 0 | 9.01000e+05 | 822241.955935 | 8.55440e+05   | 8.53697e+05   |
| 1 | 4.94537e+05 | 516527.861433 | 5.08420e+05   | 5.03458e+05   |
| 2 | 9.49404e+05 | 975953.974347 | 9.49884e+05   | 9.70672e+05   |
| 3 | 1.04039e+06 | 978179.212127 | 1.047002e+06  | 1.033746e+06  |
| 4 | 7.94010e+05 | 769437.013846 | 8.125074e+05  | 7.876537e+05  |

![img](https://i.imgur.com/U1lNrfB.png)

Dapat dilihat dari grafik-grafik dan tabel diatas bahwa performa model Random Forest adalah yang paling baik, dengan MAE dan MSE yang paling kecil, dan R2 yang paling besar, baik pada train set dan test set. AdaBoost memiliki performa kedua paling baik, dan KNN menempati posisi terakhir, dengan performa paling buruk. Terlihat juga bahwa selisih memang kecil dengan nilai asli seperti ditandakan oleh proporsi RMSE dan MAE, dan tingginya nilai R2.

Maka, sesuai dengan goals dan solution statement, kita dapat memilih model Random Forest sebagai model machine learning yang dapat memprediksi harga rumah seakurat mungkin berdasarkan fitur-fitur yang ada, untuk menyelesaikan problem statement "Bagaimana mengetahui harga rumah dengan karakteristik atau fitur tertentu?", berdasarkan performa melalui metrik MSE, MAE, dan R2 seperti terlihat pada tabel dan grafik performa ketiga model.



