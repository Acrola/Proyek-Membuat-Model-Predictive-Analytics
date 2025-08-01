# -*- coding: utf-8 -*-
"""Proyek Membuat Model Predictive Analytics.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gjocHLp48fAAH85cnUNyV3gAuyiRKXCR

# Import Library

Pada bagian ini kita mengimport library yang digunakan pada proyek.
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

"""# Data Loading

Pada bagian ini kita mengakses data yang akan digunakan pada proyek. Dataset ini dapat ditemukan pada https://www.kaggle.com/datasets/prokshitha/home-value-insights/data.
"""

df = pd.read_csv('/content/house_price_regression_dataset.csv')

"""# Exploratory Data Analysis

Pada bagian ini kita akan melakukan Exploratory Data Analysis (EDA), yang merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

**Deskripsi Variabel**
"""

# See first five rows of the data columns
df.head()

"""Menggunakan info dari Kaggle, pada dataset ini terdapat 8 fitur, yaitu:

- Square_Footage: Ukuran rumah dalam kaki persegi. Rumah yang lebih besar biasanya memiliki harga yang lebih tinggi.

- Num_Bedrooms: Jumlah kamar tidur di rumah. Lebih banyak kamar tidur umumnya meningkatkan nilai rumah.

- Num_Bathrooms: Jumlah kamar mandi di rumah. Rumah dengan lebih banyak kamar mandi biasanya memiliki harga yang lebih tinggi.

- Year_Built: Tahun rumah dibangun. Rumah yang lebih tua mungkin memiliki harga yang lebih rendah karena keausan.

- Lot_Size: Ukuran tanah tempat rumah dibangun, diukur dalam hektar. Tanah yang lebih besar cenderung menambah nilai properti.

- Garage_Size: Jumlah mobil yang dapat muat di garasi. Rumah dengan garasi yang lebih besar biasanya lebih mahal.

- Neighborhood_Quality: Peringkat kualitas lingkungan pada skala 1-10, di mana 10 menunjukkan lingkungan yang berkualitas tinggi. Lingkungan yang lebih baik biasanya memiliki harga yang lebih tinggi.

- House_Price (Variabel Target): Harga rumah, yang merupakan variabel dependen yang ingin diprediksi.
"""

# See general info of dataset
df.info()

"""Dapat dilihat bahwa terdapat 1000 baris dari 8 kolom data numerik, dengan tidak ada data null atau hilang, dan jenis data berupa int64 (6 fitur) dan float64 (2 fitur)."""

# See general statistics of dataset
df.describe()

"""Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:

- Count  adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

**Menangani Missing Value**
"""

# Check for missing values
df.isnull().sum()

"""Seperti dilihat dari df.info(), tidak ada missing value, sehingga tidak perlu diatasi.

**Menangani Outlier**
"""

# Define IQR method to find outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Find outliers using IQR method
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
outliers

"""Menggunakan metode IQR, dapat dilihat bahwa tidak terdapat outlier pada dataset ini."""

# Create a boxplot for each feature
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

"""Visualisasi distribusi data menggunakan boxplot. Dapat dilihat bahwa tidak terdapat outlier pada data.

**Univariate Analysis**

Menggunakan teknik analisis satu variabel untuk menganalisa data.
"""

# Create histograms for each feature
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns):
    plt.subplot(3, 3, i + 1)
    plt.hist(df[col], bins=10)  # Adjust the number of bins as needed
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

"""Dapat dilihat bahwa dataset memiliki distribusi yang rata. Tiga fitur (Num_Bedrooms, Num_Bathrooms, Garage_Size) memiliki distribusi yang terlihat agak aneh karena rentang nilai diskrit mereka yang kecil, namun distribusi mereka juga rata.

**Multivariate Analysis**

Menggunakan teknik multivariate untuk menunjukkan hubungan antara dua atau lebih variabel pada data.
"""

# Use pairplot to find the relationship between data
sns.pairplot(df)
plt.show()

"""Dari pairplot, kita mendapatkan grafik hubungan antar fitur pada dataset. Kita paling tertarik pada baris paling bawah, yaitu hubungan tiap fitur dengan fitur target kita, House_Price."""

# Use pairplot to find the relationship of House_Price with every other feature
sns.pairplot(df, x_vars=df.columns, y_vars='House_Price')
plt.show()

"""Dapat dilihat bahwa Square_Footage memiliki korelasi positif yang kuat, dengan semua fitur lain memiliki korelasi yang lemah karena tidak membentuk pola yang positif atau negatif."""

# Generate correlation matrix for all the features
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix", size=20)

"""Matriks korelasi yang kita buat mengonfirmasi pengamatan kita dari pairplot, dengan House_Price memiliki korelasi sangat tinggi (0.99) dengan Square_Footage, dan korelasi lemah dengan semua fitur lain. Bagi fitur dengan korelasi sangat kecil (<=0.01), kita akan melakukan drop."""

# Drop low correlation columns
df = df.drop(['Num_Bedrooms', 'Num_Bathrooms', 'Neighborhood_Quality'], axis=1)

# Show dataframe with dropped columns
df.info()

"""# Data Preparation

Data preparation merupakan tahapan penting dalam proses pengembangan model machine learning. Ini adalah tahap di mana kita melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan.

**Train-Test Split**

Disini kita membagi data menjadi set training dan test, dengan rasio yang digunakan berupa 80/20, yang berarti 80 persen data digunakan untuk pelatihan, dan 20 untuk pengujian.
"""

from sklearn.model_selection import train_test_split

# Split data into train and test sets
X = df.drop('House_Price', axis=1)
y = df['House_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80/20 split with random state

# Print number of samples in the datasets
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""**Standardisasi Data**

Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan, yang pada proyek ini menggunakan teknik StandarScaler dari library Scikitlearn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

Untuk menghindari kebocoran informasi pada data uji, kita hanya akan menerapkan fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi, kita akan melakukan standarisasi pada data uji.
"""

from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Fit on training data
scaler.fit(X_train)

# Transform training data
X_train_scaled = scaler.transform(X_train)

"""# Model Development

Model development adalah tahapan di mana kita menggunakan algoritma machine learning untuk menjawab problem statement dari tahap business understanding.

Pada tahap ini, kita akan mengembangkan model machine learning dengan tiga algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan adalah:

- K-Nearest Neighbor
- Random Forest
- Boosting Algorithm

Pertama kita akan menyiapkan dataframe untuk analisis ketiga model nantinya, menggunakan metrik MSE, MAE, dan R2.
"""

# Prepare dataframe for model analysis
models = pd.DataFrame(index=['train_mse', 'test_mse', 'train_mae', 'test_mae', 'train_r2', 'test_r2'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""Lalu kita akan melatih ketiga algoritma menggunakan data pada train set. Random Forest dan AdaBoost tidak peduli skala seperti KNN, maka set train mereka tidak menggunakan set yang distandardisasi."""

# Train the three machine learning algorithms on the train set
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

# KNN
knn = KNeighborsRegressor()
knn.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train) # No need for standardized data

# AdaBoost
ada = AdaBoostRegressor()
ada.fit(X_train, y_train) # No need for standardized data

"""# Evaluasi Model

Pada tahap ini, kita akan mengevaluasi performa ketiga model menggunakan metrik Mean Square Error (MSE), Mean Absolute Error (MAE), dan R2. Sebelum pengujian menggunakan test set, kita harus melakukan standardisasi dengan scaler yang sama seperti pada train set, agar data berskala sama.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluate KNN model
y_pred_knn_train = knn.predict(X_train_scaled)
X_test_scaled = scaler.transform(X_test) # Standardize test set with same scaling as train set
y_pred_knn_test = knn.predict(X_test_scaled)

models.loc['train_mse', 'KNN'] = mean_squared_error(y_train, y_pred_knn_train)
models.loc['test_mse', 'KNN'] = mean_squared_error(y_test, y_pred_knn_test)
models.loc['train_mae', 'KNN'] = mean_absolute_error(y_train, y_pred_knn_train)
models.loc['test_mae', 'KNN'] = mean_absolute_error(y_test, y_pred_knn_test)
models.loc['train_r2', 'KNN'] = r2_score(y_train, y_pred_knn_train)
models.loc['test_r2', 'KNN'] = r2_score(y_test, y_pred_knn_test)

# Evaluate Random Forest model
y_pred_rf_train = rf.predict(X_train)
y_pred_rf_test = rf.predict(X_test)

models.loc['train_mse', 'RandomForest'] = mean_squared_error(y_train, y_pred_rf_train)
models.loc['test_mse', 'RandomForest'] = mean_squared_error(y_test, y_pred_rf_test)
models.loc['train_mae', 'RandomForest'] = mean_absolute_error(y_train, y_pred_rf_train)
models.loc['test_mae', 'RandomForest'] = mean_absolute_error(y_test, y_pred_rf_test)
models.loc['train_r2', 'RandomForest'] = r2_score(y_train, y_pred_rf_train)
models.loc['test_r2', 'RandomForest'] = r2_score(y_test, y_pred_rf_test)

# Evaluate AdaBoost model
y_pred_ada_train = ada.predict(X_train)
y_pred_ada_test = ada.predict(X_test)

models.loc['train_mse', 'Boosting'] = mean_squared_error(y_train, y_pred_ada_train)
models.loc['test_mse', 'Boosting'] = mean_squared_error(y_test, y_pred_ada_test)
models.loc['train_mae', 'Boosting'] = mean_absolute_error(y_train, y_pred_ada_train)
models.loc['test_mae', 'Boosting'] = mean_absolute_error(y_test, y_pred_ada_test)
models.loc['train_r2', 'Boosting'] = r2_score(y_train, y_pred_ada_train)
models.loc['test_r2', 'Boosting'] = r2_score(y_test, y_pred_ada_test)

models

"""Evaluasi dilakukan dengan ketiga metrik pada set train dan test. Dapat dilihat bahwa ketiga model mengalami peningkatan MSE dan MAE pada test set dibandingkan dengan train set, yang merupakan hal normal. Performa Random Forest paling baik, namun performa Adaboost paling konsisten pada train dan test set. Perubahan R2 dari train ke test sangat kecil bagi ketiga model, menandakan model tidak overfitting."""

# Plot the model performance metrics
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Mean Squared Error (MSE)
models.loc[['train_mse', 'test_mse']].plot(kind='bar', ax=axes[0])
axes[0].set_title('Mean Squared Error (MSE)')
axes[0].set_ylabel('MSE')

# Mean Absolute Error (MAE)
models.loc[['train_mae', 'test_mae']].plot(kind='bar', ax=axes[1])
axes[1].set_title('Mean Absolute Error (MAE)')
axes[1].set_ylabel('MAE')

# R-squared (R2)
models.loc[['train_r2', 'test_r2']].plot(kind='bar', ax=axes[2])
axes[2].set_title('R-squared (R2)')
axes[2].set_ylabel('R2')

plt.tight_layout()
plt.show()

"""Maka dapat dilihat dari ketiga grafik diatas, bahwa performa model Random Forest adalah yang paling baik, dengan MAE dan MSE yang paling kecil, dan R2 yang paling besar. AdaBoost memiliki performa kedua paling baik, dan KNN menempati posisi terakhir, dengan performa paling buruk."""

# Calculate the mean of house prices
mean_house_price = np.mean(df['House_Price'])

# Get RMSE (Root Mean Square Error) values of test set
rmse_knn = np.sqrt(models.loc['test_mse', 'KNN'])
rmse_rf = np.sqrt(models.loc['test_mse', 'RandomForest'])
rmse_ada = np.sqrt(models.loc['test_mse', 'Boosting'])

# Calculate the proportion
proportion_knn = rmse_knn / mean_house_price
proportion_rf = rmse_rf / mean_house_price
proportion_ada = rmse_ada / mean_house_price

print(f"Proportion of KNN RMSE to mean of house price: {proportion_knn}")
print(f"Proportion of RF RMSE to mean of house price: {proportion_rf}")
print(f"Proportion of AB RMSE to mean of house price: {proportion_ada}")

"""Pada kode diatas, MSE diakar menjadi RMSE agar skalanya setara dengan data asli. Dapat dilihat bahwa proporsi RMSE sangat kecil jika dibandingkan dengan mean dari data harga rumah, yang menunjukkan bahwa akar dari selisih nilai prediksi dengan nilai sebenarnya yang dikuadrat sangat kecil dibandingkan mean harga rumah."""

# Get MAE values of test set
mae_knn = models.loc['test_mae', 'KNN']
mae_rf = models.loc['test_mae', 'RandomForest']
mae_ada = models.loc['test_mae', 'Boosting']

# Calculate the proportion
proportion_knn = mae_knn / mean_house_price
proportion_rf = mae_rf / mean_house_price
proportion_ada = mae_ada / mean_house_price

print(f"Proportion of KNN MAE to mean of house price: {proportion_knn}")
print(f"Proportion of RF MAE to mean of house price: {proportion_rf}")
print(f"Proportion of AB MAE to mean of house price: {proportion_ada}")

"""Proporsi MAE terhadap mean juga sangat kecil, yang menunjukkan bahwa selisih nilai prediksi dengan nilai sebenarnya sangat kecil dibandingkan mean harga rumah."""

# Create a DataFrame with the first 5 samples from the test set
comparison_df = pd.DataFrame({
    'Actual': y_test.iloc[:5],
    'KNN': y_pred_knn_test[:5],
    'RandomForest': y_pred_rf_test[:5],
    'AdaBoost': y_pred_ada_test[:5]
})

# Reset the index for better presentation
comparison_df = comparison_df.reset_index(drop=True)

# Print the comparison DataFrame
comparison_df

# Make chart to graphically show the actual vs predicted values
comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of Actual vs. Predicted House Prices (First 5 Samples)')
plt.xlabel('Sample')
plt.ylabel('House Price')
plt.xticks(rotation=0)
plt.legend(title='Model')
plt.show()

"""Tabel dan grafik diatas menunjukkan bahwa selisih memang kecil dengan nilai asli seperti ditandakan oleh proporsi RMSE dan MAE, dan tingginya nilai R2."""