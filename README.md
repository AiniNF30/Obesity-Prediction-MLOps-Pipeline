# Submission 1: Nama Proyek Anda
Nama: Aini Nurpadilah

Username dicoding: Aininrp

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Obesity Prediction](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction) |
| Masalah | Obesitas adalah kondisi yang terjadi ketika seseorang memiliki kelebihan berat badan yang dapat memengaruhi kesehatan fisik dan mental. Obesitas sering kali disebabkan oleh kombinasi faktor genetik, gaya hidup, pola makan yang tidak sehat, dan kurangnya aktivitas fisik. Meskipun sebagian orang dapat hidup dengan obesitas tanpa masalah kesehatan yang signifikan, kondisi ini dapat meningkatkan risiko berbagai penyakit serius seperti diabetes, penyakit jantung, dan hipertensi. Jika tidak diatasi, obesitas dapat memengaruhi kualitas hidup, menurunkan rasa percaya diri, dan memicu masalah kesehatan mental seperti depresi atau kecemasan. |
| Solusi machine learning |Dengan menggunakan machine learning, kita dapat memprediksi apakah seseorang berisiko mengalami obesitas atau tidak berdasarkan faktor-faktor seperti usia, jenis kelamin, tinggi badan, berat badan, tingkat aktivitas fisik, dan BMI. Prediksi ini dapat membantu dalam pengembangan kebijakan pencegahan obesitas dan memberikan rekomendasi intervensi berbasis data untuk mengurangi prevalensi obesitas di masyarakat.|
| Metode pengolahan |Dataset Obesity Prediction berisi berbagai fitur, termasuk informasi demografis dan gaya hidup. Fitur-fitur ini diproses untuk membagi dataset menjadi data pelatihan dan evaluasi dengan rasio 80:20. Fitur teks atau data kategori dikonversi menjadi nilai numerik, dan model kemudian dilatih untuk memprediksi apakah seseorang memiliki obesitas atau tidak. |
| Arsitektur model | Model yang digunakan adalah neural network dengan beberapa layer Dense. Model ini menggunakan aktivasi ReLU dan sigmoid untuk klasifikasi dua kelas (obesitas vs tidak obesitas). Fungsi loss yang digunakan adalah binary_crossentropy, dengan optimizer Adam dan metrik BinaryAccuracy. |
| Metrik evaluasi | Metrik evaluasi meliputi ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, dan BinaryAccuracy. |
| Performa model | Evaluasi model menunjukkan AUC sebesar 92%, ExampleCount 800, BinaryAccuracy 88%, dan loss 0.45. False Negatives adalah 40, False Positives 30, True Negatives 250, dan True Positives 480. Model ini menunjukkan performa yang baik namun masih dapat ditingkatkan untuk mencapai akurasi lebih tinggi. |
