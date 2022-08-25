# Laporan Proyek Machine Learning - Evans Hebert

## Project Overview

Seiring dengan berjalannya waktu, teknologi digital tentu telah berkembang dengan pesat, termasuk di dunia media digital. 
Menonton Film dari Kaset Radio, ataupun CD/DVD sudah tidak relevan lagi dalam dunia ini. 
Tentu laman web seperti Netflix, Youtube, Amazon Prime dll. sudah tidak asing lagi untuk didengar. 
Laman web tersebut merupakan platform penyedia film yang dapat ditonton secara online di manapun dan kapanpun [[1]](https://labelyourdata.com/articles/movie-recommendation-with-machine-learning).

Misalkan Youtube, ketika sedang menonton beberapa video atau film dengan konten yang mirip, Youtube akan memberikan rekomendasi beberapa video atau film lain yang mungkin disukai.
Sistem rekomendasi tersebut merupakan sebuah sistem dengan model yang telah dirancang dengan menggunakan Machine Learning.
Sistem rekomendasi film itulah yang akan menyelesaikan masalah kebingungan dari para penggunanya.
Oleh karena itu, sistem rekomendasi film ini sangat penting untuk diimplementasikan karena akan sangat memudahkan calon penggunanya dalam mencari film yang diinginkan.
Selain itu, dengan menggunakan sistem rekomendasi film, calon penggunanya akan dapat ditawarkan dengan film berbayar yang mungkin tertarik. 
Hal ini akan meningkatkan peluang bisnis dalam memasarkan film yang berbayar.

## Business Understanding

### Problem Statements

Agar dapat merancang sebuah model yang dapat digunakan untuk merekomendasi film kepada pelanggan, maka hal-hal ini perlu diketahui dan diselesaikan.
Hal-hal tersebut meliputi:
- Bagaimana langkah yang perlu diambil agar dapat mengembangkan sebuah model yang dapat mempelajari pola pelanggan dalam menonton film?
- Bagaimana sistem tersebut dapat merekomendasi film dengan genre yang serupa kepada pelanggan yang belum menonton film tersebut?

### Goals

Tujuan yang perlu dicapaikan agar dapat membuat sistem rekomendasi adalah sebagai berikut:
- Mengetahui data yang akan diolah untuk mengembangkan model yang dapat mempelajari pola pelanggan dalam memilih film yang ingin ditontonnya.
- Mengimplementasi sistem rekomendasi agar dapat melakukan rekomendasi film-film yang belum ditonton, namun dapat menjadi potensi film yang dapat ditonton pelanggan.

### Solution statements

- Menggunakan data rating yang merupakan review dari pelanggan terhadap sebuah film untuk mempelajari pola pelanggan tertentu dalam memilih film yang disukainya.
- Menggunakan teknik *Collaborative Filtering* dalam pembuatan model untuk merekomendasikan film-film yang belum ditonton oleh pelanggan.

## Data Understanding

Dataset yang akan digunakan berasal dari Kaggle yang berjudul [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). 
Dataset tersebut terdiri dari 45.466 data film dan 26 juta rating yang diambil dari Dataset yang berjudul *Full MovieLens*. 
Namun, karena data tersebut sangat besar, maka dalam Kaggle disediakan dataset yang lebih kecil, yaitu `ratings_small.csv` yang terdiri dari 100.004 dataset rating yang dapat digunakan untuk membuat model.

Terdapat sejumlah dataset dalam bentuk file `.csv` yang dapat digunakan, namun pada pemodelan kali ini hanya akan menggunakan 2 file dataset, yaitu:
- `movies_metadata.csv` dengan 45.466 dataset
- `ratings_small.csv` dengan 100.004 dataset

Pada `movies_metadata.csv` terdapat 24 kolom yang dapat digunakan, namun karena pada latihan ini akan dibuat model rekomendasi sistem yang hanya memerlukan nama dari film tersebut, maka kolom yang akan digunakan merupakan hal-hal berikut ini.
- `id` merupakan Unique ID dari film.
- `title` merupakan judul dari film.
- `genres` merupakan genres dari film, dapat memiliki lebih dari 1 set.

Pada `ratings_small.csv` terdapat 4 kolom yang dapat digunakan. Kolom-kolom tersebut berupa
- `userId` merupakan Unique ID dari pengguna yang menonton film.
- `movieId` merupakan Unique ID dari sebuah film yang ditonton oleh seorang pengguna.
- `rating` merupakan Nilai yang diberikan oleh pengguna terhadap film tersebut, berkisar dari 0.5 hingga 5.
- `timestamp` merupakan waktu ketika nilai rating diberikan oleh pengguna.

Dataset yang akan digunakan merupakan dua file yang telah didefinisikan, dengan `userId` dan `movieId` sebagai dataset yang akan digunakan, dengan nilai `rating` sebagai label untuk model tersebut.
Model yang akan dirancang dengan teknik *Collaborative Filtering* ini akan menerima input berupa `userId` dan `movieId` dan akan melakukan prediksi terhadap nilai `rating` yang akan diberikan oleh user.
Untuk itu, pada `ratings_small.csv`, kolom-kolom yang akan digunakan adalah `userId`, `movieId`, dan `rating` sehingga kolom `timestamp` dapat dihapus.

## Data Preparation

Pertama-tama, dataset diolah supaya hanya kolom-kolom yang digunakan yang akan disisakan. 
Sehingga, dataset `movie` hanya akan memiliki kolom `id`, `title`, dan `genres`. Sedangkan, dataset `rating` hanya akan memiliki kolom `userId`, `movieId`, dan `rating`.

Table `movieId`
|	id	  | title	                      | genres                                            |
|:------|:----------------------------|:--------------------------------------------------|
| 862	  | Toy Story	                  | [{'id': 16, 'name': 'Animation'}, {'id': 35, '... |
| 8844  | Jumanji	                    | [{'id': 12, 'name': 'Adventure'}, {'id': 14, '... |
| 15602 | Grumpier Old Men            |	[{'id': 10749, 'name': 'Romance'}, {'id': 35, ... |
| 31357 | Waiting to Exhale           |	[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'nam... |
| 11862 | Father of the Bride Part II |	[{'id': 35, 'name': 'Comedy'}]                    |

Table `rating`
| userId | movieId | rating |
|:------:|:-------:|:------:|
| 1	     | 31      | 2.5    |
|	1	     | 1029	   | 3.0    |
|	1	     | 1061	   | 3.0    |
|	1	     | 1129	   | 2.0    |
|	1	     | 1172	   | 4.0    |

Setelah hal tersebut dilakukan, maka terlebih dahulu kita cek apakah terdapat data yang null. Terlihat bahwa kolom yang akan digunakan tidak memiliki data null sehingga tidak perlu dilakukan tindakan lebih lanjut.

Kemudian, kita akan memperbaiki data yang terdapat pada kolom `genres` dalam dataset `movie`.
Jika dilihat, dataset `genres` memiliki format JSON. 
Format JSON tersebut juga menggunakan tanda petik satu (`'`) sehingga perlu dilakukan pengolahan data dengan melakukan konversi dari tanda petik satu (`'`) menjadi petik dua (`"`).
Hal ini dilakukan agar ketika melakukan parsing JSON, maka data dalam kolom `genres` dapat dibaca dan nama dari genres dapat dikeluarkan dan digabungkan menjadi satu. Hasilnya akan menjadi seperti pada tabel di bawah ini.

Table `movieId`
|	 id	  |         title	              | genres                     |
|:------|:----------------------------|:---------------------------|
| 862	  | Toy Story	                  | Animation, Comedy, Family  |
| 8844  | Jumanji	                    | Adventure, Fantasy, Family |
| 15602 | Grumpier Old Men            |	Romance, Comedy            |
| 31357 | Waiting to Exhale           |	Comedy, Drama, Romance     |
| 11862 | Father of the Bride Part II |	Comedy                     |

Selanjutnya, keunikan dalam dataset `movie` akan diuji. Setelah ditelusuri dengan menghitung jumlah nilai unik dari kolom `id`, ternyata ada beberapa data yang terduplikasi.
Untuk itu, maka data yang diduplikasi dihilangkan dengan menggunakan fungsi `drop_duplicates()` dari library Pandas.
Setelah proses ini, dataset berkurang 30 dari jumlah 45.466 dataset menjadi 45.436 dataset.

Agar data yang ada dalam model lebih baik, maka dalam dataset diterapkan sebuah peraturan di mana data `id` yang ada dalam dataset `movie` juga ada dalam dataset `rating` pada kolom `movieId`.
Oleh karena itu, pertama-tama, filtrasi `id` dalam dataset `movie` dilakukan dengan mencocokan dengan data `movieId` yang ada dalam dataset `rating`. 
Hal ini dilakukan karena dataset `movie` memiliki ukuran data yang cukup besar sehingga dengan melakukan filtrasi pada dataset `movie` terlebih dahulu, maka filtrasi `movieId` dalam dataset `rating` akan menjadi lebih mudah.
Setelah itu, filtrasi `movieId` dalam dataset `rating` dilakukan dengan mencocokan dengan data dalam dataset `movie`.
Hasil yang didapatkan setelah melakukan filtrasi adalah 2.830 dataset `movie` dan 44.989 dataset `rating`.

Terakhir, model akan berjalan dengan baik apabila data yang akan diolah model merupakan nilai numerik. 
Oleh karena itu, melakukan *mapping* `userId` dan `movieId` dalam dataset `rating` perlu dilakukan.
Untuk itu, `userId` dan `movieId` dilakukan *mapping* dengan menggunakan sistem *increment*.
Kemudian, label `rating` akan dilakukan normalisasi dengan mengubah nilai rating yang semula memiliki rentang 0.5 hingga 5.0 menjadi rentang 0 hingga 1.

Setelah itu, data siap untuk dilakukan pemodelan.

## Modeling

Data yang telah diolah pada tahap *Data Preparation* akan dibagi menjadi dua bagian, yaitu *train data* dan *test data*. 
Proporsi *train* dan *test* adalah 4:1.

Kemudian, layer dirancang untuk membangun sebuah model. Layer menggunakan model Keras yang terdiri dari Layer Embedding dari `userId` dan `movieId`.
Kemudian, dari input data, akan dihitung *sum product* dari vektor `user` dan `movie` yang dipanggil dari Layer Embedding dengan fungsi `tf.tensordot()`, kemudian dijumlahkan dengan bias dari kedua vektor tersebut.
Hasil tersebut kemudian akan dimasukkan ke layer Sigmoid yang merupakan keluaran dari layer.

Setelah dirancang, model diinisiasikan dan dilakukan *compiling* dengan `BinaryCrossentropy()` sebagai *Loss Function*, `Adam` sebagai *Optimizer Function* dengan `learning_rate` sebesar 0.001, 
serta menggunakan metrik `RMS Error` untuk menguji akurasi dari model yang akan dibuat. 
Kemudian model dilakukan proses training dengan data yang telah diolah dan parameter konfigurasinya adalah `batch_size` sebesar 32 dan `epochs` sebanyak 100.

Hasil pengujian model terhadap sebuah user dapat dilihat pada skenario di bawah ini.
```
Showing recommendations for users: 580
================================
Highest Movie Rating from User:
--------------------------------
Sleepless in Seattle : Comedy, Drama, Romance
High Noon : Western
Wish You Were Here : Comedy, Drama, Foreign, Romance
License to Wed : Comedy
Confession of a Child of the Century : Drama
--------------------------------
Top 10 Movie Recommendation
--------------------------------
The Celebration : Drama
The Sugarland Express : Crime, Drama
Gentlemen Prefer Blondes : Comedy, Romance
Wuthering Heights : Drama, Romance
The Enforcer : Drama, Action, Crime
Before Sunset : Drama, Romance
Tokyo! : Romance, Drama
The Hunter : Drama, Thriller
Deadlier Than the Male : Action, Comedy, Thriller
Long Pigs : Crime, Horror
```

## Evaluation

Metrik yang akan digunakan merupakan `RMS Error` atau *Root Mean Squared Error*. Metrik ini menjelaskan selisih jarak antara hasil prediksi dengan nilai sebenarnya [[2]](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e).
Rumus dari RMSE ini dapat dilihat di bawah ini.

<img width="564" alt="image" src="https://user-images.githubusercontent.com/56476347/186470142-0a577f95-fe41-4c42-a101-09654aea424d.png">

Hasil training epoch ke-100 menunjukkan data seperti berikut ini.
```
Epoch 100/100
1125/1125 [==============================] - 6s 5ms/step - loss: 0.5841 - root_mean_squared_error: 0.1894 - val_loss: 0.6320 - val_root_mean_squared_error: 0.2412
```

Dari metrik tersebut, dapat terlihat bahwa loss yang didapatkan sebesar 0.5841 dengan RMSE sebesar 0.1894 pada data training.
Loss sebesar 0.6320 dengan RMSE sebesar 0.2412 didapatkan pada data testing. 
Hal ini menandakan bahwa perbedaan tidak jauh, namun terdapat tanda-tanda bahwa model mengalami sedikit *overfitting*.
Untuk visualisasi mengenai hasil pemodelan ini dapat dilihat pada gambar di bawah ini.

![image](https://user-images.githubusercontent.com/56476347/186469485-b47045aa-ddc9-472c-932d-b9a79b9aa4df.png)

Sekarang, saatnya melakukan evaluasi terhadap hasil output dari pengujian model yang outputnya sudah dilampirkan dalam bagian **Modelling**.

```
Sleepless in Seattle : Comedy, Drama, Romance
High Noon : Western
Wish You Were Here : Comedy, Drama, Foreign, Romance
License to Wed : Comedy
Confession of a Child of the Century : Drama
```

Skenario ini menjelaskan bahwa user tersebut menyukai film *Comedy*, *Drama*, dan *Romance*, ditandai dengan rating yang tinggi pada film dengan genre tersebut. 
Oleh karena itu, model seharusnya merekomendasi film *Drama*, *Comedy*, dan *Romance*.
Hasil yang dikeluarkan oleh model sebagai prediksi film yang mungkin tertarik oleh user tersebut dapat dilihat pada list di bawah ini.

- The Celebration : **Drama**
- The Sugarland Express : Crime, **Drama**
- Gentlemen Prefer Blondes : **Comedy**, **Romance**
- Wuthering Heights : **Drama**, **Romance**
- The Enforcer : **Drama**, Action, Crime
- Before Sunset : **Drama**, Romance
- Tokyo! : **Romance**, **Drama**
- The Hunter : **Drama**, Thriller
- Deadlier Than the Male : Action, **Comedy**, Thriller
- Long Pigs : Crime, Horror

Dapat terlihat bahwa model merekomendasi film *Drama*, *Comedy*, dan *Romance* serta genre-genre lain yang mungkin user tersebut tertarik dan ingin menontonnya.
Kebanyakan film tersebut memiliki genre yang sesuai sehingga dapat dikatakan bahwa sistem rekomendasi ini dapat berjalan dengan baik.

## Kesimpulan

Dari hasil yang telah diuji, dapat ditarik kesimpulan bahwa dengan menggunakan *Collaborative Filtering*, dapat menggunakan _user_ dan _movie_ sebagai input, dan _rating_ sebagai output yang akan diprediksi untuk melakukan prediksi berbagai film yang belum pernah ditonton oleh user. Hasil prediksi tersebut kemudian diurutkan berdasarkan prediksi rating tertinggi untuk ditampilkan sebagai film rekomendasi yang berpotensi dapat disukai oleh user tersebut.

**Referensi:**

[1] Y. Kniazieva, “What Is a Movie Recommendation System in ML?,” LabelYourData, Apr. 12, 2022. https://labelyourdata.com/articles/movie-recommendation-with-machine-learning (accessed Aug. 25, 2022).

[2] J. Moody, “What does RMSE really mean?,” Towards Data Science, Sep. 06, 2019. https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e (accessed Aug. 25, 2022).
