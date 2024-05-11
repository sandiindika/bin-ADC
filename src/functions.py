# LIBRARY / MODULE / PUSTAKA

import streamlit as st

import os, pickle

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from warnings import simplefilter

simplefilter(action= "ignore", category= FutureWarning)

# DEFAULT FUNCTIONS

"""Make Space

Fungsi-fungsi untuk membuat jarak pada webpage menggunakan margin space dengan
ukuran yang bervariatif.
"""

def ms_20():
    st.markdown("<div class= \"ms-20\"></div>", unsafe_allow_html= True)

def ms_40():
    st.markdown("<div class= \"ms-40\"></div>", unsafe_allow_html= True)

def ms_60():
    st.markdown("<div class= \"ms-60\"></div>", unsafe_allow_html= True)

def ms_80():
    st.markdown("<div class= \"ms-80\"></div>", unsafe_allow_html= True)

"""Make Layout

Fungsi-fungsi untuk layouting webpage menggunakan fungsi columns() dari
Streamlit.

Returns
-------
self : object containers
    Mengembalikan layout container.
"""

def ml_center():
    left, center, right = st.columns([.3, 2.5, .3])
    return center

def ml_split():
    left, center, right = st.columns([1, .1, 1])
    return left, right

def ml_left():
    left, center, right = st.columns([2, .1, 1])
    return left, right

def ml_right():
    left, center, right = st.columns([1, .1, 2])
    return left, right

"""Cetak text

Fungsi-fungsi untuk menampilkan teks dengan berbagai gaya menggunakan method
dari Streamlit seperti title(), write(), dan caption().

Parameters
----------
text : str
    Teks yang ingin ditampilkan dalam halaman.

size : int
    Ukuran Heading untuk teks yang akan ditampilkan.

division : bool
    Kondisi yang menyatakan penambahan garis divisi teks ditampilkan.
"""

def show_title(text, division= False):
    st.title(text)
    if division:
        st.markdown("---")

def show_text(text, size= 3, division= False):
    heading = "#" if size == 1 else (
        "##" if size == 2 else (
            "###" if size == 3 else (
                "####" if size == 4 else "#####"
            )
        )
    )

    st.write(f"{heading} {text}")
    if division:
        st.markdown("---")

def show_caption(text, size= 3, division= False):
    heading = "#" if size == 1 else (
        "##" if size == 2 else (
            "###" if size == 3 else (
                "####" if size == 4 else "#####"
            )
        )
    )

    st.caption(f"{heading} {text}")
    if division:
        st.markdown("---")

def show_paragraf(text):
    st.markdown(f"<div class= \"paragraph\">{text}</div>",
                unsafe_allow_html= True)

"""Load file

Fungsi-fungsi untuk membaca file dalam lokal direktori.

Parameters
----------
filepath : str
    Jalur tempat data tersedia di lokal direktori.

Returns
-------
self : object
    Obyek dengan informasi yang berhasil didapatkan.
"""

def get_csv(filepath):
    return pd.read_csv(filepath)

def get_excel(filepath):
    return pd.read_excel(filepath)

def get_img(filepath):
    return Image.open(filepath)

def get_files(dirpath):
    filepaths, filenames, labels = [], [], []
    err = False
    for folder_label in os.listdir(dirpath):
        folder_path = os.path.join(dirpath, folder_label)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                filepaths.append(file_path)
                filenames.append(file_name)
                labels.append(folder_label)
        else:
            err = True
    if err:
        st.code(
            """Struktur data tidak sesuai ekspektasi.

main-directory
|- label-1
|  |- file-1 -> n
|
|- label-2
|  |- file-1 -> n
            """
        )
    df = pd.DataFrame({
        "filepaths": filepaths,
        "filenames": filenames,
        "labels": labels
    })
    return df

def mk_dir(dirpath):
    """Buat folder
    
    Fungsi ini akan memeriksa path folder yang diberikan. Jika tidak ada
    folder sesuai path yang dimaksud, maka folder akan dibuat.

    Parameters
    ----------
    dirpath : str
        Jalur tempat folder akan dibuat.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# CUSTOM FUNCTIONS

@st.cache_data(ttl= 3600, show_spinner= "Fetching data...")
def feature_extraction(df, duration= 30):
    """Ekstraksi Fitur

    Identifikasi dan pemilihan informasi penting dari kumpulan data.

    Parameters
    ----------
    df : object DataFrame
        Object DataFrame tempat semua file musik (path file) tersimpan.

    duration : int or float
        Durasi musik yang ingin di ekstraksi fiturnya.

    Returns
    -------
    res : object DataFrame
        DataFrame dari data musik dengan fitur yang telah di ekstraksi dan label
        genre musik.
    """
    rms_list, mfcc_list, zcr_list, s_c_list, s_r_list = [], [], [], [], []
    for _dir in df.iloc[:, 0]:
        y, sr = librosa.load(_dir, duration= duration)

        mfcc = librosa.feature.mfcc(y= y, sr= sr, n_mfcc= 13)
        zcr = librosa.feature.zero_crossing_rate(y)
        s_c = librosa.feature.spectral_centroid(y= y, sr= sr)
        s_r = librosa.feature.spectral_rolloff(y= y, sr= sr)
        
        rms_features = np.sqrt(np.mean(y ** 2))
        mfcc_features = np.mean(mfcc, axis= 1)
        zcr_features = np.mean(zcr)
        s_c_features = np.mean(s_c)
        s_r_features = np.mean(s_r)

        rms_list.append(rms_features)
        mfcc_list.append(mfcc_features)
        zcr_list.append(zcr_features)
        s_c_list.append(s_c_features)
        s_r_list.append(s_r_features)

    res = pd.DataFrame({
        "filename": df.iloc[:, 1],
        "rms": rms_list,
        **{f"mfcc_{i + 1}": [x[i] for x in mfcc_list] for i in range(13)},
        "zcr": zcr_list,
        "spectral_centroid": s_c_list,
        "spectral_rolloff": s_r_list,
        "genre": df.iloc[:, -1]
    })
    return res

def min_max_scaler(feature_names, df):
    """Transformasikan fitur dengan menskalakan setiap fitur ke rentang tertentu

    Estimator ini menskalakan dan menerjemahkan setiap fitur satu per satu
    sehingga berada dalam rentang tertentu pada set pelatihan, misalnya antara
    nol dan satu.
    
    Transformasi dilakukan dengan::
    
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    dimana min, max = feature_range.

    Transformasi ini sering digunakan sebagai alternatif terhadap mean nol,
    penskalaan unit varians.

    Parameters
    ----------
    feature_names : ndarray of shape
        Nama fitur dari kumpulan data (DataFrame). Didefinisikan hanya ketika
        `X` memiliki nama fitur yang semuanya berupa string.

    df : object DataFrame
        Object DataFrame yang menyimpan fitur musik beserta label genrenya.
    
    Returns
    -------
    self : object DataFrame
        DataFrame dari data musik dengan fitur yang telah di skalakan. 
    """
    for col in feature_names: # loop untuk setiap fitur dalam `X`
        min_ = df[col].min()
        max_ = df[col].max()
        df[col] = (df[col] - min_) / (max_ - min_)
    return df

def euclidean_distance(point1, point2):
    """Euclidean Distance

    Fungsi untuk menghitung matrix menggunakan rumus Euclidean

    Parameters
    ----------
    point1 : str or float
        Nilai untuk point pertama.

    point2 : str or float
        Nilai untuk point kedua.

    Returns
    -------
    self : float
        Nilai hasil perhitungan menggunakan rumus euclidean distance.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

def weighted_knn(X_train, y_train, X_test, neighbor):
    """Menerapkan algoritma k-Nearest Neighbors dengan bobot jarak

    Estimator ini mengklasifikasikan sampel uji dengan menghitung jarak antara
    setiap sampel uji dan sampel pelatihan, lalu memberi bobot pada kelas
    berdasarkan jaraknya. Semakin dekat suatu sampel uji dengan sampel pelatihan
    semakin besar bobotnya. Kelas dari sampel uji ditentukan dengan
    memperhitungkan bobot kelas dari tetangga terdekat.

    Parameters
    ----------
    X_train : ndarray
        Array numpy dari sampel pelatihan.

    y_train : ndarray
        Array numpy dari label kelas untuk sampel pelatihan.

    X_test : ndarray
        Array numpy dari sampel uji.

    neighbor : int
        Jumlah tetangga terdekat yang akan dipertimbangkan dalam klasifikasi.

    Returns
    -------
    y_pred : Array numpy dari label kelas yang diprediksi untuk sampel uji.
    """
    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    y_pred = np.zeros(num_test)

    for i in range(num_test):
        distance = np.zeros(num_train)
        for j in range(num_train):
            distance[j] = euclidean_distance(X_test[i], X_train[j])
        nearest_indices = np.argsort(distance)[:neighbor]
        weights = 1 / (distance[nearest_indices] + 1e-10)

        class_weights = {}
        for index, label in enumerate(y_train[nearest_indices]):
            class_weights[label] = class_weights.get(label, 0) + weights[index]
        y_pred[i] = max(class_weights, key= class_weights.get)
    return y_pred

@st.cache_data(ttl= 3600, show_spinner= "Fetching data...")
def classify(features, labels, n_fold, neighbor= 5):
    """ Menerapkan algoritma klasifikasi menggunakan metode K-Fold Cross
    Validation dengan k-Nearest Neighbors dengan bobot jarak

    Fungsi ini membagi data menjadi sejumlah lipatan (folds) dan melakukan
    klasifikasi pada setiap lipatan menggunakan algoritma k-NN dengan bobot
    jarak. Setelah itu, fungsi menghitung dan mengembalikkan rata-rata akurasi
    dari seluruh lipatan.

    Parameters
    ----------
    features : object DataFrame
        DataFrame yang berisi fitur-fitur dari data.

    labels : object DataFrame
        DataFrame yang berisi label kelas untuk setiap sampel.

    n_fold : int
        Jumlah lipatan (folds) dalam K-Fold Cross Validation.

    neighbor : int, default= 5
        Jumlah tetangga terdekat yang akan dipertimbangkan dalam klasifikasi.

    Returns
    -------
    scores : list
        Daftar score dari setiap lipatan (folds).

    avg_score : float
        Rata-rata akurasi dari klasifikasi pada setiap lipatan.
    """
    kfold = KFold(n_splits= n_fold, shuffle= True, random_state= 42)
    encoder = LabelEncoder()
    features = features.values
    labels = labels.values

    list_test, list_predict = [], []
    scores = []
    avg_score = 0

    for tr_index, ts_index in kfold.split(features):
        X_train, X_test = features[tr_index], features[ts_index]
        y_train, y_test = labels[tr_index], labels[ts_index]

        y_train_enc = encoder.fit_transform(y_train)
        
        y_pred = weighted_knn(X_train, y_train_enc, X_test, neighbor)
        y_pred = [int(x) for x in y_pred]
        y_pred = encoder.inverse_transform(y_pred)

        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        avg_score += score

        list_test.append(y_test)
        list_predict.append(y_pred)

    mk_dir("./data/pickle")
    with open("./data/pickle/actual_labels.pickle", "wb") as f:
        pickle.dump(list_test, f)
    with open("./data/pickle/predict_labels.pickle", "wb") as f:
        pickle.dump(list_predict, f)
    return scores, avg_score / n_fold

def plot_scores(scores):
    """ Memplot skor akurasi untuk setiap lipatan (fold)

    Fungsi ini menghasilkan plot skor akurasi untuk setiap lipatan dalam K-Fold
    Cross Validation. Plot menunjukkan bagaimana akurasi berubah pada setiap
    lipatan dan membantu dalam mengidentifikasi variasi kinerja model pada data
    yang berbeda.

    Parameters
    ----------
    scores : list
        List yang berisi skor akurasi untuk setiap lipatan dalam K-Fold.
    """
    fig = plt.figure(figsize= (10, 5))
    plt.plot(range(1, len(scores) + 1), scores, marker= "o")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy Score")
    plt.title("Accuracy Score across Folds")
    plt.grid(True)
    st.pyplot(fig)