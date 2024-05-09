# LIBRARY / MODULE / PUSTAKA

import streamlit as st

import os, itertools

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix

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
# ------------------------------------------------------------------------------