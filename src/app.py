# LIBRARY / MODULE / PUSTAKA

import streamlit as st
from streamlit import session_state as ss
from streamlit_option_menu import option_menu

from functions import *
from warnings import simplefilter

simplefilter(action= "ignore", category= FutureWarning)

# PAGE CONFIG

st.set_page_config(
    page_title= "App", layout= "wide", initial_sidebar_state= "expanded",
    page_icon= get_img("./assets/favicon.ico")
)

# hide menu, header, and footer
st.markdown(
    """<style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .st-emotion-cache-1jicfl2 {padding-top: 2rem;}
    </style>""",
    unsafe_allow_html= True
)

# CSS on style.css
with open("./css/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html= True)

class MyApp():
    """Class dari MyApp
    
    Parameters
    ----------
    message : bool, default= False
        Jika False, maka pesan error tidak akan ditampilkan dalam Webpage
        Sistem. Jika True, maka akan menampilkan pesan dalam Webpage Sistem.

    Attributes
    ----------
    message : bool
        Tampilkan pesan error pada Webpage Sistem atau sebaliknya.

    pathdata : str
        Path data yang disimpan dalam lokal direktori.

    menu_ : list
        Daftar menu yang akan ditampilkan dalam Webpage Sistem.

    icons_ : list
        Daftar icon menu untuk setiap menu yang ditampilkan.
    """

    def __init__(self, message= False):
        self.message = message
        self.pathdata = "./data/music"
        self.menu_ = ["Beranda", "Data Musik", "Ekstraksi Fitur", "Klasifikasi",
                      "Evaluasi"]
        self.icons_ = ["house", "music-note-beamed", "soundwave",
                       "bar-chart", "clipboard-data"]
     
    def _navigation(self):
        """Navigasi sistem / Sidebar
        
        Returns
        -------
        selected : str
            Selected menu.
        """
        with st.sidebar:
            selected = option_menu(
                menu_title= "", options= self.menu_, icons= self.icons_,
                styles= {
                    "container": {"padding": "0 !important",
                                  "background-color": "#E6E6EA"},
                    "icon": {"color": "#020122", "font-size": "18px"},
                    "nav-link": {"font-size": "16px", "text-align": "left",
                                 "margin": "0px", "color": "#020122"},
                    "nav-link-selected": {"background-color": "#F4F4F8"}
                }
            )

            ms_60()
            show_caption("Copyright © 2024 | ~", size= 5)
        return selected
    
    def _exceptionMessage(self, e):
        """Tampilkan pesan galat
        
        Parameters
        ----------
        e : exception object
            Obyek exception yang tersimpan.
        """
        ms_20()
        with ml_center():
            st.error("Terjadi masalah...")
            if self.message:
                st.exception(e)

    def _pageBeranda(self):
        """Tab beranda
        
        Halaman ini akan menampilkan judul sistem dan abstra dari proyek.
        """
        try:
            ms_20()
            show_text("Klasifikasi Musik Berdasarkan Genre Menggunakan Metode \
                      Weighted k-Nearest Neighbor", size= 2, division= True)
            
            ms_40()
            with ml_center():
                with open("./assets/abstract.txt", "r") as f:
                    abstract = f.read()
                show_paragraf(abstract)
        except Exception as e:
            self._exceptionMessage(e)

    def _pageDataMusik(self):
        """Halaman data musik

        Bagian ini akan menampilkan DataFrame yang berisi daftar musik untuk
        diolah.
        """
        try:
            ms_20()
            show_text("Data Musik", division= True)

            ms_40()
            with ml_center():
                df = get_files(self.pathdata)
                st.dataframe(df, use_container_width= True, hide_index= True)

                mk_dir("./data/dataframe")
                df.to_csv("./data/dataframe/daftar-musik.csv", index= False)
        except Exception as e:
            self._exceptionMessage(e)

    def _pageEkstraksiFitur(self):
        """Halaman ekstraksi fitur

        Halaman ini akan mengekstrak fitur-fitur data musik dengan membaca
        filepath dari DataFrame list-musik. Number input disediakan untuk
        optimasi pada durasi musik.
        """
        try:
            ms_20()
            show_text("Ekstraksi Fitur", division= True)

            df = get_csv("./data/dataframe/daftar-musik.csv")

            left, right = ml_right()
            with left:
                ms_20()
                duration = st.number_input(
                    "Durasi Musik (detik)", min_value= 1, value= 30, step= 1,
                    key= "Number input untuk nilai durasi musik"
                )

                ms_40()
                btn_extract = st.button(
                    "Submit", key= "Button fit ekstraksi fitur",
                    use_container_width= True
                )

            with right:
                ms_20()
                if btn_extract or ss.fit_extract:
                    ss.fit_extract = True
                    with st.spinner("Feature Extraction is running..."):
                        res = feature_extraction(df, duration)
                    res.to_csv("./data/dataframe/music_features.csv",
                               index= False)
                    st.dataframe(res, use_container_width= True,
                                 hide_index= True)
        except Exception as e:
            self._exceptionMessage(e)

    def _pageKlasifikasi(self):
        """Halaman klasifikasi

        Halaman ini untuk setting dan training model klasifikasi.
        """
        try:
            ms_20()
            show_text("Klasifikasi", division= True)

            df = get_csv("./data/dataframe/music_features.csv")
            features = df.iloc[:, 1:-1]
            labels = df.iloc[:, -1]
            feature_names = features.columns

            norm_features = min_max_scaler(feature_names, features)

            left, right = ml_right()
            with left:
                fold = st.selectbox(
                    "Jumlah subset Fold", [4, 5, 10], index= 1,
                    key= "Selectbox untuk nilai k-Fold"
                )

                neighbor = st.number_input(
                    "Masukkan jumlah tetangga", min_value= 3,
                    max_value= int(len(labels) / 2),
                    key= "Number input k tetangga"
                )

                ms_40()
                btn_classify = st.button(
                    "Submit", use_container_width= True,
                    key= "Button fit klasifikasi"
                )
            with right:
                ms_20()
                if btn_classify or ss.fit_classify:
                    ms_20()
# ------------------------------------------------------------------------------
        except Exception as e:
            self._exceptionMessage(e)

    def _pageEvaluasi(self):
        """Halaman evaluasi"""
        try:
            ms_20()
        except Exception as e:
            self._exceptionMessage(e)

    def main(self):
        """Main program
        
        Setting session page diatur disini dan konfigurasi setiap halaman
        dipanggil disini.
        """
        with st.container():
            selected = self._navigation()

            if "fit_extract" not in ss:
                ss.fit_extract = False
            if "fit_classify" not in ss:
                ss.fit_classify = False

            if selected == self.menu_[0]:
                self._pageBeranda()
            elif selected == self.menu_[1]:
                self._pageDataMusik()
            elif selected == self.menu_[2]:
                self._pageEkstraksiFitur()
            elif selected == self.menu_[3]:
                self._pageKlasifikasi()
            elif selected == self.menu_[4]:
                self._pageEvaluasi()

if __name__ == "__main__":
    app = MyApp(message= True)
    app.main()