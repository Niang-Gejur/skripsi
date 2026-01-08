# ===============================
# app.py — Streamlit Main App
# ===============================
import sys, os
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import prediksi_komentar
import streamlit_sentiment_app

st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("📂 Navigasi Utama")
    menu = st.radio(
        "Pilih Halaman:",
        ("🏠 Beranda", "🔍 Prediksi Komentar", "📈 Analisis Sentimen")
    )

if menu == "🏠 Beranda":
    st.title("💬 Aplikasi Analisis Sentimen Mobile Legends")

elif menu == "🔍 Prediksi Komentar":
    prediksi_komentar.main()

elif menu == "📈 Analisis Sentimen":
    streamlit_sentiment_app.main()
