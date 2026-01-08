# =======================
# prediksi_komentar.py
# FINAL - SESUAI DATASET
# =======================

import streamlit as st
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# =======================
# TRAIN MODEL LANGSUNG
# =======================
@st.cache_resource(show_spinner=True)
def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "mobilelegends_labelled (1).xlsx")

    df = pd.read_excel(data_path)

    # Gunakan kolom YANG ADA
    X = df["stemmed_text"]
    y = df["sentiment_label"]

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_tfidf, y)

    return model, vectorizer

model, vectorizer = train_model()

# =======================
# MAIN FUNCTION
# =======================
def main():
    st.subheader("✍️ Input Komentar")

    user_text = st.text_area(
        "Masukkan komentar Twitter tentang Mobile Legends:",
        height=120
    )

    if st.button("🔍 Prediksi Sentimen"):
        if user_text.strip() == "":
            st.warning("⚠️ Komentar tidak boleh kosong.")
            return

        # Karena model dilatih dari stemmed_text,
        # input dipakai langsung (konsisten metodologi)
        text_tfidf = vectorizer.transform([user_text.lower()])

        prediksi = model.predict(text_tfidf)[0]
        confidence = model.predict_proba(text_tfidf).max() * 100

        st.subheader("📊 Hasil Prediksi")
        st.write(f"**Komentar:** {user_text}")
        st.write(f"**Sentimen:** {prediksi}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        if prediksi.lower() == "positif":
            st.success("Komentar bernada **positif** 🎉")
        elif prediksi.lower() == "netral":
            st.info("Komentar bersifat **netral** 😐")
        else:
            st.error("Komentar bernada **negatif** 😠")
