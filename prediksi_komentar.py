# =======================
# prediksi_komentar.py
# Train-on-Load Naïve Bayes
# =======================

import streamlit as st
import pandas as pd
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# =======================
# NLTK SETUP (Cloud-safe)
# =======================
nltk.download("punkt")
nltk.download("stopwords")

# =======================
# PREPROCESSING
# =======================
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words("indonesian"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# =======================
# LOAD & TRAIN MODEL
# =======================
@st.cache_resource(show_spinner=True)
def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "labeled_tweets.xlsx")

    df = pd.read_excel(data_path)

    # WAJIB: sesuaikan dengan kolom dataset Anda
    # contoh umum:
    # kolom teks  : 'text' / 'tweet' / 'clean_text'
    # kolom label : 'sentiment'

    df["clean_text"] = df["text"].apply(preprocess_text)

    X = df["clean_text"]
    y = df["sentiment"]

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

        clean_text = preprocess_text(user_text)
        text_tfidf = vectorizer.transform([clean_text])

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
