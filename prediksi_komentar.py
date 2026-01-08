# =======================
# prediksi_komentar.py
# (Naïve Bayes Version)
# =======================

import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# =======================
# LOAD MODEL & VECTORIZER
# =======================
@st.cache_resource(show_spinner=True)
def load_model():
    with open("model_nb.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

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

        # Preprocessing
        clean_text = preprocess_text(user_text)
        text_tfidf = vectorizer.transform([clean_text])

        # Prediksi
        prediksi = model.predict(text_tfidf)[0]
        confidence = model.predict_proba(text_tfidf).max() * 100

        # Output
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

