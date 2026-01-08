# ============================
# prediksi_komentar.py
# NAÏVE BAYES - INFERENCE
# ============================

import pickle
import os
import streamlit as st


# ============================
# LOAD MODEL (CACHED)
# ============================
@st.cache_resource(show_spinner=False)
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_nb.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("model_nb.pkl tidak ditemukan")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


model = load_model()


# ============================
# LOGIC FUNCTION (DIPANGGIL app.py)
# ============================
def predict_sentiment(text: str):
    if text is None or text.strip() == "":
        return "Tidak valid", 0.0

    label = model.predict([text])[0]

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba([text]).max()
    else:
        confidence = 0.0

    return label, confidence


# ============================
# UI FUNCTION (DIPANGGIL app.py)
# ============================
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

        label, confidence = predict_sentiment(user_text)

        st.subheader("📊 Hasil Prediksi")
        st.write(f"**Komentar:** {user_text}")
        st.write(f"**Sentimen:** {label}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        if str(label).lower() == "positif":
            st.success("Komentar bernada **positif** 🎉")
        elif str(label).lower() == "netral":
            st.info("Komentar bersifat **netral** 😐")
        else:
            st.error("Komentar bernada **negatif** 😠")
