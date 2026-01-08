# =======================
# prediksi_komentar.py
# NB PIPELINE - INFERENCE
# =======================

import streamlit as st
import joblib
import os

# =======================
# LOAD MODEL PIPELINE
# =======================
@st.cache_resource(show_spinner=True)
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_nb.pkl")

    model = joblib.load(model_path)
    return model

model = load_model()

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

        # Pipeline: TF-IDF + Naive Bayes
        prediksi = model.predict([user_text])[0]
        confidence = model.predict_proba([user_text]).max() * 100

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
