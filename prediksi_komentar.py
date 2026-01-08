# ============================
# prediksi_komentar.py
# INFERENCE NAÏVE BAYES
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
# PREDICTION FUNCTION
# ============================
def predict_sentiment(text: str):
    """
    Memprediksi sentimen komentar Twitter Mobile Legends
    Output:
        label (str)
        confidence (float)
    """

    if text is None or text.strip() == "":
        return "Tidak valid", 0.0

    # Prediksi
    label = model.predict([text])[0]

    # Confidence
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba([text]).max()
    else:
        confidence = 0.0

    return label, confidence
