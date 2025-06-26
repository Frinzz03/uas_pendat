import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Kanker Serviks", layout="wide")
st.title("🧬 Prediksi Risiko Kanker Serviks")

# Load model, scaler, dan kolom
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Load data mentah untuk ditampilkan
try:
    df_raw = pd.read_csv("risk_factors_cervical_cancer.csv")
    df_raw.replace("?", np.nan, inplace=True)
    st.subheader("📊 Dataset Mentah")
    st.dataframe(df_raw)
except FileNotFoundError:
    st.warning("File dataset tidak ditemukan. Pastikan 'risk_factors_cervical_cancer.csv' tersedia.")

# Form input manual
st.subheader("📝 Input Data Pasien")
with st.form("form_prediksi"):
    input_data = []
    for col in columns:
        val = st.number_input(f"{col}", value=0.0, step=1.0, format="%.2f")
        input_data.append(val)
    submit = st.form_submit_button("🔍 Prediksi")

if submit:
    try:
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("📈 Hasil Prediksi")
        st.success(f"Prediksi: {'Berisiko (Biopsy = 1)' if pred == 1 else 'Tidak Berisiko (Biopsy = 0)'}")
        st.info(f"Probabilitas Risiko: {prob:.2%}")
    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi.")
        st.exception(e)
