# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Prediksi Kanker Serviks", layout="centered")
st.title("🧠 Prediksi Risiko Kanker Serviks (Biopsy)")
st.markdown("Model ini dilatih dari data asli dan dapat memprediksi hasil Biopsy berdasarkan gejala pasien.")

# === 1. Load dan Preprocessing CSV asli ===
@st.cache_data
def load_and_train_model():
    # Load data
    df = pd.read_csv("risk_factors_cervical_cancer.csv")

    # Replace '?' dan ubah ke float
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Imputasi missing values
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Fitur dan target
    X = df_imputed.drop("Biopsy", axis=1)
    y = df_imputed["Biopsy"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dan latih model Decision Tree
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns, scaler

model, feature_names, scaler = load_and_train_model()

# === 2. Form Input User ===
st.subheader("📋 Masukkan Data Pasien")

user_input = {}
for feature in feature_names:
    val = st.number_input(f"{feature}", min_value=0.0, step=0.1, value=0.0)
    user_input[feature] = val

input_df = pd.DataFrame([user_input])
st.write("Data Input Anda:")
st.dataframe(input_df)

# === 3. Preprocess input dan prediksi ===
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
pred_label = "✅ Tidak Terindikasi Kanker Serviks" if prediction == 0 else "⚠️ Terindikasi Kanker Serviks"

st.subheader("📌 Hasil Prediksi:")
st.success(pred_label)
#.