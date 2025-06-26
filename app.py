import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

st.title("Prediksi Risiko Kanker Serviks")

# === Load dataset langsung dari file lokal ===
df = pd.read_csv("risk_factors_cervical_cancer.csv")

st.subheader("Data Awal")
st.dataframe(df.head())

# === Preprocessing ===
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

if 'Age' in df_imputed.columns:
    binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    df_imputed['Age_binned'] = binner.fit_transform(df_imputed[['Age']])

if 'Biopsy' in df_imputed.columns:
    X = df_imputed.drop('Biopsy', axis=1)
    y = df_imputed['Biopsy']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # === Training dan Simpan Model ===
    model_path = "decision_tree_model.pkl"
    if not os.path.exists(model_path):
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Evaluasi Model Decision Tree")
    st.write(f"Akurasi: {acc:.4f}")
    st.text("Confusion Matrix:")
    st.write(cm)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
    except:
        st.warning("ROC-AUC tidak bisa dihitung.")

    st.success("Model siap digunakan untuk prediksi.")
else:
    st.error("Kolom 'Biopsy' tidak ditemukan dalam dataset.")
