import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# === Load models and scalers ===
depression_model = tf.keras.models.load_model("cnn_lstm_depression_model.h5")
severity_model = tf.keras.models.load_model("severity_model_cnn.h5")
scaler_depression = joblib.load("scaler.pkl")
scaler_severity = joblib.load("scaler_severity.pkl")
label_encoder_severity = joblib.load("label_encoder_severity.pkl")

# === Page config ===
st.set_page_config(page_title="🧠 EEG Mental Health App", page_icon="🧠", layout="centered")

# === CSS Styling ===
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #0f1117;
        color: #ffffff;
    }
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] {
        display: flex; justify-content: center; margin-bottom: 20px;
    }
    .stTabs [role="tab"] {
        font-weight: bold; padding: 10px 30px;
        background-color: #1e1e1e; border-radius: 10px 10px 0 0;
        color: #ccc; margin-right: 5px;
        border: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6c63ff;
        color: white !important;
        border-color: #6c63ff;
    }
    .stButton>button {
        background-color: #6c63ff; color: white; border-radius: 8px;
        padding: 10px 20px; font-weight: 600;
    }
    .stDownloadButton>button {
        background-color: #2a9d8f; color: white; border-radius: 8px;
        padding: 8px 20px; font-weight: 600;
    }
    .uploadbox {
        background-color: #1a1c23; padding: 20px;
        border-radius: 10px; margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# === App Logo ===
st.image("logo.png", width=100)  # Ensure logo.png is in the same folder
st.title("🧠 EEG Mental Health Prediction")
st.markdown("Upload your EEG data (**1024 features**) to predict either **Depression** or **Severity** using trained ML/DL models.")

# === Synthetic CSV Generator Sidebar ===
with st.sidebar:
    st.header("🧪 Generate EEG Data")
    if st.button("📄 Create Sample EEG CSV"):
        sample = pd.DataFrame(np.random.randn(1, 1024))
        st.success("✅ EEG sample generated.")
        st.download_button("⬇️ Download Sample CSV", data=sample.to_csv(index=False).encode("utf-8"),
                           file_name="sample_eeg.csv", mime="text/csv")

# === Tabs for Depression and Severity ===
tab1, tab2 = st.tabs(["💭 Depression Prediction", "📊 Severity Prediction"])

# ------------------ Depression Tab ------------------
with tab1:
    st.header("💭 Depression Prediction")
    uploaded_file_dep = st.file_uploader("📤 Upload CSV for Depression (1024 features)", type=["csv"], key="dep")

    if uploaded_file_dep:
        try:
            df_dep = pd.read_csv(uploaded_file_dep, header=None)
            if df_dep.shape[1] != 1024:
                st.error("❌ The file must have 1024 columns.")
            else:
                X_scaled = scaler_depression.transform(df_dep)
                X_reshaped = X_scaled.reshape(-1, 1024, 1)
                y_pred = depression_model.predict(X_reshaped)
                predicted = (y_pred > 0.5).astype(int).flatten()

                result_df = df_dep.copy()
                result_df["Predicted Depression"] = predicted

                st.success("✅ Prediction complete!")
                st.dataframe(result_df)

                # 📊 Prediction Chart
                st.subheader("📊 Prediction Summary")
                fig, ax = plt.subplots()
                labels = ["No Depression", "Depression"]
                counts = pd.Series(predicted).value_counts().reindex([0, 1], fill_value=0)
                ax.bar(labels, counts, color=["#4daf4a", "#e41a1c"])
                ax.set_ylabel("Count")
                st.pyplot(fig)

                # ⬇️ Download
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Depression Predictions", csv, "depression_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ------------------ Severity Tab ------------------
with tab2:
    st.header("📊 Severity Prediction")
    uploaded_file_sev = st.file_uploader("📤 Upload CSV for Severity (1024 features)", type=["csv"], key="sev")

    if uploaded_file_sev:
        try:
            df_sev = pd.read_csv(uploaded_file_sev, header=None)
            if df_sev.shape[1] != 1024:
                st.error("❌ The file must have 1024 columns.")
            else:
                X_scaled = scaler_severity.transform(df_sev)
                X_reshaped = X_scaled.reshape(-1, 1024, 1)
                y_pred = severity_model.predict(X_reshaped)

                predicted_classes = np.argmax(y_pred, axis=1)
                predicted_labels = label_encoder_severity.inverse_transform(predicted_classes)

                result_df = df_sev.copy()
                result_df["Predicted Severity"] = predicted_labels

                st.success("✅ Severity prediction complete!")
                st.dataframe(result_df)

                # 📊 Severity Distribution
                st.subheader("📊 Severity Class Distribution")
                fig, ax = plt.subplots()
                pd.Series(predicted_labels).value_counts().plot(kind='bar', color='#ffa600', ax=ax)
                ax.set_ylabel("Count")
                ax.set_xlabel("Severity Level")
                ax.set_title("Predicted Severity Distribution")
                st.pyplot(fig)

                # ⬇️ Download
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Severity Predictions", csv, "severity_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
