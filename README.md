# 🧠 EEG-Based Mental Health Analysis App

This is a Streamlit-based web application for analyzing EEG signals to classify **depression severity** and predict **HAMD scores** using deep learning models.

🚀 Try the App: [Click here to open](https://eeg-mental-health-app-khare.streamlit.app/)

---

## 📁 Dataset

- **Name**: `CIP_signal_all_FT7.mat`
- **Source**: Private research EEG dataset from a mental health study.
- **Content**:
  - EEG signal features recorded from the **FT7** channel.
  - Labels include:
    - `severity`: Multiclass depression severity levels (1–33)
    - `depression`: Binary label (0 = No Depression, 1 = Depression)
    - `HAMD`: Continuous target score for regression tasks (Hamilton Depression Rating Scale)
- **Preprocessing**:
  - Extracted feature matrix and label vectors.
  - Scaled/standardized signals for deep learning input.
  - Mapped `severity` into 3 classes: **Low**, **Mild**, **High**

---

## 🧠 Model Details

This project uses **two models** trained on EEG signal features:

### 1. 🧩 CNN Classifier
- **Task**: Multiclass classification of depression severity (`Low`, `Mild`, `High`)
- **Architecture**: 1D Convolutional Neural Network (CNN)
- **Performance**: ✅ Accuracy = **90.6%**

### 2. 📈 CNN Regression Model
- **Task**: Predict **HAMD (Hamilton Depression Rating Scale)** score
- **Architecture**: CNN with linear output
- **Metric**: Mean Squared Error (MSE) on test set

---

## 💻 App Features

- Upload `.mat` EEG files (from the FT7 channel)
- Run classification (Low/Mild/High depression)
- Predict the HAMD score
- Visual outputs, download results as CSV
- Fully interactive and browser-based with Streamlit

---

## 🚀 Installation & Run


<pre><code>git clone https://github.com/lakkshitkhare/eeg-mental-health-app.git 
cd eeg-mental-health-app pip install -r requirements.txt 
streamlit run App.py</code></pre>

##📄 License
This project is open-source and available under the MIT License.
