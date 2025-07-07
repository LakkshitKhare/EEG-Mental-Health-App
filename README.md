# 🧠 EEG Mental Health Classifier

Live Demo 👉 [https://eeg-mental-health-app-khare.streamlit.app](https://eeg-mental-health-app-khare.streamlit.app)

An end-to-end **Deep Learning web app** for predicting mental health severity levels using EEG data. Built using a **CNN model** and deployed via **Streamlit Cloud**.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Details](#model-details)
- [Setup Instructions (Local)](#setup-instructions-local)
- [Screenshots](#screenshots)
- [Credits](#credits)
- [License](#license)

---

## Overview

This web app leverages **Electroencephalography (EEG)** signal features to classify the **severity of mental health conditions** into three categories: **Low**, **Mild**, and **High**. The model used is a **Convolutional Neural Network (CNN)** trained on processed EEG features extracted from `.mat` files.

---

## Features

- 📂 Upload EEG `.mat` files directly in the app  
- 🧠 Predict severity levels of mental health conditions  
- 📊 Display predictions and allow CSV download  
- 🖥 Sleek and interactive UI with Streamlit Tabs  
- 🌐 Fully deployed online — no installation required

---

## Tech Stack

| Layer      | Technology               |
|------------|---------------------------|
| UI         | Streamlit                |
| Model      | TensorFlow + Keras (CNN) |
| Data       | EEG Signals from `.mat`  |
| Others     | Pandas, NumPy, SciPy, Joblib |

---

## Model Details

- **Dataset:** `CIP_signal_all_FT7.mat` (EEG data)
- **Features:** 1024-length vector per sample
- **Target:** 20+ severity labels → grouped into 3 (Low/Mild/High)
- **Model:** CNN
- **Accuracy:** ~90.6%
- **Preprocessing:** Normalization, Label grouping

---

## Setup Instructions (Local)

```bash
git clone https://github.com/lakkshitkhare/eeg-mental-health-app.git
cd eeg-mental-health-app
pip install -r requirements.txt
streamlit run App.py```

Credits
Developed by Lakkshit Khare

Dataset: CIP EEG Signal Data

Model: CNN trained on EEG features

Thanks to the Streamlit community and TensorFlow team

