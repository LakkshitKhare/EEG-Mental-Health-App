🧠 EEG Mental Health Classifier
Live Demo 👉 eeg-mental-health-app-khare.streamlit.app

An end-to-end Deep Learning web app for predicting mental health severity levels from EEG data using a CNN model.

📌 Table of Contents
Overview

Features

Tech Stack

Model Details

Setup Instructions

Screenshots

Credits

🔍 Overview
This project uses EEG signals to predict the severity level of mental health conditions using a Convolutional Neural Network (CNN). The app is built with Streamlit and deployed publicly for demonstration and analysis.

✅ Features
Upload EEG features from .mat file

Predict severity: Low, Mild, High

Download predictions in CSV format

Clean, tab-based UI with result visualizations

Fully deployed with no setup required

🧪 Model Details
Item	Description
Dataset	CIP_signal_all_FT7.mat (EEG Signals)
Features	1024 signal features
Labels (Target)	Severity levels (20+ classes → grouped into 3)
Model Used	CNN
Accuracy	~90.6% (on test set)
Preprocessing	Normalization, class grouping

🛠 Tech Stack
Frontend: Streamlit

Backend/Model: TensorFlow/Keras (CNN)

Deployment: Streamlit Cloud

Others: scipy, pandas, joblib, numpy

🚀 Setup Instructions (Local)
bash
Copy
Edit
git clone https://github.com/lakkshitkhare/eeg-mental-health-app.git
cd eeg-mental-health-app
pip install -r requirements.txt
streamlit run App.py
🖼 Screenshots
<details> <summary>🔎 Click to expand</summary>


</details>
🙋‍♂️ Credits
Developed by Lakkshit Khare

EEG Dataset: CIP_signal_all_FT7.mat

Thanks to open-source contributors and Streamlit community

📌 License
This project is licensed under the MIT License.

