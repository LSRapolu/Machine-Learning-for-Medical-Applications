# Machine-Learning-for-Medical-Applications
# 🧠 ML4Health – Machine Learning Pipelines for Medical Diagnostics

A unified collection of 3 machine learning pipelines tailored to structured and signal-based health diagnostics, covering:

- 🩺 Diabetes Risk Classification (Tabular Data)
- ❤️ Arrhythmia Detection (ECG)
- ⚡ Epilepsy Detection (EEG)

This project demonstrates the application of **decision trees, deep learning (LSTM, CNN), and autoencoders** to three distinct biomedical challenges using publicly available datasets.

---

## 📁 Modules

### 1. 📊 Early-Stage Diabetes Risk – Decision Trees (CART)

- **Dataset**: Early-Stage Diabetes Risk Prediction Dataset (UCI)
- **Task**: Binary classification of diabetes risk based on clinical features
- **Model**: Custom implementation of CART (Gini impurity)
- **Technologies**:
  - Python (pandas, numpy)
  - Manual tree construction logic
  - Node splitting, Gini impurity, recursive partitioning

### 2. 🫀 Arrhythmia Detection – LSTM & CNN on ECG

- **Dataset**: MIT-BIH Arrhythmia Database (PhysioNet)
- **Task**: Classify heartbeat signals into normal/abnormal
- **Model**: 1D Convolution + BiLSTM
- **Technologies**:
  - Keras (LSTM, Conv1D, Dropout, Dense)
  - Wavelet transforms for signal denoising (PyWavelets)
  - ECG annotation via WFDB
  - Data augmentation, signal slicing, and sequence modeling

### 3. 🧠 Epilepsy Detection – Autoencoders on EEG

- **Dataset**: EEG Epilepsy Dataset (Zenodo, 2020)
- **Task**: Detect epileptic patterns from EEG segments
- **Model**: 1D Convolutional Autoencoder + Classifier
- **Technologies**:
  - Keras (Conv1D, UpSampling, Flatten, Reshape)
  - Signal preprocessing and batch generation
  - Feature learning + latent space analysis via t-SNE
  - Evaluation via class separation in latent space

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python)
![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/-Seaborn-66B3BA?logo=python)
![Scikit-learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?logo=scikit-learn)
![WFDB](https://img.shields.io/badge/-WFDB-006400)
![PyWavelets](https://img.shields.io/badge/-PyWavelets-4169E1)

---

## 📌 Resume Highlights

- Built 3 diagnostic ML pipelines for structured and biomedical signal data, addressing diabetes, arrhythmia, and epilepsy.
- Implemented a full CART Decision Tree algorithm from scratch with Gini-based splitting for risk classification.
- Developed an LSTM-CNN model for ECG signal classification using PhysioNet data, achieving high interpretability and real-time readiness.
- Trained a convolutional autoencoder on EEG data to distinguish epileptic vs healthy signals using latent features and t-SNE visualization.
- Preprocessed ECG and EEG signals using wavelets, batch segmentation, and temporal padding to enable sequence learning.

---

## 📂 How to Use

Each module contains:
- A standalone `.ipynb` notebook
- Download instructions for the corresponding dataset
- Preprocessing, training, and evaluation steps
- Model interpretation via plots or accuracy metrics

---

## 📚 References

- MIT-BIH Arrhythmia Dataset – https://physionet.org/content/mitdb/1.0.0/
- EEG Epilepsy Dataset – https://zenodo.org/record/3684992
- Diabetes Dataset – https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset

---

> 🎯 This project serves as a multi-modal demo of applying classical ML and deep learning to real-world medical diagnostics using open data.
