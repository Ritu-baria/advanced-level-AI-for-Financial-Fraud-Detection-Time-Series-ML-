# advanced-level-AI-for-Financial-Fraud-Detection-Time-Series
# 🔍 AI for Financial Fraud Detection (Time Series + ML)

Detecting fraudulent transactions using Machine Learning and Time-Series based features.

## 🚀 Project Overview

This project uses **real-world financial transaction data** to detect fraudulent activity using both **supervised** (Random Forest) and **unsupervised** (Isolation Forest) machine learning techniques.

### 🎯 Key Goals:
- Detect fraud using transaction patterns and metadata
- Handle severe class imbalance using SMOTE
- Apply anomaly detection for unlabeled prediction
- Visualize model performance without external libraries like seaborn

---

## 🧠 Features & Techniques

- ✅ Time-based feature engineering (`Hour` of transaction)
- ✅ Data normalization using `StandardScaler`
- ✅ Class balancing with `SMOTE`
- ✅ Supervised model: `RandomForestClassifier`
- ✅ Unsupervised model: `IsolationForest`
- ✅ Confusion matrix plotted with `matplotlib`

---

## 📁 Dataset

- [Kaggle: Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains anonymized transaction data, including:
  - `Time`, `Amount`, `Class` (1 = fraud, 0 = normal)

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib

---

## 🧪 Model Evaluation

- **Random Forest Accuracy**: ~99%
- **F1 Score (fraud class)**: High due to class weight + SMOTE
- **ROC AUC Score**: Evaluated on test set
- **Anomaly Detection Accuracy**: Using Isolation Forest (unsupervised)

---

## 📊 Confusion Matrix Example

Plotted using only `matplotlib`:

![Confusion Matrix](confusion_matrix_example.png)

---

## 🛠️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/fraud-detection-ml.git
   cd fraud-detection-ml
