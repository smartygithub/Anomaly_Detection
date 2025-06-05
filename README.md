# 🕵️ Anomaly Detection in Credit Card Transactions

This project implements an **anomaly detection system** to identify fraudulent credit card transactions using unsupervised learning techniques. The approach focuses on detecting outliers in transaction data without relying on labeled outcomes.

## 📊 Dataset

> The original dataset used is the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
> **Note:** Due to GitHub's file size limitations, the file `creditcard.csv` is not included. Please download it from Kaggle and place it in the project directory.

- **Total Transactions**: 284,807  
- **Fraudulent**: 492 (0.17%)
- **Features**: 30 numerical features + class label

---

## 🧠 Methodology

- **Technique**: Anomaly Detection using Singular Value Decomposition (SVD)
- **Steps**:
  - Preprocess and normalize the data
  - Apply SVD to reduce dimensionality
  - Reconstruct and compute reconstruction error
  - Detect anomalies using thresholding

---

## 📈 Evaluation Metrics

- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC

---

## 🛠️ Technologies Used

- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in the project folder
3. Open and run the Jupyter notebook:

```bash
jupyter notebook Code_Final.ipynb
