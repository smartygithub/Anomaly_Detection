# 🕵️‍♂️ Credit Card Fraud Detection via Anomaly Detection

This project focuses on detecting fraudulent credit card transactions using **unsupervised anomaly detection** based on **Singular Value Decomposition (SVD)**. It also explores how supervised classification and clustering can refine and analyze fraud patterns. Developed as part of the CS 685/785 course, it applies real-world machine learning techniques to highly imbalanced data.

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Observations**: 284,807 transactions
- **Features**:
  - `Time`, `Amount`
  - 28 anonymized PCA-transformed components (`V1` to `V28`)
  - `Class` — 0: legitimate, 1: fraud

> ⚠️ **Note**: The full dataset (`creditcard.csv`) exceeds GitHub's upload limit. Please download it from Kaggle and place it manually in the project directory.

---

## 🔍 Exploratory Data Analysis & Hypothesis

- **Missing Values**: None
- **Class Imbalance**: Only ~0.17% of transactions are fraudulent
- **Univariate Analysis**:
  - t-tests and boxplots show distinct patterns between legitimate and fraudulent transactions
- **Hypothesis**:
  Fraudulent transactions are structurally different and produce **higher reconstruction errors** when projected onto a low-rank subspace via SVD.

---

## 🧠 Anomaly Detection Using SVD

1. Build data matrix \( X \) using PCA features (V1–V28)
2. Apply SVD: \( X = U \Sigma V^T \)
3. Reconstruct \( \hat{X} \) using top \( k = 10 \) singular values
4. Compute reconstruction error using L₂ norm
5. Select anomaly threshold (95th percentile of errors in legitimate class)
6. Flag transactions exceeding threshold as potential frauds

---

## 🤖 Supervised Classification (Logistic Regression)

- **Goal**: Improve anomaly detection by combining it with supervised learning
- **Features Used**: All PCA features + reconstruction error
- **Classifier**: Logistic Regression with balanced class weights
- **Evaluation**: Classification report, confusion matrix, precision-recall curve

---

## 🧩 Clustering Analysis

- **Algorithm**: K-Means (k = 2)
- **Data**: PCA-transformed features
- **Visualization**: Scatter plot of first two PCA components colored by cluster
- **Insight**: Clustering reveals partial separation of fraud and legitimate classes, reinforcing structural differences in feature space.

---

## 📁 Project Structure

