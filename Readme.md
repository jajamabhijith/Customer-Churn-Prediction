# 📊 Customer Churn Prediction

This project focuses on predicting **customer churn** (whether a customer will leave or stay) using machine learning models. The goal is to help businesses identify customers at risk of leaving early, so they can take proactive measures to improve retention and reduce revenue loss.  

---

## 🚀 Project Overview
- Dataset: **7,043 records** with **21 features** (from Kaggle).
- Task: Predict customer churn (binary classification).
- Preprocessing:
  - One-hot encoding & label encoding for categorical data.
  - Min-Max scaling for numerical features.
  - Exploratory Data Analysis (EDA) for churn distribution.
- Models Implemented:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Gaussian Mixture Model (GMM)

---

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Libraries Used:**  
  - `pandas`, `numpy` – Data handling  
  - `scikit-learn` – Machine Learning models, preprocessing  
  - `matplotlib`, `seaborn` – Visualization  

---

## 📂 Dataset Features
- **Numerical:** `tenure`, `monthly_charges`, `total_charges`  
- **Categorical:** `gender`, `partner`, `dependents`, `phone_service`, `internet_service`, `contract`, `payment_method`, etc.  

After encoding, the dataset expands to **23 features**.  

---

## 📈 Results
| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | **81.09%** | 64.5%     | 57.8%  | **60.96%** |
| Support Vector Machine | 81.03%   | **66.5%** | 52.0%  | 58.20%   |
| K-Nearest Neighbors    | 76.77%   | 55.6%     | 52.2%  | 53.78%   |
| Gaussian Mixture Model | 54.57%   | 41.5%     | 35.0%  | 37.97%   |

✅ **Logistic Regression performed the best overall** with the highest accuracy and F1-score.  

---

## 🔍 Key Insights
- Logistic Regression is the most reliable model for this dataset.  
- GMM struggles with classification due to distribution assumptions.  
- Improvements can be made by:
  - Handling class imbalance (e.g., **SMOTE**).
  - Hyperparameter tuning with regularization (L1/L2).
  - Better feature engineering.
  - Trying different distance metrics for KNN.
  - Using cross-validation for robust evaluation.  

---


