# ❤️ Heart Disease Classification and Deployment using Streamlit

## 👩‍🎓 Student Details

- **Name:** Sakshi Priya  
- **Roll No:** 2025AA05425  
- **Subject:** Machine Learning  
- **Assignment:** Assignment 2 – Classification and Deployment  
- **Program:** M.Tech  

---

## 📌 Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a real-world classification dataset. The project involves preprocessing data using pipelines, training different classification models, evaluating their performance using multiple metrics, and deploying the models using Streamlit for interactive testing and evaluation.

---

## 📊 Dataset Description

- **Dataset:** Heart Disease Dataset  
- **Source:** Kaggle / UCI Repository  
- **Problem Type:** Multiclass Classification  
- **Target Variable:** `target` (represents severity of heart disease) 
  - 0 - No Heart Disease
  - 1 - Mild Heart Disease
  - 2 - Moderate Heart Disease
  - 3 - Severe Heart Disease
  - 4 - Very Severe Heart Disease
- **Number of Features:** 13 input features  
- **Number of Instances:** 918  

### Features include:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- Slope
- Number of Major Vessels
- Thalassemia

---

## ⚙️ Preprocessing Steps

The following preprocessing steps were applied using Scikit-learn pipelines:

- Handling missing/Nan values using SimpleImputer
- One-hot encoding for categorical features
- Feature scaling using StandardScaler
- Unified preprocessing using ColumnTransformer and Pipeline

This ensured consistent preprocessing across all models.

---

## 🤖 Machine Learning Models Implemented

The following classification models were implemented and compared:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest Classifier  
6. XGBoost Classifier  

---

## 📊 Model Performance Comparison

| Model               | Accuracy | AUC Score | Precision | Recall   | F1 Score | MCC Score |
|---------------------|----------|-----------|-----------|----------|----------|-----------|
| Logistic Regression | 0.661017 | 0.814375  | 0.324532  | 0.318393 | 0.317383 | 0.343419  |
| Decision Tree       | 0.474576 | 0.548684  | 0.240888  | 0.255536 | 0.239687 | 0.155531  |
| KNN                 | 0.661017 | 0.709951  | 0.268952  | 0.290714 | 0.279403 | 0.372552  |
| Naive Bayes         | 0.347458 | 0.641720  | 0.227121  | 0.245714 | 0.192231 | 0.190497  |
| Random Forest       | 0.618644 | 0.751571  | 0.244118  | 0.262857 | 0.253134 | 0.289985  |
| XGBoost             | 0.601695 | 0.812970  | 0.244444  | 0.260536 | 0.252113 | 0.277401  |

---

## 🔎 Model Observations

| Model               | Detailed Observation |
|---------------------|---------------------|
| Logistic Regression | Logistic Regression achieved highest accuracy scores (0.661) and the highest AUC score (0.814). This indicates strong class separation capability. The MCC score (0.343) suggests balanced predictions across classes. Overall, it showed strong and stable performance despite being a simple linear model. |
| Decision Tree       | Decision Tree showed lower performance with an accuracy of 0.475 and MCC of 0.156. The lower AUC score indicates weak generalization capability. This suggests possible overfitting and reduced ability to handle complex patterns in the dataset. |
| KNN                 | KNN achieved the highest MCC score (0.373) and tied highest accuracy (0.661), indicating balanced and reliable predictions across classes. Although its AUC score was lower than Logistic Regression, the higher MCC score shows better overall classification agreement. |
| Naive Bayes         | Naive Bayes produced the lowest accuracy (0.347) and lowest F1 score (0.192), indicating weak predictive performance. This is likely due to the independence assumption, which does not hold well for correlated medical features. |
| Random Forest       | Random Forest achieved moderate performance with accuracy of 0.619 and AUC score of 0.752. As an ensemble model, it reduced variance but did not outperform Logistic Regression or KNN in this dataset. |
| XGBoost             | XGBoost achieved a high AUC score (0.813), nearly equal to Logistic Regression, showing strong ranking capability. However, its accuracy and MCC were slightly lower, indicating moderate classification consistency. |

---

## 🌐 Streamlit Application Features

The deployed Streamlit application includes:

- Upload test dataset (CSV)
- Select classification model from dropdown
- Display pre-calculated performance metrics
- Display evaluation metrics on uploaded data
- Classification report
- Confusion matrix visualization
- ROC Curve (Multiclass)
- Interactive user interface

---

## 🚀 Deployment

- **GitHub Repository:** https://github.com/priyasakshi/ml-classification-project  
- **Streamlit App:** https://heart-disease-model-comparison.streamlit.app

---

## 📁 Project Structure

ML_Assignment2/
│
├── app.py
├── requirements.txt
├── README.md
│
└── model/
    ├── logistic.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── test_data.csv

---

## Conclusion

This project successfully demonstrates the implementation, evaluation, and deployment of multiple classification models using structured machine learning pipelines. Logistic Regression and KNN showed the best overall performance. Ensemble models such as Random Forest and XGBoost also performed well. Naive Bayes showed the weakest performance due to its assumptions.
The Streamlit application provided an interactive interface to evaluate and compare models using real test data.

---

## Requirements

Install dependencies using:   
pip install -r requirements.txt

---

## Run Streamlit App Locally
streamlit run app.py

