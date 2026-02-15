# ML Classification Project

This project compares different classification models using a Kaggle dataset.

a. Problem Statement
    The objective of this project is to implement and compare multiple machine learning classification models on the Heart Disease dataset. The project demonstrates an end-to-end ML workflow including preprocessing, model training, evaluation using multiple performance metrics, and deployment through a Streamlit web application.

b. Dataset Description
    Dataset Name: Heart Disease Dataset
    Source: Kaggle 
    Number of Instances: (Enter from your notebook, e.g., 1025)
    Number of Features: (Enter actual feature count excluding target)
    Target Variable: target
    Problem Type: Multiclass Classification

c. Model Comparison
## 📊 Model Performance Comparison

| Model               | Accuracy | AUC Score | Precision | Recall   | F1 Score | MCC Score |
|---------------------|----------|-----------|-----------|----------|----------|-----------|
| Logistic Regression | 0.661017 | 0.814375  | 0.324532  | 0.318393 | 0.317383 | 0.343419  |
| Decision Tree       | 0.474576 | 0.548684  | 0.240888  | 0.255536 | 0.239687 | 0.155531  |
| KNN                 | 0.661017 | 0.709951  | 0.268952  | 0.290714 | 0.279403 | 0.372552  |
| Naive Bayes         | 0.347458 | 0.641720  | 0.227121  | 0.245714 | 0.192231 | 0.190497  |
| Random Forest       | 0.618644 | 0.751571  | 0.244118  | 0.262857 | 0.253134 | 0.289985  |
| XGBoost             | 0.601695 | 0.812970  | 0.244444  | 0.260536 | 0.252113 | 0.277401  |



| Model                   | Key Observation                                                                                                                                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | Achieved the highest AUC score (0.814) and tied highest accuracy (0.661). This indicates strong class   separation capability. The moderate MCC suggests balanced predictive power across classes.                        |
| **Decision Tree**       | Showed relatively low accuracy (0.475) and low MCC (0.156), indicating weaker generalization and possible overfitting. Performance was inferior compared to other models.                                               |
| **KNN**                 | Achieved the highest MCC score (0.373) and tied highest accuracy (0.661), indicating better balanced classification performance across all classes. Performed competitively despite lower AUC than Logistic Regression. |
| **Naive Bayes**         | Recorded the lowest accuracy (0.347) and lowest F1 score (0.192), suggesting that the independence assumption did not hold well for this dataset. Overall performance was comparatively weak.                           |
| **Random Forest**       | Delivered stable performance with good AUC (0.752) and moderate accuracy (0.619). However, it did not outperform Logistic Regression or KNN in this dataset.                                                            |
| **XGBoost**             | Achieved high AUC (0.813), nearly equal to Logistic Regression, showing strong ranking ability. However, accuracy and MCC were slightly lower, indicating moderate classification consistency.                          |


