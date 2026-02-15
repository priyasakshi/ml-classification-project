import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import numpy as np
import time
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, matthews_corrcoef

# Multiclass AUC and MCC calculation
def multiclass_metrics(y_true, y_pred, y_prob):
    # Macro-average AUC (One-vs-Rest)
    try:
        auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception as e:
        auc_macro = f"AUC error: {e}"
    # Multiclass MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Multiclass AUC (macro): {auc_macro}")
    print(f"Multiclass MCC: {mcc:.4f}")

# Common metrics calculation
def calculate_metrics(y_true, y_pred, y_prob=None):
    # Accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    # Precision, Recall, F1 ( average for multiclass)
    classes = np.unique(y_true)
    precisions, recalls, f1s = [], [], []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    f1_macro = np.mean(f1s)

    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_true, y_pred)

   
    print(f"Accuracy: {accuracy:.4f}")

    # AUC Score
    auc = None
    if y_prob is not None:
        try:
            auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            print(f"Multiclass AUC Score (macro): {auc_score:.4f}")
        except Exception as e:
            print(f"Multiclass AUC Score error: {e}")

    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"MCC Score: {mcc:.4f}")

    return accuracy, auc_score, precision_macro, recall_macro, f1_macro, mcc

# Load model metrics for display
metrics_df = pd.read_csv("model_metrics.csv")

# Page configuration
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide",
    page_icon="❤️"
)

st.title("Heart Disease Classification App")
st.markdown("Upload a test dataset and evaluate different ML models.")

# -----------------------------
# Download Sample Test Data
# -----------------------------
st.subheader("📥 Download Sample Test Dataset")

with open("data/heart_disease_test_data.csv", "rb") as file:
    st.download_button(
        label="Download Sample Test CSV",
        data=file,
        file_name="heart_disease_test_data.csv",
        mime="text/csv"
    )

# Load models
model_option = st.selectbox(
    "Select Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

model_paths = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model = joblib.load(model_paths[model_option])

# -----------------------------
# Sidebar Section
# -----------------------------
st.sidebar.header("⚙️ Settings")
#File uploader
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

st.sidebar.markdown("""
                    
---
### 📝 How to Use This App:
1. Download the sample test dataset.
2. Upload the CSV file .
3. Select a model.
4. View evaluation metrics and confusion matrix.
                    
---
### 👩‍🎓 Student Details

**Name:** Sakshi Priya  
**BITS Id:** 2025AA05425  
**Subject:** Machine Learning  
**Assignment:** ML_Assignment_2

---
""")


# -----------------------------
# Main Section
# -----------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'target' not in data.columns:
        st.error("Uploaded dataset must contain a 'target' column.")
    else:
        X_test = data.drop("target", axis=1)
        y_test = data["target"]

        with st.spinner("Model is analyzing data... Please wait ⏳"):
            time.sleep(3)   # waits for 3 seconds
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        st.success("Prediction completed successfully! ✅")


        # Predictions
        y_pred = model.predict(X_test)

        # Probabilities for ROC AUC 
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

        # Metrics
        acc_lr, auc_lr, precision_lr, recall_lr, f1_lr, mcc_lr = calculate_metrics(y_test, y_pred, y_prob)   
        # acc = accuracy_score(y_test, y_pred)
        # roc_auc_score_value = classification_report(y_test, y_pred, output_dict=True)['macro avg']['roc_auc'] if 'roc_auc' in classification_report(y_test, y_pred, output_dict=True)['macro avg'] else "N/A"
        # roc_auc_score_value = classification_report(y_test, y_pred, output_dict=True)['macro avg']['roc_auc'] if 'roc_auc' in classification_report(y_test, y_pred, output_dict=True)['macro avg'] else "N/A"
        # prec = classification_report(y_test, y_pred, output_dict=True)['macro avg']['precision']
        # rec = classification_report(y_test, y_pred, output_dict=True)['macro avg']['recall']
        # f1 = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
        # mcc = classification_report(y_test, y_pred, output_dict=True)['macro avg']['mcc'] if 'mcc' in classification_report(y_test, y_pred, output_dict=True)['macro avg'] else "N/A"

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{acc_lr:.3f}")
        col2.metric("Precision (Macro)", f"{precision_lr:.3f}")
        col3.metric("Recall (Macro)", f"{recall_lr:.3f}")
        col1.metric("F1 Score (Macro)", f"{f1_lr:.3f}")
        col2.metric("AUC Score (Macro)", f"{auc_lr:.3f}")
        col3.metric("MCC Score", f"{mcc_lr:.3f}")

        # Classification Report
        st.subheader("📋 Classification Report")

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap="Blues"))

        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

        st.subheader("📈 ROC Curve (Multiclass)")

    # Binarize the output
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    # Plot ROC curve for each class
    fig, ax = plt.subplots(figsize=(7,6))

    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"Class {classes[i]} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (One-vs-Rest)")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Please upload a test dataset to begin evaluation.")