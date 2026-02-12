import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import numpy as np
import time
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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

st.markdown("""
### 📝 How to Use This App:
1. Download the sample test dataset.
2. Upload the CSV file using the sidebar.
3. Select a model.
4. View evaluation metrics and confusion matrix.
""")


# -----------------------------
# Sidebar Section
# -----------------------------
st.sidebar.header("⚙️ Settings")

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
    "Logistic Regression": "models/logistic.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

model = joblib.load(model_paths[model_option])

#File uploader
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

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
        acc = accuracy_score(y_test, y_pred)
        roc_auc_score_value = classification_report(y_test, y_pred, output_dict=True)['macro avg']['roc_auc'] if 'roc_auc' in classification_report(y_test, y_pred, output_dict=True)['macro avg'] else "N/A"
        prec = classification_report(y_test, y_pred, output_dict=True)['macro avg']['precision']
        rec = classification_report(y_test, y_pred, output_dict=True)['macro avg']['recall']
        f1 = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
        mcc = classification_report(y_test, y_pred, output_dict=True)['macro avg']['mcc'] if 'mcc' in classification_report(y_test, y_pred, output_dict=True)['macro avg'] else "N/A"

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("Precision (Macro)", f"{prec:.3f}")
        col3.metric("Recall (Macro)", f"{rec:.3f}")
        col4.metric("F1 Score (Macro)", f"{f1:.3f}")

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