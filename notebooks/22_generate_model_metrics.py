import os

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from sklearn.model_selection import train_test_split


print("Program started")


# --------------------------------------------------
# Project paths
# --------------------------------------------------
project_folder = (
    r"C:\Users\User\OneDrive\DS_Project"
    r"\ComplaintsIntelligent"
)

data_path = os.path.join(
    project_folder,
    "data",
    "complaints_nlp_clean.csv"
)

model_folder = os.path.join(
    project_folder,
    "models"
)

results_folder = os.path.join(
    project_folder,
    "reports",
    "model_results"
)

os.makedirs(
    results_folder,
    exist_ok=True
)


product_model_path = os.path.join(
    model_folder,
    "product_classifier.pkl"
)

issue_model_path = os.path.join(
    model_folder,
    "issue_classifier.pkl"
)

response_model_path = os.path.join(
    model_folder,
    "timely_response_classifier.pkl"
)


# --------------------------------------------------
# Load dataset and models
# --------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(
    data_path,
    low_memory=False
)

print("Rows loaded:", len(df))


print("Loading trained models...")

product_model = joblib.load(
    product_model_path
)

issue_model = joblib.load(
    issue_model_path
)

response_model = joblib.load(
    response_model_path
)

print("Models loaded successfully")


summary_results = []


# ==================================================
# 1. Product classifier evaluation
# ==================================================
print("\nEvaluating product classifier...")


product_df = df[
    [
        "Consumer complaint narrative",
        "Product"
    ]
].dropna().copy()


# Merge equivalent CFPB product categories
product_replacements = {
    "Credit reporting":
        "Credit reporting or other personal consumer reports",

    "Credit reporting, credit repair services, or other personal consumer reports":
        "Credit reporting or other personal consumer reports",

    "Credit card":
        "Credit card or prepaid card",

    "Prepaid card":
        "Credit card or prepaid card",

    "Bank account or service":
        "Checking or savings account",

    "Payday loan, title loan, or personal loan":
        "Payday loan, title loan, personal loan, or advance loan"
}

product_df["Product"] = (
    product_df["Product"]
    .replace(product_replacements)
)


# Recreate the same category filtering used in training
product_counts = (
    product_df["Product"]
    .value_counts()
)

valid_products = product_counts[
    product_counts >= 300
].index

product_df = product_df[
    product_df["Product"].isin(
        valid_products
    )
].copy()


product_X = product_df[
    "Consumer complaint narrative"
]

product_y = product_df[
    "Product"
]


_, product_X_test, _, product_y_test = (
    train_test_split(
        product_X,
        product_y,
        test_size=0.20,
        random_state=42,
        stratify=product_y
    )
)


product_predictions = product_model.predict(
    product_X_test
)


product_accuracy = accuracy_score(
    product_y_test,
    product_predictions
)

product_macro_f1 = f1_score(
    product_y_test,
    product_predictions,
    average="macro",
    zero_division=0
)

product_weighted_f1 = f1_score(
    product_y_test,
    product_predictions,
    average="weighted",
    zero_division=0
)


summary_results.append({
    "Model": "Product classifier",
    "Test records": len(product_y_test),
    "Number of classes": product_y.nunique(),
    "Accuracy": product_accuracy,
    "Macro F1": product_macro_f1,
    "Weighted F1": product_weighted_f1,
    "Positive precision": None,
    "Positive recall": None,
    "Positive F1": None,
    "PR-AUC": None,
    "ROC-AUC": None,
    "Decision threshold": None
})


# Save product classification report
product_report = classification_report(
    product_y_test,
    product_predictions,
    output_dict=True,
    zero_division=0
)

product_report_df = (
    pd.DataFrame(product_report)
    .transpose()
    .reset_index()
    .rename(
        columns={
            "index": "Class"
        }
    )
)

product_report_path = os.path.join(
    results_folder,
    "product_classification_report.csv"
)

product_report_df.to_csv(
    product_report_path,
    index=False
)


# Save product confusion matrix
product_labels = list(
    product_model.classes_
)

product_matrix = confusion_matrix(
    product_y_test,
    product_predictions,
    labels=product_labels
)

product_matrix_df = pd.DataFrame(
    product_matrix,
    index=product_labels,
    columns=product_labels
)

product_matrix_df.index.name = (
    "Actual product"
)

product_matrix_path = os.path.join(
    results_folder,
    "product_confusion_matrix.csv"
)

product_matrix_df.to_csv(
    product_matrix_path
)


print(
    "Product accuracy:",
    round(product_accuracy, 4)
)

print(
    "Product macro F1:",
    round(product_macro_f1, 4)
)


# ==================================================
# 2. Issue classifier evaluation
# ==================================================
print("\nEvaluating issue classifier...")


issue_df = df[
    [
        "Consumer complaint narrative",
        "Issue"
    ]
].dropna().copy()


# Recreate the same issue filtering used in training
issue_counts = (
    issue_df["Issue"]
    .value_counts()
)

valid_issues = issue_counts[
    issue_counts >= 300
].index

issue_df = issue_df[
    issue_df["Issue"].isin(
        valid_issues
    )
].copy()


issue_X = issue_df[
    "Consumer complaint narrative"
]

issue_y = issue_df[
    "Issue"
]


_, issue_X_test, _, issue_y_test = (
    train_test_split(
        issue_X,
        issue_y,
        test_size=0.20,
        random_state=42,
        stratify=issue_y
    )
)


issue_predictions = issue_model.predict(
    issue_X_test
)


issue_accuracy = accuracy_score(
    issue_y_test,
    issue_predictions
)

issue_macro_f1 = f1_score(
    issue_y_test,
    issue_predictions,
    average="macro",
    zero_division=0
)

issue_weighted_f1 = f1_score(
    issue_y_test,
    issue_predictions,
    average="weighted",
    zero_division=0
)


summary_results.append({
    "Model": "Issue classifier",
    "Test records": len(issue_y_test),
    "Number of classes": issue_y.nunique(),
    "Accuracy": issue_accuracy,
    "Macro F1": issue_macro_f1,
    "Weighted F1": issue_weighted_f1,
    "Positive precision": None,
    "Positive recall": None,
    "Positive F1": None,
    "PR-AUC": None,
    "ROC-AUC": None,
    "Decision threshold": None
})


# Save issue classification report
issue_report = classification_report(
    issue_y_test,
    issue_predictions,
    output_dict=True,
    zero_division=0
)

issue_report_df = (
    pd.DataFrame(issue_report)
    .transpose()
    .reset_index()
    .rename(
        columns={
            "index": "Class"
        }
    )
)

issue_report_path = os.path.join(
    results_folder,
    "issue_classification_report.csv"
)

issue_report_df.to_csv(
    issue_report_path,
    index=False
)


# Save issue confusion matrix
issue_labels = list(
    issue_model.classes_
)

issue_matrix = confusion_matrix(
    issue_y_test,
    issue_predictions,
    labels=issue_labels
)

issue_matrix_df = pd.DataFrame(
    issue_matrix,
    index=issue_labels,
    columns=issue_labels
)

issue_matrix_df.index.name = (
    "Actual issue"
)

issue_matrix_path = os.path.join(
    results_folder,
    "issue_confusion_matrix.csv"
)

issue_matrix_df.to_csv(
    issue_matrix_path
)


print(
    "Issue accuracy:",
    round(issue_accuracy, 4)
)

print(
    "Issue macro F1:",
    round(issue_macro_f1, 4)
)


# ==================================================
# 3. Untimely-response model evaluation
# ==================================================
print("\nEvaluating untimely-response model...")


response_columns = [
    "Consumer complaint narrative",
    "Product",
    "Issue",
    "State",
    "Submitted via",
    "Timely response?"
]

response_df = df[
    response_columns
].copy()


response_df = response_df.dropna(
    subset=[
        "Consumer complaint narrative",
        "Timely response?"
    ]
)


# No means untimely
response_df["Untimely"] = (
    response_df["Timely response?"]
    .map({
        "Yes": 0,
        "No": 1
    })
)

response_df = response_df.dropna(
    subset=["Untimely"]
)

response_df["Untimely"] = (
    response_df["Untimely"]
    .astype(int)
)


response_category_columns = [
    "Product",
    "Issue",
    "State",
    "Submitted via"
]

for column in response_category_columns:
    response_df[column] = (
        response_df[column]
        .fillna("Unknown")
        .astype(str)
    )


response_X = response_df[
    [
        "Consumer complaint narrative",
        "Product",
        "Issue",
        "State",
        "Submitted via"
    ]
]

response_y = response_df[
    "Untimely"
]


_, response_X_test, _, response_y_test = (
    train_test_split(
        response_X,
        response_y,
        test_size=0.20,
        random_state=42,
        stratify=response_y
    )
)


response_probabilities = (
    response_model.predict_proba(
        response_X_test
    )[:, 1]
)


# Use your selected threshold
response_threshold = 0.70

response_predictions = (
    response_probabilities
    >= response_threshold
).astype(int)


response_accuracy = accuracy_score(
    response_y_test,
    response_predictions
)

response_precision = precision_score(
    response_y_test,
    response_predictions,
    zero_division=0
)

response_recall = recall_score(
    response_y_test,
    response_predictions,
    zero_division=0
)

response_f1 = f1_score(
    response_y_test,
    response_predictions,
    zero_division=0
)

response_pr_auc = average_precision_score(
    response_y_test,
    response_probabilities
)

response_roc_auc = roc_auc_score(
    response_y_test,
    response_probabilities
)


summary_results.append({
    "Model": "Untimely-response classifier",
    "Test records": len(response_y_test),
    "Number of classes": 2,
    "Accuracy": response_accuracy,
    "Macro F1": None,
    "Weighted F1": None,
    "Positive precision": response_precision,
    "Positive recall": response_recall,
    "Positive F1": response_f1,
    "PR-AUC": response_pr_auc,
    "ROC-AUC": response_roc_auc,
    "Decision threshold": response_threshold
})


# Save response classification report
response_report = classification_report(
    response_y_test,
    response_predictions,
    target_names=[
        "Timely",
        "Untimely"
    ],
    output_dict=True,
    zero_division=0
)

response_report_df = (
    pd.DataFrame(response_report)
    .transpose()
    .reset_index()
    .rename(
        columns={
            "index": "Class"
        }
    )
)

response_report_path = os.path.join(
    results_folder,
    "response_classification_report.csv"
)

response_report_df.to_csv(
    response_report_path,
    index=False
)


# Save response confusion matrix
response_matrix = confusion_matrix(
    response_y_test,
    response_predictions,
    labels=[
        0,
        1
    ]
)

response_matrix_df = pd.DataFrame(
    response_matrix,
    index=[
        "Actual timely",
        "Actual untimely"
    ],
    columns=[
        "Predicted timely",
        "Predicted untimely"
    ]
)

response_matrix_path = os.path.join(
    results_folder,
    "response_confusion_matrix.csv"
)

response_matrix_df.to_csv(
    response_matrix_path
)


print(
    "Untimely precision:",
    round(response_precision, 4)
)

print(
    "Untimely recall:",
    round(response_recall, 4)
)

print(
    "Untimely F1:",
    round(response_f1, 4)
)

print(
    "PR-AUC:",
    round(response_pr_auc, 4)
)

print(
    "ROC-AUC:",
    round(response_roc_auc, 4)
)


# ==================================================
# Save combined model summary
# ==================================================
summary_df = pd.DataFrame(
    summary_results
)

metric_columns = [
    "Accuracy",
    "Macro F1",
    "Weighted F1",
    "Positive precision",
    "Positive recall",
    "Positive F1",
    "PR-AUC",
    "ROC-AUC",
    "Decision threshold"
]

summary_df[metric_columns] = (
    summary_df[metric_columns]
    .round(4)
)


summary_path = os.path.join(
    results_folder,
    "model_metrics.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


print("\nModel performance summary:")

print(
    summary_df.to_string(
        index=False
    )
)


print("\nResults saved inside:")
print(results_folder)

print("\nModel evaluation completed successfully")