import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)


print("Program started")

data_path = (
    r"C:\Users\User\OneDrive\DS_Project"
    r"\ComplaintsIntelligent\data"
    r"\complaints_nlp_clean.csv"
)

print("Loading dataset...")

df = pd.read_csv(
    data_path,
    low_memory=False
)

print("Dataset loaded")
print("Rows:", len(df))


# Keep required columns
required_columns = [
    "Consumer complaint narrative",
    "Product",
    "Issue",
    "State",
    "Submitted via",
    "Timely response?"
]

model_df = df[required_columns].copy()


# Remove rows without target or narrative
model_df = model_df.dropna(
    subset=[
        "Consumer complaint narrative",
        "Timely response?"
    ]
)


# Convert target:
# No = untimely = 1
# Yes = timely = 0
model_df["Untimely"] = (
    model_df["Timely response?"]
    .map({
        "Yes": 0,
        "No": 1
    })
)

model_df = model_df.dropna(
    subset=["Untimely"]
)

model_df["Untimely"] = model_df["Untimely"].astype(int)


# Fill missing categorical values
categorical_columns = [
    "Product",
    "Issue",
    "State",
    "Submitted via"
]

for column in categorical_columns:
    model_df[column] = (
        model_df[column]
        .fillna("Unknown")
        .astype(str)
    )


print("\nTarget distribution:")
print(model_df["Untimely"].value_counts())

print("\nTarget percentage:")
print(
    model_df["Untimely"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


X = model_df[
    [
        "Consumer complaint narrative",
        "Product",
        "Issue",
        "State",
        "Submitted via"
    ]
]

y = model_df["Untimely"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# Prepare text and categorical features
preprocessor = ColumnTransformer([
    (
        "text",
        TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            min_df=3
        ),
        "Consumer complaint narrative"
    ),
    (
        "categories",
        OneHotEncoder(
            handle_unknown="ignore"
        ),
        categorical_columns
    )
])


model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


print("\nTraining untimely-response model...")

model.fit(
    X_train,
    y_train
)

print("Training completed")


predictions = model.predict(X_test)

probabilities = model.predict_proba(
    X_test
)[:, 1]


print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))


print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Timely",
            "Untimely"
        ],
        zero_division=0
    )
)


print("PR-AUC:")
print(
    round(
        average_precision_score(
            y_test,
            probabilities
        ),
        4
    )
)

print("ROC-AUC:")
print(
    round(
        roc_auc_score(
            y_test,
            probabilities
        ),
        4
    )
)


# Save the model
model_folder = (
    r"C:\Users\User\OneDrive\DS_Project"
    r"\ComplaintsIntelligent\models"
)

os.makedirs(
    model_folder,
    exist_ok=True
)

model_path = os.path.join(
    model_folder,
    "timely_response_classifier.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved to:")
print(model_path)