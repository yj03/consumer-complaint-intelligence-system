import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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


# Remove rows without issue or narrative
df = df.dropna(
    subset=[
        "Issue",
        "Consumer complaint narrative"
    ]
)


# Display the most common issues
print("\nTop 15 issues:")

print(
    df["Issue"]
    .value_counts()
    .head(15)
)


# Keep issues with at least 300 complaints
issue_counts = df["Issue"].value_counts()

valid_issues = issue_counts[
    issue_counts >= 300
].index

model_df = df[
    df["Issue"].isin(valid_issues)
].copy()


print("\nRows used:", len(model_df))

print(
    "Number of issue categories:",
    model_df["Issue"].nunique()
)

print("\nIssue distribution:")

print(
    model_df["Issue"]
    .value_counts()
)


# Input and target
X = model_df["Consumer complaint narrative"]
y = model_df["Issue"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# Build model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=3
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


print("\nTraining issue classifier...")

model.fit(
    X_train,
    y_train
)

print("Training completed")


# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel accuracy:")
print(round(accuracy, 4))

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# Save model
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
    "issue_classifier.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved to:")
print(model_path)