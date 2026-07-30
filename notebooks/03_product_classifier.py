import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


print("Program started")

folder_path = (
    r"C:\Users\User\OneDrive\DS_Project"
    r"\ComplaintsIntelligent\data"
)

file_path = os.path.join(
    folder_path,
    "complaints_nlp_clean.csv"
)

print("Loading clean dataset...")

df = pd.read_csv(
    file_path,
    low_memory=False
)

print("Dataset loaded")
print("Rows:", df.shape[0])

# Merge old CFPB product names into newer categories
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

df["Product"] = df["Product"].replace(product_replacements)

print("\nProducts after merging:")
print(df["Product"].value_counts().head(15))

# Keep only product categories with at least 100 complaints
product_counts = df["Product"].value_counts()

valid_products = product_counts[
    product_counts >= 300
].index

model_df = df[
    df["Product"].isin(valid_products)
].copy()


print("\nRows used for modelling:", model_df.shape[0])
print("Number of product categories:", model_df["Product"].nunique())

print("\nProduct distribution:")
print(model_df["Product"].value_counts())


# Input text and target label
X = model_df["Consumer complaint narrative"]
y = model_df["Product"]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# Create NLP model
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


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed")


# Make predictions
predictions = model.predict(X_test)


# Evaluate model
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
    "product_classifier.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\nModel saved to:")
print(model_path)