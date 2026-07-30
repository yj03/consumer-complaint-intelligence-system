import os
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)
from sklearn.decomposition import NMF


print("Program started")

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

os.makedirs(model_folder, exist_ok=True)


# ---------------------------------
# Load dataset
# ---------------------------------
print("Loading complaint dataset...")

df = pd.read_csv(
    data_path,
    low_memory=False
)

df = df.dropna(
    subset=["Consumer complaint narrative"]
).copy()

print("Narratives loaded:", len(df))


# ---------------------------------
# Clean complaint narratives
# ---------------------------------
def clean_narrative(text):
    text = str(text).lower()

    # Remove anonymized placeholders such as XX and XXXX
    text = re.sub(
        r"\b[xX]{2,}\b",
        " ",
        text
    )

    # Remove currency amounts
    text = re.sub(
        r"\{\$[\d,.]+\}",
        " ",
        text
    )

    # Remove URLs and email addresses
    text = re.sub(
        r"http\S+|www\S+|\S+@\S+",
        " ",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\b\d+\b",
        " ",
        text
    )

    # Keep letters and spaces only
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove repeated spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("Cleaning complaint narratives...")

df["Clean narrative"] = (
    df["Consumer complaint narrative"]
    .apply(clean_narrative)
)

# Remove narratives that become too short
df["Clean word count"] = (
    df["Clean narrative"]
    .str.split()
    .str.len()
)

df = df[
    df["Clean word count"] >= 20
].copy()

print("Narratives after cleaning:", len(df))


# ---------------------------------
# Create additional stop words
# ---------------------------------
custom_stop_words = {
    "complaint",
    "consumer",
    "company",
    "said",
    "told",
    "asked",
    "received",
    "sent",
    "provided",
    "information",
    "regarding",
    "matter",
    "please",
    "thank",
    "section",
    "states",
    "code",
    "law",
    "violation",
    "violations",
    "fact",
    "doctrine"
}

stop_words = list(
    ENGLISH_STOP_WORDS.union(
        custom_stop_words
    )
)


# ---------------------------------
# Create TF-IDF features
# ---------------------------------
print("\nCreating improved text features...")

vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    max_features=15000,
    min_df=10,
    max_df=0.70,
    ngram_range=(1, 2),
    sublinear_tf=True
)

text_features = vectorizer.fit_transform(
    df["Clean narrative"]
)

print("Feature matrix shape:", text_features.shape)


# ---------------------------------
# Train improved topic model
# ---------------------------------
number_of_topics = 10

print(
    f"\nTraining improved model with "
    f"{number_of_topics} topics..."
)

topic_model = NMF(
    n_components=number_of_topics,
    random_state=42,
    init="nndsvda",
    max_iter=500
)

topic_scores = topic_model.fit_transform(
    text_features
)

print("Topic modelling completed")


# ---------------------------------
# Display topic keywords
# ---------------------------------
feature_names = vectorizer.get_feature_names_out()

topic_keywords = []

print("\nImproved topics:")

for topic_number, weights in enumerate(
    topic_model.components_
):
    top_indexes = weights.argsort()[-12:][::-1]

    top_words = [
        feature_names[index]
        for index in top_indexes
    ]

    keywords = ", ".join(top_words)

    topic_keywords.append({
        "Topic number": topic_number,
        "Topic": f"Topic {topic_number}",
        "Keywords": keywords
    })

    print(f"\nTopic {topic_number}:")
    print(keywords)


# ---------------------------------
# Assign topic to each complaint
# ---------------------------------
df["Topic number"] = topic_scores.argmax(axis=1)

df["Topic"] = (
    "Topic "
    + df["Topic number"].astype(str)
)

df["Topic confidence"] = topic_scores.max(axis=1)


print("\nTopic distribution:")

print(
    df["Topic"]
    .value_counts()
    .sort_index()
)


# ---------------------------------
# Save results
# ---------------------------------
complaints_output = os.path.join(
    project_folder,
    "data",
    "complaints_with_improved_topics.csv"
)

keywords_output = os.path.join(
    project_folder,
    "data",
    "improved_topic_keywords.csv"
)

df.to_csv(
    complaints_output,
    index=False
)

pd.DataFrame(
    topic_keywords
).to_csv(
    keywords_output,
    index=False
)

joblib.dump(
    topic_model,
    os.path.join(
        model_folder,
        "improved_topic_model.pkl"
    )
)

joblib.dump(
    vectorizer,
    os.path.join(
        model_folder,
        "improved_topic_vectorizer.pkl"
    )
)

print("\nResults saved to:")
print(complaints_output)
print(keywords_output)

print("\nImproved topic model saved successfully")