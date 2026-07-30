from pathlib import Path
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

project_folder = Path(__file__).resolve().parents[1]

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

import os
import pandas as pd


print("Program started")

project_folder = Path(__file__).resolve().parents[1]

input_path = os.path.join(
    project_folder,
    "data",
    "complaints_with_improved_topics.csv"
)

output_path = os.path.join(
    project_folder,
    "data",
    "complaints_with_named_topics.csv"
)


print("Loading topic dataset...")

df = pd.read_csv(
    input_path,
    low_memory=False
)

print("Dataset loaded")
print("Rows:", len(df))


# Detailed name for each discovered topic
topic_names = {
    0: "FCRA Privacy and Permissible Purpose Disputes",

    1: "Credit Report Inaccuracies and Disputes",

    2: "Account, Payment and Late Reporting Problems",

    3: "Unauthorized Credit Reporting and Consent",

    4: "Digital Payment Fraud and Dispute Handling",

    5: "Unverified Debt and Debt Validation",

    6: "Credit Denial Caused by Reporting Errors",

    7: "FCRA Rights and Privacy Complaints",

    8: "False Credit Reporting and Unresolved Disputes",

    9: "Identity Theft and Unauthorized Accounts"
}


# Broader groups for charts and dashboards
topic_families = {
    0: "Credit Reporting Rights",

    1: "Credit Report Accuracy",

    2: "Account and Payment Problems",

    3: "Credit Reporting Rights",

    4: "Digital Payment Fraud",

    5: "Debt Collection",

    6: "Credit Report Accuracy",

    7: "Credit Reporting Rights",

    8: "Credit Report Accuracy",

    9: "Identity Theft"
}


df["Topic name"] = (
    df["Topic number"]
    .map(topic_names)
)

df["Topic family"] = (
    df["Topic number"]
    .map(topic_families)
)


# Check whether any topic numbers were not mapped
missing_names = df["Topic name"].isna().sum()

print("\nRows without a topic name:")
print(missing_names)


print("\nDetailed topic distribution:")

print(
    df["Topic name"]
    .value_counts()
    .to_string()
)


print("\nTopic family distribution:")

print(
    df["Topic family"]
    .value_counts()
    .to_string()
)


df.to_csv(
    output_path,
    index=False
)


print("\nNamed topic dataset saved to:")
print(output_path)

print("\nTopic mapping completed successfully")

import os
import pandas as pd


print("Program started")

project_folder = Path(__file__).resolve().parents[1]

input_path = os.path.join(
    project_folder,
    "data",
    "complaints_with_named_topics.csv"
)

output_path = os.path.join(
    project_folder,
    "data",
    "monthly_topic_growth.csv"
)


# ---------------------------------
# Load dataset
# ---------------------------------
print("Loading named-topic dataset...")

df = pd.read_csv(
    input_path,
    low_memory=False
)

print("Dataset loaded")
print("Rows:", len(df))


# ---------------------------------
# Prepare dates
# ---------------------------------
df["Date received"] = pd.to_datetime(
    df["Date received"],
    errors="coerce",
    utc=True
)

df = df.dropna(
    subset=[
        "Date received",
        "Topic family"
    ]
).copy()


# Convert each date to the first day of its month
df["Month"] = (
    df["Date received"]
    .dt.tz_convert(None)
    .dt.to_period("M")
    .dt.to_timestamp()
)


print("\nEarliest month:", df["Month"].min())
print("Latest month:", df["Month"].max())


# ---------------------------------
# Count complaints by month and topic
# ---------------------------------
monthly_counts = (
    df.groupby(
        [
            "Month",
            "Topic family"
        ]
    )
    .size()
    .reset_index(
        name="Complaint count"
    )
)


# ---------------------------------
# Add missing months as zero
# ---------------------------------
all_months = pd.date_range(
    start=df["Month"].min(),
    end=df["Month"].max(),
    freq="MS"
)

all_topics = sorted(
    df["Topic family"].unique()
)

complete_index = pd.MultiIndex.from_product(
    [
        all_months,
        all_topics
    ],
    names=[
        "Month",
        "Topic family"
    ]
)

monthly_counts = (
    monthly_counts
    .set_index(
        [
            "Month",
            "Topic family"
        ]
    )
    .reindex(
        complete_index,
        fill_value=0
    )
    .reset_index()
)


monthly_counts = monthly_counts.sort_values(
    [
        "Topic family",
        "Month"
    ]
)


# ---------------------------------
# Calculate previous 6-month average
# ---------------------------------
monthly_counts["Previous 6-month average"] = (
    monthly_counts
    .groupby("Topic family")["Complaint count"]
    .transform(
        lambda values:
        values.shift(1)
        .rolling(
            window=6,
            min_periods=3
        )
        .mean()
    )
)


# Previous 6-month standard deviation
monthly_counts["Previous 6-month standard deviation"] = (
    monthly_counts
    .groupby("Topic family")["Complaint count"]
    .transform(
        lambda values:
        values.shift(1)
        .rolling(
            window=6,
            min_periods=3
        )
        .std()
    )
)


# ---------------------------------
# Calculate growth percentage
# ---------------------------------
monthly_counts["Growth percentage"] = (
    (
        monthly_counts["Complaint count"]
        - monthly_counts["Previous 6-month average"]
    )
    / monthly_counts["Previous 6-month average"]
    * 100
)


# Calculate Z-score
monthly_counts["Z-score"] = (
    (
        monthly_counts["Complaint count"]
        - monthly_counts["Previous 6-month average"]
    )
    / monthly_counts[
        "Previous 6-month standard deviation"
    ]
)


# Replace infinite values
monthly_counts["Growth percentage"] = (
    monthly_counts["Growth percentage"]
    .replace(
        [
            float("inf"),
            float("-inf")
        ],
        pd.NA
    )
)

monthly_counts["Z-score"] = (
    monthly_counts["Z-score"]
    .replace(
        [
            float("inf"),
            float("-inf")
        ],
        pd.NA
    )
)


# ---------------------------------
# Classify emerging topics
# ---------------------------------
def classify_topic(row):

    growth = row["Growth percentage"]
    z_score = row["Z-score"]
    complaint_count = row["Complaint count"]

    if pd.isna(growth):
        return "Insufficient history"

    # Avoid flagging very small topic volumes
    if complaint_count < 10:
        return "Low volume"

    if growth >= 100 or z_score >= 3:
        return "Critical emerging topic"

    if growth >= 50 or z_score >= 2:
        return "High emerging topic"

    if growth >= 25 or z_score >= 1:
        return "Moderate growth"

    if growth <= -25:
        return "Decreasing"

    return "Stable"


monthly_counts["Topic status"] = (
    monthly_counts.apply(
        classify_topic,
        axis=1
    )
)


# ---------------------------------
# Save results
# ---------------------------------
monthly_counts.to_csv(
    output_path,
    index=False
)


# ---------------------------------
# Find the latest sufficiently complete month
# ---------------------------------

monthly_total = (
    df.groupby("Month")
    .size()
    .sort_index()
)

# Typical number of complaints in a month
median_monthly_total = monthly_total.median()

# A month must contain at least 25% of the normal monthly volume
minimum_required_count = max(
    20,
    int(median_monthly_total * 0.25)
)

valid_months = monthly_total[
    monthly_total >= minimum_required_count
].index

if len(valid_months) == 0:
    raise ValueError(
        "No month has enough complaints for trend analysis."
    )

latest_month = valid_months.max()

print("\nLatest date in dataset:")
print(monthly_counts["Month"].max())

print("\nLatest usable month:")
print(latest_month)

print("\nComplaints in latest usable month:")
print(monthly_total.loc[latest_month])

print("\nMinimum complaints required:")
print(minimum_required_count)


latest_results = monthly_counts[
    monthly_counts["Month"] == latest_month
].copy()


display_columns = [
    "Topic family",
    "Month",
    "Complaint count",
    "Previous 6-month average",
    "Growth percentage",
    "Z-score",
    "Topic status"
]


print("\nLatest topic growth results:")

latest_display = latest_results[
    display_columns
].copy()

numeric_columns = [
    "Complaint count",
    "Previous 6-month average",
    "Growth percentage",
    "Z-score"
]

latest_display[numeric_columns] = (
    latest_display[numeric_columns]
    .round(2)
)

print(
    latest_display.to_string(
        index=False
    )
)

# Show only emerging topics
emerging_topics = latest_results[
    latest_results["Topic status"].isin(
        [
            "Moderate growth",
            "High emerging topic",
            "Critical emerging topic"
        ]
    )
]


print("\nEmerging topics:")

if emerging_topics.empty:
    print("No emerging topics detected.")

else:
    print(
        emerging_topics[display_columns]
        .round(2)
        .to_string(index=False)
    )


print("\nResults saved to:")
print(output_path)

print("\nTopic growth analysis completed")