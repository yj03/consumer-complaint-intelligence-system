import os
import pandas as pd


print("Program started")

project_folder = (
    r"C:\Users\User\OneDrive\DS_Project"
    r"\ComplaintsIntelligent"
)

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