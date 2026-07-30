import os
import numpy as np
import pandas as pd


print("Program started")

project_folder = (
    r"C:\Users\User\OneDrive\DS_Project"
    r"\ComplaintsIntelligent"
)

input_path = os.path.join(
    project_folder,
    "data",
    "latest_company_risk_features.csv"
)

output_path = os.path.join(
    project_folder,
    "data",
    "company_risk_scores.csv"
)


# ---------------------------------
# Load company risk features
# ---------------------------------
print("Loading company risk features...")

df = pd.read_csv(
    input_path,
    low_memory=False
)

print("Rows loaded:", len(df))


# ---------------------------------
# Remove non-company placeholders
# ---------------------------------
excluded_names = [
    "Pending Company Match",
    "Unknown",
    "No Company Found",
    "Company Not Provided"
]

df = df[
    ~df["Company"].isin(excluded_names)
].copy()


# Require a reasonable number of complaints
minimum_complaints = 100

df = df[
    df["Complaint_count"] >= minimum_complaints
].copy()

print("Companies used for scoring:", len(df))


# ---------------------------------
# Convert columns to numeric
# ---------------------------------
numeric_columns = [
    "Complaint_count",
    "Complaint_growth_percentage",
    "Complaint_z_score",
    "Untimely_rate",
    "Previous_6_month_untimely_rate",
    "Monthly_complaint_share"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


# ---------------------------------
# Component 1:
# Complaint-volume anomaly
# ---------------------------------
# Negative Z-scores receive zero.
# A Z-score of 5 or above receives 100.

df["Volume_anomaly_score"] = (
    df["Complaint_z_score"]
    .clip(lower=0, upper=5)
    / 5
    * 100
)


# ---------------------------------
# Component 2:
# Complaint growth
# ---------------------------------
# Negative growth receives zero.
# Growth of 100% or above receives 100.

df["Growth_score"] = (
    df["Complaint_growth_percentage"]
    .clip(lower=0, upper=100)
)


# ---------------------------------
# Component 3:
# Complaint-volume share
# ---------------------------------
# Convert complaint share into a percentile
# among the companies being scored.

df["Complaint_share_score"] = (
    df["Monthly_complaint_share"]
    .rank(
        method="average",
        pct=True
    )
    * 100
)


# ---------------------------------
# Component 4:
# Untimely-response performance
# ---------------------------------
# Use whichever is worse:
# current month or previous six months.

df["Response_risk_rate"] = df[
    [
        "Untimely_rate",
        "Previous_6_month_untimely_rate"
    ]
].max(axis=1)


# An untimely rate of 5% or more receives 100.
df["Response_risk_score"] = (
    df["Response_risk_rate"]
    .clip(lower=0, upper=5)
    / 5
    * 100
)


# ---------------------------------
# Calculate total company risk score
# ---------------------------------
df["Company_risk_score"] = (
    df["Volume_anomaly_score"] * 0.35
    + df["Growth_score"] * 0.25
    + df["Complaint_share_score"] * 0.25
    + df["Response_risk_score"] * 0.15
)


df["Company_risk_score"] = (
    df["Company_risk_score"]
    .clip(lower=0, upper=100)
    .round(2)
)


# ---------------------------------
# Assign risk levels
# ---------------------------------
def assign_risk_level(score):

    if score >= 75:
        return "Critical"

    if score >= 50:
        return "High"

    if score >= 25:
        return "Moderate"

    return "Low"


df["Risk_level"] = (
    df["Company_risk_score"]
    .apply(assign_risk_level)
)


# ---------------------------------
# Create rank
# ---------------------------------
df = df.sort_values(
    "Company_risk_score",
    ascending=False
).reset_index(drop=True)

df["Risk_rank"] = (
    df.index + 1
)


# ---------------------------------
# Create explanation
# ---------------------------------
def create_reason(row):

    reasons = []

    if row["Complaint_z_score"] >= 3:
        reasons.append("unusual complaint-volume spike")

    elif row["Complaint_z_score"] >= 2:
        reasons.append("moderately unusual complaint volume")

    if row["Complaint_growth_percentage"] >= 50:
        reasons.append("rapid complaint growth")

    elif row["Complaint_growth_percentage"] >= 25:
        reasons.append("moderate complaint growth")

    # Use actual share, not percentile rank, for the explanation
    if row["Monthly_complaint_share"] >= 10:
        reasons.append("very large share of all complaints")

    elif row["Monthly_complaint_share"] >= 1:
        reasons.append("meaningful share of all complaints")

    if row["Response_risk_rate"] >= 5:
        reasons.append("very high untimely-response rate")

    elif row["Response_risk_rate"] >= 1:
        reasons.append("elevated untimely-response rate")

    if not reasons:
        reasons.append("multiple smaller monitoring indicators")

    return ", ".join(reasons)


df["Main_risk_reason"] = df.apply(
    create_reason,
    axis=1
)


# ---------------------------------
# Round output columns
# ---------------------------------
columns_to_round = [
    "Complaint_growth_percentage",
    "Complaint_z_score",
    "Monthly_complaint_share",
    "Volume_anomaly_score",
    "Growth_score",
    "Complaint_share_score",
    "Response_risk_rate",
    "Response_risk_score"
]

df[columns_to_round] = (
    df[columns_to_round]
    .round(2)
)


# ---------------------------------
# Display highest-risk companies
# ---------------------------------
display_columns = [
    "Risk_rank",
    "Company",
    "Company_risk_score",
    "Risk_level",
    "Complaint_count",
    "Complaint_growth_percentage",
    "Complaint_z_score",
    "Monthly_complaint_share",
    "Response_risk_rate",
    "Main_risk_reason"
]


print("\nHighest-risk companies:")

print(
    df[display_columns]
    .head(20)
    .to_string(index=False)
)


print("\nRisk-level distribution:")

print(
    df["Risk_level"]
    .value_counts()
    .reindex(
        [
            "Critical",
            "High",
            "Moderate",
            "Low"
        ],
        fill_value=0
    )
)


# ---------------------------------
# Save results
# ---------------------------------
df.to_csv(
    output_path,
    index=False
)


print("\nCompany risk scores saved to:")
print(output_path)

print("\nCompany risk scoring completed")