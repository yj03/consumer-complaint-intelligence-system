import os
import numpy as np
import pandas as pd


print("Program started")

project_folder = (
    r"C:\Users\User\OneDrive\DS_Project"
    r"\ComplaintsIntelligent"
)

company_input_path = os.path.join(
    project_folder,
    "data",
    "full_monthly_company_counts.csv"
)

monthly_input_path = os.path.join(
    project_folder,
    "data",
    "full_monthly_complaint_counts.csv"
)

all_features_output = os.path.join(
    project_folder,
    "data",
    "full_company_risk_features.csv"
)

latest_features_output = os.path.join(
    project_folder,
    "data",
    "latest_company_risk_features.csv"
)


# ---------------------------------
# Load aggregated data
# ---------------------------------
print("Loading full company data...")

company_df = pd.read_csv(
    company_input_path,
    low_memory=False
)

monthly_df = pd.read_csv(
    monthly_input_path,
    low_memory=False
)

company_df["Month"] = pd.to_datetime(
    company_df["Month"],
    errors="coerce"
)

monthly_df["Month"] = pd.to_datetime(
    monthly_df["Month"],
    errors="coerce"
)

company_df = company_df.dropna(
    subset=[
        "Month",
        "Company"
    ]
)

monthly_df = monthly_df.dropna(
    subset=["Month"]
)

print("Company records:", len(company_df))
print("Companies:", company_df["Company"].nunique())


# ---------------------------------
# Select latest completed month
# ---------------------------------
current_month = (
    pd.Timestamp.today()
    .normalize()
    .replace(day=1)
)

completed_months = monthly_df[
    monthly_df["Month"] < current_month
]

if completed_months.empty:
    raise ValueError(
        "No completed month was found."
    )

analysis_month = completed_months[
    "Month"
].max()

analysis_month_total = int(
    monthly_df.loc[
        monthly_df["Month"] == analysis_month,
        "Complaint_count"
    ].iloc[0]
)

print("\nCurrent month:")
print(current_month)

print("\nAnalysis month:")
print(analysis_month)

print("\nTotal complaints in analysis month:")
print(f"{analysis_month_total:,}")


# ---------------------------------
# Choose currently relevant companies
# ---------------------------------
# Rank companies using the latest 12 completed months
ranking_start_month = (
    analysis_month
    - pd.DateOffset(months=11)
)

ranking_data = company_df[
    (
        company_df["Month"]
        >= ranking_start_month
    )
    &
    (
        company_df["Month"]
        <= analysis_month
    )
]

number_of_companies = 50

top_companies = (
    ranking_data
    .groupby("Company")[
        "Complaint_count"
    ]
    .sum()
    .nlargest(number_of_companies)
    .index
)

print("\nCompanies included:")
print(len(top_companies))


# ---------------------------------
# Prepare 13 months of history
# ---------------------------------
history_start_month = (
    analysis_month
    - pd.DateOffset(months=12)
)

history_df = company_df[
    (
        company_df["Month"]
        >= history_start_month
    )
    &
    (
        company_df["Month"]
        <= analysis_month
    )
    &
    (
        company_df["Company"]
        .isin(top_companies)
    )
].copy()


# Add missing company-month combinations
all_months = pd.date_range(
    start=history_start_month,
    end=analysis_month,
    freq="MS"
)

complete_index = pd.MultiIndex.from_product(
    [
        all_months,
        top_companies
    ],
    names=[
        "Month",
        "Company"
    ]
)

history_df = (
    history_df
    .set_index([
        "Month",
        "Company"
    ])
    .reindex(
        complete_index,
        fill_value=0
    )
    .reset_index()
)

history_df = history_df.sort_values(
    [
        "Company",
        "Month"
    ]
).reset_index(drop=True)


# ---------------------------------
# Current untimely-response rate
# ---------------------------------
history_df["Untimely_rate"] = np.where(
    history_df["Complaint_count"] > 0,
    (
        history_df["Untimely_count"]
        / history_df["Complaint_count"]
        * 100
    ),
    0
)


# ---------------------------------
# Previous six-month volume features
# ---------------------------------
history_df["Previous_6_month_average"] = (
    history_df
    .groupby("Company")[
        "Complaint_count"
    ]
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

history_df["Previous_6_month_std"] = (
    history_df
    .groupby("Company")[
        "Complaint_count"
    ]
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
# Complaint growth
# ---------------------------------
history_df["Complaint_growth_percentage"] = np.where(
    history_df["Previous_6_month_average"] > 0,
    (
        (
            history_df["Complaint_count"]
            - history_df["Previous_6_month_average"]
        )
        / history_df["Previous_6_month_average"]
        * 100
    ),
    np.nan
)


# ---------------------------------
# Complaint-volume Z-score
# ---------------------------------
history_df["Complaint_z_score"] = np.where(
    history_df["Previous_6_month_std"] > 0,
    (
        (
            history_df["Complaint_count"]
            - history_df["Previous_6_month_average"]
        )
        / history_df["Previous_6_month_std"]
    ),
    np.nan
)


# ---------------------------------
# Previous six-month untimely rate
# ---------------------------------
history_df[
    "Previous_6_month_complaints"
] = (
    history_df
    .groupby("Company")[
        "Complaint_count"
    ]
    .transform(
        lambda values:
        values.shift(1)
        .rolling(
            window=6,
            min_periods=3
        )
        .sum()
    )
)

history_df[
    "Previous_6_month_untimely"
] = (
    history_df
    .groupby("Company")[
        "Untimely_count"
    ]
    .transform(
        lambda values:
        values.shift(1)
        .rolling(
            window=6,
            min_periods=3
        )
        .sum()
    )
)

history_df[
    "Previous_6_month_untimely_rate"
] = np.where(
    history_df[
        "Previous_6_month_complaints"
    ] > 0,
    (
        history_df[
            "Previous_6_month_untimely"
        ]
        / history_df[
            "Previous_6_month_complaints"
        ]
        * 100
    ),
    0
)


# ---------------------------------
# Untimely-rate change
# ---------------------------------
history_df["Untimely_rate_change"] = (
    history_df["Untimely_rate"]
    - history_df[
        "Previous_6_month_untimely_rate"
    ]
)


# ---------------------------------
# Share of all complaints
# ---------------------------------
history_df["Monthly_complaint_share"] = (
    history_df["Complaint_count"]
    / analysis_month_total
    * 100
)


# Replace infinite values
numeric_columns_to_clean = [
    "Complaint_growth_percentage",
    "Complaint_z_score",
    "Untimely_rate_change"
]

for column in numeric_columns_to_clean:
    history_df[column] = (
        history_df[column]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )


# ---------------------------------
# Select analysis-month results
# ---------------------------------
latest_results = history_df[
    history_df["Month"] == analysis_month
].copy()

latest_results = latest_results.sort_values(
    [
        "Complaint_z_score",
        "Complaint_growth_percentage"
    ],
    ascending=False
)


display_columns = [
    "Company",
    "Complaint_count",
    "Previous_6_month_average",
    "Complaint_growth_percentage",
    "Complaint_z_score",
    "Untimely_count",
    "Untimely_rate",
    "Previous_6_month_untimely_rate",
    "Untimely_rate_change",
    "Monthly_complaint_share"
]


round_columns = [
    "Previous_6_month_average",
    "Complaint_growth_percentage",
    "Complaint_z_score",
    "Untimely_rate",
    "Previous_6_month_untimely_rate",
    "Untimely_rate_change",
    "Monthly_complaint_share"
]

latest_results[round_columns] = (
    latest_results[round_columns]
    .round(2)
)


print("\nTop companies by unusual complaint growth:")

print(
    latest_results[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ---------------------------------
# Save results
# ---------------------------------
history_df.to_csv(
    all_features_output,
    index=False
)

latest_results.to_csv(
    latest_features_output,
    index=False
)


print("\nAll features saved to:")
print(all_features_output)

print("\nLatest company features saved to:")
print(latest_features_output)

print("\nFull company risk feature analysis completed")