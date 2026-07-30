from pathlib import Path
import os
import pandas as pd


print("Program started")

project_folder = Path(__file__).resolve().parents[1]

input_path = os.path.join(
    project_folder,
    "data",
    "complaints.csv"
)

company_output_path = os.path.join(
    project_folder,
    "data",
    "full_monthly_company_counts.csv"
)

monthly_output_path = os.path.join(
    project_folder,
    "data",
    "full_monthly_complaint_counts.csv"
)


# Only load the columns required for company risk analysis
columns_to_load = [
    "Date received",
    "Company",
    "Timely response?"
]

chunk_size = 500_000

company_results = []
monthly_results = []

total_rows_processed = 0
chunk_number = 0


print("Reading full dataset in chunks...")


for chunk in pd.read_csv(
    input_path,
    usecols=columns_to_load,
    chunksize=chunk_size,
    low_memory=False
):
    chunk_number += 1
    total_rows_processed += len(chunk)

    print(
        f"Processing chunk {chunk_number} | "
        f"Total rows processed: {total_rows_processed:,}"
    )

    # Convert date
    chunk["Date received"] = pd.to_datetime(
        chunk["Date received"],
        errors="coerce",
        utc=True
    )

    # Remove rows without date or company
    chunk = chunk.dropna(
        subset=[
            "Date received",
            "Company"
        ]
    ).copy()

    # Convert date into month
    chunk["Month"] = (
        chunk["Date received"]
        .dt.tz_convert(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # No means the company response was untimely
    chunk["Untimely"] = (
        chunk["Timely response?"]
        .map({
            "Yes": 0,
            "No": 1
        })
        .fillna(0)
        .astype("int8")
    )

    # Aggregate by month and company
    company_chunk = (
        chunk.groupby(
            [
                "Month",
                "Company"
            ]
        )
        .agg(
            Complaint_count=(
                "Company",
                "size"
            ),
            Untimely_count=(
                "Untimely",
                "sum"
            )
        )
        .reset_index()
    )

    company_results.append(company_chunk)

    # Aggregate total complaints per month
    monthly_chunk = (
        chunk.groupby("Month")
        .size()
        .reset_index(
            name="Complaint_count"
        )
    )

    monthly_results.append(monthly_chunk)


print("\nCombining chunk results...")


# Combine company results from all chunks
full_company = pd.concat(
    company_results,
    ignore_index=True
)

full_company = (
    full_company.groupby(
        [
            "Month",
            "Company"
        ],
        as_index=False
    )
    .agg(
        Complaint_count=(
            "Complaint_count",
            "sum"
        ),
        Untimely_count=(
            "Untimely_count",
            "sum"
        )
    )
)


# Calculate untimely-response rate
full_company["Untimely_rate"] = (
    full_company["Untimely_count"]
    / full_company["Complaint_count"]
    * 100
)


# Combine total monthly results
full_monthly = pd.concat(
    monthly_results,
    ignore_index=True
)

full_monthly = (
    full_monthly.groupby(
        "Month",
        as_index=False
    )["Complaint_count"]
    .sum()
    .sort_values("Month")
)


# Sort company results
full_company = full_company.sort_values(
    [
        "Month",
        "Complaint_count"
    ],
    ascending=[
        True,
        False
    ]
)


# Save results
full_company.to_csv(
    company_output_path,
    index=False
)

full_monthly.to_csv(
    monthly_output_path,
    index=False
)


print("\nAggregation completed successfully")

print("\nTotal rows processed:")
print(f"{total_rows_processed:,}")

print("\nNumber of companies:")
print(full_company["Company"].nunique())

print("\nEarliest month:")
print(full_monthly["Month"].min())

print("\nLatest month:")
print(full_monthly["Month"].max())

print("\nLatest 12 monthly totals:")
print(
    full_monthly
    .tail(12)
    .to_string(index=False)
)

print("\nCompany results saved to:")
print(company_output_path)

print("\nMonthly totals saved to:")
print(monthly_output_path)