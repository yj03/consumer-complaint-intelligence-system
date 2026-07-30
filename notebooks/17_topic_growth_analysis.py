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