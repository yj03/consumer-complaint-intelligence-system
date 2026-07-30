from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_FOLDER / "data"

COMPANY_INPUT_PATH = (
    DATA_FOLDER / "full_monthly_company_counts.csv"
)

MONTHLY_INPUT_PATH = (
    DATA_FOLDER / "full_monthly_complaint_counts.csv"
)

ALL_FEATURES_OUTPUT = (
    DATA_FOLDER / "full_company_risk_features.csv"
)

LATEST_FEATURES_OUTPUT = (
    DATA_FOLDER / "latest_company_risk_features.csv"
)

RISK_SCORES_OUTPUT = (
    DATA_FOLDER / "company_risk_scores.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------
NUMBER_OF_COMPANIES = 50
MINIMUM_COMPLAINTS = 100

EXCLUDED_COMPANY_NAMES = {
    "Pending Company Match",
    "Unknown",
    "No Company Found",
    "Company Not Provided",
}


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Raise a clear error when required columns are missing."""
    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_text}"
        )


def load_aggregated_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the full monthly company and total complaint data."""
    print("\n" + "=" * 60)
    print("STEP 1: LOAD AGGREGATED COMPANY DATA")
    print("=" * 60)

    if not COMPANY_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Company input file was not found: "
            f"{COMPANY_INPUT_PATH}"
        )

    if not MONTHLY_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Monthly input file was not found: "
            f"{MONTHLY_INPUT_PATH}"
        )

    print("Loading full company data...")

    company_df = pd.read_csv(
        COMPANY_INPUT_PATH,
        low_memory=False,
    )

    monthly_df = pd.read_csv(
        MONTHLY_INPUT_PATH,
        low_memory=False,
    )

    validate_columns(
        company_df,
        {
            "Month",
            "Company",
            "Complaint_count",
            "Untimely_count",
        },
        "Company aggregation dataset",
    )

    validate_columns(
        monthly_df,
        {
            "Month",
            "Complaint_count",
        },
        "Monthly complaint dataset",
    )

    company_df["Month"] = pd.to_datetime(
        company_df["Month"],
        errors="coerce",
    )

    monthly_df["Month"] = pd.to_datetime(
        monthly_df["Month"],
        errors="coerce",
    )

    company_df = company_df.dropna(
        subset=[
            "Month",
            "Company",
        ]
    ).copy()

    monthly_df = monthly_df.dropna(
        subset=["Month"]
    ).copy()

    for column in [
        "Complaint_count",
        "Untimely_count",
    ]:
        company_df[column] = pd.to_numeric(
            company_df[column],
            errors="coerce",
        ).fillna(0)

    monthly_df["Complaint_count"] = (
        pd.to_numeric(
            monthly_df["Complaint_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    print("Company records:", len(company_df))
    print(
        "Companies:",
        company_df["Company"].nunique(),
    )

    return company_df, monthly_df


def find_analysis_month(
    monthly_df: pd.DataFrame,
) -> tuple[pd.Timestamp, int]:
    """Select the latest fully completed calendar month."""
    current_month = (
        pd.Timestamp.today()
        .normalize()
        .replace(day=1)
    )

    completed_months = monthly_df[
        monthly_df["Month"] < current_month
    ].copy()

    if completed_months.empty:
        raise ValueError(
            "No completed month was found."
        )

    analysis_month = (
        completed_months["Month"].max()
    )

    matching_totals = monthly_df.loc[
        monthly_df["Month"] == analysis_month,
        "Complaint_count",
    ]

    if matching_totals.empty:
        raise ValueError(
            "The complaint total for the analysis "
            "month was not found."
        )

    analysis_month_total = int(
        matching_totals.iloc[0]
    )

    if analysis_month_total <= 0:
        raise ValueError(
            "The complaint total for the analysis "
            "month must be greater than zero."
        )

    print("\nCurrent month:")
    print(current_month)

    print("\nAnalysis month:")
    print(analysis_month)

    print(
        "\nTotal complaints in analysis month:"
    )
    print(f"{analysis_month_total:,}")

    return (
        analysis_month,
        analysis_month_total,
    )


def build_company_risk_features(
    company_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create historical and latest-month company risk features."""
    print("\n" + "=" * 60)
    print("STEP 2: BUILD COMPANY RISK FEATURES")
    print("=" * 60)

    (
        analysis_month,
        analysis_month_total,
    ) = find_analysis_month(monthly_df)

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
    ].copy()

    top_companies = (
        ranking_data
        .groupby("Company")[
            "Complaint_count"
        ]
        .sum()
        .nlargest(
            NUMBER_OF_COMPANIES
        )
        .index
    )

    print("\nCompanies included:")
    print(len(top_companies))

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

    all_months = pd.date_range(
        start=history_start_month,
        end=analysis_month,
        freq="MS",
    )

    complete_index = (
        pd.MultiIndex.from_product(
            [
                all_months,
                top_companies,
            ],
            names=[
                "Month",
                "Company",
            ],
        )
    )

    history_df = (
        history_df
        .set_index(
            [
                "Month",
                "Company",
            ]
        )
        .reindex(
            complete_index,
            fill_value=0,
        )
        .reset_index()
        .sort_values(
            [
                "Company",
                "Month",
            ]
        )
        .reset_index(drop=True)
    )

    history_df["Untimely_rate"] = (
        np.where(
            history_df[
                "Complaint_count"
            ] > 0,
            (
                history_df[
                    "Untimely_count"
                ]
                / history_df[
                    "Complaint_count"
                ]
                * 100
            ),
            0,
        )
    )

    history_df[
        "Previous_6_month_average"
    ] = (
        history_df
        .groupby("Company")[
            "Complaint_count"
        ]
        .transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    window=6,
                    min_periods=3,
                )
                .mean()
            )
        )
    )

    history_df[
        "Previous_6_month_std"
    ] = (
        history_df
        .groupby("Company")[
            "Complaint_count"
        ]
        .transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    window=6,
                    min_periods=3,
                )
                .std()
            )
        )
    )

    history_df[
        "Complaint_growth_percentage"
    ] = np.where(
        history_df[
            "Previous_6_month_average"
        ] > 0,
        (
            (
                history_df[
                    "Complaint_count"
                ]
                - history_df[
                    "Previous_6_month_average"
                ]
            )
            / history_df[
                "Previous_6_month_average"
            ]
            * 100
        ),
        np.nan,
    )

    history_df[
        "Complaint_z_score"
    ] = np.where(
        history_df[
            "Previous_6_month_std"
        ] > 0,
        (
            (
                history_df[
                    "Complaint_count"
                ]
                - history_df[
                    "Previous_6_month_average"
                ]
            )
            / history_df[
                "Previous_6_month_std"
            ]
        ),
        np.nan,
    )

    history_df[
        "Previous_6_month_complaints"
    ] = (
        history_df
        .groupby("Company")[
            "Complaint_count"
        ]
        .transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    window=6,
                    min_periods=3,
                )
                .sum()
            )
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
            lambda values: (
                values.shift(1)
                .rolling(
                    window=6,
                    min_periods=3,
                )
                .sum()
            )
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
        0,
    )

    history_df[
        "Untimely_rate_change"
    ] = (
        history_df["Untimely_rate"]
        - history_df[
            "Previous_6_month_untimely_rate"
        ]
    )

    history_df[
        "Monthly_complaint_share"
    ] = (
        history_df["Complaint_count"]
        / analysis_month_total
        * 100
    )

    numeric_columns_to_clean = [
        "Complaint_growth_percentage",
        "Complaint_z_score",
        "Untimely_rate_change",
    ]

    for column in numeric_columns_to_clean:
        history_df[column] = (
            history_df[column]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

    latest_results = history_df[
        history_df["Month"]
        == analysis_month
    ].copy()

    latest_results = (
        latest_results.sort_values(
            [
                "Complaint_z_score",
                "Complaint_growth_percentage",
            ],
            ascending=False,
        )
    )

    round_columns = [
        "Previous_6_month_average",
        "Complaint_growth_percentage",
        "Complaint_z_score",
        "Untimely_rate",
        "Previous_6_month_untimely_rate",
        "Untimely_rate_change",
        "Monthly_complaint_share",
    ]

    latest_results[round_columns] = (
        latest_results[round_columns]
        .round(2)
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
        "Monthly_complaint_share",
    ]

    print(
        "\nTop companies by unusual "
        "complaint growth:"
    )

    print(
        latest_results[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    DATA_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_df.to_csv(
        ALL_FEATURES_OUTPUT,
        index=False,
    )

    latest_results.to_csv(
        LATEST_FEATURES_OUTPUT,
        index=False,
    )

    print("\nAll features saved to:")
    print(ALL_FEATURES_OUTPUT)

    print(
        "\nLatest company features "
        "saved to:"
    )
    print(LATEST_FEATURES_OUTPUT)

    return history_df, latest_results


def assign_risk_level(
    score: float,
) -> str:
    """Convert a numerical score into a risk category."""
    if score >= 75:
        return "Critical"

    if score >= 50:
        return "High"

    if score >= 25:
        return "Moderate"

    return "Low"


def create_risk_reason(
    row: pd.Series,
) -> str:
    """Create a readable explanation for a company risk score."""
    reasons = []

    if row["Complaint_z_score"] >= 3:
        reasons.append(
            "unusual complaint-volume spike"
        )
    elif row["Complaint_z_score"] >= 2:
        reasons.append(
            "moderately unusual complaint volume"
        )

    if (
        row[
            "Complaint_growth_percentage"
        ]
        >= 50
    ):
        reasons.append(
            "rapid complaint growth"
        )
    elif (
        row[
            "Complaint_growth_percentage"
        ]
        >= 25
    ):
        reasons.append(
            "moderate complaint growth"
        )

    if (
        row[
            "Monthly_complaint_share"
        ]
        >= 10
    ):
        reasons.append(
            "very large share of all complaints"
        )
    elif (
        row[
            "Monthly_complaint_share"
        ]
        >= 1
    ):
        reasons.append(
            "meaningful share of all complaints"
        )

    if row["Response_risk_rate"] >= 5:
        reasons.append(
            "very high untimely-response rate"
        )
    elif row["Response_risk_rate"] >= 1:
        reasons.append(
            "elevated untimely-response rate"
        )

    if not reasons:
        reasons.append(
            "multiple smaller monitoring indicators"
        )

    return ", ".join(reasons)


def score_company_risk(
    latest_features: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the final 0-100 company risk scores."""
    print("\n" + "=" * 60)
    print("STEP 3: CALCULATE COMPANY RISK SCORES")
    print("=" * 60)

    risk_df = latest_features.copy()

    risk_df = risk_df[
        ~risk_df["Company"].isin(
            EXCLUDED_COMPANY_NAMES
        )
    ].copy()

    risk_df = risk_df[
        risk_df["Complaint_count"]
        >= MINIMUM_COMPLAINTS
    ].copy()

    print(
        "Companies used for scoring:",
        len(risk_df),
    )

    numeric_columns = [
        "Complaint_count",
        "Complaint_growth_percentage",
        "Complaint_z_score",
        "Untimely_rate",
        "Previous_6_month_untimely_rate",
        "Monthly_complaint_share",
    ]

    for column in numeric_columns:
        risk_df[column] = (
            pd.to_numeric(
                risk_df[column],
                errors="coerce",
            )
            .fillna(0)
        )

    risk_df[
        "Volume_anomaly_score"
    ] = (
        risk_df["Complaint_z_score"]
        .clip(
            lower=0,
            upper=5,
        )
        / 5
        * 100
    )

    risk_df["Growth_score"] = (
        risk_df[
            "Complaint_growth_percentage"
        ]
        .clip(
            lower=0,
            upper=100,
        )
    )

    risk_df[
        "Complaint_share_score"
    ] = (
        risk_df[
            "Monthly_complaint_share"
        ]
        .rank(
            method="average",
            pct=True,
        )
        * 100
    )

    risk_df[
        "Response_risk_rate"
    ] = (
        risk_df[
            [
                "Untimely_rate",
                "Previous_6_month_untimely_rate",
            ]
        ]
        .max(axis=1)
    )

    risk_df[
        "Response_risk_score"
    ] = (
        risk_df["Response_risk_rate"]
        .clip(
            lower=0,
            upper=5,
        )
        / 5
        * 100
    )

    risk_df[
        "Company_risk_score"
    ] = (
        risk_df[
            "Volume_anomaly_score"
        ]
        * 0.35
        + risk_df["Growth_score"]
        * 0.25
        + risk_df[
            "Complaint_share_score"
        ]
        * 0.25
        + risk_df[
            "Response_risk_score"
        ]
        * 0.15
    )

    risk_df[
        "Company_risk_score"
    ] = (
        risk_df["Company_risk_score"]
        .clip(
            lower=0,
            upper=100,
        )
        .round(2)
    )

    risk_df["Risk_level"] = (
        risk_df["Company_risk_score"]
        .apply(assign_risk_level)
    )

    risk_df = (
        risk_df.sort_values(
            "Company_risk_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    risk_df["Risk_rank"] = (
        risk_df.index + 1
    )

    risk_df["Main_risk_reason"] = (
        risk_df.apply(
            create_risk_reason,
            axis=1,
        )
    )

    columns_to_round = [
        "Complaint_growth_percentage",
        "Complaint_z_score",
        "Monthly_complaint_share",
        "Volume_anomaly_score",
        "Growth_score",
        "Complaint_share_score",
        "Response_risk_rate",
        "Response_risk_score",
    ]

    risk_df[columns_to_round] = (
        risk_df[columns_to_round]
        .round(2)
    )

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
        "Main_risk_reason",
    ]

    print("\nHighest-risk companies:")

    print(
        risk_df[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nRisk-level distribution:")

    print(
        risk_df["Risk_level"]
        .value_counts()
        .reindex(
            [
                "Critical",
                "High",
                "Moderate",
                "Low",
            ],
            fill_value=0,
        )
    )

    risk_df.to_csv(
        RISK_SCORES_OUTPUT,
        index=False,
    )

    print(
        "\nCompany risk scores saved to:"
    )
    print(RISK_SCORES_OUTPUT)

    return risk_df


def main() -> None:
    """Run the complete company risk pipeline."""
    print("Company risk pipeline started")
    print("Project folder:", PROJECT_FOLDER)

    company_df, monthly_df = (
        load_aggregated_data()
    )

    _, latest_features = (
        build_company_risk_features(
            company_df,
            monthly_df,
        )
    )

    score_company_risk(
        latest_features
    )

    print("\n" + "=" * 60)
    print(
        "COMPANY RISK PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()