import os
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------
# Page settings
# ---------------------------------
st.set_page_config(
    page_title="Consumer Complaint Intelligence",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------
# File paths
# ---------------------------------
project_folder = (
    r"C:\Users\User\OneDrive\DS_Project"
    r"\ComplaintsIntelligent"
)

risk_score_path = os.path.join(
    project_folder,
    "data",
    "company_risk_scores.csv"
)

monthly_counts_path = os.path.join(
    project_folder,
    "data",
    "full_monthly_complaint_counts.csv"
)


# ---------------------------------
# Load data
# ---------------------------------
@st.cache_data
def load_data():
    risk_df = pd.read_csv(
        risk_score_path,
        low_memory=False
    )

    monthly_df = pd.read_csv(
        monthly_counts_path,
        low_memory=False
    )

    monthly_df["Month"] = pd.to_datetime(
        monthly_df["Month"],
        errors="coerce"
    )

    return risk_df, monthly_df


try:
    risk_df, monthly_df = load_data()

except FileNotFoundError as error:
    st.error(
        "A required data file could not be found."
    )
    st.code(str(error))
    st.stop()


# ---------------------------------
# Dashboard title
# ---------------------------------
st.title("Consumer Complaint Intelligence System")

st.caption(
    "Monitor complaint trends, company-level risk indicators "
    "and unusual complaint-volume changes."
)

st.info(
    "Risk scores are monitoring indicators for prioritizing "
    "further review. They are not proof of wrongdoing."
)


# ---------------------------------
# Sidebar filters
# ---------------------------------
st.sidebar.header("Dashboard Filters")

available_risk_levels = [
    level
    for level in [
        "Critical",
        "High",
        "Moderate",
        "Low"
    ]
    if level in risk_df["Risk_level"].unique()
]

selected_risk_levels = st.sidebar.multiselect(
    "Risk level",
    options=available_risk_levels,
    default=available_risk_levels
)

minimum_score = st.sidebar.slider(
    "Minimum risk score",
    min_value=0,
    max_value=100,
    value=0,
    step=5
)


filtered_df = risk_df[
    (
        risk_df["Risk_level"]
        .isin(selected_risk_levels)
    )
    &
    (
        risk_df["Company_risk_score"]
        >= minimum_score
    )
].copy()


# ---------------------------------
# Summary metrics
# ---------------------------------
st.subheader("Risk Overview")

critical_count = (
    risk_df["Risk_level"]
    .eq("Critical")
    .sum()
)

high_count = (
    risk_df["Risk_level"]
    .eq("High")
    .sum()
)

highest_risk_score = (
    risk_df["Company_risk_score"]
    .max()
)

highest_risk_company = (
    risk_df.sort_values(
        "Company_risk_score",
        ascending=False
    )
    .iloc[0]["Company"]
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Companies monitored",
    f"{len(risk_df):,}"
)

metric_2.metric(
    "Critical risk",
    f"{critical_count:,}"
)

metric_3.metric(
    "High risk",
    f"{high_count:,}"
)

metric_4.metric(
    "Highest risk score",
    f"{highest_risk_score:.1f}"
)

st.caption(
    f"Highest-ranked company: {highest_risk_company}"
)


# ---------------------------------
# Risk score chart
# ---------------------------------
st.subheader("Highest-Risk Companies")

top_n = st.slider(
    "Number of companies to display",
    min_value=5,
    max_value=min(30, len(filtered_df)),
    value=min(15, len(filtered_df))
)

chart_df = (
    filtered_df
    .sort_values(
        "Company_risk_score",
        ascending=False
    )
    .head(top_n)
    .sort_values(
        "Company_risk_score",
        ascending=True
    )
)

risk_chart = px.bar(
    chart_df,
    x="Company_risk_score",
    y="Company",
    orientation="h",
    color="Risk_level",
    title="Company Risk Scores",
    labels={
        "Company_risk_score": "Risk score",
        "Company": "Company",
        "Risk_level": "Risk level"
    },
    hover_data={
        "Complaint_count": ":,",
        "Complaint_growth_percentage": ":.2f",
        "Complaint_z_score": ":.2f",
        "Main_risk_reason": True
    },
    category_orders={
        "Risk_level": [
            "Critical",
            "High",
            "Moderate",
            "Low"
        ]
    }
)

risk_chart.update_layout(
    height=550,
    yaxis_title=None
)

st.plotly_chart(
    risk_chart,
    use_container_width=True
)


# ---------------------------------
# Risk-level distribution
# ---------------------------------
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Risk-Level Distribution")

    risk_distribution = (
        risk_df["Risk_level"]
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
        .reset_index()
    )

    risk_distribution.columns = [
        "Risk level",
        "Company count"
    ]

    distribution_chart = px.pie(
        risk_distribution,
        names="Risk level",
        values="Company count",
        hole=0.45
    )

    st.plotly_chart(
        distribution_chart,
        use_container_width=True
    )


# ---------------------------------
# Monthly complaint trend
# ---------------------------------
with right_column:
    st.subheader("Monthly Complaint Trend")

    completed_monthly_df = monthly_df.copy()

    current_month = (
        pd.Timestamp.today()
        .normalize()
        .replace(day=1)
    )

    completed_monthly_df = completed_monthly_df[
        completed_monthly_df["Month"] < current_month
    ]

    trend_chart = px.line(
        completed_monthly_df.tail(24),
        x="Month",
        y="Complaint_count",
        markers=True,
        labels={
            "Month": "Month",
            "Complaint_count": "Complaints"
        }
    )

    st.plotly_chart(
        trend_chart,
        use_container_width=True
    )


# ---------------------------------
# Company details table
# ---------------------------------
st.subheader("Company Risk Details")

table_columns = [
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

display_df = (
    filtered_df[table_columns]
    .sort_values(
        "Company_risk_score",
        ascending=False
    )
    .copy()
)

display_df = display_df.rename(
    columns={
        "Risk_rank": "Rank",
        "Company_risk_score": "Risk score",
        "Risk_level": "Risk level",
        "Complaint_count": "Complaints",
        "Complaint_growth_percentage": "Growth (%)",
        "Complaint_z_score": "Volume Z-score",
        "Monthly_complaint_share": "Complaint share (%)",
        "Response_risk_rate": "Untimely rate (%)",
        "Main_risk_reason": "Main risk reason"
    }
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Risk score": st.column_config.ProgressColumn(
            "Risk score",
            min_value=0,
            max_value=100,
            format="%.2f"
        ),
        "Complaints": st.column_config.NumberColumn(
            "Complaints",
            format="%d"
        ),
        "Growth (%)": st.column_config.NumberColumn(
            "Growth (%)",
            format="%.2f%%"
        ),
        "Volume Z-score": st.column_config.NumberColumn(
            "Volume Z-score",
            format="%.2f"
        ),
        "Complaint share (%)": st.column_config.NumberColumn(
            "Complaint share (%)",
            format="%.2f%%"
        ),
        "Untimely rate (%)": st.column_config.NumberColumn(
            "Untimely rate (%)",
            format="%.2f%%"
        )
    }
)