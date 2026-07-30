from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Emerging Complaint Topics",
    page_icon="📈",
    layout="wide"
)


# --------------------------------------------------
# File paths
# --------------------------------------------------
project_folder = Path(__file__).resolve().parents[2]

topic_growth_path = os.path.join(
    project_folder,
    "data",
    "monthly_topic_growth.csv"
)

named_topics_path = os.path.join(
    project_folder,
    "data",
    "complaints_with_named_topics.csv"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_topic_data():

    growth_df = pd.read_csv(
        topic_growth_path,
        low_memory=False
    )

    complaints_df = pd.read_csv(
        named_topics_path,
        low_memory=False
    )

    growth_df["Month"] = pd.to_datetime(
        growth_df["Month"],
        errors="coerce"
    )

    complaints_df["Date received"] = pd.to_datetime(
        complaints_df["Date received"],
        errors="coerce",
        utc=True
    )

    return growth_df, complaints_df


try:
    growth_df, complaints_df = load_topic_data()

except FileNotFoundError as error:
    st.error(
        "One or more topic-analysis files could not be found."
    )
    st.code(str(error))
    st.stop()

except Exception as error:
    st.error(
        "The topic data could not be loaded."
    )
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# Clean data
# --------------------------------------------------
growth_df = growth_df.dropna(
    subset=[
        "Month",
        "Topic family"
    ]
).copy()

numeric_columns = [
    "Complaint count",
    "Previous 6-month average",
    "Growth percentage",
    "Z-score"
]

for column in numeric_columns:
    growth_df[column] = pd.to_numeric(
        growth_df[column],
        errors="coerce"
    )


# --------------------------------------------------
# Select latest usable completed month
# --------------------------------------------------
current_month = (
    pd.Timestamp.today()
    .normalize()
    .replace(day=1)
)

completed_growth = growth_df[
    growth_df["Month"] < current_month
].copy()

monthly_totals = (
    completed_growth
    .groupby("Month")["Complaint count"]
    .sum()
    .sort_index()
)

median_monthly_total = monthly_totals.median()

minimum_required_count = max(
    20,
    int(median_monthly_total * 0.25)
)

usable_months = monthly_totals[
    monthly_totals >= minimum_required_count
].index

if len(usable_months) == 0:
    st.error(
        "No month contains enough topic data for analysis."
    )
    st.stop()

latest_month = usable_months.max()

latest_df = completed_growth[
    completed_growth["Month"] == latest_month
].copy()


# --------------------------------------------------
# Page title
# --------------------------------------------------
st.title("Emerging Complaint Topics")

st.write(
    "Monitor complaint themes that are increasing compared "
    "with their previous six-month average."
)

st.info(
    "Topic trends use complaints containing consumer narratives. "
    "They do not include every complaint in the full CFPB dataset."
)

st.caption(
    f"Latest usable month: {latest_month.strftime('%B %Y')}"
)


# --------------------------------------------------
# Summary metrics
# --------------------------------------------------
emerging_statuses = [
    "Moderate growth",
    "High emerging topic",
    "Critical emerging topic"
]

emerging_count = (
    latest_df["Topic status"]
    .isin(emerging_statuses)
    .sum()
)

critical_count = (
    latest_df["Topic status"]
    .eq("Critical emerging topic")
    .sum()
)

highest_growth_row = (
    latest_df
    .dropna(subset=["Growth percentage"])
    .sort_values(
        "Growth percentage",
        ascending=False
    )
)

if not highest_growth_row.empty:
    highest_growth_topic = (
        highest_growth_row.iloc[0]["Topic family"]
    )

    highest_growth_value = (
        highest_growth_row.iloc[0]["Growth percentage"]
    )

else:
    highest_growth_topic = "Not available"
    highest_growth_value = 0


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Topic families monitored",
    latest_df["Topic family"].nunique()
)

metric_2.metric(
    "Emerging topics",
    int(emerging_count)
)

metric_3.metric(
    "Critical topics",
    int(critical_count)
)

metric_4.metric(
    "Highest growth",
    f"{highest_growth_value:.1f}%"
)

st.caption(
    f"Fastest-growing topic: {highest_growth_topic}"
)


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Topic Filters")

all_topics = sorted(
    growth_df["Topic family"]
    .dropna()
    .unique()
)

selected_topics = st.sidebar.multiselect(
    "Topic families",
    options=all_topics,
    default=all_topics
)

available_statuses = [
    status
    for status in [
        "Critical emerging topic",
        "High emerging topic",
        "Moderate growth",
        "Stable",
        "Decreasing",
        "Low volume",
        "Insufficient history"
    ]
    if status in growth_df["Topic status"].unique()
]

selected_statuses = st.sidebar.multiselect(
    "Topic status",
    options=available_statuses,
    default=available_statuses
)


filtered_latest = latest_df[
    latest_df["Topic family"].isin(
        selected_topics
    )
    &
    latest_df["Topic status"].isin(
        selected_statuses
    )
].copy()


# --------------------------------------------------
# Latest topic growth chart
# --------------------------------------------------
st.subheader("Latest Topic Growth")

chart_df = (
    filtered_latest
    .dropna(subset=["Growth percentage"])
    .sort_values(
        "Growth percentage",
        ascending=True
    )
)

if chart_df.empty:
    st.warning(
        "No topics match the selected filters."
    )

else:
    growth_chart = px.bar(
        chart_df,
        x="Growth percentage",
        y="Topic family",
        orientation="h",
        color="Topic status",
        title=(
            "Topic Growth Compared with Previous "
            "Six-Month Average"
        ),
        labels={
            "Growth percentage": "Growth (%)",
            "Topic family": "Topic",
            "Topic status": "Status"
        },
        hover_data={
            "Complaint count": True,
            "Previous 6-month average": ":.2f",
            "Z-score": ":.2f"
        }
    )

    growth_chart.add_vline(
        x=0,
        line_dash="dash"
    )

    growth_chart.update_layout(
        height=500,
        yaxis_title=None
    )

    st.plotly_chart(
        growth_chart,
        use_container_width=True
    )


# --------------------------------------------------
# Topic trend over time
# --------------------------------------------------
st.subheader("Topic Trends Over Time")

selected_trend_topics = st.multiselect(
    "Select topics to compare",
    options=all_topics,
    default=all_topics[:3]
)

trend_df = completed_growth[
    completed_growth["Topic family"].isin(
        selected_trend_topics
    )
].copy()

if trend_df.empty:
    st.warning(
        "Select at least one topic to display the trend."
    )

else:
    trend_chart = px.line(
        trend_df,
        x="Month",
        y="Complaint count",
        color="Topic family",
        markers=True,
        labels={
            "Month": "Month",
            "Complaint count": "Complaints",
            "Topic family": "Topic"
        }
    )

    trend_chart.update_layout(
        height=520
    )

    st.plotly_chart(
        trend_chart,
        use_container_width=True
    )


# --------------------------------------------------
# Emerging-topic table
# --------------------------------------------------
st.subheader("Latest Topic Details")

table_columns = [
    "Topic family",
    "Complaint count",
    "Previous 6-month average",
    "Growth percentage",
    "Z-score",
    "Topic status"
]

display_df = (
    filtered_latest[table_columns]
    .sort_values(
        "Growth percentage",
        ascending=False
    )
    .copy()
)

display_df = display_df.rename(
    columns={
        "Topic family": "Topic",
        "Complaint count": "Complaints",
        "Previous 6-month average":
            "Previous 6-month average",
        "Growth percentage": "Growth (%)",
        "Z-score": "Z-score",
        "Topic status": "Status"
    }
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Complaints":
            st.column_config.NumberColumn(
                "Complaints",
                format="%d"
            ),

        "Previous 6-month average":
            st.column_config.NumberColumn(
                "Previous 6-month average",
                format="%.2f"
            ),

        "Growth (%)":
            st.column_config.NumberColumn(
                "Growth (%)",
                format="%.2f%%"
            ),

        "Z-score":
            st.column_config.NumberColumn(
                "Z-score",
                format="%.2f"
            )
    }
)


# --------------------------------------------------
# Topic distribution
# --------------------------------------------------
st.subheader("Overall Topic Distribution")

topic_distribution = (
    complaints_df["Topic family"]
    .value_counts()
    .reset_index()
)

topic_distribution.columns = [
    "Topic",
    "Complaint count"
]

distribution_chart = px.bar(
    topic_distribution.sort_values(
        "Complaint count",
        ascending=True
    ),
    x="Complaint count",
    y="Topic",
    orientation="h",
    labels={
        "Complaint count": "Complaints",
        "Topic": "Topic"
    }
)

distribution_chart.update_layout(
    height=450,
    yaxis_title=None
)

st.plotly_chart(
    distribution_chart,
    use_container_width=True
)


# --------------------------------------------------
# Topic interpretation
# --------------------------------------------------
with st.expander(
    "How are emerging topics detected?"
):
    st.write(
        """
        Each topic's current complaint count is compared with
        its previous six-month average.

        - Moderate growth: at least 25% growth or a Z-score of 1.
        - High emerging topic: at least 50% growth or a Z-score of 2.
        - Critical emerging topic: at least 100% growth or a Z-score of 3.
        - Low-volume topics are not treated as emerging alerts.
        """
    )