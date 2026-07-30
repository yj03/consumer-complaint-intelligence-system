import os

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Company Details",
    page_icon="🏢",
    layout="wide"
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------
project_folder = (
    r"C:\Users\User\OneDrive\DS_Project"
    r"\ComplaintsIntelligent"
)

risk_score_path = os.path.join(
    project_folder,
    "data",
    "company_risk_scores.csv"
)

history_path = os.path.join(
    project_folder,
    "data",
    "full_company_risk_features.csv"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_company_data():
    risk_df = pd.read_csv(
        risk_score_path,
        low_memory=False
    )

    history_df = pd.read_csv(
        history_path,
        low_memory=False
    )

    risk_df["Month"] = pd.to_datetime(
        risk_df["Month"],
        errors="coerce"
    )

    history_df["Month"] = pd.to_datetime(
        history_df["Month"],
        errors="coerce"
    )

    return risk_df, history_df


try:
    risk_df, history_df = load_company_data()

except FileNotFoundError as error:
    st.error(
        "One or more company risk files could not be found."
    )
    st.code(str(error))
    st.stop()

except Exception as error:
    st.error(
        "The company data could not be loaded."
    )
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# Page heading
# --------------------------------------------------
st.title("Company Risk Details")

st.write(
    "Select a company to review its complaint activity, "
    "risk indicators and historical trends."
)

st.info(
    "Risk scores are monitoring indicators for prioritizing "
    "further review. They are not proof of wrongdoing."
)


# --------------------------------------------------
# Company selector
# --------------------------------------------------
company_options = (
    risk_df
    .sort_values(
        "Company_risk_score",
        ascending=False
    )["Company"]
    .dropna()
    .unique()
    .tolist()
)

selected_company = st.selectbox(
    "Select a company",
    options=company_options
)


# --------------------------------------------------
# Get selected company data
# --------------------------------------------------
company_score_rows = risk_df[
    risk_df["Company"] == selected_company
].copy()

company_history = history_df[
    history_df["Company"] == selected_company
].copy()

company_history = company_history.sort_values(
    "Month"
)

if company_score_rows.empty:
    st.warning(
        "No risk-score information is available "
        "for the selected company."
    )
    st.stop()

company_score = company_score_rows.iloc[0]


# --------------------------------------------------
# Company title
# --------------------------------------------------
st.subheader(selected_company)

analysis_month = company_score.get(
    "Month",
    pd.NaT
)

if pd.notna(analysis_month):
    st.caption(
        "Analysis month: "
        f"{analysis_month.strftime('%B %Y')}"
    )


# --------------------------------------------------
# Main metrics
# --------------------------------------------------
metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Risk score",
    f"{company_score['Company_risk_score']:.1f}/100"
)

metric_2.metric(
    "Risk level",
    str(company_score["Risk_level"])
)

metric_3.metric(
    "Complaint count",
    f"{int(company_score['Complaint_count']):,}"
)

metric_4.metric(
    "Risk rank",
    f"#{int(company_score['Risk_rank'])}"
)


metric_5, metric_6, metric_7, metric_8 = st.columns(4)

metric_5.metric(
    "Complaint growth",
    f"{company_score['Complaint_growth_percentage']:.1f}%"
)

metric_6.metric(
    "Volume Z-score",
    f"{company_score['Complaint_z_score']:.2f}"
)

metric_7.metric(
    "Complaint share",
    f"{company_score['Monthly_complaint_share']:.2f}%"
)

metric_8.metric(
    "Untimely-response rate",
    f"{company_score['Response_risk_rate']:.2f}%"
)


# --------------------------------------------------
# Risk reason
# --------------------------------------------------
st.subheader("Main Risk Explanation")

risk_reason = company_score.get(
    "Main_risk_reason",
    "No explanation available."
)

if company_score["Risk_level"] == "Critical":
    st.error(risk_reason)

elif company_score["Risk_level"] == "High":
    st.warning(risk_reason)

else:
    st.info(risk_reason)


# --------------------------------------------------
# Risk-score components
# --------------------------------------------------
st.subheader("Risk-Score Breakdown")

component_data = pd.DataFrame({
    "Risk component": [
        "Volume anomaly",
        "Complaint growth",
        "Complaint share",
        "Response risk"
    ],
    "Component score": [
        company_score.get(
            "Volume_anomaly_score",
            0
        ),
        company_score.get(
            "Growth_score",
            0
        ),
        company_score.get(
            "Complaint_share_score",
            0
        ),
        company_score.get(
            "Response_risk_score",
            0
        )
    ],
    "Weight": [
        "35%",
        "25%",
        "25%",
        "15%"
    ]
})

component_chart = px.bar(
    component_data,
    x="Risk component",
    y="Component score",
    text="Weight",
    labels={
        "Risk component": "Risk component",
        "Component score": "Score"
    },
    title="Risk Components Before Weighting"
)

component_chart.update_traces(
    textposition="outside"
)

component_chart.update_layout(
    yaxis_range=[0, 110],
    height=430
)

st.plotly_chart(
    component_chart,
    use_container_width=True
)


# --------------------------------------------------
# Monthly complaint history
# --------------------------------------------------
st.subheader("Monthly Complaint Volume")

if company_history.empty:
    st.warning(
        "No monthly history is available "
        "for this company."
    )

else:
    complaint_chart = px.line(
        company_history,
        x="Month",
        y="Complaint_count",
        markers=True,
        labels={
            "Month": "Month",
            "Complaint_count": "Complaints"
        },
        title=(
            f"Monthly Complaints for "
            f"{selected_company}"
        )
    )

    complaint_chart.add_scatter(
        x=company_history["Month"],
        y=company_history[
            "Previous_6_month_average"
        ],
        mode="lines",
        name="Previous 6-month average",
        line={
            "dash": "dash"
        }
    )

    complaint_chart.update_layout(
        height=500
    )

    st.plotly_chart(
        complaint_chart,
        use_container_width=True
    )


# --------------------------------------------------
# Growth and Z-score charts
# --------------------------------------------------
left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Complaint Growth")

    growth_chart = px.bar(
        company_history,
        x="Month",
        y="Complaint_growth_percentage",
        labels={
            "Month": "Month",
            "Complaint_growth_percentage":
                "Growth (%)"
        }
    )

    growth_chart.add_hline(
        y=0,
        line_dash="dash"
    )

    growth_chart.update_layout(
        height=420
    )

    st.plotly_chart(
        growth_chart,
        use_container_width=True
    )


with right_chart:
    st.subheader("Complaint Volume Z-Score")

    z_score_chart = px.line(
        company_history,
        x="Month",
        y="Complaint_z_score",
        markers=True,
        labels={
            "Month": "Month",
            "Complaint_z_score": "Z-score"
        }
    )

    z_score_chart.add_hline(
        y=2,
        line_dash="dash",
        annotation_text="Unusual volume"
    )

    z_score_chart.add_hline(
        y=3,
        line_dash="dot",
        annotation_text="Highly unusual"
    )

    z_score_chart.update_layout(
        height=420
    )

    st.plotly_chart(
        z_score_chart,
        use_container_width=True
    )


# --------------------------------------------------
# Untimely-response history
# --------------------------------------------------
st.subheader("Untimely-Response History")

response_chart = px.line(
    company_history,
    x="Month",
    y=[
        "Untimely_rate",
        "Previous_6_month_untimely_rate"
    ],
    markers=True,
    labels={
        "Month": "Month",
        "value": "Untimely-response rate (%)",
        "variable": "Measurement"
    }
)

response_chart.for_each_trace(
    lambda trace: trace.update(
        name={
            "Untimely_rate":
                "Current monthly rate",
            "Previous_6_month_untimely_rate":
                "Previous 6-month rate"
        }.get(
            trace.name,
            trace.name
        )
    )
)

response_chart.update_layout(
    height=480
)

st.plotly_chart(
    response_chart,
    use_container_width=True
)


# --------------------------------------------------
# Historical data table
# --------------------------------------------------
st.subheader("Monthly Company Data")

history_columns = [
    "Month",
    "Complaint_count",
    "Previous_6_month_average",
    "Complaint_growth_percentage",
    "Complaint_z_score",
    "Untimely_count",
    "Untimely_rate",
    "Previous_6_month_untimely_rate",
    "Monthly_complaint_share"
]

available_history_columns = [
    column
    for column in history_columns
    if column in company_history.columns
]

display_history = company_history[
    available_history_columns
].copy()

display_history = display_history.sort_values(
    "Month",
    ascending=False
)

display_history = display_history.rename(
    columns={
        "Month": "Month",
        "Complaint_count": "Complaints",
        "Previous_6_month_average":
            "Previous 6-month average",
        "Complaint_growth_percentage":
            "Growth (%)",
        "Complaint_z_score": "Z-score",
        "Untimely_count":
            "Untimely responses",
        "Untimely_rate":
            "Untimely rate (%)",
        "Previous_6_month_untimely_rate":
            "Previous 6-month untimely rate (%)",
        "Monthly_complaint_share":
            "Complaint share (%)"
    }
)

st.dataframe(
    display_history,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Month":
            st.column_config.DateColumn(
                "Month",
                format="MMM YYYY"
            ),

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
            ),

        "Untimely rate (%)":
            st.column_config.NumberColumn(
                "Untimely rate (%)",
                format="%.2f%%"
            ),

        "Previous 6-month untimely rate (%)":
            st.column_config.NumberColumn(
                "Previous 6-month untimely rate (%)",
                format="%.2f%%"
            ),

        "Complaint share (%)":
            st.column_config.NumberColumn(
                "Complaint share (%)",
                format="%.2f%%"
            )
    }
)


# --------------------------------------------------
# Explanation
# --------------------------------------------------
with st.expander(
    "How is the company risk score calculated?"
):
    st.markdown(
        """
        The company risk score combines four monitoring indicators:

        - **35% Volume anomaly:** whether complaint volume is
          unusually high compared with recent history.
        - **25% Complaint growth:** change compared with the
          previous six-month average.
        - **25% Complaint share:** the company's complaint-volume
          rank among the monitored companies.
        - **15% Response risk:** current or recent untimely-response
          performance.

        The score is designed to prioritize companies for further
        review. It is not a regulatory conclusion or proof of
        misconduct.
        """
    )