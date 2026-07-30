from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Model Performance",
    page_icon="📋",
    layout="wide"
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------
project_folder = Path(__file__).resolve().parents[2]

results_folder = os.path.join(
    project_folder,
    "reports",
    "model_results"
)

metrics_path = os.path.join(
    results_folder,
    "model_metrics.csv"
)

product_report_path = os.path.join(
    results_folder,
    "product_classification_report.csv"
)

issue_report_path = os.path.join(
    results_folder,
    "issue_classification_report.csv"
)

response_report_path = os.path.join(
    results_folder,
    "response_classification_report.csv"
)

response_matrix_path = os.path.join(
    results_folder,
    "response_confusion_matrix.csv"
)


# --------------------------------------------------
# Load model results
# --------------------------------------------------
@st.cache_data
def load_model_results():

    metrics_df = pd.read_csv(
        metrics_path,
        low_memory=False
    )

    product_report_df = pd.read_csv(
        product_report_path,
        low_memory=False
    )

    issue_report_df = pd.read_csv(
        issue_report_path,
        low_memory=False
    )

    response_report_df = pd.read_csv(
        response_report_path,
        low_memory=False
    )

    response_matrix_df = pd.read_csv(
        response_matrix_path,
        index_col=0
    )

    return (
        metrics_df,
        product_report_df,
        issue_report_df,
        response_report_df,
        response_matrix_df
    )


try:
    (
        metrics_df,
        product_report_df,
        issue_report_df,
        response_report_df,
        response_matrix_df
    ) = load_model_results()

except FileNotFoundError as error:
    st.error(
        "One or more model-performance files could not be found."
    )
    st.code(str(error))
    st.stop()

except Exception as error:
    st.error(
        "The model-performance data could not be loaded."
    )
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# Helper function
# --------------------------------------------------
def get_model_row(model_name):

    matching_rows = metrics_df[
        metrics_df["Model"] == model_name
    ]

    if matching_rows.empty:
        return None

    return matching_rows.iloc[0]


product_metrics = get_model_row(
    "Product classifier"
)

issue_metrics = get_model_row(
    "Issue classifier"
)

response_metrics = get_model_row(
    "Untimely-response classifier"
)


# --------------------------------------------------
# Page title
# --------------------------------------------------
st.title("About & Model Performance")

st.write(
    "This page summarizes the machine-learning models used "
    "by the Consumer Complaint Intelligence System."
)

st.info(
    "The system is a portfolio prototype designed for decision "
    "support. Model predictions should be reviewed by a human "
    "before operational use."
)


# --------------------------------------------------
# Project overview
# --------------------------------------------------
st.subheader("System Overview")

overview_column_1, overview_column_2 = st.columns(2)

with overview_column_1:
    st.markdown(
        """
        **Complaint-level intelligence**

        - Predicts the financial product
        - Predicts the complaint issue
        - Estimates untimely-response risk
        - Recommends a review priority
        """
    )

with overview_column_2:
    st.markdown(
        """
        **Portfolio-level intelligence**

        - Tracks complaint volumes over time
        - Detects unusual complaint spikes
        - Identifies emerging complaint topics
        - Calculates company monitoring risk scores
        """
    )


# --------------------------------------------------
# Model summary cards
# --------------------------------------------------
st.subheader("Model Summary")


if product_metrics is not None:

    st.markdown("### Product Classifier")

    product_column_1, product_column_2, product_column_3 = (
        st.columns(3)
    )

    product_column_1.metric(
        "Accuracy",
        f"{product_metrics['Accuracy']:.1%}"
    )

    product_column_2.metric(
        "Macro F1",
        f"{product_metrics['Macro F1']:.3f}"
    )

    product_column_3.metric(
        "Product categories",
        int(product_metrics["Number of classes"])
    )

    st.caption(
        "The product classifier predicts the broad financial "
        "product from the complaint narrative."
    )


if issue_metrics is not None:

    st.markdown("### Issue Classifier")

    issue_column_1, issue_column_2, issue_column_3 = (
        st.columns(3)
    )

    issue_column_1.metric(
        "Accuracy",
        f"{issue_metrics['Accuracy']:.1%}"
    )

    issue_column_2.metric(
        "Macro F1",
        f"{issue_metrics['Macro F1']:.3f}"
    )

    issue_column_3.metric(
        "Issue categories",
        int(issue_metrics["Number of classes"])
    )

    st.caption(
        "Issue prediction is more difficult because many issue "
        "categories contain similar language."
    )


if response_metrics is not None:

    st.markdown("### Untimely-Response Classifier")

    response_column_1, response_column_2, response_column_3, response_column_4 = (
        st.columns(4)
    )

    response_column_1.metric(
        "Untimely precision",
        f"{response_metrics['Positive precision']:.1%}"
    )

    response_column_2.metric(
        "Untimely recall",
        f"{response_metrics['Positive recall']:.1%}"
    )

    response_column_3.metric(
        "PR-AUC",
        f"{response_metrics['PR-AUC']:.4f}"
    )

    response_column_4.metric(
        "ROC-AUC",
        f"{response_metrics['ROC-AUC']:.4f}"
    )

    st.warning(
        "Overall accuracy is not emphasized for this model because "
        "untimely responses represent only about 1.13% of the data. "
        "A model predicting every complaint as timely would already "
        "achieve approximately 98.87% accuracy."
    )


# --------------------------------------------------
# Combined performance table
# --------------------------------------------------
st.subheader("Combined Model Metrics")

metrics_display = metrics_df.copy()

percentage_columns = [
    "Accuracy",
    "Macro F1",
    "Weighted F1",
    "Positive precision",
    "Positive recall",
    "Positive F1",
    "PR-AUC",
    "ROC-AUC",
    "Decision threshold"
]

for column in percentage_columns:
    metrics_display[column] = pd.to_numeric(
        metrics_display[column],
        errors="coerce"
    )

metrics_display = metrics_display.rename(
    columns={
        "Test records": "Test records",
        "Number of classes": "Classes",
        "Macro F1": "Macro F1",
        "Weighted F1": "Weighted F1",
        "Positive precision": "Untimely precision",
        "Positive recall": "Untimely recall",
        "Positive F1": "Untimely F1",
        "Decision threshold": "Threshold"
    }
)

st.dataframe(
    metrics_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Test records":
            st.column_config.NumberColumn(
                "Test records",
                format="%d"
            ),

        "Classes":
            st.column_config.NumberColumn(
                "Classes",
                format="%d"
            ),

        "Accuracy":
            st.column_config.NumberColumn(
                "Accuracy",
                format="%.4f"
            ),

        "Macro F1":
            st.column_config.NumberColumn(
                "Macro F1",
                format="%.4f"
            ),

        "Weighted F1":
            st.column_config.NumberColumn(
                "Weighted F1",
                format="%.4f"
            ),

        "Untimely precision":
            st.column_config.NumberColumn(
                "Untimely precision",
                format="%.4f"
            ),

        "Untimely recall":
            st.column_config.NumberColumn(
                "Untimely recall",
                format="%.4f"
            ),

        "Untimely F1":
            st.column_config.NumberColumn(
                "Untimely F1",
                format="%.4f"
            ),

        "PR-AUC":
            st.column_config.NumberColumn(
                "PR-AUC",
                format="%.4f"
            ),

        "ROC-AUC":
            st.column_config.NumberColumn(
                "ROC-AUC",
                format="%.4f"
            ),

        "Threshold":
            st.column_config.NumberColumn(
                "Threshold",
                format="%.2f"
            )
    }
)


# --------------------------------------------------
# Product classifier performance
# --------------------------------------------------
st.subheader("Product Classifier by Category")

product_summary_rows = [
    "accuracy",
    "macro avg",
    "weighted avg"
]

product_class_df = product_report_df[
    ~product_report_df["Class"].isin(
        product_summary_rows
    )
].copy()

for column in [
    "precision",
    "recall",
    "f1-score",
    "support"
]:
    product_class_df[column] = pd.to_numeric(
        product_class_df[column],
        errors="coerce"
    )


product_chart = px.bar(
    product_class_df.sort_values(
        "f1-score",
        ascending=True
    ),
    x="f1-score",
    y="Class",
    orientation="h",
    hover_data={
        "precision": ":.3f",
        "recall": ":.3f",
        "support": ":.0f"
    },
    labels={
        "f1-score": "F1-score",
        "Class": "Product category"
    }
)

product_chart.update_layout(
    height=520,
    yaxis_title=None
)

st.plotly_chart(
    product_chart,
    use_container_width=True
)


# --------------------------------------------------
# Issue classifier performance
# --------------------------------------------------
st.subheader("Issue Classifier by Category")

issue_summary_rows = [
    "accuracy",
    "macro avg",
    "weighted avg"
]

issue_class_df = issue_report_df[
    ~issue_report_df["Class"].isin(
        issue_summary_rows
    )
].copy()

for column in [
    "precision",
    "recall",
    "f1-score",
    "support"
]:
    issue_class_df[column] = pd.to_numeric(
        issue_class_df[column],
        errors="coerce"
    )


issue_chart = px.bar(
    issue_class_df.sort_values(
        "f1-score",
        ascending=True
    ),
    x="f1-score",
    y="Class",
    orientation="h",
    hover_data={
        "precision": ":.3f",
        "recall": ":.3f",
        "support": ":.0f"
    },
    labels={
        "f1-score": "F1-score",
        "Class": "Issue category"
    }
)

issue_chart.update_layout(
    height=720,
    yaxis_title=None
)

st.plotly_chart(
    issue_chart,
    use_container_width=True
)


# --------------------------------------------------
# Untimely-response confusion matrix
# --------------------------------------------------
st.subheader("Untimely-Response Confusion Matrix")

matrix_values = response_matrix_df.values

matrix_chart = go.Figure(
    data=go.Heatmap(
        z=matrix_values,
        x=response_matrix_df.columns,
        y=response_matrix_df.index,
        text=matrix_values,
        texttemplate="%{text:,}",
        hovertemplate=(
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Complaints: %{z:,}"
            "<extra></extra>"
        ),
        colorscale="Blues"
    )
)

matrix_chart.update_layout(
    xaxis_title="Predicted class",
    yaxis_title="Actual class",
    height=450
)

st.plotly_chart(
    matrix_chart,
    use_container_width=True
)

st.write(
    """
    At the selected threshold of **0.70**, the model correctly
    identified 24 of the 100 untimely complaints in the test set.
    It also generated 147 false-positive alerts.

    This means the model is most suitable as a **risk-ranking and
    prioritization tool**, rather than an automatic decision system.
    """
)


# --------------------------------------------------
# Model limitations
# --------------------------------------------------
st.subheader("Important Limitations")

st.markdown(
    """
    1. **Narrative availability:** Most complaints do not contain a
       public consumer narrative, so the NLP models use a smaller
       subset of the full dataset.

    2. **Class imbalance:** Untimely responses are extremely rare,
       which makes precise prediction difficult.

    3. **Topic quality:** Some topic clusters reflect repeated legal
       templates instead of unique consumer problems.

    4. **Random sampling:** Initial NLP development used a random
       200,000-row sample before selecting complaints with narratives.

    5. **Monitoring score:** The company risk score is a transparent
       prioritization rule, not a regulatory judgment.

    6. **Human review:** Predictions and alerts should always be
       reviewed before action is taken.
    """
)


# --------------------------------------------------
# Technical stack
# --------------------------------------------------
st.subheader("Technical Stack")

stack_column_1, stack_column_2, stack_column_3 = st.columns(3)

with stack_column_1:
    st.markdown(
        """
        **Data processing**

        - Python
        - Pandas
        - Chunked CSV processing
        """
    )

with stack_column_2:
    st.markdown(
        """
        **Machine learning**

        - Scikit-learn
        - TF-IDF
        - Logistic regression
        - NMF topic modelling
        """
    )

with stack_column_3:
    st.markdown(
        """
        **Application**

        - Streamlit
        - Plotly
        - Joblib
        """
    )