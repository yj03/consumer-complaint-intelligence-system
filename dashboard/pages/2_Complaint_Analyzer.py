import os

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Complaint Analyzer",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------
project_folder = (
    r"C:\Users\User\OneDrive\DS_Project"
    r"\ComplaintsIntelligent"
)

model_folder = os.path.join(
    project_folder,
    "models"
)

product_model_path = os.path.join(
    model_folder,
    "product_classifier.pkl"
)

issue_model_path = os.path.join(
    model_folder,
    "issue_classifier.pkl"
)

response_model_path = os.path.join(
    model_folder,
    "timely_response_classifier.pkl"
)


# --------------------------------------------------
# Load trained models
# --------------------------------------------------
@st.cache_resource
def load_models():
    product_model = joblib.load(
        product_model_path
    )

    issue_model = joblib.load(
        issue_model_path
    )

    response_model = joblib.load(
        response_model_path
    )

    return (
        product_model,
        issue_model,
        response_model
    )


try:
    (
        product_model,
        issue_model,
        response_model
    ) = load_models()

except FileNotFoundError as error:
    st.error(
        "One or more model files could not be found."
    )
    st.code(str(error))
    st.stop()

except Exception as error:
    st.error(
        "The trained models could not be loaded."
    )
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# Page title
# --------------------------------------------------
st.title("Complaint Analyzer")

st.write(
    "Enter a consumer complaint to predict its financial "
    "product, issue category and untimely-response risk."
)

st.info(
    "These predictions are decision-support indicators "
    "and should not replace human review."
)


# --------------------------------------------------
# Complaint input
# --------------------------------------------------
complaint_text = st.text_area(
    label="Consumer complaint narrative",
    height=220,
    placeholder=(
        "Example: My mortgage provider charged me late fees "
        "even though I paid before the deadline."
    )
)


input_column_1, input_column_2 = st.columns(2)

with input_column_1:
    state = st.text_input(
        label="State abbreviation",
        value="Unknown",
        placeholder="Example: CA"
    )

with input_column_2:
    submitted_via = st.selectbox(
        label="Submission channel",
        options=[
            "Web",
            "Phone",
            "Referral",
            "Postal mail",
            "Email",
            "Fax",
            "Unknown"
        ]
    )


analyze_button = st.button(
    label="Analyze Complaint",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Analyze complaint
# --------------------------------------------------
if analyze_button:

    complaint_text = complaint_text.strip()
    word_count = len(complaint_text.split())

    if word_count < 5:
        st.warning(
            "Please enter a longer complaint containing "
            "at least five words."
        )

    else:
        try:
            with st.spinner("Analyzing complaint..."):

                # ------------------------------------------
                # Product prediction
                # ------------------------------------------
                predicted_product = product_model.predict(
                    [complaint_text]
                )[0]

                product_probabilities = (
                    product_model.predict_proba(
                        [complaint_text]
                    )[0]
                )

                product_results = pd.DataFrame({
                    "Product": product_model.classes_,
                    "Probability": product_probabilities
                })

                product_results = (
                    product_results
                    .sort_values(
                        "Probability",
                        ascending=False
                    )
                    .head(3)
                    .reset_index(drop=True)
                )


                # ------------------------------------------
                # Issue prediction
                # ------------------------------------------
                predicted_issue = issue_model.predict(
                    [complaint_text]
                )[0]

                issue_probabilities = (
                    issue_model.predict_proba(
                        [complaint_text]
                    )[0]
                )

                issue_results = pd.DataFrame({
                    "Issue": issue_model.classes_,
                    "Probability": issue_probabilities
                })

                issue_results = (
                    issue_results
                    .sort_values(
                        "Probability",
                        ascending=False
                    )
                    .head(3)
                    .reset_index(drop=True)
                )


                # ------------------------------------------
                # Prepare state value
                # ------------------------------------------
                cleaned_state = state.strip().upper()

                if not cleaned_state:
                    cleaned_state = "Unknown"


                # ------------------------------------------
                # Response-risk input
                # ------------------------------------------
                response_input = pd.DataFrame({
                    "Consumer complaint narrative": [
                        complaint_text
                    ],
                    "Product": [
                        predicted_product
                    ],
                    "Issue": [
                        predicted_issue
                    ],
                    "State": [
                        cleaned_state
                    ],
                    "Submitted via": [
                        submitted_via
                    ]
                })


                # ------------------------------------------
                # Untimely-response prediction
                # ------------------------------------------
                untimely_score = (
                    response_model.predict_proba(
                        response_input
                    )[0][1]
                )


                # ------------------------------------------
                # Risk classification
                # ------------------------------------------
                if untimely_score >= 0.70:
                    risk_level = "High"
                    recommended_action = (
                        "Send this complaint for priority review."
                    )

                elif untimely_score >= 0.40:
                    risk_level = "Medium"
                    recommended_action = (
                        "Monitor this complaint and review it soon."
                    )

                else:
                    risk_level = "Low"
                    recommended_action = (
                        "Use the standard complaint-processing workflow."
                    )


            # --------------------------------------------------
            # Analysis completed message
            # --------------------------------------------------
            st.success(
                "Analysis completed successfully."
            )


            # --------------------------------------------------
            # Main prediction cards
            # --------------------------------------------------
            result_column_1, result_column_2, result_column_3 = (
                st.columns(3)
            )


            with result_column_1:
                with st.container(border=True):
                    st.caption("Predicted product")

                    st.markdown(
                        f"##### {predicted_product}"
                    )


            with result_column_2:
                with st.container(border=True):
                    st.caption("Predicted issue")

                    st.markdown(
                        f"##### {predicted_issue}"
                    )


            with result_column_3:
                with st.container(border=True):
                    st.caption("Untimely-response risk")

                    st.markdown(
                        f"### {untimely_score:.1%}"
                    )


            # --------------------------------------------------
            # Response-risk assessment
            # --------------------------------------------------
            st.subheader(
                "Response-Risk Assessment"
            )


            if risk_level == "High":
                st.error(
                    f"Risk level: {risk_level}"
                )

            elif risk_level == "Medium":
                st.warning(
                    f"Risk level: {risk_level}"
                )

            else:
                st.success(
                    f"Risk level: {risk_level}"
                )


            with st.container(border=True):
                st.markdown(
                    "**Recommended action**"
                )

                st.write(
                    recommended_action
                )


            st.caption(
                "The response-risk value is a model ranking score. "
                "It should not be interpreted as a guaranteed "
                "probability that the company will respond late."
            )


            # --------------------------------------------------
            # Top prediction tables
            # --------------------------------------------------
            product_column, issue_column = st.columns(2)


            with product_column:
                st.subheader(
                    "Top Product Predictions"
                )

                product_display = product_results.copy()

                product_display["Probability"] = (
                    product_display["Probability"]
                    * 100
                )

                st.dataframe(
                    product_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Product":
                            st.column_config.TextColumn(
                                "Product",
                                width="large"
                            ),

                        "Probability":
                            st.column_config.ProgressColumn(
                                "Confidence",
                                min_value=0,
                                max_value=100,
                                format="%.1f%%"
                            )
                    }
                )


            with issue_column:
                st.subheader(
                    "Top Issue Predictions"
                )

                issue_display = issue_results.copy()

                issue_display["Probability"] = (
                    issue_display["Probability"]
                    * 100
                )

                st.dataframe(
                    issue_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Issue":
                            st.column_config.TextColumn(
                                "Issue",
                                width="large"
                            ),

                        "Probability":
                            st.column_config.ProgressColumn(
                                "Confidence",
                                min_value=0,
                                max_value=100,
                                format="%.1f%%"
                            )
                    }
                )


            # --------------------------------------------------
            # Input details
            # --------------------------------------------------
            with st.expander(
                "View analysis input"
            ):
                detail_column_1, detail_column_2 = (
                    st.columns(2)
                )

                detail_column_1.write(
                    f"**State:** {cleaned_state}"
                )

                detail_column_2.write(
                    f"**Submitted via:** {submitted_via}"
                )

                st.write(
                    f"**Narrative word count:** {word_count}"
                )

                st.text_area(
                    label="Submitted complaint",
                    value=complaint_text,
                    height=180,
                    disabled=True
                )


        except Exception as error:
            st.error(
                "The complaint could not be analyzed."
            )

            st.code(str(error))