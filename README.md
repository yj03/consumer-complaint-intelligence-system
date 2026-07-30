# Consumer Complaint Intelligence System

An end-to-end NLP and risk-monitoring system for analyzing consumer-finance complaints, detecting emerging complaint topics, and identifying companies with unusual complaint activity.

The project processes more than 17 million consumer complaints and provides classification models, topic modelling, company risk scoring, trend analysis, and an interactive Streamlit dashboard.

## Project Features

- Classifies complaint narratives by financial product
- Classifies complaint narratives by issue
- Predicts the risk of an untimely company response
- Identifies recurring complaint themes using NMF topic modelling
- Detects emerging complaint topics using growth rates and Z-scores
- Calculates company risk scores from complaint volume, growth, market share, and response performance
- Provides an interactive multi-page Streamlit dashboard

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- NMF Topic Modelling
- Joblib
- Streamlit
- Plotly
- Matplotlib

## Model Performance

| Model | Main Result |
|---|---:|
| Product classifier | 84.81% accuracy |
| Product classifier macro F1 | 0.73 |
| Issue classifier | 61.17% accuracy |
| Issue classifier macro F1 | 0.58 |
| Untimely-response classifier ROC-AUC | 0.85 |
| Untimely-response classifier PR-AUC | 0.08 |

The untimely-response dataset is highly imbalanced, with approximately 1.13% of complaints labelled as untimely. ROC-AUC, recall, precision, and PR-AUC are therefore more informative than overall accuracy.

## Company Risk Score

Each company receives a transparent score from 0 to 100 using:

- Complaint-volume anomaly: 35%
- Complaint growth: 25%
- Complaint-volume share: 25%
- Untimely-response performance: 15%

Risk levels are classified as:

| Score | Risk Level |
|---|---|
| 75–100 | Critical |
| 50–74.99 | High |
| 25–49.99 | Moderate |
| 0–24.99 | Low |

## Project Structure

```text
consumer-complaint-intelligence-system/
│
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 2_Complaint_Analyzer.py
│       ├── 3_Emerging_Topics.py
│       ├── 4_Company_Details.py
│       └── 5_Model_Performance.py
│
├── src/
│   ├── prepare_nlp_data.py
│   ├── train_classifiers.py
│   ├── topic_pipeline.py
│   ├── aggregate_full_data.py
│   ├── company_risk_pipeline.py
│   └── evaluate_models.py
│
├── data/
│   └── Processed dashboard datasets
│
├── models/
│   └── README.md
│
├── reports/
│   └── model_results/
│
├── .gitignore
├── requirements.txt
└── README.md