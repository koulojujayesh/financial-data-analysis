from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src.preprocess import clean_data, load_data
from analysis import calculate_metrics, get_summary_statistics, monthly_analysis, profit_margin
from anomaly import anomaly_summary, detect_anomalies
from prediction import predict_income
from utils.helpers import format_currency, month_name


st.set_page_config(page_title="Financial Analytics", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
            color: #111111;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div,
        .stApp li,
        .stApp .stMarkdown,
        .stApp .stText,
        .stApp [data-testid="stMarkdownContainer"] {
            color: #111111;
        }
        .stApp [data-testid="stSidebar"] * {
            color: #111111;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Financial Analytics Dashboard")
st.caption("Upload a CSV or use the bundled sample dataset to explore income, expense, forecasting, and anomaly detection.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


def load_input_dataframe() -> pd.DataFrame:
    if uploaded_file is not None:
        return load_data(io.BytesIO(uploaded_file.getvalue()))
    return load_data("data/sample_data.csv")


try:
    raw_df = load_input_dataframe()
    cleaned_df = clean_data(raw_df)
except Exception as exc:
    st.error(f"Unable to load or clean the data: {exc}")
    st.stop()

if cleaned_df.empty:
    st.warning("No rows available after cleaning.")
    st.stop()

metrics_df, metrics = calculate_metrics(cleaned_df)

metric_columns = st.columns(4)
metric_columns[0].metric("Total Income", format_currency(metrics["total_income"]))
metric_columns[1].metric("Total Expense", format_currency(metrics["total_expense"]))
metric_columns[2].metric("Total Profit", format_currency(metrics["total_profit"]))
metric_columns[3].metric("Average Profit", format_currency(metrics["average_profit"]))

tab_overview, tab_prediction, tab_anomaly = st.tabs(["Overview", "Prediction", "Anomaly Detection"])

with tab_overview:
    st.subheader("Cleaned Data")
    st.dataframe(metrics_df, use_container_width=True)

    monthly_df = monthly_analysis(cleaned_df)
    if not monthly_df.empty:
        chart_df = monthly_df.copy()
        chart_df["month_label"] = chart_df["month"].apply(month_name)
        fig = px.line(chart_df, x="month_label", y=["income", "expense", "profit"], markers=True)
        fig.update_layout(xaxis_title="Month", yaxis_title="Amount")
        st.plotly_chart(fig, use_container_width=True)

    summary_df = get_summary_statistics(cleaned_df)
    st.subheader("Summary Statistics")
    st.dataframe(summary_df, use_container_width=True)

    margin_df = profit_margin(cleaned_df)
    if "Profit_Margin" in margin_df.columns:
        st.subheader("Profit Margin")
        st.dataframe(margin_df[["date", "income", "expense", "profit", "Profit_Margin"]], use_container_width=True)

with tab_prediction:
    st.subheader("Income Forecast")
    future_periods = st.slider("Forecast periods", min_value=1, max_value=12, value=3)
    model_type = st.selectbox("Model", ["linear_regression", "random_forest"])
    normalize = st.checkbox("Normalize features", value=False)

    try:
        prediction_result = predict_income(
            cleaned_df,
            future_periods=future_periods,
            model_type=model_type,
            normalize=normalize,
        )
        st.write(f"Model score: {prediction_result['model_score']:.4f}")
        st.write(f"Trend: {prediction_result['trend']}")
        st.write(pd.DataFrame({"Forecast": prediction_result["predictions"]}))
    except Exception as exc:
        st.error(f"Forecasting failed: {exc}")

with tab_anomaly:
    st.subheader("Expense Anomalies")
    try:
        anomalies_df = detect_anomalies(cleaned_df)
        summary = anomaly_summary(cleaned_df, anomalies_df)
        summary_cols = st.columns(3)
        summary_cols[0].metric("Anomalies", summary["total_anomalies"])
        summary_cols[1].metric("Share", f"{summary['percentage_of_anomalies']:.2f}%")
        summary_cols[2].metric("Highest anomaly", format_currency(summary["highest_anomaly_value"]))
        st.dataframe(anomalies_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Anomaly detection failed: {exc}")
