from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.preprocess import clean_data, load_data
except Exception:  # pragma: no cover - fallback for alternate project layouts
    from preprocess import clean_data, load_data

try:
    from src.analysis import calculate_metrics
except Exception:  # pragma: no cover - fallback for alternate project layouts
    from analysis import calculate_metrics

try:
    from src.prediction import predict_income
except Exception:  # pragma: no cover - fallback for alternate project layouts
    from prediction import predict_income

try:
    from src.anomaly import detect_anomalies
except Exception:  # pragma: no cover - fallback for alternate project layouts
    from anomaly import detect_anomalies

try:
    from utils.helpers import format_currency
except Exception:  # pragma: no cover - fallback for alternate project layouts
    def format_currency(value: float) -> str:
        return f"${value:,.2f}"


st.set_page_config(page_title="Financial Data Analytics of Business Income", page_icon="📊", layout="wide")
st.title("📊 Financial Data Analytics Dashboard")
st.caption("Upload financial data, clean it, analyze performance, forecast income, and detect anomalies.")


APP_ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = APP_ROOT / "data" / "sample_data.csv"
NAV_OPTIONS = ["Upload Data", "Dashboard", "Prediction", "Anomaly Detection", "Download Report"]


def initialize_session_state() -> None:
    """Create the session state keys used by the dashboard."""
    defaults: dict[str, Any] = {
        "raw_df": None,
        "processed_df": None,
        "metrics": None,
        "prediction_result": None,
        "anomaly_df": None,
        "source_name": "",
        "upload_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_sample_dataset() -> pd.DataFrame:
    """Load the bundled sample dataset."""
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample dataset not found at {SAMPLE_PATH}")
    return load_data(SAMPLE_PATH)


def read_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV file into a dataframe."""
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")
    if not str(uploaded_file.name).lower().endswith(".csv"):
        raise ValueError("Please upload a valid CSV file.")
    return load_data(uploaded_file)


def dataframe_preview(df: pd.DataFrame, rows: int = 8) -> None:
    """Show a friendly preview of a dataframe."""
    if df is None or df.empty:
        st.warning("No data available to preview.")
        return

    display_df = df.copy()
    rename_map = {"date": "Date", "income": "Income", "expense": "Expense", "profit": "Profit"}
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
    st.dataframe(display_df.head(rows), use_container_width=True)


def process_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the dataset and calculate financial metrics."""
    cleaned_df = clean_data(raw_df)
    processed_df, metrics = calculate_metrics(cleaned_df)
    return processed_df, metrics


def build_charts(df: pd.DataFrame) -> None:
    """Render Plotly charts for income, expense, and profit."""
    if df is None or df.empty:
        st.warning("No processed data available for charts.")
        return

    chart_df = df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date"])

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.line(chart_df, x="date", y="income", markers=True, title="Income vs Date")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(chart_df, x="date", y="expense", markers=True, title="Expense vs Date")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.line(chart_df, x="date", y="profit", markers=True, title="Profit vs Date")
        st.plotly_chart(fig, use_container_width=True)


def render_metrics(metrics: dict[str, Any]) -> None:
    """Display core KPI cards."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", format_currency(metrics.get("total_income", 0.0)))
    col2.metric("Total Expense", format_currency(metrics.get("total_expense", 0.0)))
    col3.metric("Total Profit", format_currency(metrics.get("total_profit", 0.0)))


def render_prediction_section(df: pd.DataFrame) -> None:
    """Render the prediction controls and results."""
    st.subheader("🔮 Prediction")
    if df is None or df.empty:
        st.warning("Cleaned data is required before running predictions.")
        return

    periods = st.slider("Future periods to predict", min_value=1, max_value=24, value=3, key="prediction_periods")
    model_choice = st.selectbox(
        "Prediction model",
        options=["linear_regression", "random_forest"],
        index=0,
        format_func=lambda value: "Linear Regression" if value == "linear_regression" else "Random Forest",
        key="prediction_model",
    )
    normalize = st.checkbox("Normalize features", value=False, key="prediction_normalize")

    if st.button("Run Prediction", type="primary"):
        with st.spinner("Generating income forecast..."):
            try:
                result = predict_income(df, future_periods=periods, model_type=model_choice, normalize=normalize)
                st.session_state["prediction_result"] = result
            except Exception as exc:
                st.error(str(exc))
                return

    result = st.session_state.get("prediction_result")
    if not result:
        st.info("Click 'Run Prediction' to generate forecast values.")
        return

    predictions = result.get("predictions", [])
    trend = result.get("trend", "stable")
    model_score = float(result.get("model_score", 0.0))

    prediction_frame = pd.DataFrame(
        {
            "Period": [f"Future {index + 1}" for index in range(len(predictions))],
            "Predicted Income": predictions,
        }
    )

    st.write("### Predicted Values")
    st.dataframe(prediction_frame, use_container_width=True)
    st.metric("Trend", trend.replace("_", " ").title())
    st.metric("Model Score (R²)", f"{model_score:.4f}")


def render_anomaly_section(df: pd.DataFrame) -> None:
    """Render anomaly detection results."""
    st.subheader("⚠️ Anomaly Detection")
    if df is None or df.empty:
        st.warning("Cleaned data is required before anomaly detection.")
        return

    if st.button("Detect Anomalies", type="primary"):
        with st.spinner("Scanning for unusual expense values..."):
            try:
                anomaly_df = detect_anomalies(df)
                st.session_state["anomaly_df"] = anomaly_df
            except Exception as exc:
                st.error(str(exc))
                return

    anomaly_df = st.session_state.get("anomaly_df")
    if anomaly_df is None:
        st.info("Click 'Detect Anomalies' to inspect unusual expense values.")
        return

    if anomaly_df.empty:
        st.success("No anomalies detected.")
        return

    st.success(f"Detected {len(anomaly_df)} anomaly row(s).")
    display_df = anomaly_df.copy()
    display_df = display_df.rename(columns={"date": "Date", "income": "Income", "expense": "Expense", "profit": "Profit"})
    st.dataframe(display_df, use_container_width=True)


def render_download_section(df: pd.DataFrame) -> None:
    """Render the download report section."""
    st.subheader("⬇️ Download Report")
    if df is None or df.empty:
        st.warning("No processed data available for download.")
        return

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Processed CSV",
        data=csv_data,
        file_name="financial_data_processed.csv",
        mime="text/csv",
    )


initialize_session_state()

st.sidebar.header("Navigation")
selected_page = st.sidebar.radio("Go to", NAV_OPTIONS, index=0)
st.sidebar.divider()
st.sidebar.subheader("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.sidebar.button("Load Sample Data")

if uploaded_file is not None:
    signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("upload_signature") != signature:
        try:
            with st.spinner("Loading uploaded file..."):
                st.session_state["raw_df"] = read_uploaded_dataset(uploaded_file)
                st.session_state["source_name"] = uploaded_file.name
                st.session_state["upload_signature"] = signature
                st.session_state["processed_df"] = None
                st.session_state["metrics"] = None
                st.session_state["prediction_result"] = None
                st.session_state["anomaly_df"] = None
                st.sidebar.success("File loaded and stored in session state.")
        except Exception as exc:
            st.sidebar.error(str(exc))

if use_sample or st.session_state.get("raw_df") is None:
    try:
        with st.spinner("Loading sample data..."):
            st.session_state["raw_df"] = load_sample_dataset()
            st.session_state["source_name"] = SAMPLE_PATH.name
            st.session_state["upload_signature"] = (SAMPLE_PATH.name, SAMPLE_PATH.stat().st_size)
            st.session_state["processed_df"] = None
            st.session_state["metrics"] = None
            st.session_state["prediction_result"] = None
            st.session_state["anomaly_df"] = None
            if use_sample:
                st.sidebar.success("Sample data loaded.")
    except Exception as exc:
        st.sidebar.error(str(exc))

raw_df = st.session_state.get("raw_df")

if raw_df is not None:
    with st.spinner("Cleaning and preparing dataset..."):
        try:
            processed_df, metrics = process_dataset(raw_df)
            st.session_state["processed_df"] = processed_df
            st.session_state["metrics"] = metrics
            st.success("Data cleaned successfully.")
        except Exception as exc:
            st.error(str(exc))
            st.stop()

processed_df = st.session_state.get("processed_df")
metrics = st.session_state.get("metrics") or {}

if processed_df is None or processed_df.empty:
    st.warning("Upload a valid CSV file or load the sample data to continue.")
    st.stop()

st.write(f"**Current data source:** {st.session_state.get('source_name', 'Unknown')}")

if selected_page == "Upload Data":
    st.subheader("📤 Upload Data")
    st.write("Upload a CSV file with Date, Income, and Expense columns.")
    st.write("Preview of the raw dataset:")
    dataframe_preview(raw_df)
    st.write("Preview of the cleaned dataset:")
    dataframe_preview(processed_df)

elif selected_page == "Dashboard":
    st.subheader("📈 Dashboard")
    render_metrics(metrics)
    st.markdown("---")
    build_charts(processed_df)
    st.markdown("### Cleaned Data Preview")
    dataframe_preview(processed_df)

elif selected_page == "Prediction":
    render_prediction_section(processed_df)

elif selected_page == "Anomaly Detection":
    render_anomaly_section(processed_df)

elif selected_page == "Download Report":
    render_download_section(processed_df)
