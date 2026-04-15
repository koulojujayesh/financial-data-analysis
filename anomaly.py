from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"expense"}
COLUMN_ALIASES = {
    "date": "date",
    "income": "income",
    "expense": "expense",
    "expenses": "expense",
    "profit": "profit",
}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names.

    The helper trims whitespace, lowercases headers, and maps a few common
    aliases to the expected financial column names.
    """
    standardized = df.copy()
    normalized_columns = [str(column).strip().lower() for column in standardized.columns]
    standardized.columns = [COLUMN_ALIASES.get(column, column) for column in normalized_columns]
    return standardized


def _validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the input dataframe."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    standardized = standardize_column_names(df)
    missing_columns = REQUIRED_COLUMNS - set(standardized.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Missing required columns after standardization: "
            f"{missing_text}. Expected at least an Expense column."
        )

    return standardized


def calculate_threshold(df: pd.DataFrame) -> float:
    """Return the anomaly threshold based on mean + 2 * standard deviation."""
    cleaned = _validate_input_dataframe(df)
    if cleaned.empty:
        raise ValueError("Input dataframe is empty.")

    expense_values = pd.to_numeric(cleaned["expense"], errors="coerce").dropna()
    if expense_values.empty:
        raise ValueError("Expense column does not contain any numeric values.")

    threshold = float(expense_values.mean() + 2 * expense_values.std(ddof=0))
    return threshold


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect expense anomalies using a mean + 2 standard deviation threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing an Expense column and optionally Date,
        Income, and Profit columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only anomaly rows with an ``Is_Anomaly`` column.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the dataframe is empty or Expense is missing.
    RuntimeError
        If an unexpected error occurs during anomaly detection.
    """
    try:
        cleaned = _validate_input_dataframe(df)
        if cleaned.empty:
            raise ValueError("Input dataframe is empty.")

        cleaned = cleaned.copy()
        cleaned["expense"] = pd.to_numeric(cleaned["expense"], errors="coerce")
        cleaned = cleaned.dropna(subset=["expense"]).reset_index(drop=True)

        if cleaned.empty:
            raise ValueError("Expense column does not contain any numeric values.")

        threshold = calculate_threshold(cleaned)
        anomaly_mask = cleaned["expense"] > threshold
        anomalies = cleaned.loc[anomaly_mask].copy().reset_index(drop=True)
        anomalies["Is_Anomaly"] = True

        logger.info("Threshold value: %.2f", threshold)
        logger.info("Number of anomalies detected: %d", len(anomalies))
        print(f"Threshold value: {threshold:.2f}")
        print(f"Number of anomalies detected: {len(anomalies)}")

        return anomalies

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while detecting anomalies: {exc}") from exc


def anomaly_summary(df: pd.DataFrame, anomalies: pd.DataFrame) -> dict[str, Any]:
    """Return a summary of anomaly detection results."""
    cleaned = _validate_input_dataframe(df)
    total_rows = len(cleaned)
    total_anomalies = len(anomalies) if anomalies is not None else 0
    percentage = (total_anomalies / total_rows * 100.0) if total_rows else 0.0
    highest_anomaly_value = 0.0

    if anomalies is not None and not anomalies.empty and "expense" in anomalies.columns:
        anomaly_expenses = pd.to_numeric(anomalies["expense"], errors="coerce").dropna()
        if not anomaly_expenses.empty:
            highest_anomaly_value = float(anomaly_expenses.max())

    return {
        "total_anomalies": int(total_anomalies),
        "percentage_of_anomalies": float(percentage),
        "highest_anomaly_value": float(highest_anomaly_value),
    }


def detect_anomalies_zscore(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """Detect anomalies using the Z-score method.

    Values with absolute Z-score greater than the threshold are marked as
    anomalies.
    """
    cleaned = _validate_input_dataframe(df)
    if cleaned.empty:
        raise ValueError("Input dataframe is empty.")

    cleaned = cleaned.copy()
    cleaned["expense"] = pd.to_numeric(cleaned["expense"], errors="coerce")
    cleaned = cleaned.dropna(subset=["expense"]).reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("Expense column does not contain any numeric values.")

    expense_values = cleaned["expense"]
    mean_value = expense_values.mean()
    std_value = expense_values.std(ddof=0)

    if std_value == 0 or pd.isna(std_value):
        cleaned["Is_Anomaly"] = False
        return cleaned.iloc[0:0].copy()

    z_scores = (expense_values - mean_value) / std_value
    anomalies = cleaned.loc[z_scores.abs() > z_threshold].copy().reset_index(drop=True)
    anomalies["Is_Anomaly"] = True
    return anomalies
