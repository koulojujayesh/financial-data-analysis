from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "income", "expense"}
COLUMN_ALIASES = {
    "date": "date",
    "income": "income",
    "expense": "expense",
    "expenses": "expense",
    "revenue": "income",
}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized, project-standard column names.

    This helper trims whitespace, lowercases the headers, and maps a few common
    financial aliases to the expected names: ``date``, ``income``, and
    ``expense``.
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
            f"{missing_text}. Expected Date, Income, Expense."
        )

    return standardized


def _prepare_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert income and expense columns to numeric values safely."""
    prepared = df.copy()
    prepared["income"] = pd.to_numeric(prepared["income"], errors="coerce")
    prepared["expense"] = pd.to_numeric(prepared["expense"], errors="coerce")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    return prepared


def calculate_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate core financial metrics for a cleaned business income dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned input data containing Date, Income, and Expense columns.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, Any]]
        A tuple containing the updated dataframe with a Profit column and a
        dictionary of calculated financial metrics.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If required columns are missing.
    RuntimeError
        If an unexpected processing error occurs.
    """
    try:
        cleaned = _validate_input_dataframe(df)
        cleaned = _prepare_numeric_columns(cleaned)

        if cleaned.empty:
            logger.info("Total records processed: 0")
            empty_metrics = {
                "total_income": 0.0,
                "total_expense": 0.0,
                "total_profit": 0.0,
                "average_income": 0.0,
                "average_expense": 0.0,
                "average_profit": 0.0,
            }
            cleaned["profit"] = pd.Series(dtype="float64")
            return cleaned, empty_metrics

        rows_before = len(cleaned)
        logger.info("Total records processed: %d", rows_before)
        print(f"Total records processed: {rows_before}")

        cleaned["profit"] = cleaned["income"] - cleaned["expense"]

        total_income = float(cleaned["income"].sum(skipna=True))
        total_expense = float(cleaned["expense"].sum(skipna=True))
        total_profit = float(cleaned["profit"].sum(skipna=True))
        average_income = float(cleaned["income"].mean(skipna=True)) if not cleaned["income"].empty else 0.0
        average_expense = float(cleaned["expense"].mean(skipna=True)) if not cleaned["expense"].empty else 0.0
        average_profit = float(cleaned["profit"].mean(skipna=True)) if not cleaned["profit"].empty else 0.0

        metrics = {
            "total_income": total_income,
            "total_expense": total_expense,
            "total_profit": total_profit,
            "average_income": average_income,
            "average_expense": average_expense,
            "average_profit": average_profit,
        }

        logger.info("Total Income: %.2f", total_income)
        logger.info("Total Expense: %.2f", total_expense)
        logger.info("Total Profit: %.2f", total_profit)
        logger.info("Average Income: %.2f", average_income)
        logger.info("Average Expense: %.2f", average_expense)
        logger.info("Average Profit: %.2f", average_profit)
        print(f"Total Income: {total_income:.2f}")
        print(f"Total Expense: {total_expense:.2f}")
        print(f"Total Profit: {total_profit:.2f}")
        print(f"Average Income: {average_income:.2f}")
        print(f"Average Expense: {average_expense:.2f}")
        print(f"Average Profit: {average_profit:.2f}")

        cleaned = cleaned.sort_values(by="date").reset_index(drop=True)
        return cleaned, metrics

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while calculating metrics: {exc}") from exc


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return min, max, and mean statistics for Income, Expense, and Profit."""
    cleaned = _validate_input_dataframe(df)
    cleaned = _prepare_numeric_columns(cleaned)
    cleaned["profit"] = cleaned["income"] - cleaned["expense"]

    if cleaned.empty:
        return pd.DataFrame(columns=["metric", "income", "expense", "profit"])

    summary = pd.DataFrame(
        {
            "metric": ["min", "max", "mean"],
            "income": [cleaned["income"].min(), cleaned["income"].max(), cleaned["income"].mean()],
            "expense": [cleaned["expense"].min(), cleaned["expense"].max(), cleaned["expense"].mean()],
            "profit": [cleaned["profit"].min(), cleaned["profit"].max(), cleaned["profit"].mean()],
        }
    )
    return summary


def monthly_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Group the dataset by month and calculate monthly totals."""
    cleaned = _validate_input_dataframe(df)
    cleaned = _prepare_numeric_columns(cleaned)
    cleaned["profit"] = cleaned["income"] - cleaned["expense"]

    if cleaned.empty:
        return pd.DataFrame(columns=["month", "income", "expense", "profit"])

    monthly_df = cleaned.dropna(subset=["date"]).copy()
    monthly_df["month"] = monthly_df["date"].dt.to_period("M").astype(str)

    grouped = (
        monthly_df.groupby("month", as_index=False)[["income", "expense", "profit"]]
        .sum(numeric_only=True)
        .sort_values(by="month")
        .reset_index(drop=True)
    )
    return grouped


def profit_margin(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Profit_Margin column calculated as (Profit / Income) * 100."""
    cleaned = _validate_input_dataframe(df)
    cleaned = _prepare_numeric_columns(cleaned)
    cleaned["profit"] = cleaned["income"] - cleaned["expense"]

    if cleaned.empty:
        cleaned["profit_margin"] = pd.Series(dtype="float64")
        cleaned.rename(columns={"profit_margin": "Profit_Margin"}, inplace=True)
        return cleaned

    cleaned["Profit_Margin"] = cleaned.apply(
        lambda row: (row["profit"] / row["income"] * 100.0) if pd.notna(row["income"]) and row["income"] != 0 else 0.0,
        axis=1,
    )
    return cleaned


def highest_profit_day(df: pd.DataFrame) -> pd.Series:
    """Return the row corresponding to the highest profit day."""
    cleaned, _ = calculate_metrics(df)
    if cleaned.empty:
        return pd.Series(dtype="object")
    return cleaned.loc[cleaned["profit"].idxmax()]


def lowest_income_day(df: pd.DataFrame) -> pd.Series:
    """Return the row corresponding to the lowest income day."""
    cleaned, _ = calculate_metrics(df)
    if cleaned.empty:
        return pd.Series(dtype="object")
    return cleaned.loc[cleaned["income"].idxmin()]
