from __future__ import annotations

import logging
from typing import Iterable

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

    This helper strips surrounding whitespace, lowercases the headers, and
    maps a few common financial aliases to the required names: ``date``,
    ``income``, and ``expense``.
    """
    standardized = df.copy()
    normalized_columns = [str(column).strip().lower() for column in standardized.columns]
    standardized.columns = [COLUMN_ALIASES.get(column, column) for column in normalized_columns]
    return standardized


def load_data(path_or_buffer) -> pd.DataFrame:
    """Load CSV data into a DataFrame."""
    return pd.read_csv(path_or_buffer)


def _count_new_missing_values(series: pd.Series, coerced: pd.Series) -> int:
    """Count values made missing by coercion, excluding values already missing."""
    return int(coerced.isna().sum() - series.isna().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess a financial dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing Date, Income, and Expense columns. Column names
        are treated case-insensitively and common aliases are supported.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with standardized columns, valid dates, numeric
        income and expense values, duplicates removed, rows sorted by date, and
        index reset.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    ValueError
        If required columns are missing after standardization.
    RuntimeError
        If an unexpected error occurs during cleaning.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        rows_before = len(df)
        cleaned = standardize_column_names(df)

        missing_columns = REQUIRED_COLUMNS - set(cleaned.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Missing required columns after standardization: "
                f"{missing_text}. Expected Date, Income, Expense."
            )

        duplicate_count = int(cleaned.duplicated().sum())
        if duplicate_count:
            logger.info("Dropping %d duplicate rows.", duplicate_count)
            print(f"Dropped {duplicate_count} duplicate rows.")
        cleaned = cleaned.drop_duplicates()

        date_series = pd.to_datetime(cleaned["date"], errors="coerce")
        invalid_dates = int(date_series.isna().sum())
        if invalid_dates:
            logger.warning("Converted %d invalid date values to NaT.", invalid_dates)
            print(f"Converted {invalid_dates} invalid date values to NaT.")
        cleaned["date"] = date_series

        for column in ("income", "expense"):
            numeric_series = pd.to_numeric(cleaned[column], errors="coerce")
            coerced_count = _count_new_missing_values(cleaned[column], numeric_series)
            if coerced_count > 0:
                logger.warning(
                    "Converted %d non-numeric values to NaN in column '%s'.",
                    coerced_count,
                    column,
                )
                print(f"Converted {coerced_count} non-numeric values to NaN in column '{column}'.")
            cleaned[column] = numeric_series

        missing_before_fill = int(cleaned.isna().sum().sum())
        if missing_before_fill:
            logger.info("Forward-filling missing values (%d cells).", missing_before_fill)
            print(f"Forward-filling missing values in {missing_before_fill} cells.")
        cleaned = cleaned.ffill()

        rows_before_dropna = len(cleaned)
        cleaned = cleaned.dropna(subset=["date", "income", "expense"])
        dropped_after_fill = rows_before_dropna - len(cleaned)
        if dropped_after_fill:
            logger.info(
                "Dropped %d rows that still had missing required values after forward fill.",
                dropped_after_fill,
            )
            print(
                "Dropped "
                f"{dropped_after_fill} rows that still had missing required values after forward fill."
            )

        cleaned = cleaned.sort_values(by="date").reset_index(drop=True)
        rows_after = len(cleaned)

        logger.info("Rows before cleaning: %d", rows_before)
        logger.info("Rows after cleaning: %d", rows_after)
        print(f"Rows before cleaning: {rows_before}")
        print(f"Rows after cleaning: {rows_after}")

        return cleaned

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while cleaning data: {exc}") from exc
