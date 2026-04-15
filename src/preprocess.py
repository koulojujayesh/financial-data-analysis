from __future__ import annotations

import logging

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

    This helper trims whitespace, lowercases column names, and maps common
    financial aliases to the expected names: ``date``, ``income``, ``expense``.
    """
    standardized = df.copy()
    normalized_columns = [str(col).strip().lower() for col in standardized.columns]
    remapped_columns = [COLUMN_ALIASES.get(col, col) for col in normalized_columns]
    standardized.columns = remapped_columns
    return standardized


def load_data(path_or_buffer) -> pd.DataFrame:
    """Load CSV data into a DataFrame."""
    return pd.read_csv(path_or_buffer)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess financial business income data.

    The function expects a dataset containing Date, Income, and Expense columns
    (case-insensitive, with basic alias support), then applies common
    preprocessing operations used in financial analytics workflows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset to clean.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset sorted by date with reset index.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If required columns are missing or cleaning fails validation.
    RuntimeError
        If an unexpected preprocessing error occurs.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        rows_before = len(df)
        cleaned = standardize_column_names(df)

        missing_columns = REQUIRED_COLUMNS - set(cleaned.columns)
        if missing_columns:
            missing_display = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Missing required columns after standardization: "
                f"{missing_display}. Expected Date, Income, Expense."
            )

        duplicate_count = int(cleaned.duplicated().sum())
        if duplicate_count:
            logger.info("Dropping %d duplicate rows.", duplicate_count)
        cleaned = cleaned.drop_duplicates()

        invalid_dates = int(pd.to_datetime(cleaned["date"], errors="coerce").isna().sum())
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        if invalid_dates:
            logger.warning("Converted %d invalid date values to NaT.", invalid_dates)

        for column in ("income", "expense"):
            numeric_series = pd.to_numeric(cleaned[column], errors="coerce")
            coercion_count = int(numeric_series.isna().sum() - cleaned[column].isna().sum())
            if coercion_count > 0:
                logger.warning(
                    "Converted %d non-numeric values to NaN in column '%s'.",
                    coercion_count,
                    column,
                )
            cleaned[column] = numeric_series

        missing_before_fill = int(cleaned.isna().sum().sum())
        if missing_before_fill:
            logger.info("Forward-filling missing values (%d cells).", missing_before_fill)
        cleaned = cleaned.ffill()

        rows_before_dropna = len(cleaned)
        cleaned = cleaned.dropna(subset=["date", "income", "expense"])
        rows_dropped_after_fill = rows_before_dropna - len(cleaned)
        if rows_dropped_after_fill:
            logger.info(
                "Dropped %d rows that still had missing required values after forward fill.",
                rows_dropped_after_fill,
            )

        cleaned = cleaned.sort_values(by="date").reset_index(drop=True)
        rows_after = len(cleaned)

        logger.info("Rows before cleaning: %d", rows_before)
        logger.info("Rows after cleaning: %d", rows_after)

        return cleaned

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while cleaning data: {exc}") from exc
