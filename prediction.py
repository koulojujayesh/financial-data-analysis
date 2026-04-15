from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "income"}
COLUMN_ALIASES = {
    "date": "date",
    "income": "income",
    "expense": "expense",
    "expenses": "expense",
    "revenue": "income",
}

ModelType = Literal["linear_regression", "random_forest"]


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized, project-standard column names."""
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
            f"{missing_text}. Expected Date and Income."
        )

    return standardized


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date and income values to model-ready types."""
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["income"] = pd.to_numeric(prepared["income"], errors="coerce")
    prepared = prepared.dropna(subset=["date", "income"]).sort_values(by="date").reset_index(drop=True)
    return prepared


def _build_model(model_type: ModelType = "linear_regression", normalize: bool = False):
    """Create a regression model for income forecasting."""
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        return Pipeline(
            steps=[
                ("scaler", StandardScaler() if normalize else "passthrough"),
                ("model", model),
            ]
        )

    model = LinearRegression()
    if normalize:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
    return model


def evaluate_model(model, X, y) -> float:
    """Return the R2 score for the fitted model."""
    predictions = model.predict(X)
    if len(y) < 2:
        return 0.0
    return float(r2_score(y, predictions))


def get_trend(predictions) -> str:
    """Return an easy-to-read trend label from the prediction sequence."""
    if predictions is None:
        return "stable"

    values = list(predictions)
    if len(values) < 2:
        return "stable"

    first_value = float(values[0])
    last_value = float(values[-1])
    if last_value > first_value:
        return "upward trend"
    if last_value < first_value:
        return "downward trend"
    return "stable"


def predict_income(
    df: pd.DataFrame,
    future_periods: int = 3,
    model_type: ModelType = "linear_regression",
    normalize: bool = False,
) -> dict[str, Any]:
    """Predict future income using a time-based regression model.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing Date and Income columns. Column names are
        treated case-insensitively and common aliases are supported.
    future_periods : int, default=3
        Number of future values to predict.
    model_type : {"linear_regression", "random_forest"}, default="linear_regression"
        Regression model to use for forecasting.
    normalize : bool, default=False
        Whether to scale the time feature before fitting the model.

    Returns
    -------
    dict[str, Any]
        Dictionary containing predictions, model score, trend, and metadata.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If required columns are missing, the dataframe is empty, or parameters
        are invalid.
    RuntimeError
        If an unexpected error occurs during forecasting.
    """
    try:
        if future_periods < 1:
            raise ValueError("future_periods must be at least 1.")

        cleaned = _validate_input_dataframe(df)
        cleaned = _prepare_data(cleaned)

        if cleaned.empty:
            raise ValueError("Input dataframe is empty after cleaning.")

        if len(cleaned) < 2:
            income_value = float(cleaned["income"].iloc[-1])
            predictions = [income_value for _ in range(future_periods)]
            logger.info("Total records processed: %d", len(cleaned))
            logger.info("Prediction values: %s", predictions)
            print(f"Total records processed: {len(cleaned)}")
            print(f"Prediction values: {predictions}")
            return {
                "predictions": predictions,
                "model_score": 0.0,
                "trend": "stable",
                "model_type": model_type,
            }

        cleaned["time_index"] = np.arange(len(cleaned), dtype=float)
        X = cleaned[["time_index"]]
        y = cleaned["income"].astype(float)

        model = _build_model(model_type=model_type, normalize=normalize)
        model.fit(X, y)
        model_score = evaluate_model(model, X, y)

        future_index = np.arange(len(cleaned), len(cleaned) + future_periods, dtype=float)
        future_X = pd.DataFrame({"time_index": future_index})
        forecast_values = model.predict(future_X)
        predictions = [float(round(value, 2)) for value in forecast_values]

        trend_label = get_trend(predictions)
        trend = "increasing" if trend_label == "upward trend" else "decreasing" if trend_label == "downward trend" else "stable"

        logger.info("Total records processed: %d", len(cleaned))
        logger.info("Model score: %.4f", model_score)
        logger.info("Prediction values: %s", predictions)
        print(f"Total records processed: {len(cleaned)}")
        print(f"Model score: {model_score:.4f}")
        print(f"Prediction values: {predictions}")

        return {
            "predictions": predictions,
            "model_score": model_score,
            "trend": trend,
            "model_type": model_type,
        }

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Unexpected error while predicting income: {exc}") from exc
