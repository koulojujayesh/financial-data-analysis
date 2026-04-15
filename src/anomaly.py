from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.15) -> pd.DataFrame:
    """Flag anomalous rows based on revenue and expenses."""
    if df.empty or not {"revenue", "expenses"}.issubset(df.columns):
        return df.copy()

    result = df.copy()
    model = IsolationForest(contamination=contamination, random_state=42)
    features = result[["revenue", "expenses"]]
    result["anomaly_flag"] = model.fit_predict(features)
    result["is_anomaly"] = result["anomaly_flag"] == -1
    return result
