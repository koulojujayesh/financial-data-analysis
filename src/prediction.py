from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def forecast_revenue(df: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
    """Forecast future revenue using a simple linear trend model."""
    if df.empty or "revenue" not in df.columns:
        return pd.DataFrame(columns=["period", "forecast_revenue"])

    y = df["revenue"].to_numpy(dtype=float)
    x = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    future_x = np.arange(len(y), len(y) + periods).reshape(-1, 1)
    forecast = model.predict(future_x)

    return pd.DataFrame(
        {
            "period": [f"T+{idx}" for idx in range(1, periods + 1)],
            "forecast_revenue": forecast.round(2),
        }
    )
