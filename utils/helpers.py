from __future__ import annotations

import pandas as pd


def format_currency(value: float) -> str:
    """Format numeric values as USD currency."""
    return f"${value:,.2f}"


def month_name(dt_value) -> str:
    """Return month label from a datetime-like value."""
    try:
        return pd.to_datetime(dt_value).strftime("%b %Y")
    except Exception:
        return str(dt_value)
