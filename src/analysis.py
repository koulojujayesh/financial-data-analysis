from __future__ import annotations

import pandas as pd


def add_profit_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with computed profit."""
    analyzed = df.copy()
    if {"revenue", "expenses"}.issubset(analyzed.columns):
        analyzed["profit"] = analyzed["revenue"] - analyzed["expenses"]
    return analyzed


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute summary KPIs for dashboard cards."""
    if df.empty:
        return {"total_revenue": 0.0, "total_expenses": 0.0, "total_profit": 0.0, "margin_pct": 0.0}

    total_revenue = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
    total_expenses = float(df["expenses"].sum()) if "expenses" in df.columns else 0.0
    total_profit = total_revenue - total_expenses
    margin_pct = (total_profit / total_revenue * 100.0) if total_revenue else 0.0

    return {
        "total_revenue": total_revenue,
        "total_expenses": total_expenses,
        "total_profit": total_profit,
        "margin_pct": margin_pct,
    }
