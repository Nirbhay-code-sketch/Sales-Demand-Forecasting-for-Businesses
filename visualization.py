"""
decomposition.py
------------------
A lightweight, dependency-free trend + seasonality decomposition.

Real business stakeholders want to know: "is the business growing?"
and "are there predictable patterns (weekday, month) we can plan
around?". This module answers both without needing `statsmodels` --
just pandas rolling windows and group-by averages, which keeps the
whole project runnable with zero extra installs.

decompose() returns:
    trend      -> smoothed long-run level (centered rolling mean)
    seasonal   -> the average recurring effect of each calendar period
                  (e.g. each day-of-week) after removing the trend
    residual   -> what's left over (noise / one-off events)
"""

import numpy as np
import pandas as pd


def decompose(df: pd.DataFrame, target_col: str = "sales",
              seasonal_period: int = 7, seasonal_key: str = "day_of_week") -> pd.DataFrame:
    """Additive decomposition: sales = trend + seasonal + residual.

    Args:
        df: DataFrame with a datetime 'date' column and the target column.
        target_col: column to decompose.
        seasonal_period: window size used to smooth out the trend
            (7 for weekly seasonality, 30 for monthly).
        seasonal_key: the column to group by when estimating the
            recurring seasonal effect (e.g. 'day_of_week' or 'month').
    """
    out = df.copy()

    # Trend: centered rolling mean smooths out the seasonal wiggle,
    # leaving the underlying growth/decline pattern.
    out["trend"] = out[target_col].rolling(
        window=seasonal_period, center=True, min_periods=max(2, seasonal_period // 2)
    ).mean()
    out["trend"] = out["trend"].interpolate(limit_direction="both")

    # Detrended series
    detrended = out[target_col] - out["trend"]

    # Seasonal: average detrended value per seasonal bucket (e.g. per weekday)
    seasonal_avg = detrended.groupby(out[seasonal_key]).transform("mean")
    out["seasonal"] = seasonal_avg

    # Residual: whatever the trend + seasonal pattern doesn't explain
    out["residual"] = out[target_col] - out["trend"] - out["seasonal"]

    return out


def seasonal_summary(df: pd.DataFrame, target_col: str = "sales") -> dict:
    """Produce plain-English seasonality facts for a business audience."""
    decomposed = decompose(df, target_col=target_col, seasonal_period=7, seasonal_key="day_of_week")

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_effect = decomposed.groupby("day_of_week")["seasonal"].mean()
    best_day = dow_names[int(dow_effect.idxmax())]
    worst_day = dow_names[int(dow_effect.idxmin())]

    month_decomposed = decompose(df, target_col=target_col, seasonal_period=30, seasonal_key="month")
    month_names = ["January", "February", "March", "April", "May", "June", "July",
                   "August", "September", "October", "November", "December"]
    month_effect = month_decomposed.groupby("month")["seasonal"].mean()
    best_month = month_names[int(month_effect.idxmax()) - 1]
    worst_month = month_names[int(month_effect.idxmin()) - 1]

    # Overall trend direction: compare first vs last smoothed trend value
    trend_series = decomposed["trend"].dropna()
    trend_change = trend_series.iloc[-1] - trend_series.iloc[0]
    trend_direction = "growing" if trend_change > 0 else ("declining" if trend_change < 0 else "flat")
    pct_change = (trend_change / trend_series.iloc[0] * 100) if trend_series.iloc[0] != 0 else 0.0

    return {
        "best_day": best_day,
        "worst_day": worst_day,
        "best_month": best_month,
        "worst_month": worst_month,
        "trend_direction": trend_direction,
        "trend_pct_change": round(float(pct_change), 1),
        "dow_effect": {dow_names[i]: round(float(v), 1) for i, v in dow_effect.items()},
        "month_effect": {month_names[i - 1]: round(float(v), 1) for i, v in month_effect.items()},
    }
