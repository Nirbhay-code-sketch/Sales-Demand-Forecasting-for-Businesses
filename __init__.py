"""
report.py
---------
Turns model metrics, seasonality facts, and the forecast itself into a
plain-English business summary -- the part a store manager or CFO
actually reads.
"""

import os
import pandas as pd


def build_business_summary(metrics_df: pd.DataFrame, seasonal_facts: dict,
                            forecast_df: pd.DataFrame, avg_actual: float, horizon: int) -> str:
    best = metrics_df.iloc[0]
    total_forecast = forecast_df["forecast"].sum()
    avg_forecast = forecast_df["forecast"].mean()
    change_vs_history = (avg_forecast - avg_actual) / avg_actual * 100 if avg_actual else 0.0

    lines = []
    lines.append(f"BUSINESS SUMMARY — next {horizon} days")
    lines.append("-" * 50)
    lines.append(
        f"Best model: {best['model']} (avg error {best['mae']:.0f} units/day, "
        f"~{best['mape']:.1f}% of typical daily sales)."
    )
    lines.append(
        f"Business is currently {seasonal_facts['trend_direction']} "
        f"({seasonal_facts['trend_pct_change']:+.1f}% over the observed history)."
    )
    lines.append(
        f"Busiest day of the week: {seasonal_facts['best_day']}. "
        f"Slowest day: {seasonal_facts['worst_day']}."
    )
    lines.append(
        f"Strongest month historically: {seasonal_facts['best_month']}. "
        f"Weakest month: {seasonal_facts['worst_month']}."
    )
    lines.append(
        f"Forecasted total demand for the next {horizon} days: {total_forecast:,.0f} units "
        f"(average {avg_forecast:,.0f}/day, {change_vs_history:+.1f}% vs. recent historical average)."
    )
    lines.append("")
    lines.append("Suggested actions:")
    lines.append(f"  - Stock up ahead of {seasonal_facts['best_day']}s and the {seasonal_facts['best_month']} peak season.")
    lines.append(f"  - Consider leaner staffing/inventory around {seasonal_facts['worst_day']}s and {seasonal_facts['worst_month']}.")
    lines.append(f"  - Build a buffer of about {best['mape']:.0f}% into inventory plans to cover forecast uncertainty.")

    return "\n".join(lines)


def export_forecast_csv(forecast_df: pd.DataFrame, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    forecast_df.to_csv(out_path, index=False)
    return out_path


def export_metrics_csv(metrics_df: pd.DataFrame, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    metrics_df.to_csv(out_path)
    return out_path
