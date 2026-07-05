"""
visualization.py
------------------
Business-friendly charts: everything a stakeholder needs to trust and
act on the forecast, without needing to read any code or metrics table.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


COLORS = {
    "actual": "#2b3a55",
    "forecast": "#c97a2b",
    "trend": "#5b7fb5",
    "seasonal": "#8fb339",
    "residual": "#b5533c",
    "grid": "#e6e2d8",
}


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def plot_actual_vs_predicted(dates, actual, predicted, model_name, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, actual, label="Actual sales", color=COLORS["actual"], linewidth=2)
    ax.plot(dates, predicted, label=f"{model_name} prediction", color=COLORS["forecast"],
            linewidth=2, linestyle="--")
    ax.set_title(f"Actual vs Predicted — {model_name} (test period)")
    ax.set_ylabel("Sales")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    _style_ax(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_full_history_with_forecast(history_dates, history_values, forecast_dates, forecast_values,
                                     save_path=None):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(history_dates, history_values, label="Historical sales", color=COLORS["actual"], linewidth=1.6)
    ax.plot(forecast_dates, forecast_values, label="Forecast", color=COLORS["forecast"],
            linewidth=2.2, linestyle="--")
    ax.axvline(history_dates.iloc[-1] if hasattr(history_dates, "iloc") else history_dates[-1],
               color="#999", linestyle=":", linewidth=1)
    ax.set_title("Sales History and Forecast")
    ax.set_ylabel("Sales")
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _style_ax(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_decomposition(decomposed_df, target_col="sales", save_path=None):
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(decomposed_df["date"], decomposed_df[target_col], color=COLORS["actual"], linewidth=1.4)
    axes[0].set_title("Observed sales")

    axes[1].plot(decomposed_df["date"], decomposed_df["trend"], color=COLORS["trend"], linewidth=1.8)
    axes[1].set_title("Trend (smoothed long-run level)")

    axes[2].plot(decomposed_df["date"], decomposed_df["seasonal"], color=COLORS["seasonal"], linewidth=1.2)
    axes[2].set_title("Seasonality (recurring weekly pattern)")

    axes[3].scatter(decomposed_df["date"], decomposed_df["residual"], color=COLORS["residual"], s=6, alpha=0.6)
    axes[3].axhline(0, color="#999", linewidth=1)
    axes[3].set_title("Residual (unexplained noise / one-off events)")

    for ax in axes:
        _style_ax(ax)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_model_comparison(metrics_df, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = metrics_df["model"][::-1]
    mape_vals = metrics_df["mape"][::-1]
    bars = ax.barh(models, mape_vals, color=COLORS["trend"])
    for bar, val in zip(bars, mape_vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", fontsize=9)
    ax.set_xlabel("Forecast error (MAPE %) — lower is better")
    ax.set_title("Model Comparison")
    _style_ax(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_weekday_effect(dow_effect: dict, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    days = list(dow_effect.keys())
    values = list(dow_effect.values())
    colors = [COLORS["seasonal"] if v >= 0 else COLORS["residual"] for v in values]
    ax.bar(days, values, color=colors)
    ax.axhline(0, color="#999", linewidth=1)
    ax.set_title("Average Sales Effect by Day of Week")
    ax.set_ylabel("Effect vs. trend")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _style_ax(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
