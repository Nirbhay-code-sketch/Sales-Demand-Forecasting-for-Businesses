#!/usr/bin/env python3
"""
main.py
-------
Command-line entry point for the Sales & Demand Forecasting System.

Usage:
    python main.py --data data/sales_history.csv --horizon 30

Runs the full pipeline:
    1. Load & clean historical sales data
    2. Engineer time-based features (trend, seasonality, lags)
    3. Train & compare 4 forecasting approaches
    4. Evaluate accuracy on a held-out recent period
    5. Forecast the next `horizon` days with the best model
    6. Save charts + a plain-English business summary to outputs/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import build_feature_frame, get_feature_columns
from src.decomposition import decompose, seasonal_summary
from src.forecasting import time_based_split, run_all_models, forecast_future
from src.evaluation import evaluate_all_models
from src.report import build_business_summary, export_forecast_csv, export_metrics_csv
from src.visualization import (
    plot_actual_vs_predicted, plot_full_history_with_forecast,
    plot_decomposition, plot_model_comparison, plot_weekday_effect,
)


def main():
    parser = argparse.ArgumentParser(description="Sales & Demand Forecasting System")
    parser.add_argument("--data", default="data/sales_history.csv", help="Path to historical sales CSV")
    parser.add_argument("--date-col", default="date", help="Name of the date column")
    parser.add_argument("--target-col", default="sales", help="Name of the sales/demand column")
    parser.add_argument("--horizon", type=int, default=30, help="Number of future days to forecast")
    parser.add_argument("--test-size", type=int, default=60, help="Number of most-recent days held out for evaluation")
    parser.add_argument("--out", default="outputs", help="Directory to save charts/CSVs")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Data file not found: {args.data}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print("Loading and preparing data...")
    df = build_feature_frame(args.data, date_col=args.date_col, target_col=args.target_col)
    feature_cols = get_feature_columns(df)

    if len(df) <= args.test_size + 30:
        print("Not enough historical data for the requested test size. Reduce --test-size.")
        sys.exit(1)

    print("Splitting into train/test (time-ordered)...")
    train, test = time_based_split(df, args.test_size)

    print("Training models: Naive, Moving Average, Linear Regression, Random Forest...")
    predictions, fitted_models = run_all_models(train, test, feature_cols, target_col="sales")

    print("Evaluating models on the held-out period...")
    metrics_df = evaluate_all_models(test["sales"].values, predictions)
    print("\nMODEL COMPARISON (sorted by MAPE, lower is better):")
    print(metrics_df.to_string())

    best_model_name = metrics_df.iloc[0]["model"]
    print(f"\nBest model: {best_model_name}")

    print("Analyzing trend & seasonality...")
    decomposed = decompose(df, target_col="sales")
    facts = seasonal_summary(df, target_col="sales")

    print(f"Forecasting the next {args.horizon} days...")
    if best_model_name in fitted_models:
        best_model = fitted_models[best_model_name]
        forecast_df = forecast_future(df, best_model, feature_cols, horizon=args.horizon, target_col="sales")
    else:
        # Fall back to Random Forest if the winning model was a
        # non-regression baseline (Naive / Moving Average can't
        # recursively forecast far into the future on their own).
        fallback_name = "Random Forest" if "Random Forest" in fitted_models else list(fitted_models.keys())[0]
        print(f"({best_model_name} can't project multiple steps ahead by itself; "
              f"using {fallback_name} to generate the forward-looking forecast.)")
        forecast_df = forecast_future(df, fitted_models[fallback_name], feature_cols,
                                       horizon=args.horizon, target_col="sales")

    avg_actual = df["sales"].mean()
    summary_text = build_business_summary(metrics_df, facts, forecast_df, avg_actual, args.horizon)
    print("\n" + "=" * 60)
    print(summary_text)
    print("=" * 60)

    # ---- Save outputs ----
    print("\nSaving charts and results...")

    best_preds = predictions[best_model_name]
    plot_actual_vs_predicted(test["date"], test["sales"], best_preds, best_model_name,
                              save_path=os.path.join(args.out, "actual_vs_predicted.png"))
    plot_full_history_with_forecast(df["date"], df["sales"], forecast_df["date"], forecast_df["forecast"],
                                     save_path=os.path.join(args.out, "forecast.png"))
    plot_decomposition(decomposed, save_path=os.path.join(args.out, "decomposition.png"))
    plot_model_comparison(metrics_df, save_path=os.path.join(args.out, "model_comparison.png"))
    plot_weekday_effect(facts["dow_effect"], save_path=os.path.join(args.out, "weekday_effect.png"))

    export_forecast_csv(forecast_df, os.path.join(args.out, "forecast.csv"))
    export_metrics_csv(metrics_df, os.path.join(args.out, "model_metrics.csv"))

    with open(os.path.join(args.out, "business_summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\nAll outputs saved to: {args.out}/")
    print("  - forecast.csv, model_metrics.csv, business_summary.txt")
    print("  - actual_vs_predicted.png, forecast.png, decomposition.png,")
    print("    model_comparison.png, weekday_effect.png")


if __name__ == "__main__":
    main()
