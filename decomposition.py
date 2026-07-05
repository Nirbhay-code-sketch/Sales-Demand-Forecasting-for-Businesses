"""
evaluation.py
-------------
Standard forecast-accuracy metrics, plus a plain-English translation
of what they mean for business planning.
"""

import numpy as np
import pandas as pd


def mae(actual, predicted):
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def r_squared(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def evaluate_all_models(actual, model_predictions: dict) -> pd.DataFrame:
    """Build a comparison table across every model.

    Args:
        actual: the true values for the test period
        model_predictions: {model_name: predicted_values_array}

    Returns:
        DataFrame sorted by MAPE ascending (best model first), with columns:
        model, mae, rmse, mape, r2
    """
    rows = []
    for name, preds in model_predictions.items():
        rows.append({
            "model": name,
            "mae": round(mae(actual, preds), 2),
            "rmse": round(rmse(actual, preds), 2),
            "mape": round(mape(actual, preds), 2),
            "r2": round(r_squared(actual, preds), 3),
        })
    result = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
    result.index = result.index + 1
    result.index.name = "rank"
    return result


def explain_metrics(best_model_row: pd.Series, avg_actual: float) -> str:
    """Turn the winning model's metrics into a business-friendly sentence."""
    mape_val = best_model_row["mape"]
    mae_val = best_model_row["mae"]
    return (
        f"The best-performing model, {best_model_row['model']}, is off by an average of "
        f"{mae_val:,.0f} units per day ({mape_val:.1f}% of typical daily sales of "
        f"{avg_actual:,.0f} units). In practice this means forecasts can be trusted for "
        f"planning purposes, with a margin of roughly ±{mape_val:.0f}% to build into "
        f"inventory and staffing decisions."
    )
