"""
database.py
------------
Lightweight SQLite persistence layer for the forecasting app.

Why SQLite?
- Zero setup: no server to install or configure, the whole database is
  one file (data/forecasting.db), which keeps the project runnable
  anywhere Python runs.
- Good enough for this use case: a single business user (or small team)
  saving datasets and forecast runs, not a high-concurrency system.

Tables:
    datasets          -> one row per uploaded/used dataset (name, when, row count)
    sales_records     -> the actual historical rows belonging to a dataset
    forecast_runs     -> one row per "Run forecast" click (model, horizon, metrics)
    forecast_points   -> the predicted values produced by a run
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "forecasting.db")
DB_PATH = os.path.abspath(DB_PATH)


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source TEXT NOT NULL,             -- 'sample' or 'upload'
            row_count INTEGER,
            date_range_start TEXT,
            date_range_end TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            sales REAL NOT NULL,
            extra_json TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            dataset_id INTEGER NOT NULL,
            best_model TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            test_size INTEGER NOT NULL,
            mae REAL, rmse REAL, mape REAL, r2 REAL,
            trend_direction TEXT,
            best_day TEXT,
            worst_day TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            date TEXT NOT NULL,
            forecast_value REAL NOT NULL,
            FOREIGN KEY (run_id) REFERENCES forecast_runs (run_id)
        )
    """)

    conn.commit()
    conn.close()


def save_dataset(name: str, source: str, df) -> int:
    """Persist a dataset's historical rows. Returns the new dataset_id."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO datasets (name, source, row_count, date_range_start, date_range_end, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, source, len(df), str(df["date"].min()), str(df["date"].max()),
         datetime.utcnow().isoformat()),
    )
    dataset_id = cur.lastrowid

    extra_cols = [c for c in df.columns if c not in ("date", "sales")]
    rows = []
    for _, row in df.iterrows():
        extra = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in extra_cols}
        rows.append((dataset_id, str(row["date"].date()), float(row["sales"]), json.dumps(extra)))

    cur.executemany(
        "INSERT INTO sales_records (dataset_id, date, sales, extra_json) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return dataset_id


def save_forecast_run(run_id: str, dataset_id: int, best_model: str, horizon: int, test_size: int,
                       metrics: dict, seasonality: dict, forecast_df) -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """INSERT INTO forecast_runs
           (run_id, dataset_id, best_model, horizon, test_size, mae, rmse, mape, r2,
            trend_direction, best_day, worst_day, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, dataset_id, best_model, horizon, test_size,
         metrics.get("mae"), metrics.get("rmse"), metrics.get("mape"), metrics.get("r2"),
         seasonality.get("trend_direction"), seasonality.get("best_day"), seasonality.get("worst_day"),
         datetime.utcnow().isoformat()),
    )

    rows = [(run_id, str(row["date"].date()), float(row["forecast"])) for _, row in forecast_df.iterrows()]
    cur.executemany(
        "INSERT INTO forecast_points (run_id, date, forecast_value) VALUES (?, ?, ?)",
        rows,
    )

    conn.commit()
    conn.close()


def get_recent_runs(limit: int = 10):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT fr.run_id, d.name AS dataset_name, fr.best_model, fr.horizon,
               fr.mape, fr.r2, fr.trend_direction, fr.created_at
        FROM forecast_runs fr
        JOIN datasets d ON d.id = fr.dataset_id
        ORDER BY fr.created_at DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_forecast_points(run_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT date, forecast_value FROM forecast_points WHERE run_id = ? ORDER BY date", (run_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
