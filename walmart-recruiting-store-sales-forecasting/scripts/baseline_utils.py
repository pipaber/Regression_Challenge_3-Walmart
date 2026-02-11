from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from phase1_local_linear import attach_climatology, compute_climatology

TARGET_COL = "sales_anom"
LAG_ORDERS = (1, 2)
EXOG_FEATURE_COLS = [
    "temp_anom",
    "fuel_anom",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
]
ANOM_FEATURE_COLS = [
    "lag1",
    "lag2",
    "week_of_year",
    "month",
    "is_holiday_int",
    *EXOG_FEATURE_COLS,
]


def load_data(train_path: Path, test_path: Path, max_series: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(train_path).copy()
    test = pd.read_parquet(test_path).copy()

    for frame in (train, test):
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Store"] = frame["Store"].astype(int)
        frame["Dept"] = frame["Dept"].astype(int)
        frame["series_id"] = frame["Store"].astype(str) + "_" + frame["Dept"].astype(str)
        frame["week_of_year"] = frame["Date"].dt.isocalendar().week.astype(int)
        frame["month"] = frame["Date"].dt.month.astype(int)
        frame["is_holiday_int"] = frame["IsHoliday"].astype(int)

    if max_series is not None:
        keep_ids = train["series_id"].drop_duplicates().sort_values().head(max_series).tolist()
        train = train[train["series_id"].isin(keep_ids)].copy()
        test = test[test["series_id"].isin(keep_ids)].copy()

    train = train.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)
    test = test.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)
    return train, test


def build_forward_folds(train_df: pd.DataFrame, n_folds: int, val_weeks: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    unique_dates = np.array(sorted(train_df["Date"].dropna().unique()))
    initial_train = len(unique_dates) - (n_folds * val_weeks)
    if initial_train <= 0:
        raise ValueError(
            f"Not enough unique weeks ({len(unique_dates)}) for n_folds={n_folds}, val_weeks={val_weeks}."
        )

    folds: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        train_end = initial_train + fold * val_weeks
        val_end = train_end + val_weeks
        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end:val_end]
        folds.append((fold + 1, train_dates, val_dates))
    return folds


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def wmae(y_true: np.ndarray, y_pred: np.ndarray, holiday: np.ndarray) -> float:
    weights = np.where(holiday, 5.0, 1.0)
    return float(np.average(np.abs(y_true - y_pred), weights=weights))


def summarize_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["Weekly_Sales"].to_numpy(dtype=float)
    y_pred = df["pred_sales"].to_numpy(dtype=float)
    holiday = df["IsHoliday"].to_numpy(dtype=bool)
    metrics = {
        "wmae": wmae(y_true, y_pred, holiday),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "n_rows": float(df.shape[0]),
    }
    if "sales_anom" in df.columns and "pred_anom" in df.columns:
        metrics["anom_mae"] = float(np.mean(np.abs(df["sales_anom"] - df["pred_anom"])))
    return metrics


def _get_lag_value(hist: deque[float], lag: int) -> float:
    if len(hist) >= lag:
        return float(hist[-lag])
    return np.nan


def _robust_bounds(values: np.ndarray, clip_scale: float = 2.0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (-1.0, 1.0)
    q01, q99 = np.quantile(arr, [0.01, 0.99])
    std = float(np.std(arr))
    span = max(float(q99 - q01), std, 1.0)
    lower = float(q01 - clip_scale * span)
    upper = float(q99 + clip_scale * span)
    if lower >= upper:
        center = float(np.median(arr))
        lower, upper = center - 1.0, center + 1.0
    return (lower, upper)


def initialize_target_history(train_df: pd.DataFrame, max_lag: int) -> dict[str, deque[float]]:
    history: dict[str, deque[float]] = {}
    ordered = train_df.sort_values(["series_id", "Date"])
    for series_id, group in ordered.groupby("series_id", sort=False):
        values = group[TARGET_COL].dropna().to_numpy(dtype=float)
        history[series_id] = deque(values.tolist(), maxlen=max_lag)
    return history


def make_supervised_lag_frame(
    train_df: pd.DataFrame,
    lag_orders: tuple[int, ...] = LAG_ORDERS,
    feature_cols: list[str] = ANOM_FEATURE_COLS,
) -> tuple[pd.DataFrame, np.ndarray]:
    out = train_df.sort_values(["series_id", "Date"]).copy()
    grouped = out.groupby("series_id")[TARGET_COL]
    for lag in lag_orders:
        out[f"lag{lag}"] = grouped.shift(lag)

    X = out[feature_cols].copy()
    y = out[TARGET_COL].to_numpy(dtype=float)
    valid_mask = np.isfinite(y) & np.all(np.isfinite(X.to_numpy(dtype=float)), axis=1)
    return X.loc[valid_mask].reset_index(drop=True), y[valid_mask]


def recursive_predict_tabular_anomaly(
    future_df: pd.DataFrame,
    model,
    history: dict[str, deque[float]],
    lag_orders: tuple[int, ...] = LAG_ORDERS,
    feature_cols: list[str] = ANOM_FEATURE_COLS,
    lower: float = -np.inf,
    upper: float = np.inf,
) -> pd.DataFrame:
    max_lag = max(lag_orders)
    ordered = future_df.sort_values(["Date", "series_id"]).copy()
    ordered["row_id"] = np.arange(ordered.shape[0], dtype=int)

    preds: list[tuple[int, float, float]] = []
    for _, block in ordered.groupby("Date", sort=True):
        rows = []
        row_ids = block["row_id"].to_numpy(dtype=int)
        series_ids = block["series_id"].tolist()
        clim_sales = block["clim_sales"].to_numpy(dtype=float)

        for row in block.itertuples(index=False):
            hist = history.get(row.series_id)
            if hist is None:
                hist = deque(maxlen=max_lag)
                history[row.series_id] = hist
            lag_features = {f"lag{lag}": _get_lag_value(hist, lag) for lag in lag_orders}
            rows.append(
                {
                    **lag_features,
                    "week_of_year": float(row.week_of_year),
                    "month": float(row.month),
                    "is_holiday_int": float(row.is_holiday_int),
                    "temp_anom": float(row.temp_anom),
                    "fuel_anom": float(row.fuel_anom),
                    "MarkDown1": float(row.MarkDown1),
                    "MarkDown2": float(row.MarkDown2),
                    "MarkDown3": float(row.MarkDown3),
                    "MarkDown4": float(row.MarkDown4),
                    "MarkDown5": float(row.MarkDown5),
                    "CPI": float(row.CPI),
                    "Unemployment": float(row.Unemployment),
                }
            )

        X_step = pd.DataFrame(rows, columns=feature_cols)
        pred_anom_step = np.asarray(model.predict(X_step), dtype=float)
        pred_anom_step = np.clip(pred_anom_step, lower, upper)
        pred_sales_step = pred_anom_step + clim_sales

        for row_id, series_id, pred_sales, pred_anom in zip(
            row_ids, series_ids, pred_sales_step, pred_anom_step
        ):
            preds.append((int(row_id), float(pred_sales), float(pred_anom)))
            history[series_id].append(float(pred_anom))

    pred_df = pd.DataFrame(preds, columns=["row_id", "pred_sales", "pred_anom"])
    return ordered.merge(pred_df, on="row_id", how="left")


def run_cv_tabular_baseline_anomaly(
    train_df: pd.DataFrame,
    model_builder: Callable[[], object],
    n_folds: int,
    val_weeks: int,
    lag_orders: tuple[int, ...] = LAG_ORDERS,
    feature_cols: list[str] = ANOM_FEATURE_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = build_forward_folds(train_df, n_folds=n_folds, val_weeks=val_weeks)
    metrics_rows: list[dict[str, float | int | str]] = []
    oof_parts: list[pd.DataFrame] = []
    max_lag = max(lag_orders)

    for fold_id, train_dates, val_dates in folds:
        print(f"\nFold {fold_id}: train={len(train_dates)} weeks, val={len(val_dates)} weeks")
        t0 = time.perf_counter()

        fold_train_raw = train_df[train_df["Date"].isin(train_dates)].copy()
        fold_val_raw = train_df[train_df["Date"].isin(val_dates)].copy()

        climatology = compute_climatology(fold_train_raw)
        fold_train = attach_climatology(fold_train_raw, climatology).sort_values(["series_id", "Date"]).copy()
        fold_val = attach_climatology(fold_val_raw, climatology).sort_values(["Date", "series_id"]).copy()

        X_train, y_train = make_supervised_lag_frame(fold_train, lag_orders=lag_orders, feature_cols=feature_cols)
        model = model_builder()
        model.fit(X_train, y_train)

        history = initialize_target_history(fold_train, max_lag=max_lag)
        lower, upper = _robust_bounds(fold_train[TARGET_COL].to_numpy(dtype=float), clip_scale=2.0)
        fold_val_pred = recursive_predict_tabular_anomaly(
            fold_val,
            model,
            history,
            lag_orders=lag_orders,
            feature_cols=feature_cols,
            lower=lower,
            upper=upper,
        )

        fold_metrics = summarize_metrics(fold_val_pred)
        fold_metrics["runtime_sec"] = float(time.perf_counter() - t0)
        fold_metrics["fold"] = fold_id
        fold_metrics["train_start"] = str(pd.Timestamp(train_dates[0]).date())
        fold_metrics["train_end"] = str(pd.Timestamp(train_dates[-1]).date())
        fold_metrics["val_start"] = str(pd.Timestamp(val_dates[0]).date())
        fold_metrics["val_end"] = str(pd.Timestamp(val_dates[-1]).date())
        metrics_rows.append(fold_metrics)

        print(
            f"  wmae={fold_metrics['wmae']:.4f} mae={fold_metrics['mae']:.4f} "
            f"rmse={fold_metrics['rmse']:.4f} runtime={fold_metrics['runtime_sec']:.1f}s"
        )

        keep_cols = [
            "Store",
            "Dept",
            "Date",
            "IsHoliday",
            "Weekly_Sales",
            "sales_anom",
            "pred_anom",
            "pred_sales",
            "series_id",
        ]
        part = fold_val_pred[keep_cols].copy()
        part["fold"] = fold_id
        oof_parts.append(part)

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        mean_row = {"fold": "mean"}
        for col in ["wmae", "mae", "rmse", "anom_mae", "n_rows", "runtime_sec"]:
            if col in metrics_df.columns:
                mean_row[col] = float(metrics_df[col].mean())
        metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return metrics_df, oof_df


def fit_full_train_and_predict_test_tabular_anomaly(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_builder: Callable[[], object],
    lag_orders: tuple[int, ...] = LAG_ORDERS,
    feature_cols: list[str] = ANOM_FEATURE_COLS,
) -> pd.DataFrame:
    climatology = compute_climatology(train_df)
    full_train = attach_climatology(train_df.copy(), climatology).sort_values(["series_id", "Date"]).copy()
    full_test = attach_climatology(test_df.copy(), climatology).sort_values(["Date", "series_id"]).copy()

    X_train, y_train = make_supervised_lag_frame(full_train, lag_orders=lag_orders, feature_cols=feature_cols)
    model = model_builder()
    model.fit(X_train, y_train)

    history = initialize_target_history(full_train, max_lag=max(lag_orders))
    lower, upper = _robust_bounds(full_train[TARGET_COL].to_numpy(dtype=float), clip_scale=2.0)
    test_pred = recursive_predict_tabular_anomaly(
        full_test,
        model,
        history,
        lag_orders=lag_orders,
        feature_cols=feature_cols,
        lower=lower,
        upper=upper,
    )
    return test_pred[
        ["Store", "Dept", "Date", "IsHoliday", "pred_sales", "pred_anom", "series_id"]
    ].copy()
