from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from phase1_local_linear import attach_climatology, compute_climatology
from baseline_utils import (
    build_forward_folds,
    ensure_output_dir,
    load_data,
    summarize_metrics,
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: ETS anomaly baseline with forward-chaining validation."
    )
    parser.add_argument("--train-path", default="train_feat.parquet")
    parser.add_argument("--test-path", default="test_feat.parquet")
    parser.add_argument("--output-dir", default="outputs/baselines/ets")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-weeks", type=int, default=13)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--seasonal-periods", type=int, default=52)
    parser.add_argument("--exog-alpha", type=float, default=1.0)
    return parser.parse_args()


def make_exog_model(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _safe_score(aic: float | None, sse: float | None) -> float:
    if aic is not None and np.isfinite(aic):
        return float(aic)
    if sse is not None and np.isfinite(sse):
        return float(sse)
    return float("inf")


def forecast_ets_with_fallback(
    train_values: np.ndarray,
    horizon: int,
    seasonal_periods: int,
    fallback_value: float,
) -> tuple[np.ndarray, str]:
    y = np.asarray(train_values, dtype=float)
    y = y[np.isfinite(y)]

    if horizon <= 0:
        return np.empty(0, dtype=float), "empty-horizon"

    if y.size == 0:
        return np.full(horizon, fallback_value, dtype=float), "global-fallback"

    if y.size < 3:
        return np.full(horizon, float(y[-1]), dtype=float), "last-value-fallback"

    candidate_configs: list[dict[str, object]] = [
        {"trend": None, "seasonal": None, "damped_trend": False, "tag": "level"},
        {"trend": "add", "seasonal": None, "damped_trend": False, "tag": "level-trend"},
    ]
    if y.size >= seasonal_periods + 4:
        candidate_configs.append(
            {"trend": None, "seasonal": "add", "damped_trend": False, "tag": "level-seasonal"}
        )
        candidate_configs.append(
            {"trend": "add", "seasonal": "add", "damped_trend": True, "tag": "trend-seasonal-damped"}
        )

    best_forecast: np.ndarray | None = None
    best_score = float("inf")
    best_tag = "last-value-fallback"

    for cfg in candidate_configs:
        try:
            model = ExponentialSmoothing(
                y,
                trend=cfg["trend"],
                seasonal=cfg["seasonal"],
                damped_trend=bool(cfg["damped_trend"]),
                seasonal_periods=seasonal_periods if cfg["seasonal"] else None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True, use_brute=False)
            pred = np.asarray(fit.forecast(horizon), dtype=float)
            score = _safe_score(getattr(fit, "aic", None), getattr(fit, "sse", None))
            if np.all(np.isfinite(pred)) and score < best_score:
                best_score = score
                best_forecast = pred
                best_tag = str(cfg["tag"])
        except Exception:
            continue

    if best_forecast is None:
        return np.full(horizon, float(y[-1]), dtype=float), "last-value-fallback"
    return best_forecast, best_tag


def run_cv(
    train_df: pd.DataFrame,
    n_folds: int,
    val_weeks: int,
    seasonal_periods: int,
    exog_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    folds = build_forward_folds(train_df, n_folds=n_folds, val_weeks=val_weeks)
    metrics_rows: list[dict[str, float | int | str]] = []
    oof_parts: list[pd.DataFrame] = []
    mode_counts: dict[str, int] = {}

    for fold_id, train_dates, val_dates in folds:
        print(f"\nFold {fold_id}: train={len(train_dates)} weeks, val={len(val_dates)} weeks")
        t0 = time.perf_counter()

        fold_train_raw = train_df[train_df["Date"].isin(train_dates)].copy()
        fold_val_raw = train_df[train_df["Date"].isin(val_dates)].copy()
        climatology = compute_climatology(fold_train_raw)
        fold_train = attach_climatology(fold_train_raw, climatology)
        fold_val = attach_climatology(fold_val_raw, climatology)
        exog_model = make_exog_model(exog_alpha)
        exog_model.fit(fold_train[EXOG_FEATURE_COLS], fold_train["sales_anom"].to_numpy(dtype=float))
        fold_train["exog_anom_pred"] = np.asarray(exog_model.predict(fold_train[EXOG_FEATURE_COLS]), dtype=float)
        fold_val["exog_anom_pred"] = np.asarray(exog_model.predict(fold_val[EXOG_FEATURE_COLS]), dtype=float)
        fold_train["resid_anom"] = fold_train["sales_anom"] - fold_train["exog_anom_pred"]
        fallback = float(fold_train["resid_anom"].median())

        fold_preds: list[pd.DataFrame] = []
        val_series_ids = fold_val["series_id"].drop_duplicates().tolist()
        total_series = len(val_series_ids)
        for i, series_id in enumerate(val_series_ids, start=1):
            train_series = fold_train[fold_train["series_id"] == series_id].sort_values("Date")
            val_series = fold_val[fold_val["series_id"] == series_id].sort_values("Date").copy()
            horizon = val_series.shape[0]

            if train_series.empty:
                pred = np.full(horizon, fallback, dtype=float)
                mode = "global-fallback"
            else:
                pred, mode = forecast_ets_with_fallback(
                    train_series["resid_anom"].to_numpy(dtype=float),
                    horizon=horizon,
                    seasonal_periods=seasonal_periods,
                    fallback_value=fallback,
                )
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            val_series["pred_resid_anom"] = pred
            val_series["pred_anom"] = val_series["exog_anom_pred"].to_numpy(dtype=float) + val_series[
                "pred_resid_anom"
            ].to_numpy(dtype=float)
            val_series["pred_sales"] = val_series["pred_anom"] + val_series["clim_sales"]
            fold_preds.append(val_series)

            if i % 400 == 0:
                print(f"  processed {i}/{total_series} series...")

        fold_val_pred = pd.concat(fold_preds, ignore_index=True) if fold_preds else fold_val.assign(pred_sales=np.nan)
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
            mean_row[col] = float(metrics_df[col].mean())
        metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return metrics_df, oof_df, mode_counts


def fit_full_and_predict_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seasonal_periods: int,
    exog_alpha: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    climatology = compute_climatology(train_df)
    full_train = attach_climatology(train_df.copy(), climatology)
    full_test = attach_climatology(test_df.copy(), climatology)
    exog_model = make_exog_model(exog_alpha)
    exog_model.fit(full_train[EXOG_FEATURE_COLS], full_train["sales_anom"].to_numpy(dtype=float))
    full_train["exog_anom_pred"] = np.asarray(exog_model.predict(full_train[EXOG_FEATURE_COLS]), dtype=float)
    full_test["exog_anom_pred"] = np.asarray(exog_model.predict(full_test[EXOG_FEATURE_COLS]), dtype=float)
    full_train["resid_anom"] = full_train["sales_anom"] - full_train["exog_anom_pred"]
    fallback = float(full_train["resid_anom"].median())
    out_parts: list[pd.DataFrame] = []
    mode_counts: dict[str, int] = {}

    test_series_ids = full_test["series_id"].drop_duplicates().tolist()
    total_series = len(test_series_ids)
    for i, series_id in enumerate(test_series_ids, start=1):
        train_series = full_train[full_train["series_id"] == series_id].sort_values("Date")
        test_series = full_test[full_test["series_id"] == series_id].sort_values("Date").copy()
        horizon = test_series.shape[0]

        if train_series.empty:
            pred = np.full(horizon, fallback, dtype=float)
            mode = "global-fallback"
        else:
            pred, mode = forecast_ets_with_fallback(
                train_series["resid_anom"].to_numpy(dtype=float),
                horizon=horizon,
                seasonal_periods=seasonal_periods,
                fallback_value=fallback,
            )
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        test_series["pred_resid_anom"] = pred
        test_series["pred_anom"] = test_series["exog_anom_pred"].to_numpy(dtype=float) + test_series[
            "pred_resid_anom"
        ].to_numpy(dtype=float)
        test_series["pred_sales"] = test_series["pred_anom"] + test_series["clim_sales"]
        out_parts.append(
            test_series[["Store", "Dept", "Date", "IsHoliday", "pred_sales", "pred_anom", "series_id"]]
        )

        if i % 400 == 0:
            print(f"  processed {i}/{total_series} series...")

    pred_df = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    return pred_df, mode_counts


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output_dir = ensure_output_dir(Path(args.output_dir))

    print("Loading data...")
    train_df, test_df = load_data(Path(args.train_path), Path(args.test_path), max_series=args.max_series)
    print(
        f"train rows={train_df.shape[0]}, test rows={test_df.shape[0]}, "
        f"train series={train_df['series_id'].nunique()}, test series={test_df['series_id'].nunique()}"
    )

    print("\nRunning forward-chaining CV...")
    metrics_df, oof_df, cv_mode_counts = run_cv(
        train_df=train_df,
        n_folds=args.n_folds,
        val_weeks=args.val_weeks,
        seasonal_periods=args.seasonal_periods,
        exog_alpha=args.exog_alpha,
    )

    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics -> {metrics_path}")

    oof_path = output_dir / "oof.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"Saved OOF predictions -> {oof_path}")

    print("\nFitting full train and forecasting test...")
    test_pred, test_mode_counts = fit_full_and_predict_test(
        train_df=train_df,
        test_df=test_df,
        seasonal_periods=args.seasonal_periods,
        exog_alpha=args.exog_alpha,
    )
    test_path = output_dir / "test.parquet"
    test_pred.to_parquet(test_path, index=False)
    print(f"Saved test predictions -> {test_path}")

    config = {
        "model": "ets",
        "target": "sales_anom (via exogenous + ETS residual)",
        "seasonal_periods": args.seasonal_periods,
        "exogenous_features": EXOG_FEATURE_COLS,
        "exog_alpha": args.exog_alpha,
        "n_folds": args.n_folds,
        "val_weeks": args.val_weeks,
        "max_series": args.max_series,
        "cv_mode_counts": cv_mode_counts,
        "test_mode_counts": test_mode_counts,
        "train_path": args.train_path,
        "test_path": args.test_path,
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved config -> {config_path}")


if __name__ == "__main__":
    main()
