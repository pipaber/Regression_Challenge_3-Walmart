from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COL = "sales_anom"
LAG_ORDERS = (1, 2)
LAG_FEATURE_COLS = [f"sales_anom_lag{lag}" for lag in LAG_ORDERS]
FULL_EXOG_FEATURE_COLS = [
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
LITE_EXOG_FEATURE_COLS = [
    "temp_anom",
    "fuel_anom",
]
CORE_FEATURE_COLS = [
    "sales_anom_lag1",
    "sales_anom_lag2",
    "is_holiday_int",
]
INTERACTION_FEATURE_COLS = [
    "holiday_x_lag1",
]
WEEK_PERIOD = 53


@dataclass(slots=True)
class LocalLinearConfig:
    kernel: str
    bandwidth: int
    min_samples: int
    ridge: float
    coef_clip: float
    anom_clip_scale: float
    use_interactions: bool
    feature_mode: str


@dataclass(slots=True)
class FittedLocalLinear:
    local_params: dict[tuple[str, int], np.ndarray]
    available_weeks: dict[str, np.ndarray]
    series_fallback: dict[str, np.ndarray]
    global_fallback: np.ndarray
    series_bounds: dict[str, tuple[float, float]]
    global_bounds: tuple[float, float]
    feature_cols: list[str]
    feature_means: np.ndarray
    feature_stds: np.ndarray


def get_feature_cols(config: LocalLinearConfig) -> list[str]:
    cols = list(get_exog_feature_cols(config)) + list(CORE_FEATURE_COLS)
    if config.use_interactions:
        cols.extend(INTERACTION_FEATURE_COLS)
    return cols


def get_exog_feature_cols(config: LocalLinearConfig) -> list[str]:
    if config.feature_mode == "lite":
        return list(LITE_EXOG_FEATURE_COLS)
    return list(FULL_EXOG_FEATURE_COLS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Local linear regression on anomalies with forward-chaining CV."
    )
    parser.add_argument("--train-path", default="train_feat.parquet")
    parser.add_argument("--test-path", default="test_feat.parquet")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--kernel", choices=["tricube", "exponential"], default="tricube")
    parser.add_argument("--bandwidth", type=int, default=6, help="Neighborhood width in weeks.")
    parser.add_argument("--min-samples", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--coef-clip", type=float, default=6.0, help="Absolute clip for non-intercept coefficients.")
    parser.add_argument(
        "--anom-clip-scale",
        type=float,
        default=2.0,
        help="How wide recursive anomaly clipping bounds are around train quantiles.",
    )
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-weeks", type=int, default=13)
    parser.add_argument("--max-series", type=int, default=None, help="Optional debug subset.")
    parser.add_argument(
        "--use-interactions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use targeted interaction term (holiday x lag1).",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["full", "lite"],
        default="full",
        help="`full` uses all exogenous features; `lite` keeps only temp/fuel exogenous features.",
    )
    return parser.parse_args()


def weighted_ridge_fit(X: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    if X.size == 0:
        return np.zeros(X.shape[1] + 1, dtype=float)

    design = np.column_stack([np.ones(X.shape[0], dtype=float), X])
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
    Xw = design * sqrt_w[:, None]
    yw = y * sqrt_w

    reg = np.eye(Xw.shape[1], dtype=float) * ridge
    reg[0, 0] = 0.0

    xtx = Xw.T @ Xw + reg
    xty = Xw.T @ yw
    beta = np.linalg.pinv(xtx) @ xty
    return beta.astype(float, copy=False)


def regularize_beta(beta: np.ndarray, coef_clip: float) -> np.ndarray:
    out = beta.copy()
    out[1:] = np.clip(out[1:], -coef_clip, coef_clip)
    return out


def cyclic_week_distance(weeks: np.ndarray, target_week: int) -> np.ndarray:
    delta = np.abs(weeks - target_week)
    return np.minimum(delta, WEEK_PERIOD - delta)


def kernel_weights(distance: np.ndarray, bandwidth: int, kernel: str) -> np.ndarray:
    if kernel == "tricube":
        scaled = np.clip(distance / float(bandwidth), 0.0, 1.0)
        weights = (1.0 - scaled**3) ** 3
        weights[distance > bandwidth] = 0.0
        return weights
    if kernel == "exponential":
        return np.exp(-0.5 * (distance / float(bandwidth)) ** 2)
    raise ValueError(f"Unsupported kernel: {kernel}")


def robust_bounds(values: np.ndarray, clip_scale: float) -> tuple[float, float]:
    if values.size == 0:
        return (-1.0, 1.0)
    q01, q99 = np.quantile(values, [0.01, 0.99])
    std = float(np.std(values))
    span = max(float(q99 - q01), std, 1.0)
    lower = float(q01 - clip_scale * span)
    upper = float(q99 + clip_scale * span)
    if lower >= upper:
        center = float(np.median(values))
        lower, upper = center - 1.0, center + 1.0
    return (lower, upper)


def fit_standard_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    bad = (~np.isfinite(stds)) | (stds < 1e-8)
    stds[bad] = 1.0
    means[~np.isfinite(means)] = 0.0
    return means.astype(float, copy=False), stds.astype(float, copy=False)


def apply_standard_scaler(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (X - means) / stds


def load_data(train_path: Path, test_path: Path, max_series: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    for frame in (train, test):
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Store"] = frame["Store"].astype(int)
        frame["Dept"] = frame["Dept"].astype(int)
        frame["series_id"] = frame["Store"].astype(str) + "_" + frame["Dept"].astype(str)
        frame["week_of_year"] = frame["Date"].dt.isocalendar().week.astype(int)
        frame["is_holiday_int"] = frame["IsHoliday"].astype(int)

    if max_series is not None:
        keep_ids = (
            train["series_id"]
            .drop_duplicates()
            .sort_values()
            .head(max_series)
            .tolist()
        )
        train = train[train["series_id"].isin(keep_ids)].copy()
        test = test[test["series_id"].isin(keep_ids)].copy()

    train = train.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)
    test = test.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)
    return train, test


def build_forward_folds(
    train_df: pd.DataFrame, n_folds: int, val_weeks: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
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


def compute_climatology(train_df: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, float]]:
    series = (
        train_df.groupby(["series_id", "week_of_year"], as_index=False)[
            ["Weekly_Sales", "Temperature", "Fuel_Price"]
        ]
        .median()
        .rename(
            columns={
                "Weekly_Sales": "clim_sales_series",
                "Temperature": "clim_temp_series",
                "Fuel_Price": "clim_fuel_series",
            }
        )
    )
    store = (
        train_df.groupby(["Store", "week_of_year"], as_index=False)[
            ["Weekly_Sales", "Temperature", "Fuel_Price"]
        ]
        .median()
        .rename(
            columns={
                "Weekly_Sales": "clim_sales_store",
                "Temperature": "clim_temp_store",
                "Fuel_Price": "clim_fuel_store",
            }
        )
    )
    global_week = (
        train_df.groupby("week_of_year", as_index=False)[["Weekly_Sales", "Temperature", "Fuel_Price"]]
        .median()
        .rename(
            columns={
                "Weekly_Sales": "clim_sales_global",
                "Temperature": "clim_temp_global",
                "Fuel_Price": "clim_fuel_global",
            }
        )
    )
    overall = {
        "sales": float(train_df["Weekly_Sales"].median()),
        "temp": float(train_df["Temperature"].median()),
        "fuel": float(train_df["Fuel_Price"].median()),
    }
    return {"series": series, "store": store, "global_week": global_week, "overall": overall}


def attach_climatology(
    df: pd.DataFrame, climatology: dict[str, pd.DataFrame | dict[str, float]]
) -> pd.DataFrame:
    out = df.merge(
        climatology["series"], on=["series_id", "week_of_year"], how="left"  # type: ignore[arg-type]
    )
    out = out.merge(climatology["store"], on=["Store", "week_of_year"], how="left")  # type: ignore[arg-type]
    out = out.merge(climatology["global_week"], on=["week_of_year"], how="left")  # type: ignore[arg-type]

    overall = climatology["overall"]  # type: ignore[assignment]
    out["clim_sales"] = (
        out["clim_sales_series"].fillna(out["clim_sales_store"]).fillna(out["clim_sales_global"]).fillna(overall["sales"])
    )
    out["clim_temp"] = (
        out["clim_temp_series"].fillna(out["clim_temp_store"]).fillna(out["clim_temp_global"]).fillna(overall["temp"])
    )
    out["clim_fuel"] = (
        out["clim_fuel_series"].fillna(out["clim_fuel_store"]).fillna(out["clim_fuel_global"]).fillna(overall["fuel"])
    )

    out["temp_anom"] = out["Temperature"] - out["clim_temp"]
    out["fuel_anom"] = out["Fuel_Price"] - out["clim_fuel"]
    if "Weekly_Sales" in out.columns:
        out["sales_anom"] = out["Weekly_Sales"] - out["clim_sales"]

    drop_cols = [
        "clim_sales_series",
        "clim_temp_series",
        "clim_fuel_series",
        "clim_sales_store",
        "clim_temp_store",
        "clim_fuel_store",
        "clim_sales_global",
        "clim_temp_global",
        "clim_fuel_global",
    ]
    return out.drop(columns=drop_cols, errors="ignore")


def add_anomaly_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["series_id", "Date"]).copy()
    grp = out.groupby("series_id")["sales_anom"]
    for lag in LAG_ORDERS:
        out[f"sales_anom_lag{lag}"] = grp.shift(lag)
    out["holiday_x_lag1"] = out["is_holiday_int"] * out["sales_anom_lag1"]
    return out


def fit_local_models(train_df: pd.DataFrame, config: LocalLinearConfig) -> FittedLocalLinear:
    feature_cols = get_feature_cols(config)
    valid = train_df.copy()
    # Keep early rows by treating unavailable long-lag anomalies as neutral (0).
    for c in LAG_FEATURE_COLS + INTERACTION_FEATURE_COLS:
        if c in valid.columns:
            valid[c] = valid[c].fillna(0.0)
    for c in get_exog_feature_cols(config):
        if c in valid.columns and valid[c].isna().any():
            median = float(valid[c].median())
            if not np.isfinite(median):
                median = 0.0
            valid[c] = valid[c].fillna(median)
    valid = valid.dropna(subset=feature_cols + [TARGET_COL]).copy()
    if valid.empty:
        zero_beta = np.zeros(len(feature_cols) + 1, dtype=float)
        return FittedLocalLinear(
            {},
            {},
            {},
            zero_beta,
            {},
            (-1.0, 1.0),
            feature_cols,
            np.zeros(len(feature_cols), dtype=float),
            np.ones(len(feature_cols), dtype=float),
        )

    all_X_raw = valid[feature_cols].to_numpy(dtype=float)
    scaler_means, scaler_stds = fit_standard_scaler(all_X_raw)
    all_X = apply_standard_scaler(all_X_raw, scaler_means, scaler_stds)
    all_y = valid[TARGET_COL].to_numpy(dtype=float)
    global_fallback = regularize_beta(
        weighted_ridge_fit(
        all_X, all_y, np.ones(all_y.shape[0], dtype=float), ridge=config.ridge
        ),
        coef_clip=config.coef_clip,
    )

    anom_values = train_df[TARGET_COL].dropna().to_numpy(dtype=float)
    global_bounds = robust_bounds(anom_values, clip_scale=config.anom_clip_scale)
    series_bounds: dict[str, tuple[float, float]] = {}

    local_params: dict[tuple[str, int], np.ndarray] = {}
    available_weeks: dict[str, np.ndarray] = {}
    series_fallback: dict[str, np.ndarray] = {}

    grouped = valid.groupby("series_id", sort=False)
    for i, (series_id, group) in enumerate(grouped, start=1):
        g = group.sort_values("Date")
        X_raw = g[feature_cols].to_numpy(dtype=float)
        X = apply_standard_scaler(X_raw, scaler_means, scaler_stds)
        y = g[TARGET_COL].to_numpy(dtype=float)
        weeks = g["week_of_year"].to_numpy(dtype=int)
        series_bounds[series_id] = robust_bounds(y, clip_scale=config.anom_clip_scale)

        if y.shape[0] >= 3:
            series_fallback[series_id] = regularize_beta(
                weighted_ridge_fit(X, y, np.ones(y.shape[0], dtype=float), ridge=config.ridge),
                coef_clip=config.coef_clip,
            )
        else:
            series_fallback[series_id] = global_fallback

        learned_weeks: list[int] = []
        for target_week in range(1, WEEK_PERIOD + 1):
            distance = cyclic_week_distance(weeks, target_week)
            weights = kernel_weights(distance, config.bandwidth, config.kernel)
            active = weights > 0.0
            if int(active.sum()) < config.min_samples:
                continue
            beta = regularize_beta(
                weighted_ridge_fit(X[active], y[active], weights[active], ridge=config.ridge),
                coef_clip=config.coef_clip,
            )
            local_params[(series_id, target_week)] = beta
            learned_weeks.append(target_week)

        if learned_weeks:
            available_weeks[series_id] = np.array(learned_weeks, dtype=int)

        if i % 300 == 0:
            print(f"  fitted models for {i} series...")

    return FittedLocalLinear(
        local_params,
        available_weeks,
        series_fallback,
        global_fallback,
        series_bounds,
        global_bounds,
        feature_cols,
        scaler_means,
        scaler_stds,
    )


def _lag_from_history(hist: deque[float], lag: int) -> float:
    if len(hist) >= lag:
        return float(hist[-lag])
    return 0.0


def initialize_history(train_df: pd.DataFrame) -> dict[str, deque[float]]:
    history: dict[str, deque[float]] = {}
    max_lag = max(LAG_ORDERS)
    ordered = train_df.sort_values(["series_id", "Date"])
    for series_id, group in ordered.groupby("series_id", sort=False):
        values = group["sales_anom"].dropna().to_numpy(dtype=float)
        history[series_id] = deque(values.tolist(), maxlen=max_lag)
    return history


def pick_beta(model: FittedLocalLinear, series_id: str, week_of_year: int) -> np.ndarray:
    direct = model.local_params.get((series_id, week_of_year))
    if direct is not None:
        return direct

    learned = model.available_weeks.get(series_id)
    if learned is not None and learned.size > 0:
        delta = np.abs(learned - week_of_year)
        cyclic = np.minimum(delta, WEEK_PERIOD - delta)
        nearest_week = int(learned[np.argmin(cyclic)])
        return model.local_params[(series_id, nearest_week)]

    series_fb = model.series_fallback.get(series_id)
    if series_fb is not None:
        return series_fb
    return model.global_fallback


def _build_feature_vector(row, hist: deque[float], feature_cols: list[str]) -> np.ndarray:
    def safe_float(value: float) -> float:
        value = float(value)
        if np.isfinite(value):
            return value
        return 0.0

    lag_vals = {
        "sales_anom_lag1": _lag_from_history(hist, 1),
        "sales_anom_lag2": _lag_from_history(hist, 2),
    }
    feat = {
        "temp_anom": safe_float(row.temp_anom),
        "fuel_anom": safe_float(row.fuel_anom),
        "MarkDown1": safe_float(row.MarkDown1),
        "MarkDown2": safe_float(row.MarkDown2),
        "MarkDown3": safe_float(row.MarkDown3),
        "MarkDown4": safe_float(row.MarkDown4),
        "MarkDown5": safe_float(row.MarkDown5),
        "CPI": safe_float(row.CPI),
        "Unemployment": safe_float(row.Unemployment),
        "is_holiday_int": safe_float(row.is_holiday_int),
        **lag_vals,
        "holiday_x_lag1": safe_float(row.is_holiday_int) * lag_vals["sales_anom_lag1"],
    }
    return np.array([feat[c] for c in feature_cols], dtype=float)


def recursive_predict(
    future_df: pd.DataFrame, model: FittedLocalLinear, history: dict[str, deque[float]]
) -> pd.DataFrame:
    ordered = future_df.sort_values(["Date", "series_id"]).copy()
    preds: list[tuple[int, float, float]] = []
    max_lag = max(LAG_ORDERS)

    for row in ordered.itertuples(index=False):
        series_id = row.series_id
        hist = history.get(series_id)
        if hist is None:
            hist = deque(maxlen=max_lag)
            history[series_id] = hist

        beta = pick_beta(model, series_id, int(row.week_of_year))
        x_raw = _build_feature_vector(row, hist, model.feature_cols)
        x = apply_standard_scaler(x_raw, model.feature_means, model.feature_stds)
        pred_anom = float(beta[0] + np.dot(beta[1:], x))
        lower, upper = model.series_bounds.get(series_id, model.global_bounds)
        pred_anom = float(np.clip(pred_anom, lower, upper))
        pred_sales = float(pred_anom + row.clim_sales)

        preds.append((int(row.row_id), pred_sales, pred_anom))
        hist.append(pred_anom)

    pred_df = pd.DataFrame(preds, columns=["row_id", "pred_sales", "pred_anom"])
    return future_df.merge(pred_df, on="row_id", how="left")


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


def run_cv(
    train_df: pd.DataFrame, config: LocalLinearConfig, n_folds: int, val_weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = build_forward_folds(train_df, n_folds=n_folds, val_weeks=val_weeks)
    metrics_rows: list[dict[str, float | int | str]] = []
    oof_parts: list[pd.DataFrame] = []

    for fold_id, train_dates, val_dates in folds:
        print(f"\nFold {fold_id}: train={len(train_dates)} weeks, val={len(val_dates)} weeks")

        fold_train = train_df[train_df["Date"].isin(train_dates)].copy()
        fold_val = train_df[train_df["Date"].isin(val_dates)].copy()
        fold_val["row_id"] = np.arange(fold_val.shape[0], dtype=int)

        climatology = compute_climatology(fold_train)
        fold_train = add_anomaly_lags(attach_climatology(fold_train, climatology))

        model = fit_local_models(fold_train, config)
        history = initialize_history(fold_train)

        fold_val = attach_climatology(fold_val, climatology)
        fold_val_pred = recursive_predict(fold_val, model, history)

        fold_metrics = summarize_metrics(fold_val_pred)
        fold_metrics["fold"] = fold_id
        fold_metrics["train_start"] = str(pd.Timestamp(train_dates[0]).date())
        fold_metrics["train_end"] = str(pd.Timestamp(train_dates[-1]).date())
        fold_metrics["val_start"] = str(pd.Timestamp(val_dates[0]).date())
        fold_metrics["val_end"] = str(pd.Timestamp(val_dates[-1]).date())
        metrics_rows.append(fold_metrics)

        print(
            f"  wmae={fold_metrics['wmae']:.4f} mae={fold_metrics['mae']:.4f} "
            f"rmse={fold_metrics['rmse']:.4f}"
        )

        keep_cols = [
            "Store",
            "Dept",
            "Date",
            "IsHoliday",
            "Weekly_Sales",
            "sales_anom",
            "pred_sales",
            "pred_anom",
        ]
        part = fold_val_pred[keep_cols].copy()
        part["fold"] = fold_id
        oof_parts.append(part)

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        mean_row = {"fold": "mean"}
        for col in ["wmae", "mae", "rmse", "anom_mae", "n_rows"]:
            if col in metrics_df.columns:
                mean_row[col] = float(metrics_df[col].mean())
        metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return metrics_df, oof_df


def fit_full_and_predict_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: LocalLinearConfig
) -> pd.DataFrame:
    train_full = train_df.copy()
    test_full = test_df.copy()
    test_full["row_id"] = np.arange(test_full.shape[0], dtype=int)

    climatology = compute_climatology(train_full)
    train_full = add_anomaly_lags(attach_climatology(train_full, climatology))
    model = fit_local_models(train_full, config)
    history = initialize_history(train_full)

    test_full = attach_climatology(test_full, climatology)
    pred = recursive_predict(test_full, model, history)
    return pred


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "root": base,
        "metrics": base / "metrics",
        "preds": base / "preds",
        "submissions": base / "submissions",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def make_submission(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["Id"] = (
        out["Store"].astype(int).astype(str)
        + "_"
        + out["Dept"].astype(int).astype(str)
        + "_"
        + out["Date"].dt.strftime("%Y-%m-%d")
    )
    submission = out[["Id", "pred_sales"]].rename(columns={"pred_sales": "Weekly_Sales"})
    return submission


def main() -> None:
    args = parse_args()
    config = LocalLinearConfig(
        kernel=args.kernel,
        bandwidth=args.bandwidth,
        min_samples=args.min_samples,
        ridge=args.ridge,
        coef_clip=args.coef_clip,
        anom_clip_scale=args.anom_clip_scale,
        use_interactions=args.use_interactions,
        feature_mode=args.feature_mode,
    )

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    output_paths = ensure_dirs(Path(args.output_dir))

    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path, max_series=args.max_series)
    print(
        f"train rows={train_df.shape[0]}, test rows={test_df.shape[0]}, "
        f"series train={train_df['series_id'].nunique()}, series test={test_df['series_id'].nunique()}"
    )

    print("\nRunning forward-chaining CV...")
    metrics_df, oof_df = run_cv(train_df, config=config, n_folds=args.n_folds, val_weeks=args.val_weeks)

    metrics_path = output_paths["metrics"] / "local_linear_cv.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved CV metrics -> {metrics_path}")

    oof_path = output_paths["preds"] / "local_linear_oof.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"Saved OOF predictions -> {oof_path}")

    print("\nTraining on full train and forecasting test recursively...")
    test_pred = fit_full_and_predict_test(train_df, test_df, config=config)

    test_pred_path = output_paths["preds"] / "local_linear_test.parquet"
    test_pred.to_parquet(test_pred_path, index=False)
    print(f"Saved test predictions -> {test_pred_path}")

    submission = make_submission(test_pred)
    submission_path = output_paths["submissions"] / "local_linear_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission -> {submission_path}")

    config_path = output_paths["metrics"] / "local_linear_config.json"
    config_payload = {
        "kernel": args.kernel,
        "bandwidth": args.bandwidth,
        "min_samples": args.min_samples,
        "ridge": args.ridge,
        "coef_clip": args.coef_clip,
        "anom_clip_scale": args.anom_clip_scale,
        "lags": list(LAG_ORDERS),
        "feature_mode": args.feature_mode,
        "exogenous_features": get_exog_feature_cols(config),
        "use_interactions": args.use_interactions,
        "interaction_cols": INTERACTION_FEATURE_COLS if args.use_interactions else [],
        "feature_cols": get_feature_cols(config),
        "standard_scaler": True,
        "n_folds": args.n_folds,
        "val_weeks": args.val_weeks,
        "max_series": args.max_series,
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    print(f"Saved run config -> {config_path}")


if __name__ == "__main__":
    main()
