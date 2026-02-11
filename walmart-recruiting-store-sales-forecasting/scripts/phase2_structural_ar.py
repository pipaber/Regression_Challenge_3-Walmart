from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

from phase1_local_linear import (
    add_anomaly_lags,
    attach_climatology,
    build_forward_folds,
    compute_climatology,
    load_data,
    wmae,
)

TARGET_COL = "sales_anom"
PERIOD_WEEKS = 52.0
LAG_ORDERS = (1, 2)
LAG_COLS = [f"sales_anom_lag{lag}" for lag in LAG_ORDERS]
EXOG_COLS = [
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


@dataclass(slots=True)
class ScaleStats:
    mean: float
    std: float


@dataclass(slots=True)
class StructuralARConfig:
    n_folds: int
    val_weeks: int
    max_eval: int
    random_seed: int
    pred_draws: int
    fourier_order: int
    sigma_clusters: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Structural AR + trend + seasonal Fourier model with recursive validation."
    )
    parser.add_argument("--train-path", default="train_feat.parquet")
    parser.add_argument("--test-path", default="test_feat.parquet")
    parser.add_argument("--output-dir", default="outputs/phase2_structural_ar")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-weeks", type=int, default=13)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=8927)
    parser.add_argument("--pred-draws", type=int, default=80)
    parser.add_argument("--fourier-order", type=int, default=3)
    parser.add_argument(
        "--sigma-clusters",
        type=int,
        default=0,
        help="Number of series-level noise clusters for heteroskedastic sigma (0 disables).",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=200,
        help="Subset of Store-Dept series for tractable PyMC structural model.",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {"root": base, "metrics": base / "metrics", "preds": base / "preds"}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def fit_standardizer(series: pd.Series) -> ScaleStats:
    mean = float(series.mean())
    std = float(series.std())
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return ScaleStats(mean=mean, std=std)


def zscore(values: np.ndarray, stats: ScaleStats) -> np.ndarray:
    return (values - stats.mean) / stats.std


def add_structural_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fourier_order: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    min_date = train_df["Date"].min()
    for frame in (train_df, val_df):
        frame["t_week"] = ((frame["Date"] - min_date).dt.days // 7).astype(float)
        frame["trend"] = frame["t_week"]

    fourier_cols: list[str] = []
    for k in range(1, fourier_order + 1):
        sin_col = f"ff_sin_{k}"
        cos_col = f"ff_cos_{k}"
        angle_train = 2.0 * np.pi * k * train_df["t_week"] / PERIOD_WEEKS
        angle_val = 2.0 * np.pi * k * val_df["t_week"] / PERIOD_WEEKS
        train_df[sin_col] = np.sin(angle_train)
        train_df[cos_col] = np.cos(angle_train)
        val_df[sin_col] = np.sin(angle_val)
        val_df[cos_col] = np.cos(angle_val)
        fourier_cols.extend([sin_col, cos_col])

    return train_df, val_df, fourier_cols


def prepare_fold_data(
    train_df: pd.DataFrame,
    train_dates: np.ndarray,
    val_dates: np.ndarray,
    fourier_order: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fold_train_raw = train_df[train_df["Date"].isin(train_dates)].copy()
    fold_val_raw = train_df[train_df["Date"].isin(val_dates)].copy()
    fold_train_raw["split"] = "train"
    fold_val_raw["split"] = "val"

    climatology = compute_climatology(fold_train_raw)
    combined = pd.concat([fold_train_raw, fold_val_raw], ignore_index=True)
    combined = attach_climatology(combined, climatology)
    combined = add_anomaly_lags(combined)
    combined = combined.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)

    fold_train = combined[combined["split"] == "train"].copy()
    fold_val = combined[combined["split"] == "val"].copy()

    # Keep early rows by treating unavailable long-lag anomalies as neutral (0).
    for c in LAG_COLS:
        fold_train[c] = fold_train[c].fillna(0.0)
        fold_val[c] = fold_val[c].fillna(0.0)

    for c in EXOG_COLS:
        median = float(fold_train[c].median())
        if not np.isfinite(median):
            median = 0.0
        fold_train[c] = fold_train[c].fillna(median)
        fold_val[c] = fold_val[c].fillna(median)

    fold_train = fold_train.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    fold_val = fold_val.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    fold_train, fold_val, fourier_cols = add_structural_features(
        fold_train, fold_val, fourier_order=fourier_order
    )
    return fold_train, fold_val, fourier_cols


def _lag_from_history(hist: deque[float], lag: int) -> float:
    if len(hist) >= lag:
        return float(hist[-lag])
    return 0.0


def initialize_history(train_part: pd.DataFrame) -> dict[str, deque[float]]:
    history: dict[str, deque[float]] = {}
    max_lag = max(LAG_ORDERS)
    ordered = train_part.sort_values(["series_id", "Date"])
    for series_id, grp in ordered.groupby("series_id", sort=False):
        vals = grp[TARGET_COL].dropna().to_numpy(dtype=float)
        history[series_id] = deque(vals.tolist(), maxlen=max_lag)
    return history


def build_sigma_clusters(train_part: pd.DataFrame, n_clusters: int) -> dict[str, int]:
    if n_clusters <= 1:
        return {}
    series_stats = (
        train_part.groupby("series_id", as_index=False)[TARGET_COL]
        .std()
        .rename(columns={TARGET_COL: "anom_std"})
    )
    series_stats["anom_std"] = series_stats["anom_std"].fillna(0.0)
    k = int(max(1, min(n_clusters, series_stats.shape[0])))
    if k == 1:
        return {sid: 0 for sid in series_stats["series_id"].tolist()}
    # Quantile bins by anomaly volatility to capture heteroskedastic groups.
    labels = pd.qcut(series_stats["anom_std"], q=k, labels=False, duplicates="drop")
    labels = labels.fillna(0).astype(int)
    return dict(zip(series_stats["series_id"], labels))


def fit_structural_model(
    train_part: pd.DataFrame,
    fourier_cols: list[str],
    cfg: StructuralARConfig,
) -> dict:
    y_stats = fit_standardizer(train_part[TARGET_COL])
    lag_stats = [fit_standardizer(train_part[c]) for c in LAG_COLS]
    exog_stats = [fit_standardizer(train_part[c]) for c in EXOG_COLS]
    trend_stats = fit_standardizer(train_part["trend"])

    y_train = zscore(train_part[TARGET_COL].to_numpy(dtype=float), y_stats)
    lag_z_matrix = np.column_stack(
        [
            zscore(train_part[col].to_numpy(dtype=float), stats)
            for col, stats in zip(LAG_COLS, lag_stats)
        ]
    )
    exog_z_matrix = np.column_stack(
        [
            zscore(train_part[col].to_numpy(dtype=float), stats)
            for col, stats in zip(EXOG_COLS, exog_stats)
        ]
    )
    trend_z = zscore(train_part["trend"].to_numpy(dtype=float), trend_stats)
    holiday = train_part["is_holiday_int"].to_numpy(dtype=float)
    ff_matrix = train_part[fourier_cols].to_numpy(dtype=float)

    series_ids = train_part["series_id"].drop_duplicates().sort_values().tolist()
    series_to_idx = {sid: i for i, sid in enumerate(series_ids)}
    train_series_idx = train_part["series_id"].map(series_to_idx).to_numpy(dtype=int)
    n_series = len(series_ids)
    series_to_cluster: dict[str, int] = {}
    cluster_idx: np.ndarray | None = None
    n_cluster = 0
    if cfg.sigma_clusters > 1:
        series_to_cluster = build_sigma_clusters(train_part, cfg.sigma_clusters)
        if series_to_cluster:
            cluster_idx = train_part["series_id"].map(series_to_cluster).fillna(0).to_numpy(dtype=int)
            n_cluster = int(np.max(cluster_idx)) + 1

    with pm.Model() as model:
        alpha_mu = pm.Normal("alpha_mu", mu=0.0, sigma=1.0)
        alpha_sigma = pm.HalfNormal("alpha_sigma", sigma=1.0)
        z_alpha = pm.Normal("z_alpha", mu=0.0, sigma=1.0, shape=n_series)
        alpha_series = pm.Deterministic("alpha_series", alpha_mu + alpha_sigma * z_alpha)

        beta_lags = pm.Normal("beta_lags", mu=0.0, sigma=0.6, shape=len(LAG_COLS))
        beta_exog = pm.Normal("beta_exog", mu=0.0, sigma=0.5, shape=len(EXOG_COLS))
        beta_holiday = pm.Normal("beta_holiday", mu=0.0, sigma=0.8)
        beta_trend = pm.Normal("beta_trend", mu=0.0, sigma=0.5)
        beta_fourier = pm.Normal("beta_fourier", mu=0.0, sigma=0.5, shape=ff_matrix.shape[1])
        if cluster_idx is not None and n_cluster > 1:
            sigma_cluster = pm.HalfNormal("sigma_cluster", sigma=0.8, shape=n_cluster)
            sigma_obs = sigma_cluster[cluster_idx]
        else:
            sigma = pm.HalfNormal("sigma", sigma=0.8)
            sigma_obs = sigma

        mu = (
            alpha_series[train_series_idx]
            + pm.math.dot(lag_z_matrix, beta_lags)
            + pm.math.dot(exog_z_matrix, beta_exog)
            + beta_holiday * holiday
            + beta_trend * trend_z
            + pm.math.dot(ff_matrix, beta_fourier)
        )
        pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y_train)

        map_point = pm.find_MAP(
            progressbar=False,
            maxeval=cfg.max_eval,
            return_raw=False,
            seed=cfg.random_seed,
        )

    train_vals = train_part[TARGET_COL].to_numpy(dtype=float)
    q01, q99 = np.quantile(train_vals, [0.01, 0.99])
    span = max(float(q99 - q01), float(np.std(train_vals)), 1.0)
    lower = float(q01 - 2.0 * span)
    upper = float(q99 + 2.0 * span)

    fitted = {
        "map_point": map_point,
        "series_to_idx": series_to_idx,
        "y_stats": y_stats,
        "lag_stats": lag_stats,
        "exog_stats": exog_stats,
        "exog_cols": EXOG_COLS,
        "trend_stats": trend_stats,
        "fourier_cols": fourier_cols,
        "history_base": initialize_history(train_part),
        "clip_bounds": (lower, upper),
        "series_to_cluster": series_to_cluster,
    }
    return fitted


def recursive_predict_with_uncertainty(
    val_part: pd.DataFrame,
    fitted: dict,
    cfg: StructuralARConfig,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = (
        val_part.reset_index()
        .rename(columns={"index": "orig_idx"})
        .sort_values(["Date", "series_id"])
        .reset_index(drop=True)
    )

    mp = fitted["map_point"]
    series_to_idx = fitted["series_to_idx"]
    alpha_series = np.asarray(mp["alpha_series"], dtype=float)
    alpha_mu = float(mp["alpha_mu"])

    beta_lags = np.asarray(mp["beta_lags"], dtype=float)
    beta_exog = np.asarray(mp["beta_exog"], dtype=float)
    beta_holiday = float(mp["beta_holiday"])
    beta_trend = float(mp["beta_trend"])
    beta_fourier = np.asarray(mp["beta_fourier"], dtype=float)
    sigma_global = float(mp["sigma"]) if "sigma" in mp else None
    sigma_cluster = np.asarray(mp["sigma_cluster"], dtype=float) if "sigma_cluster" in mp else None

    y_stats = fitted["y_stats"]
    lag_stats = fitted["lag_stats"]
    exog_stats = fitted["exog_stats"]
    exog_cols = fitted["exog_cols"]
    trend_stats = fitted["trend_stats"]
    fourier_cols = fitted["fourier_cols"]
    series_to_cluster = fitted.get("series_to_cluster", {})
    lower, upper = fitted["clip_bounds"]

    sid_arr = ordered["series_id"].to_numpy()
    exog_arr = ordered[exog_cols].to_numpy(dtype=float)
    holiday_arr = ordered["is_holiday_int"].to_numpy(dtype=float)
    trend_arr = ordered["trend"].to_numpy(dtype=float)
    ff_arr = ordered[fourier_cols].to_numpy(dtype=float)
    orig_idx = ordered["orig_idx"].to_numpy(dtype=int)

    n = len(ordered)
    draws = max(1, cfg.pred_draws)
    draw_preds = np.empty((draws, n), dtype=float)
    rng = np.random.default_rng(cfg.random_seed)

    for d in range(draws):
        history = {sid: deque(vals, maxlen=max(LAG_ORDERS)) for sid, vals in fitted["history_base"].items()}
        for i in range(n):
            sid = sid_arr[i]
            hist = history.get(sid)
            if hist is None:
                hist = deque(maxlen=max(LAG_ORDERS))
                history[sid] = hist
            sid_idx = series_to_idx.get(sid)
            alpha = alpha_series[sid_idx] if sid_idx is not None else alpha_mu

            lag_z_vals = [
                (_lag_from_history(hist, lag) - stats.mean) / stats.std
                for lag, stats in zip(LAG_ORDERS, lag_stats)
            ]
            exog_z_vals = [
                (exog_arr[i, j] - stats.mean) / stats.std
                for j, stats in enumerate(exog_stats)
            ]
            trend_z = (trend_arr[i] - trend_stats.mean) / trend_stats.std

            pred_z = (
                alpha
                + float(np.dot(lag_z_vals, beta_lags))
                + float(np.dot(exog_z_vals, beta_exog))
                + beta_holiday * holiday_arr[i]
                + beta_trend * trend_z
                + float(np.dot(ff_arr[i], beta_fourier))
            )
            if sigma_cluster is not None and series_to_cluster:
                c_idx = int(series_to_cluster.get(sid, 0))
                sigma_draw = float(sigma_cluster[min(c_idx, len(sigma_cluster) - 1)])
            else:
                sigma_draw = float(sigma_global if sigma_global is not None else 1.0)
            pred_z += float(rng.normal(loc=0.0, scale=sigma_draw))
            pred_raw = float(pred_z * y_stats.std + y_stats.mean)
            pred_raw = float(np.clip(pred_raw, lower, upper))

            draw_preds[d, i] = pred_raw
            hist.append(pred_raw)

    mean_ordered = draw_preds.mean(axis=0)
    sd_ordered = draw_preds.std(axis=0, ddof=0)
    mean_series = pd.Series(mean_ordered, index=orig_idx)
    sd_series = pd.Series(sd_ordered, index=orig_idx)
    pred_mean = mean_series.reindex(val_part.index).to_numpy(dtype=float)
    pred_sd = sd_series.reindex(val_part.index).to_numpy(dtype=float)
    return pred_mean, pred_sd


def evaluate_fold(val_df: pd.DataFrame) -> dict[str, float]:
    y_true = val_df["Weekly_Sales"].to_numpy(dtype=float)
    y_pred = val_df["pred_sales"].to_numpy(dtype=float)
    holiday = val_df["IsHoliday"].to_numpy(dtype=bool)
    return {
        "wmae": wmae(y_true, y_pred, holiday),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "anom_mae": float(np.mean(np.abs(val_df["sales_anom"] - val_df["pred_anom"]))),
        "n_rows": float(len(val_df)),
    }


def run_cv(train_df: pd.DataFrame, cfg: StructuralARConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = build_forward_folds(train_df, n_folds=cfg.n_folds, val_weeks=cfg.val_weeks)
    metrics_rows: list[dict] = []
    oof_parts: list[pd.DataFrame] = []

    for fold_id, train_dates, val_dates in folds:
        print(f"\nFold {fold_id}: train={len(train_dates)} weeks, val={len(val_dates)} weeks")
        fold_train, fold_val, fourier_cols = prepare_fold_data(
            train_df, train_dates, val_dates, fourier_order=cfg.fourier_order
        )
        print(f"  rows used -> train={len(fold_train)}, val={len(fold_val)}")

        t0 = time.perf_counter()
        fitted = fit_structural_model(fold_train, fourier_cols=fourier_cols, cfg=cfg)
        pred_anom, pred_anom_sd = recursive_predict_with_uncertainty(fold_val, fitted=fitted, cfg=cfg)
        elapsed = time.perf_counter() - t0

        fold_val = fold_val.copy()
        fold_val["pred_anom"] = pred_anom
        fold_val["pred_anom_sd"] = pred_anom_sd
        fold_val["pred_sales"] = fold_val["pred_anom"] + fold_val["clim_sales"]
        fold_val["pred_sales_sd"] = fold_val["pred_anom_sd"]
        fold_val["pred_sales_lower"] = fold_val["pred_sales"] - 1.96 * fold_val["pred_sales_sd"]
        fold_val["pred_sales_upper"] = fold_val["pred_sales"] + 1.96 * fold_val["pred_sales_sd"]
        fold_val["fold"] = fold_id

        fold_metrics = evaluate_fold(fold_val)
        fold_metrics["fold"] = fold_id
        fold_metrics["train_start"] = str(pd.Timestamp(train_dates[0]).date())
        fold_metrics["train_end"] = str(pd.Timestamp(train_dates[-1]).date())
        fold_metrics["val_start"] = str(pd.Timestamp(val_dates[0]).date())
        fold_metrics["val_end"] = str(pd.Timestamp(val_dates[-1]).date())
        fold_metrics["runtime_sec"] = float(elapsed)
        metrics_rows.append(fold_metrics)
        print(
            f"  wmae={fold_metrics['wmae']:.4f} mae={fold_metrics['mae']:.4f} "
            f"rmse={fold_metrics['rmse']:.4f} runtime={elapsed:.1f}s"
        )

        keep_cols = [
            "Store",
            "Dept",
            "Date",
            "IsHoliday",
            "Weekly_Sales",
            "sales_anom",
            "pred_anom",
            "pred_anom_sd",
            "pred_sales",
            "pred_sales_sd",
            "pred_sales_lower",
            "pred_sales_upper",
            "fold",
        ]
        oof_parts.append(fold_val[keep_cols].copy())

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        mean_row = {"fold": "mean"}
        for col in ["wmae", "mae", "rmse", "anom_mae", "n_rows", "runtime_sec"]:
            mean_row[col] = float(metrics_df[col].mean())
        metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return metrics_df, oof_df


def main() -> None:
    args = parse_args()
    out = ensure_dirs(Path(args.output_dir))
    cfg = StructuralARConfig(
        n_folds=args.n_folds,
        val_weeks=args.val_weeks,
        max_eval=args.max_eval,
        random_seed=args.random_seed,
        pred_draws=args.pred_draws,
        fourier_order=args.fourier_order,
        sigma_clusters=args.sigma_clusters,
    )

    print("Loading data...")
    train_df, _ = load_data(
        Path(args.train_path),
        Path(args.test_path),
        max_series=args.max_series,
    )
    print(
        f"rows={len(train_df)}, series={train_df['series_id'].nunique()}, "
        f"max_series={args.max_series}"
    )

    metrics_df, oof_df = run_cv(train_df, cfg=cfg)
    metrics_path = out["metrics"] / "structural_ar_cv.csv"
    oof_path = out["preds"] / "structural_ar_oof.parquet"
    metrics_df.to_csv(metrics_path, index=False)
    oof_df.to_parquet(oof_path, index=False)

    cfg_payload = {
        "n_folds": args.n_folds,
        "val_weeks": args.val_weeks,
        "max_eval": args.max_eval,
        "random_seed": args.random_seed,
        "pred_draws": args.pred_draws,
        "fourier_order": args.fourier_order,
        "lag_orders": list(LAG_ORDERS),
        "exogenous_features": EXOG_COLS,
        "sigma_clusters": args.sigma_clusters,
        "max_series": args.max_series,
        "note": "Structural AR model with hierarchical series intercept, trend, Fourier seasonality, and recursive validation.",
    }
    cfg_path = out["metrics"] / "structural_ar_config.json"
    cfg_path.write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved OOF preds -> {oof_path}")
    print(f"Saved config -> {cfg_path}")


if __name__ == "__main__":
    main()
