from __future__ import annotations

import argparse
import json
import time
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
CONT_COLS = ["temp_anom", "fuel_anom", "sales_anom_lag1", "sales_anom_lag2"]


@dataclass(slots=True)
class ScaleStats:
    mean: float
    std: float


@dataclass(slots=True)
class HSGPConfig:
    n_folds: int
    val_weeks: int
    lengthscale_min: float
    lengthscale_max: float
    max_eval: int
    random_seed: int
    pred_draws: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: HSGP model for sales anomalies with forward-chaining CV."
    )
    parser.add_argument("--train-path", default="train_feat.parquet")
    parser.add_argument("--test-path", default="test_feat.parquet")
    parser.add_argument("--output-dir", default="outputs/phase2_hsgp")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-weeks", type=int, default=13)
    parser.add_argument("--lengthscale-min", type=float, default=0.2)
    parser.add_argument("--lengthscale-max", type=float, default=2.5)
    parser.add_argument("--max-eval", type=int, default=2500)
    parser.add_argument("--pred-draws", type=int, default=60)
    parser.add_argument("--random-seed", type=int, default=8927)
    parser.add_argument(
        "--max-series",
        type=int,
        default=200,
        help="Subset of Store-Dept series for tractable Phase 2 prototyping.",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "root": base,
        "metrics": base / "metrics",
        "preds": base / "preds",
    }
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


def prepare_fold_data(
    train_df: pd.DataFrame,
    train_dates: np.ndarray,
    val_dates: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    fold_train = fold_train.dropna(subset=CONT_COLS + [TARGET_COL]).reset_index(drop=True)
    # Recursive validation rebuilds lag features from model predictions.
    fold_val = fold_val.dropna(subset=["temp_anom", "fuel_anom", TARGET_COL]).reset_index(drop=True)
    return fold_train, fold_val


def initialize_history(train_part: pd.DataFrame) -> dict[str, tuple[float, float]]:
    history: dict[str, tuple[float, float]] = {}
    ordered = train_part.sort_values(["series_id", "Date"])
    for series_id, grp in ordered.groupby("series_id", sort=False):
        vals = grp[TARGET_COL].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            history[series_id] = (0.0, 0.0)
        elif vals.size == 1:
            history[series_id] = (float(vals[-1]), 0.0)
        else:
            history[series_id] = (float(vals[-1]), float(vals[-2]))
    return history


def hsgp_predict_fold(
    train_part: pd.DataFrame,
    val_part: pd.DataFrame,
    cfg: HSGPConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    y_stats = fit_standardizer(train_part[TARGET_COL])
    y_train = zscore(train_part[TARGET_COL].to_numpy(dtype=float), y_stats)

    feature_stats: dict[str, ScaleStats] = {}
    x_train: dict[str, np.ndarray] = {}
    for col in CONT_COLS:
        st = fit_standardizer(train_part[col])
        feature_stats[col] = st
        x_train[col] = zscore(train_part[col].to_numpy(dtype=float), st)

    holiday_train = train_part["is_holiday_int"].to_numpy(dtype=float)
    val_ordered = (
        val_part.reset_index()
        .rename(columns={"index": "orig_idx"})
        .sort_values(["Date", "series_id"])
        .reset_index(drop=True)
    )
    holiday_val = val_ordered["is_holiday_int"].to_numpy(dtype=float)

    min_date = train_part["Date"].min()
    t_train = (train_part["Date"] - min_date).dt.days.to_numpy(dtype=float) / 7.0
    t_val = (val_ordered["Date"] - min_date).dt.days.to_numpy(dtype=float) / 7.0
    t_stats = fit_standardizer(pd.Series(t_train))
    t_train_z = zscore(t_train, t_stats)
    t_val_z = zscore(t_val, t_stats)
    t_train_centered = t_train_z - float(np.mean(t_train_z))
    t_val_centered = t_val_z - float(np.mean(t_train_z))

    x_range = [float(np.min(t_train_centered)), float(np.max(t_train_centered))]
    m_t, c_t = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
        x_range=x_range,
        lengthscale_range=[cfg.lengthscale_min, cfg.lengthscale_max],
        cov_func="matern32",
    )

    with pm.Model() as model:
        ls = pm.HalfNormal("ls", sigma=1.5)
        eta = pm.HalfNormal("eta", sigma=1.0)
        cov = eta**2 * pm.gp.cov.Matern32(1, ls=ls)
        hsgp = pm.gp.HSGP(cov_func=cov, m=[int(m_t)], c=float(c_t), drop_first=False)

        f_train = hsgp.prior("f_train", X=t_train_centered[:, None])
        f_val = hsgp.conditional("f_val", Xnew=t_val_centered[:, None])

        b0 = pm.Normal("b0", mu=0.0, sigma=1.0)
        b_temp = pm.Normal("b_temp", mu=0.0, sigma=1.0)
        b_fuel = pm.Normal("b_fuel", mu=0.0, sigma=1.0)
        b_lag1 = pm.Normal("b_lag1", mu=0.0, sigma=1.0)
        b_lag2 = pm.Normal("b_lag2", mu=0.0, sigma=1.0)
        b_holiday = pm.Normal("b_holiday", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu_train = (
            b0
            + f_train
            + b_temp * x_train["temp_anom"]
            + b_fuel * x_train["fuel_anom"]
            + b_lag1 * x_train["sales_anom_lag1"]
            + b_lag2 * x_train["sales_anom_lag2"]
            + b_holiday * holiday_train
        )
        pm.Normal("y_obs", mu=mu_train, sigma=sigma, observed=y_train)

        pm.Deterministic("f_val_det", f_val)

        map_point = pm.find_MAP(
            progressbar=False,
            maxeval=cfg.max_eval,
            return_raw=False,
            seed=cfg.random_seed,
        )
        ppc_trace = [map_point for _ in range(max(1, cfg.pred_draws))]
        ppc = pm.sample_posterior_predictive(
            ppc_trace,
            var_names=["f_val_det"],
            predictions=True,
            random_seed=cfg.random_seed,
            progressbar=False,
        )
        f_draws_raw = np.asarray(ppc.predictions["f_val_det"], dtype=float)

    if f_draws_raw.ndim == 1:
        f_draws = f_draws_raw[None, :]
    elif f_draws_raw.ndim == 2:
        f_draws = f_draws_raw
    else:
        f_draws = f_draws_raw.reshape(-1, f_draws_raw.shape[-1])

    b0 = float(map_point["b0"])
    b_temp = float(map_point["b_temp"])
    b_fuel = float(map_point["b_fuel"])
    b_lag1 = float(map_point["b_lag1"])
    b_lag2 = float(map_point["b_lag2"])
    b_holiday = float(map_point["b_holiday"])
    sigma = float(map_point["sigma"])

    history = initialize_history(train_part)
    train_vals = train_part[TARGET_COL].to_numpy(dtype=float)
    q01, q99 = np.quantile(train_vals, [0.01, 0.99])
    span = max(float(q99 - q01), float(np.std(train_vals)), 1.0)
    lower = float(q01 - 2.0 * span)
    upper = float(q99 + 2.0 * span)

    n_obs = len(val_ordered)
    n_draws = f_draws.shape[0]
    draw_preds = np.empty((n_draws, n_obs), dtype=float)
    rng = np.random.default_rng(cfg.random_seed)

    for d in range(n_draws):
        history = initialize_history(train_part)
        for i, row in enumerate(val_ordered.itertuples(index=False)):
            sid = row.series_id
            lag1_raw, lag2_raw = history.get(sid, (0.0, 0.0))

            temp_z = float(zscore(np.array([row.temp_anom], dtype=float), feature_stats["temp_anom"])[0])
            fuel_z = float(zscore(np.array([row.fuel_anom], dtype=float), feature_stats["fuel_anom"])[0])
            lag1_z = float(zscore(np.array([lag1_raw], dtype=float), feature_stats["sales_anom_lag1"])[0])
            lag2_z = float(zscore(np.array([lag2_raw], dtype=float), feature_stats["sales_anom_lag2"])[0])

            pred_z = (
                b0
                + float(f_draws[d, i])
                + b_temp * temp_z
                + b_fuel * fuel_z
                + b_lag1 * lag1_z
                + b_lag2 * lag2_z
                + b_holiday * float(holiday_val[i])
            )
            pred_z += float(rng.normal(loc=0.0, scale=sigma))
            pred_raw = float(pred_z * y_stats.std + y_stats.mean)
            pred_raw = float(np.clip(pred_raw, lower, upper))
            draw_preds[d, i] = pred_raw
            history[sid] = (pred_raw, lag1_raw)

    pred_mean_ordered = draw_preds.mean(axis=0)
    pred_sd_ordered = draw_preds.std(axis=0, ddof=0)

    order_idx = val_ordered["orig_idx"].to_numpy()
    pred_mean_series = pd.Series(pred_mean_ordered, index=order_idx)
    pred_sd_series = pd.Series(pred_sd_ordered, index=order_idx)
    pred_anom = pred_mean_series.reindex(val_part.index).to_numpy(dtype=float)
    pred_anom_sd = pred_sd_series.reindex(val_part.index).to_numpy(dtype=float)
    meta = {"m_t": float(m_t), "c_t": float(c_t)}
    return pred_anom, pred_anom_sd, meta


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


def run_cv(
    train_df: pd.DataFrame,
    cfg: HSGPConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = build_forward_folds(train_df, n_folds=cfg.n_folds, val_weeks=cfg.val_weeks)
    metrics_rows: list[dict] = []
    oof_parts: list[pd.DataFrame] = []

    for fold_id, train_dates, val_dates in folds:
        print(f"\nFold {fold_id}: train={len(train_dates)} weeks, val={len(val_dates)} weeks")
        fold_train, fold_val = prepare_fold_data(train_df, train_dates, val_dates)
        print(f"  rows used -> train={len(fold_train)}, val={len(fold_val)}")

        t0 = time.perf_counter()
        pred_anom, pred_anom_sd, gp_meta = hsgp_predict_fold(fold_train, fold_val, cfg)
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
        fold_metrics.update(gp_meta)
        metrics_rows.append(fold_metrics)
        print(
            f"  wmae={fold_metrics['wmae']:.4f} mae={fold_metrics['mae']:.4f} "
            f"rmse={fold_metrics['rmse']:.4f} runtime={elapsed:.1f}s"
        )

        keep = [
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
        oof_parts.append(fold_val[keep].copy())

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        mean_row = {"fold": "mean"}
        for col in ["wmae", "mae", "rmse", "anom_mae", "n_rows", "runtime_sec", "m_t", "c_t"]:
            mean_row[col] = float(metrics_df[col].mean())
        metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return metrics_df, oof_df


def main() -> None:
    args = parse_args()
    out = ensure_dirs(Path(args.output_dir))

    cfg = HSGPConfig(
        n_folds=args.n_folds,
        val_weeks=args.val_weeks,
        lengthscale_min=args.lengthscale_min,
        lengthscale_max=args.lengthscale_max,
        max_eval=args.max_eval,
        random_seed=args.random_seed,
        pred_draws=args.pred_draws,
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

    metrics_df, oof_df = run_cv(train_df, cfg)
    metrics_path = out["metrics"] / "hsgp_cv.csv"
    oof_path = out["preds"] / "hsgp_oof.parquet"

    metrics_df.to_csv(metrics_path, index=False)
    oof_df.to_parquet(oof_path, index=False)

    cfg_path = out["metrics"] / "hsgp_config.json"
    cfg_payload = {
        "n_folds": args.n_folds,
        "val_weeks": args.val_weeks,
        "lengthscale_min": args.lengthscale_min,
        "lengthscale_max": args.lengthscale_max,
        "max_eval": args.max_eval,
        "pred_draws": args.pred_draws,
        "random_seed": args.random_seed,
        "max_series": args.max_series,
        "note": "Phase 2 uses recursive validation (predicted anomalies are fed back as lag features).",
    }
    cfg_path.write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved OOF preds -> {oof_path}")
    print(f"Saved config -> {cfg_path}")


if __name__ == "__main__":
    main()
