from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from phase1_local_linear import build_forward_folds, load_data
from phase2_structural_ar import (
    initialize_history,
    prepare_fold_data,
    fit_standardizer,
    zscore,
)


TARGET_COL = "sales_anom"


@dataclass(slots=True)
class ModelData:
    y_train: np.ndarray
    lag1_z: np.ndarray
    lag2_z: np.ndarray
    temp_z: np.ndarray
    fuel_z: np.ndarray
    trend_z: np.ndarray
    holiday: np.ndarray
    ff_matrix: np.ndarray
    train_series_idx: np.ndarray
    n_series: int
    n_fourier: int
    series_to_idx: dict[str, int]
    y_stats: object
    lag1_stats: object
    lag2_stats: object
    temp_stats: object
    fuel_stats: object
    trend_stats: object
    clip_bounds: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MAP vs NUTS chains for one fold of the structural AR model."
    )
    parser.add_argument("--train-path", default="train_feat.parquet")
    parser.add_argument("--test-path", default="test_feat.parquet")
    parser.add_argument("--output-dir", default="outputs/phase2_structural_ar_map_vs_chains")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-weeks", type=int, default=13)
    parser.add_argument("--fold-id", type=int, default=1, help="1-based fold index to evaluate.")
    parser.add_argument("--max-series", type=int, default=200)
    parser.add_argument("--fourier-order", type=int, default=3)
    parser.add_argument("--max-eval", type=int, default=2500)
    parser.add_argument("--random-seed", type=int, default=8927)

    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--nuts-sampler", choices=["pymc", "numpyro", "blackjax", "nutpie"], default="pymc")
    parser.add_argument(
        "--posterior-draw-preds",
        type=int,
        default=60,
        help="Number of posterior parameter draws to average for recursive prediction.",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {"root": base, "metrics": base / "metrics", "preds": base / "preds", "logs": base / "logs"}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def compute_wmae(y_true: np.ndarray, y_pred: np.ndarray, holiday: np.ndarray) -> float:
    weights = np.where(holiday, 5.0, 1.0)
    return float(np.average(np.abs(y_true - y_pred), weights=weights))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, holiday: np.ndarray) -> dict[str, float]:
    return {
        "wmae": compute_wmae(y_true, y_pred, holiday),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def build_model_data(train_part: pd.DataFrame, fourier_cols: list[str]) -> ModelData:
    y_stats = fit_standardizer(train_part[TARGET_COL])
    lag1_stats = fit_standardizer(train_part["sales_anom_lag1"])
    lag2_stats = fit_standardizer(train_part["sales_anom_lag2"])
    temp_stats = fit_standardizer(train_part["temp_anom"])
    fuel_stats = fit_standardizer(train_part["fuel_anom"])
    trend_stats = fit_standardizer(train_part["trend"])

    y_train = zscore(train_part[TARGET_COL].to_numpy(dtype=float), y_stats)
    lag1_z = zscore(train_part["sales_anom_lag1"].to_numpy(dtype=float), lag1_stats)
    lag2_z = zscore(train_part["sales_anom_lag2"].to_numpy(dtype=float), lag2_stats)
    temp_z = zscore(train_part["temp_anom"].to_numpy(dtype=float), temp_stats)
    fuel_z = zscore(train_part["fuel_anom"].to_numpy(dtype=float), fuel_stats)
    trend_z = zscore(train_part["trend"].to_numpy(dtype=float), trend_stats)
    holiday = train_part["is_holiday_int"].to_numpy(dtype=float)
    ff_matrix = train_part[fourier_cols].to_numpy(dtype=float)

    series_ids = train_part["series_id"].drop_duplicates().sort_values().tolist()
    series_to_idx = {sid: i for i, sid in enumerate(series_ids)}
    train_series_idx = train_part["series_id"].map(series_to_idx).to_numpy(dtype=int)

    train_vals = train_part[TARGET_COL].to_numpy(dtype=float)
    q01, q99 = np.quantile(train_vals, [0.01, 0.99])
    span = max(float(q99 - q01), float(np.std(train_vals)), 1.0)
    clip_bounds = (float(q01 - 2.0 * span), float(q99 + 2.0 * span))

    return ModelData(
        y_train=y_train,
        lag1_z=lag1_z,
        lag2_z=lag2_z,
        temp_z=temp_z,
        fuel_z=fuel_z,
        trend_z=trend_z,
        holiday=holiday,
        ff_matrix=ff_matrix,
        train_series_idx=train_series_idx,
        n_series=len(series_ids),
        n_fourier=ff_matrix.shape[1],
        series_to_idx=series_to_idx,
        y_stats=y_stats,
        lag1_stats=lag1_stats,
        lag2_stats=lag2_stats,
        temp_stats=temp_stats,
        fuel_stats=fuel_stats,
        trend_stats=trend_stats,
        clip_bounds=clip_bounds,
    )


def build_model(md: ModelData) -> pm.Model:
    with pm.Model() as model:
        alpha_mu = pm.Normal("alpha_mu", mu=0.0, sigma=1.0)
        alpha_sigma = pm.HalfNormal("alpha_sigma", sigma=1.0)
        z_alpha = pm.Normal("z_alpha", mu=0.0, sigma=1.0, shape=md.n_series)
        alpha_series = pm.Deterministic("alpha_series", alpha_mu + alpha_sigma * z_alpha)

        beta_lag1 = pm.Normal("beta_lag1", mu=0.0, sigma=0.6)
        beta_lag2 = pm.Normal("beta_lag2", mu=0.0, sigma=0.6)
        beta_temp = pm.Normal("beta_temp", mu=0.0, sigma=0.5)
        beta_fuel = pm.Normal("beta_fuel", mu=0.0, sigma=0.5)
        beta_holiday = pm.Normal("beta_holiday", mu=0.0, sigma=0.8)
        beta_trend = pm.Normal("beta_trend", mu=0.0, sigma=0.5)
        beta_fourier = pm.Normal("beta_fourier", mu=0.0, sigma=0.5, shape=md.n_fourier)
        sigma = pm.HalfNormal("sigma", sigma=0.8)

        mu = (
            alpha_series[md.train_series_idx]
            + beta_lag1 * md.lag1_z
            + beta_lag2 * md.lag2_z
            + beta_temp * md.temp_z
            + beta_fuel * md.fuel_z
            + beta_holiday * md.holiday
            + beta_trend * md.trend_z
            + pm.math.dot(md.ff_matrix, beta_fourier)
        )
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=md.y_train)
    return model


def point_to_params(point: dict) -> dict:
    return {
        "alpha_mu": float(point["alpha_mu"]),
        "alpha_series": np.asarray(point["alpha_series"], dtype=float),
        "beta_lag1": float(point["beta_lag1"]),
        "beta_lag2": float(point["beta_lag2"]),
        "beta_temp": float(point["beta_temp"]),
        "beta_fuel": float(point["beta_fuel"]),
        "beta_holiday": float(point["beta_holiday"]),
        "beta_trend": float(point["beta_trend"]),
        "beta_fourier": np.asarray(point["beta_fourier"], dtype=float),
        "sigma": float(point["sigma"]),
    }


def posterior_mean_params(idata) -> dict:
    post = idata.posterior
    return {
        "alpha_mu": float(post["alpha_mu"].mean(dim=("chain", "draw")).values),
        "alpha_series": np.asarray(post["alpha_series"].mean(dim=("chain", "draw")).values, dtype=float),
        "beta_lag1": float(post["beta_lag1"].mean(dim=("chain", "draw")).values),
        "beta_lag2": float(post["beta_lag2"].mean(dim=("chain", "draw")).values),
        "beta_temp": float(post["beta_temp"].mean(dim=("chain", "draw")).values),
        "beta_fuel": float(post["beta_fuel"].mean(dim=("chain", "draw")).values),
        "beta_holiday": float(post["beta_holiday"].mean(dim=("chain", "draw")).values),
        "beta_trend": float(post["beta_trend"].mean(dim=("chain", "draw")).values),
        "beta_fourier": np.asarray(post["beta_fourier"].mean(dim=("chain", "draw")).values, dtype=float),
        "sigma": float(post["sigma"].mean(dim=("chain", "draw")).values),
    }


def extract_posterior_draw_params(idata, draw_count: int, random_seed: int) -> list[dict]:
    post = idata.posterior.stack(sample=("chain", "draw"))
    n_total = int(post.sizes["sample"])
    n_pick = min(draw_count, n_total)
    rng = np.random.default_rng(random_seed)
    pick = np.sort(rng.choice(n_total, size=n_pick, replace=False))

    alpha_mu_vals = np.asarray(post["alpha_mu"].values, dtype=float)[pick]
    alpha_series_vals = np.asarray(
        post["alpha_series"].transpose("sample", "alpha_series_dim_0").values, dtype=float
    )[pick, :]
    beta_fourier_vals = np.asarray(
        post["beta_fourier"].transpose("sample", "beta_fourier_dim_0").values, dtype=float
    )[pick, :]

    scalars = {
        "beta_lag1": np.asarray(post["beta_lag1"].values, dtype=float)[pick],
        "beta_lag2": np.asarray(post["beta_lag2"].values, dtype=float)[pick],
        "beta_temp": np.asarray(post["beta_temp"].values, dtype=float)[pick],
        "beta_fuel": np.asarray(post["beta_fuel"].values, dtype=float)[pick],
        "beta_holiday": np.asarray(post["beta_holiday"].values, dtype=float)[pick],
        "beta_trend": np.asarray(post["beta_trend"].values, dtype=float)[pick],
        "sigma": np.asarray(post["sigma"].values, dtype=float)[pick],
    }

    out: list[dict] = []
    for i in range(n_pick):
        out.append(
            {
                "alpha_mu": float(alpha_mu_vals[i]),
                "alpha_series": alpha_series_vals[i],
                "beta_lag1": float(scalars["beta_lag1"][i]),
                "beta_lag2": float(scalars["beta_lag2"][i]),
                "beta_temp": float(scalars["beta_temp"][i]),
                "beta_fuel": float(scalars["beta_fuel"][i]),
                "beta_holiday": float(scalars["beta_holiday"][i]),
                "beta_trend": float(scalars["beta_trend"][i]),
                "beta_fourier": beta_fourier_vals[i],
                "sigma": float(scalars["sigma"][i]),
            }
        )
    return out


def recursive_predict(
    val_part: pd.DataFrame,
    train_part: pd.DataFrame,
    fourier_cols: list[str],
    params: dict,
    md: ModelData,
) -> np.ndarray:
    ordered = (
        val_part.reset_index()
        .rename(columns={"index": "orig_idx"})
        .sort_values(["Date", "series_id"])
        .reset_index(drop=True)
    )

    sid_arr = ordered["series_id"].to_numpy()
    temp_arr = ordered["temp_anom"].to_numpy(dtype=float)
    fuel_arr = ordered["fuel_anom"].to_numpy(dtype=float)
    holiday_arr = ordered["is_holiday_int"].to_numpy(dtype=float)
    trend_arr = ordered["trend"].to_numpy(dtype=float)
    ff_arr = ordered[fourier_cols].to_numpy(dtype=float)
    orig_idx = ordered["orig_idx"].to_numpy(dtype=int)

    lower, upper = md.clip_bounds
    history = initialize_history(train_part)
    pred_ordered = np.empty(len(ordered), dtype=float)

    for i in range(len(ordered)):
        sid = sid_arr[i]
        lag1_raw, lag2_raw = history.get(sid, (0.0, 0.0))
        sid_idx = md.series_to_idx.get(sid)
        alpha = params["alpha_series"][sid_idx] if sid_idx is not None else params["alpha_mu"]

        lag1_z = (lag1_raw - md.lag1_stats.mean) / md.lag1_stats.std
        lag2_z = (lag2_raw - md.lag2_stats.mean) / md.lag2_stats.std
        temp_z = (temp_arr[i] - md.temp_stats.mean) / md.temp_stats.std
        fuel_z = (fuel_arr[i] - md.fuel_stats.mean) / md.fuel_stats.std
        trend_z = (trend_arr[i] - md.trend_stats.mean) / md.trend_stats.std

        pred_z = (
            alpha
            + params["beta_lag1"] * lag1_z
            + params["beta_lag2"] * lag2_z
            + params["beta_temp"] * temp_z
            + params["beta_fuel"] * fuel_z
            + params["beta_holiday"] * holiday_arr[i]
            + params["beta_trend"] * trend_z
            + float(np.dot(ff_arr[i], params["beta_fourier"]))
        )
        pred_raw = float(pred_z * md.y_stats.std + md.y_stats.mean)
        pred_raw = float(np.clip(pred_raw, lower, upper))
        pred_ordered[i] = pred_raw
        history[sid] = (pred_raw, lag1_raw)

    pred_series = pd.Series(pred_ordered, index=orig_idx)
    return pred_series.reindex(val_part.index).to_numpy(dtype=float)


def log_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = ensure_dirs(Path(args.output_dir))
    rng = np.random.default_rng(args.random_seed)

    train_df, _ = load_data(
        Path(args.train_path),
        Path(args.test_path),
        max_series=args.max_series,
    )
    folds = build_forward_folds(train_df, n_folds=args.n_folds, val_weeks=args.val_weeks)
    if args.fold_id < 1 or args.fold_id > len(folds):
        raise ValueError(f"--fold-id must be in [1, {len(folds)}]")

    fold_id, train_dates, val_dates = folds[args.fold_id - 1]
    fold_train, fold_val, fourier_cols = prepare_fold_data(
        train_df,
        train_dates,
        val_dates,
        fourier_order=args.fourier_order,
    )

    md = build_model_data(fold_train, fourier_cols=fourier_cols)
    model = build_model(md)

    y_true = fold_val["Weekly_Sales"].to_numpy(dtype=float)
    holiday = fold_val["IsHoliday"].to_numpy(dtype=bool)

    metrics_rows: list[dict] = []
    pred_frame = fold_val[["Store", "Dept", "Date", "IsHoliday", "Weekly_Sales", "clim_sales", "sales_anom"]].copy()

    # MAP fit
    with model:
        t0 = time.perf_counter()
        map_point = pm.find_MAP(
            progressbar=False,
            maxeval=args.max_eval,
            return_raw=False,
            seed=args.random_seed,
        )
        map_runtime = time.perf_counter() - t0
    map_params = point_to_params(map_point)
    pred_map_anom = recursive_predict(fold_val, fold_train, fourier_cols, map_params, md)
    pred_map_sales = pred_map_anom + fold_val["clim_sales"].to_numpy(dtype=float)
    m_map = evaluate(y_true, pred_map_sales, holiday)
    m_map.update({"method": "map", "runtime_sec": map_runtime})
    metrics_rows.append(m_map)
    pred_frame["pred_map"] = pred_map_sales

    # NUTS fit
    with model:
        t1 = time.perf_counter()
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.random_seed,
            progressbar=True,
            nuts_sampler=args.nuts_sampler,
        )
        nuts_runtime = time.perf_counter() - t1

    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    diag_summary = az.summary(
        idata,
        var_names=["beta_lag1", "beta_lag2", "beta_temp", "beta_fuel", "beta_holiday", "beta_trend", "sigma"],
    )
    rhat_max = float(diag_summary["r_hat"].max())
    ess_bulk_min = float(diag_summary["ess_bulk"].min())

    # NUTS posterior mean params
    nuts_mean_params = posterior_mean_params(idata)
    pred_nuts_mean_anom = recursive_predict(fold_val, fold_train, fourier_cols, nuts_mean_params, md)
    pred_nuts_mean_sales = pred_nuts_mean_anom + fold_val["clim_sales"].to_numpy(dtype=float)
    m_nuts_mean = evaluate(y_true, pred_nuts_mean_sales, holiday)
    m_nuts_mean.update({"method": "nuts_mean_params", "runtime_sec": nuts_runtime})
    metrics_rows.append(m_nuts_mean)
    pred_frame["pred_nuts_mean"] = pred_nuts_mean_sales

    # NUTS posterior draw averaged predictions
    draw_params = extract_posterior_draw_params(
        idata,
        draw_count=args.posterior_draw_preds,
        random_seed=args.random_seed,
    )
    draw_preds = []
    for p in draw_params:
        draw_pred_anom = recursive_predict(fold_val, fold_train, fourier_cols, p, md)
        draw_preds.append(draw_pred_anom + fold_val["clim_sales"].to_numpy(dtype=float))
    draw_preds_arr = np.vstack(draw_preds)
    pred_nuts_draw_mean_sales = draw_preds_arr.mean(axis=0)
    pred_nuts_draw_sd = draw_preds_arr.std(axis=0, ddof=0)
    m_nuts_draw = evaluate(y_true, pred_nuts_draw_mean_sales, holiday)
    m_nuts_draw.update({"method": "nuts_draw_mean", "runtime_sec": nuts_runtime})
    metrics_rows.append(m_nuts_draw)
    pred_frame["pred_nuts_draw_mean"] = pred_nuts_draw_mean_sales
    pred_frame["pred_nuts_draw_sd"] = pred_nuts_draw_sd
    pred_frame["pred_nuts_draw_lower"] = pred_nuts_draw_mean_sales - 1.96 * pred_nuts_draw_sd
    pred_frame["pred_nuts_draw_upper"] = pred_nuts_draw_mean_sales + 1.96 * pred_nuts_draw_sd

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out["metrics"] / "fold_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    pred_path = out["preds"] / "fold_predictions.parquet"
    pred_frame.to_parquet(pred_path, index=False)

    log_lines = [
        "Structural AR Fold Comparison: MAP vs NUTS",
        f"timestamp_seed={args.random_seed}",
        "",
        "Data setup",
        f"max_series={args.max_series}",
        f"fold_id={fold_id}",
        f"train_start={pd.Timestamp(train_dates[0]).date()}",
        f"train_end={pd.Timestamp(train_dates[-1]).date()}",
        f"val_start={pd.Timestamp(val_dates[0]).date()}",
        f"val_end={pd.Timestamp(val_dates[-1]).date()}",
        f"train_rows={len(fold_train)}",
        f"val_rows={len(fold_val)}",
        f"fourier_order={args.fourier_order}",
        "",
        "Inference setup",
        f"map_maxeval={args.max_eval}",
        f"nuts_sampler={args.nuts_sampler}",
        f"chains={args.chains}",
        f"draws={args.draws}",
        f"tune={args.tune}",
        f"target_accept={args.target_accept}",
        f"posterior_draw_preds={args.posterior_draw_preds}",
        "",
        "Diagnostics",
        f"nuts_runtime_sec={nuts_runtime:.3f}",
        f"map_runtime_sec={map_runtime:.3f}",
        f"divergences={div}",
        f"rhat_max={rhat_max:.4f}",
        f"ess_bulk_min={ess_bulk_min:.2f}",
        "",
        "Metrics",
        metrics_df.to_string(index=False),
    ]
    log_path = out["logs"] / "fold_comparison_log.txt"
    log_text(log_path, "\n".join(log_lines))

    summary = {
        "fold_id": int(fold_id),
        "map_runtime_sec": float(map_runtime),
        "nuts_runtime_sec": float(nuts_runtime),
        "divergences": int(div),
        "rhat_max": float(rhat_max),
        "ess_bulk_min": float(ess_bulk_min),
        "best_method_by_wmae": str(metrics_df.sort_values("wmae").iloc[0]["method"]),
    }
    summary_path = out["metrics"] / "fold_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:")
    print(metrics_path)
    print(pred_path)
    print(log_path)
    print(summary_path)


if __name__ == "__main__":
    main()
