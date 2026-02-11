from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import pandas as pd

from phase1_local_linear import (
    LocalLinearConfig,
    fit_full_and_predict_test,
    load_data,
    make_submission,
    run_cv,
)


def parse_csv_list(value: str, cast) -> list:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small-grid search for Phase 1 local linear anomaly model."
    )
    parser.add_argument("--train-path", default="train_feat.parquet")
    parser.add_argument("--test-path", default="test_feat.parquet")
    parser.add_argument("--output-dir", default="outputs/search")

    parser.add_argument("--kernels", default="tricube,exponential")
    parser.add_argument("--bandwidths", default="4,6,8")
    parser.add_argument("--min-samples-grid", default="12,16")

    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--coef-clip", type=float, default=6.0)
    parser.add_argument("--anom-clip-scale", type=float, default=2.0)

    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-weeks", type=int, default=13)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument(
        "--use-interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use targeted interaction terms (holiday x lag1, holiday x lag13).",
    )
    parser.add_argument(
        "--refit-best",
        action="store_true",
        help="Train full model on best config and generate best-grid submission artifacts.",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "root": base,
        "fold_metrics": base / "fold_metrics",
        "best_preds": base / "best_preds",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(Path(args.output_dir))

    kernels = parse_csv_list(args.kernels, str)
    bandwidths = parse_csv_list(args.bandwidths, int)
    min_samples_grid = parse_csv_list(args.min_samples_grid, int)
    combos = list(product(kernels, bandwidths, min_samples_grid))
    if not combos:
        raise ValueError("Grid is empty. Check kernels/bandwidths/min-samples-grid arguments.")

    print("Loading data once for all grid runs...")
    train_df, test_df = load_data(
        Path(args.train_path),
        Path(args.test_path),
        max_series=args.max_series,
    )
    print(
        f"Grid size={len(combos)}, train rows={len(train_df)}, test rows={len(test_df)}, "
        f"series train={train_df['series_id'].nunique()}"
    )

    summary_rows: list[dict] = []
    all_fold_rows: list[pd.DataFrame] = []

    for run_id, (kernel, bandwidth, min_samples) in enumerate(combos, start=1):
        config = LocalLinearConfig(
            kernel=kernel,
            bandwidth=bandwidth,
            min_samples=min_samples,
            ridge=args.ridge,
            coef_clip=args.coef_clip,
            anom_clip_scale=args.anom_clip_scale,
            use_interactions=args.use_interactions,
        )

        print(
            f"\n[{run_id}/{len(combos)}] kernel={kernel} bandwidth={bandwidth} "
            f"min_samples={min_samples}"
        )
        t0 = time.perf_counter()
        metrics_df, _ = run_cv(
            train_df,
            config=config,
            n_folds=args.n_folds,
            val_weeks=args.val_weeks,
        )
        elapsed = time.perf_counter() - t0

        fold_df = metrics_df[metrics_df["fold"].astype(str) != "mean"].copy()
        fold_df["kernel"] = kernel
        fold_df["bandwidth"] = bandwidth
        fold_df["min_samples"] = min_samples
        fold_df["ridge"] = args.ridge
        fold_df["coef_clip"] = args.coef_clip
        fold_df["anom_clip_scale"] = args.anom_clip_scale
        fold_df["run_id"] = run_id
        fold_df["runtime_sec"] = elapsed
        all_fold_rows.append(fold_df)

        row = {
            "run_id": run_id,
            "kernel": kernel,
            "bandwidth": bandwidth,
            "min_samples": min_samples,
            "ridge": args.ridge,
            "coef_clip": args.coef_clip,
            "anom_clip_scale": args.anom_clip_scale,
            "mean_wmae": float(fold_df["wmae"].mean()),
            "mean_mae": float(fold_df["mae"].mean()),
            "mean_rmse": float(fold_df["rmse"].mean()),
            "mean_anom_mae": float(fold_df["anom_mae"].mean()),
            "worst_fold_wmae": float(fold_df["wmae"].max()),
            "runtime_sec": float(elapsed),
        }
        summary_rows.append(row)
        print(
            f"  -> mean_wmae={row['mean_wmae']:.4f} "
            f"worst_fold_wmae={row['worst_fold_wmae']:.4f} "
            f"runtime={row['runtime_sec']:.1f}s"
        )

        # Save progress after each run.
        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["mean_wmae", "worst_fold_wmae", "runtime_sec"], ascending=[True, True, True]
        )
        summary_df.to_csv(paths["root"] / "local_linear_grid_results.csv", index=False)

        fold_metrics_df = pd.concat(all_fold_rows, ignore_index=True)
        fold_metrics_df.to_csv(paths["root"] / "local_linear_grid_fold_metrics.csv", index=False)

    final_summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_wmae", "worst_fold_wmae", "runtime_sec"], ascending=[True, True, True]
    )
    best = final_summary.iloc[0].to_dict()
    best_payload = {
        "kernel": str(best["kernel"]),
        "bandwidth": int(best["bandwidth"]),
        "min_samples": int(best["min_samples"]),
        "ridge": float(best["ridge"]),
        "coef_clip": float(best["coef_clip"]),
        "anom_clip_scale": float(best["anom_clip_scale"]),
        "n_folds": int(args.n_folds),
        "val_weeks": int(args.val_weeks),
        "selection_metric": "mean_wmae_then_worst_fold_wmae",
        "selected_run_id": int(best["run_id"]),
        "use_interactions": bool(args.use_interactions),
    }
    best_path = paths["root"] / "best_local_linear_config.json"
    best_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    print("\nBest config:")
    print(json.dumps(best_payload, indent=2))
    print(f"Saved -> {best_path}")

    if args.refit_best:
        print("\nRefitting best config on full train and forecasting test...")
        best_cfg = LocalLinearConfig(
            kernel=best_payload["kernel"],
            bandwidth=best_payload["bandwidth"],
            min_samples=best_payload["min_samples"],
            ridge=best_payload["ridge"],
            coef_clip=best_payload["coef_clip"],
            anom_clip_scale=best_payload["anom_clip_scale"],
            use_interactions=bool(best_payload["use_interactions"]),
        )
        pred_test = fit_full_and_predict_test(train_df, test_df, config=best_cfg)
        pred_path = paths["best_preds"] / "local_linear_best_test.parquet"
        pred_test.to_parquet(pred_path, index=False)
        submission = make_submission(pred_test)
        sub_path = paths["best_preds"] / "local_linear_best_submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Saved best test predictions -> {pred_path}")
        print(f"Saved best submission -> {sub_path}")


if __name__ == "__main__":
    main()
