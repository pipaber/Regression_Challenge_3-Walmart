from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.ensemble import AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from baseline_utils import (
    ANOM_FEATURE_COLS,
    LAG_ORDERS,
    ensure_output_dir,
    fit_full_train_and_predict_test_tabular_anomaly,
    load_data,
    run_cv_tabular_baseline_anomaly,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7: AdaBoost anomaly baseline with recursive forward-chaining evaluation."
    )
    parser.add_argument("--train-path", default="train_feat.parquet")
    parser.add_argument("--test-path", default="test_feat.parquet")
    parser.add_argument("--output-dir", default="outputs/baselines/adaboost")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-weeks", type=int, default=13)
    parser.add_argument("--max-series", type=int, default=None)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--loss", default="linear", choices=["linear", "square", "exponential"])
    return parser.parse_args()


def make_model(
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    loss: str,
):
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    # Backward-compatible constructor: newer sklearn uses `estimator`,
    # older versions use `base_estimator`.
    try:
        model = AdaBoostRegressor(
            estimator=tree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=42,
        )
    except TypeError:
        model = AdaBoostRegressor(
            base_estimator=tree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=42,
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(Path(args.output_dir))

    print("Loading data...")
    train_df, test_df = load_data(Path(args.train_path), Path(args.test_path), max_series=args.max_series)
    print(
        f"train rows={train_df.shape[0]}, test rows={test_df.shape[0]}, "
        f"train series={train_df['series_id'].nunique()}, test series={test_df['series_id'].nunique()}"
    )

    print("\nRunning recursive forward-chaining CV...")
    metrics_df, oof_df = run_cv_tabular_baseline_anomaly(
        train_df,
        model_builder=lambda: make_model(
            args.n_estimators,
            args.learning_rate,
            args.max_depth,
            args.loss,
        ),
        n_folds=args.n_folds,
        val_weeks=args.val_weeks,
        lag_orders=LAG_ORDERS,
        feature_cols=ANOM_FEATURE_COLS,
    )

    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics -> {metrics_path}")

    oof_path = output_dir / "oof.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"Saved OOF predictions -> {oof_path}")

    print("\nFitting full train and forecasting test recursively...")
    test_pred = fit_full_train_and_predict_test_tabular_anomaly(
        train_df,
        test_df,
        model_builder=lambda: make_model(
            args.n_estimators,
            args.learning_rate,
            args.max_depth,
            args.loss,
        ),
        lag_orders=LAG_ORDERS,
        feature_cols=ANOM_FEATURE_COLS,
    )
    test_path = output_dir / "test.parquet"
    test_pred.to_parquet(test_path, index=False)
    print(f"Saved test predictions -> {test_path}")

    config = {
        "model": "adaboost",
        "target": "sales_anom",
        "features": ANOM_FEATURE_COLS,
        "lags": list(LAG_ORDERS),
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "loss": args.loss,
        "n_folds": args.n_folds,
        "val_weeks": args.val_weeks,
        "max_series": args.max_series,
        "train_path": args.train_path,
        "test_path": args.test_path,
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved config -> {config_path}")


if __name__ == "__main__":
    main()
