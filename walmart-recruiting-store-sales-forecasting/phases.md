# Walmart Weekly Sales Forecasting Phases

## Phase 1: Local Linear Anomaly Model (`statsmodels`-style weighted regression) [DONE]
- Objective: anomaly-based forecasting with local seasonal kernels and recursive rollout.
- Status: completed.
- Artifacts:
  - `outputs/metrics/local_linear_cv.csv`
  - `outputs/preds/local_linear_oof.parquet`
  - `outputs/preds/local_linear_test.parquet`
  - `outputs/submissions/local_linear_submission.csv`
  - `outputs/search/local_linear_grid_results.csv`
  - `outputs/search/best_local_linear_config.json`

## Phase 2: Bayesian Structural AR Model (`PyMC`) [DONE]
- Objective: recursive probabilistic model with AR lags, trend, Fourier seasonality, and uncertainty bands.
- Status: completed (subset-200 prototype + diagnostics + full-data run).
- Artifacts:
  - `outputs/phase2_structural_ar_prototype/metrics/structural_ar_cv.csv`
  - `outputs/phase2_structural_ar_prototype/preds/structural_ar_oof.parquet`
  - `outputs/phase2_structural_ar_prototype/metrics/structural_ar_config.json`
  - `outputs/phase2_structural_ar_map_vs_chains/metrics/fold_comparison_metrics.csv`
  - `outputs/phase2_structural_ar_map_vs_chains/logs/fold_comparison_log.txt`
  - `outputs/phase2_structural_ar_full/metrics/structural_ar_cv.csv`
  - `outputs/phase2_structural_ar_full/preds/structural_ar_oof.parquet`
  - `outputs/phase2_structural_ar_full/metrics/structural_ar_config.json`

## Phase 3: Elastic Net Baseline (`scikit-learn`) [DONE]
- Objective: establish a simple linear regularized anomaly baseline on supervised lag-anomaly features.
- Status: completed.
- Data key: keep `Store + Dept` as series identifier.
- Features:
  - Lags of sales anomalies (`lag1`, `lag2`).
  - Calendar (`week_of_year`, month proxy, `IsHoliday`).
  - Exogenous (`temp_anom`, `fuel_anom`, `MarkDown1-5`, `CPI`, `Unemployment`).
- Validation:
  - Forward-chaining with recursive forecasting over validation horizon.
  - Metrics: `WMAE` (primary), `MAE`, `RMSE`.
- Artifacts:
  - `outputs/baselines/elastic_net/metrics.csv`
  - `outputs/baselines/elastic_net/oof.parquet`
  - `outputs/baselines/elastic_net/test.parquet`
  - `outputs/baselines/elastic_net/config.json`

## Phase 4: Random Forest Baseline (`scikit-learn`) [DONE]
- Objective: test a simple nonlinear tabular baseline with same feature interface as Phase 1.
- Status: completed.
- Features:
  - Same lag-anomaly/calendar/exogenous set as Elastic Net for fair comparison.
- Validation:
  - Same forward-chaining and recursive protocol as Phase 1.
  - Same metrics and artifact format.
- Artifacts:
  - `outputs/baselines/random_forest/metrics.csv`
  - `outputs/baselines/random_forest/oof.parquet`
  - `outputs/baselines/random_forest/test.parquet`
  - `outputs/baselines/random_forest/config.json`

## Phase 5: ETS Baseline (`statsmodels`) [DONE]
- Objective: test a classical time-series baseline on anomaly series with climatology reconstruction.
- Status: completed.
- Model:
  - `ExponentialSmoothing` (ETS), additive trend/seasonality candidate settings on residual anomalies after an exogenous linear component (`temp_anom`, `fuel_anom`, `MarkDown1-5`, `CPI`, `Unemployment`).
  - Per-series fitting for `Store + Dept` with fallback for sparse series.
- Validation:
  - Same forward-chaining fold dates and recursive multi-step forecasting.
  - Same metrics for direct ranking.
- Artifacts:
  - `outputs/baselines/ets/metrics.csv`
  - `outputs/baselines/ets/oof.parquet`
  - `outputs/baselines/ets/test.parquet`
  - `outputs/baselines/ets/config.json`

## Phase 6: Comparative Report Update [DONE]
- Objective: update `report/model_comparison.qmd` with baseline sections and final ranking.
- Status: completed.
- Contents:
  - Per-model setup summary.
  - Fold-level and mean metric tables.
  - Key plots and final comparative chart.
  - Short decision note on which baseline is best and why.

## Phase 7: AdaBoost Baseline (`scikit-learn`) [DONE]
- Objective: add a boosted tree anomaly baseline under the same recursive protocol for fair comparison.
- Status: completed.
- Data key: keep `Store + Dept` as series identifier.
- Features:
  - Same lag-anomaly/calendar/exogenous set (`lag1`, `lag2`, `week_of_year`, `month`, `IsHoliday`, `temp_anom`, `fuel_anom`, `MarkDown1-5`, `CPI`, `Unemployment`).
- Validation:
  - Same forward-chaining and recursive protocol as other tabular anomaly baselines.
  - Same metrics and artifact format.
- Artifacts:
  - `outputs/baselines/adaboost/metrics.csv`
  - `outputs/baselines/adaboost/oof.parquet`
  - `outputs/baselines/adaboost/test.parquet`
  - `outputs/baselines/adaboost/config.json`
