# ADR Regression — Rating Criteria & Top 5 Runs

## 1. Rating Criteria

Models are evaluated on the held-out test set (20% split, stratified on `ADR==0`, fixed random seed = 42) using the City Hotel H2 dataset (79,330 rows × 31 columns). All metrics are reported in **original ADR units** (predictions clipped at ≥ 0).

### Primary Metric: RMSE (Root Mean Squared Error)

RMSE is the ranking metric — lower is better. It penalizes large errors more heavily than MAE, which matters for ADR where extreme rates (e.g., the single ADR=5,400 outlier) can dominate business impact.

### Secondary Metrics

| Metric | Purpose |
|--------|---------|
| **RMSE** | Primary ranking metric (original ADR units) |
| **MAE** | Mean Absolute Error — less sensitive to outliers than RMSE |
| **R²** | Coefficient of determination — proportion of variance explained |

### Evaluation Safeguards (Leakage-Free)

Every run must pass a **four-path leakage checker** before being logged:

1. **Row disjointness** — train and test fingerprints (content hashes) must have zero overlap, even after cleaning/deduplication.
2. **Column allowlist** — no leakage-flagged columns (`ReservationStatus`, `ReservationStatusDate`, `BookingChanges`, `DaysInWaitingList`, `AssignedRoomType`, `ADR`) may appear in the feature matrix for the strict `after` variant.
3. **Fit manifest** — a `ManifestFitter` records every row chunk each learned transform (imputer, scaler, one-hot encoder) was fit on; the union must be a subset of training fingerprints only.
4. **Tuning purity** — CV folds refit the same manifest pipeline, proving test fingerprints never entered any fold fit.

**Only runs with `leak_check_passed = True` are eligible for ranking.**

### Variants

| Variant | Description |
|---------|-------------|
| `before` | Pre-truncation feature set — includes post-booking fields (`IsCanceled`, `BookingChanges`, `DaysInWaitingList`, `AssignedRoomType`) and `Company`. Used to quantify how removed data skewed ADR. |
| `after` | Truncated booking-time-only feature set — strictly leak-free. Fields knowable only at booking creation. This is the production-validated variant. |

### Models Evaluated

LinearRegression, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, HistGradientBoosting, GradientBoosting, SVR_RBF, MLP_small, MLP_deep.

Tuned models use `RandomizedSearchCV` inside cross-validation on the training split only; best parameters are recorded per run.

---

## 2. Top 5 Runs (by RMSE, lower is better)

| Rank | Run Label | Variant | Model | Tuned | RMSE | MAE | R² | Leak Passed |
|------|-----------|---------|-------|-------|------|-----|-----|-------------|
| 1 | `before_RandomForest_baseline` | before | RandomForest | No | **15.56** | 8.18 | 0.857 | ✅ |
| 2 | `before_HistGradientBoosting_baseline` | before | HistGradientBoosting | No | **16.33** | 10.33 | 0.843 | ✅ |
| 3 | `before_RandomForest_tuned` | before | RandomForest | Yes | **16.40** | 9.77 | 0.842 | ✅ |
| 4 | `before_GradientBoosting_baseline` | before | GradientBoosting | No | **17.83** | 11.87 | 0.813 | ✅ |
| 5 | `before_DecisionTree_tuned` | before | DecisionTree | Yes | **18.76** | 10.57 | 0.793 | ✅ |

### Key Takeaways

- **Best overall:** `before_RandomForest_baseline` (RMSE 15.56, R² 0.857) — however, this is the `before` variant which includes post-booking leakage features; it represents an upper-bound (optimistic) result.
- **Best leak-free (`after`) variant:** `after_DecisionTree_baseline` (RMSE 21.67, R² 0.723) — this is the honest, production-safe result with no post-booking information.
- **Tuning impact:** Tuning helped DecisionTree (20.81 → 18.76 RMSE) and RandomForest CV (17.18 → 16.40 test RMSE on `before`), but RandomForest baseline (15.56) outperformed its tuned version (16.40), indicating overfitting risk.
- **Before vs after gap:** For DecisionTree, removing post-booking data increased RMSE from 18.76 → 21.67 (delta −2.92), confirming the leaked features carried signal but are unavailable at booking time.

### Files

- Full run log: `outputs/results/regression/runs.csv` (19 unique runs, all leak-checked)
- Per-run JSON: `outputs/results/regression/<label>.json`
- Conclusions: `outputs/results/regression/conclusions.md`
- HTML report: `outputs/report/regression_report.html`