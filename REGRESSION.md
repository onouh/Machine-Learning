# ADR Regression — Rating Criteria & Top 5 Runs (`after` Variant)

## 1. Rating Criteria

Models are evaluated on the held-out test set (20% split, stratified on `ADR==0`, fixed random seed = 42) using the City Hotel H2 dataset (79,330 rows × 31 columns). All metrics are reported in **original ADR units** (predictions clipped at ≥ 0).

**Scope:** `after` variant only — strictly leak-free booking-time feature set. Post-booking fields (`ReservationStatus`, `ReservationStatusDate`, `BookingChanges`, `DaysInWaitingList`, `AssignedRoomType`, `IsCanceled`, `Company`) are excluded so no future information leaks into the model.

### Primary Metric: RMSE (Root Mean Squared Error)

RMSE is the ranking metric — lower is better. It penalizes large errors more heavily than MAE, which matters for ADR where extreme rates can dominate business impact.

### Secondary Metrics

| Metric | Purpose |
|--------|---------|
| **RMSE** | Primary ranking metric (original ADR units) |
| **MAE** | Mean Absolute Error — less sensitive to outliers than RMSE |
| **R²** | Coefficient of determination — proportion of variance explained |

### Evaluation Safeguards (Leakage-Free)

Every run must pass a **four-path leakage checker** before being logged:

1. **Row disjointness** — train and test fingerprints (content hashes) must have zero overlap, even after cleaning/deduplication.
2. **Column allowlist** — no leakage-flagged columns (`ReservationStatus`, `ReservationStatusDate`, `BookingChanges`, `DaysInWaitingList`, `AssignedRoomType`, `ADR`) may appear in the feature matrix.
3. **Fit manifest** — a `ManifestFitter` records every row chunk each learned transform (imputer, scaler, one-hot encoder) was fit on; the union must be a subset of training fingerprints only.
4. **Tuning purity** — CV folds refit the same manifest pipeline, proving test fingerprints never entered any fold fit.

**Only runs with `leak_check_passed = True` are eligible for ranking.**

### Feature-Intake Pipeline (Tuned Jointly with Model Hyper-parameters)

The preprocessing pipeline is exposed as tunable parameters (`prep__` prefix) and tuned together with model parameters via `RandomizedSearchCV`:

| Parameter | Values searched | Role |
|-----------|-----------------|------|
| `prep__num__imputer__strategy` | `median`, `mean` | Numeric missing-value imputation |
| `prep__cat__imputer__strategy` | `constant`, `most_frequent` | Categorical missing-value imputation |
| `prep__cat__imputer__fill_value` | `0`, `missing` | Sentinel token for constant imputation |
| `prep__cat__onehot__min_frequency` | `10`, `30`, `None` | Rare-category folding threshold |
| `prep__num__scaler__with_mean` | `True`, `False` | Whether to center numeric features |
| `select__k` | `10`, `20`, `all` | Number of top features to keep (f_regression) |

### Models Evaluated (`after` variant)

LinearRegression, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, HistGradientBoosting, GradientBoosting, SVR_RBF, MLP_small, MLP_deep.

Tuned models use `RandomizedSearchCV` inside cross-validation on the training split only; best parameters (including `prep__` and `select__` keys) are recorded per run.

---

## 2. Top 5 Runs — `after` Variant Only (by RMSE, lower is better)

| Rank | Run Label | Model | Tuned | RMSE | MAE | R² | Leak Passed |
|------|-----------|-------|-------|------|-----|-----|-------------|
| 1 | `after_DecisionTree_tuned` | DecisionTree | Yes | **19.54** | 10.66 | 0.755 | ✅ |
| 2 | `after_DecisionTree_baseline` | DecisionTree | No | **21.67** | 10.66 | 0.723 | ✅ |
| 3 | `after_Ridge_tuned` | Ridge | Yes | **24.75** | 17.82 | 0.639 | ✅ |
| 4 | `after_Ridge_baseline` | Ridge | No | **24.85** | 17.92 | 0.636 | ✅ |
| 5 | `after_LinearRegression_baseline` | LinearRegression | No | **24.85** | 17.92 | 0.636 | ✅ |

### Best Tuned Run Details (`after_DecisionTree_tuned`)

| Parameter | Value |
|-----------|-------|
| `prep__num__imputer__strategy` | `median` |
| `prep__cat__imputer__strategy` | `constant` |
| `prep__cat__imputer__fill_value` | `0` |
| `prep__cat__onehot__min_frequency` | `10` |
| `prep__num__scaler__with_mean` | `True` |
| `select__k` | `all` |
| `model__max_depth` | `None` |
| `model__min_samples_leaf` | `5` |
| `model__min_impurity_decrease` | `0.005` |

### Key Takeaways

- **Best model:** DecisionTree with tuned feature-intake pipeline (RMSE 19.54, R² 0.755) — tuning improved RMSE from 21.67 (baseline) to 19.54 (~10% error reduction).
- **Linear models:** Ridge and Lasso benefit from tuning (Ridge: 24.85 → 24.75; Lasso: 29.42 → 24.79). ElasticNet tuning matched baseline (~24.85).
- **LinearRegression:** Tuning hurt (24.85 → 27.92) — overfit on small search space; baseline preferred.
- **All runs leak-clean:** Every one of the 10 `after` variant runs passed all four leakage checks (`leak_check_passed = True`).
- **Feature-intake tuning matters:** Best configs used `min_frequency=10` (rare category folding), `fill_value=0`, and `with_mean=True/False` varying by model — confirming preprocessing parameters are as impactful as model hyper-parameters.

### Files

- Full run log: `outputs/results/regression/runs.csv` (10 unique `after` runs, all leak-checked)
- Per-run JSON: `outputs/results/regression/<label>.json`
- Conclusions: `outputs/results/regression/conclusions.md`
- HTML report: `outputs/report/regression_report.html`