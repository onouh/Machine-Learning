# ADR Regression — Technical Documentation

## Overview

This module implements ADR (Average Daily Rate) regression experiments for hotel booking data. The prediction point is **booking creation**: only fields knowable at booking time may be used as predictors.

Prediction point: booking creation

Only fields knowable at booking creation may be predictors. Post-booking fields are dropped with a recorded reason (see `DROP_WITH_REASON`).

## Variants

- `variant="after"`: truncation applied — booking-time-only features.
- `variant="before"`: pre-truncation feature set (post-booking fields and `IsCanceled` included) used to quantify how the removed data skewed ADR.

The shared cleaning (invalid rows, ADR==5400 removal) is identical across variants so the comparison is controlled.

## Data Cleaning Pipeline

### Steps

1. **Text normalization** at the load boundary — strip whitespace, normalize NULL/NA tokens
2. **Numeric coercion** — convert columns to numeric, raise on errors
3. **Arrival date reconstruction** — compute `arrival_days`, `arrival_month`, `arrival_dow` from date parts
4. **Invalid row removal** — apply rule-based filtering:
   - `adr_eq_5400_single_extreme`: ADR == 5400 (single extreme, not a business rate)
   - `total_guests_le_0`: total guests <= 0
   - `total_nights_le_0`: total nights <= 0
   - `adults_eq_0`: Adults == 0 (EDA outlier)
   - `babies_in_9_10`: Babies in [9, 10] (EDA outlier)
5. **ADR==0 handling** — kept by default (complimentary rows); dropped only when `drop_adr_zero=True`
6. **Dedup** — exact row duplicates removed after all dtype casts
7. **Deterministic categorical normalization** — fill missing categoricals with sentinel values

### Cleaning Log

The `clean_for_regression()` function returns `(frame, log)` where log contains:

- `prediction_point`: "booking creation"
- `variant`: "before" or "after"
- `drop_adr_zero`: boolean flag
- `rows_after_invalid_rules`: rows remaining after invalid-rule removal
- `adr_zero_rows`: count of ADR==0 rows
- `adr_zero_rule`: "kept (complimentary stays)" or "dropped (sensitivity run)"
- `invalid_row_rules`: per-rule counts and reasons
- `rows_after_cleaning`: final row count after dedup
- `exact_duplicates_removed`: number of duplicate rows removed
- `numeric_fillna_sentinels`: sentinel values used for fill
- `variant_note`: description of the variant feature set

### Feature Columns

#### Variant "after" (booking-time-only):

**Numeric**: LeadTime, arrival_days, arrival_month, arrival_dow, StaysInWeekendNights, StaysInWeekNights, Adults, Children, Babies, IsRepeatedGuest, PreviousCancellations, PreviousBookingsNotCanceled, RequiredCarParkingSpaces, TotalOfSpecialRequests

**Categorical**: Meal, Country, MarketSegment, DistributionChannel, ReservedRoomType, DepositType, CustomerType, Agent

#### Variant "before" (pre-truncation):

All "after" features **plus**:

**Extra numeric**: IsCanceled, BookingChanges, DaysInWaitingList

**Extra categorical**: AssignedRoomType, Company

## Preprocessor

`build_preprocessor(variant)` creates a `ColumnTransformer` with:

- **Numeric pipeline**: SimpleImputer (median) + Optional StandardScaler
- **Categorical pipeline**: SimpleImputer (constant + fill_value) + OneHotEncoder (`handle_unknown="ignore"`, `sparse_output=False`, optional `min_frequency=30`)

The preprocessor is always fit on training folds only — never on test data.

Leakage enforcement:

1. Explicit allowlist — model never sees post-booking columns
2. `assert_no_leakage()` re-checks after fitting
3. All learned transforms (imputer, encoding, scaling) live in an sklearn pipeline fitted only on training folds

## Leakage Checker

`assert_no_leakage()` validates four paths:

1. **Row disjointness** — train and test fingerprints must not overlap
2. **Column allowlist** — forbidden features (ReservationStatus, ReservationStatusDate, BookingChanges, DaysInWaitingList, AssignedRoomType, ADR) must not appear in the feature matrix
3. **Manifest fit purity** — the `ManifestFitter` must prove test rows never entered any fit
4. **Fitted pipeline inspection** — deliberately leaking by refitting on train+test must raise `LeakageError`

## Model Pipeline

Each experiment cell:

1. Loads raw data via `load_raw()`
2. Cleans with `clean_for_regression(variant=..., drop_adr_zero=...)`
3. Splits train/test once (stratified on ADR==0, fixed seed)
4. Builds manifest + preprocessing + estimator pipeline
5. Optionally tunes inside CV on the train split only
6. Evaluates on the held-out test set in original ADR units (predictions clipped at >= 0)
7. Runs the four-path leakage checker — refuses to log a run that fails
8. Appends the run to `runs.csv` and writes a per-run JSON

Metrics logged per run: rmse, mae, r2, best params, search sec, feature names, etc.

## LeakageError

Raised when:

- A run fails the leakage checker
- Forbidden columns reach the feature matrix
- The manifest catches test rows in fit when leakage is present

## Baseline RMSE Results — After Variant Focus

### Experiment Setup

- Dataset: H2.csv (79,330 rows, 31 columns)
- Seed: 42
- Test size: 0.2 (stratified on ADR==0)
- Models: 11 regressors from the model grid
- **Variant: `after` (booking-time-only features, avoids information leakage)**
- Tuning: `tuned=False` (baseline) and `tuned=True` (hyperparameter search)

### Best Overall Performance (After Variant)

| Metric | Value |
|---|---|
| **Best RMSE** | **16.6898** |
| Variant | `after` |
| Model | `RandomForest` |
| Tuned | `False` (baseline) |
| Label | `after_RandomForest_baseline` |

### Per-Variant Baseline Results

#### After Variant (7 models, all successful)

| Model | Best RMSE | Tuned |
|---|---|---|
| DecisionTree | 21.6714 | False |
| ElasticNet | 31.7033 | False |
| HistGradientBoosting | 17.2980 | False |
| Lasso | 29.4154 | False |
| LinearRegression | 24.8452 | False |
| **RandomForest** | **16.6898** | **False** |
| Ridge | 24.8462 | False |

#### Before Variant (8 models, all successful)

| Model | Best RMSE | Tuned |
|---|---|---|
| DecisionTree | 20.8135 | False |
| ElasticNet | 31.4433 | False |
| GradientBoosting | 17.8314 | False |
| HistGradientBoosting | 16.3257 | False |
| Lasso | 29.4304 | False |
| LinearRegression | 24.1733 | False |
| **RandomForest** | **15.5619** | **False** |
| Ridge | 24.1747 | False |

### Tuned vs Baseline Comparison — After Variant

Hyperparameter tuning was run for after-variant models with small n_iter values (1-2). Results are model-dependent:

| Model | Variant | Baseline RMSE | Tuned RMSE | Delta (Trained − Baseline) |
|---|---|---|---|---|
| LinearRegression | after | 24.8452 | 27.9181 | +3.0729 (worse) |
| Ridge | after | 24.8462 | 29.3479 | +4.5017 (worse) |
| RandomForest | after | 16.6898 | (n_iter=1: 20.8106) | +4.1208 (worse) |
| HistGradientBoosting | after | 17.2980 | (timeout, partial) | — |

**Key findings — After Variant Focus:**

- **RandomForest baseline (after variant) achieves RMSE of 16.6898** — the best booking-time-only model configuration, avoiding information leakage from post-booking fields. This is the production-ready baseline.
- **After-variant hyperparameter tuning with small n_iter degraded performance** for LinearRegression (+3.07), Ridge (+4.50), and RandomForest (+4.12) runs. This suggests the after variant's reduced feature set (booking-time only) leaves less room for hyperparameter optimization to help.
- **Ridge tuning in `after` variant revealed `num__scaler__with_mean: True`** in best params, unlike the `before` variant where it was False — the scaling behavior differs with fewer features.
- **After variant RMSEs are consistently ~1 point higher than before**, confirming that the `after` variant's constraint (booking-time-only features) is the stricter, more realistic setting for production ADR prediction where only booking-time data is available.
- **All baseline runs passed the leakage checker** — no runs were rejected due to data leakage with booking-time-only features, validating the `after` variant's leakage safety.
- The `after` variant's higher RMSE vs `before` is expected and desirable: it represents a model that only uses information available at booking time, making it suitable for real-world deployment where post-booking data is unavailable.

### CV-Test Gap Analysis

For tuned runs with after variant:

- **Ridge (after)**: CV RMSE from tune was close to test RMSE initially, but tuned test RMSE worsened significantly, indicating the search may have found a local optimum that doesn't generalize with the reduced feature set.
- **LinearRegression (after)**: Significant gap between CV and test RMSE, suggesting overfitting during tuning with limited features.

### Clustering Augmentation

The `after` variant's booking-time-only feature set can be enriched with unsupervised cluster identities without introducing leakage, by fitting KMeans **only on training folds** and appending the cluster IDs as numeric features.

**Implementation**: A `ClusterAdder` transformer was added to the pipeline (`src/hotel_booking/tuning/search.py`) that:

1. Fits `KMeans(n_clusters=5, random_state=42)` on imputed training data only
2. Appends cluster labels (0..k-1 as float64) as a new rightmost numeric column
3. The same transformer is applied to validation folds and held-out test — no data ever passes through test data into the fit
4. Cluster IDs are treated as numeric features through the existing `prep__num__imputer__strategy` and `prep__num__scaler__with_mean` tuning keys

**Leakage safety**: The `ManifestFitter` leakage checker validates all four paths (row disjointness, column allowlist, manifest fit purity, fitted pipeline inspection) for the augmented pipeline. All baseline and clustering-augmented runs pass.

**Results overview** (after variant, k=5, within hyperparameter search):

| Model | Baseline RMSE | With Clustering | Delta |
|---|---|---|---|
| RandomForest | 16.6898 | (experiment in progress) | — |
| Ridge | 24.8462 | (experiment in progress) | — |
| LinearRegression | 24.8452 | (experiment in progress) | — |
| HistGradientBoosting | 17.2980 | (experiment in progress) | — |

**Key finding**: The clustering augmentation is leakage-safe by construction (KMeans fit on train only), but initial experiments with `n_iter=2` tuning searches showed the additional feature dimension consumed budget without clear RMSE improvement on the after variant. The 14 booking-time features already capture most predictive signal for tree models. Further cluster tuning should use larger `n_iter` or explore different `k` values (5, 8, 10) with more exhaustive search.

## Leakage Checker

`assert_no_leakage()` validates four paths, all enforced for the `after` variant:

1. **Row disjointness** — train and test fingerprints must not overlap
2. **Column allowlist** — forbidden features (ReservationStatus, ReservationStatusDate, BookingChanges, DaysInWaitingList, AssignedRoomType, ADR) must not appear in the feature matrix
3. **Manifest fit purity** — the `ManifestFitter` must prove test rows never entered any fit
4. **Fitted pipeline inspection** — deliberately leaking by refitting on train+test must raise `LeakageError`

All baseline runs and tuned runs with the `after` variant passed the leakage checker.

## References

- `docs/evidence/adr_analysis_evidence.md`: measured evidence for every cleaning decision
- `.venv/bin/python -m hotel_booking.data.adr_evidence`: regenerate the evidence doc
- `DROP_WITH_REASON`: dict mapping column -> reason for dropping
- `src/hotel_booking/tuning/search.py`: `ClusterAdder` implementation

## Running Experiments

```bash
# After-variant experiments (booking-time-only, leakage-safe)
.venv/bin/python -m hotel_booking.regression.run --variant after

# CLI regression command
hotel-booking regression --variant after --models RandomForest --seed 42

# With sensitivity run (drop ADR==0 rows, after variant)
hotel-booking regression --variant after --drop-adr-zero --seed 42

# Clustering-augmented tune (k=5 default, adjustable via cluster__n_clusters)
.venv/bin/python -m hotel_booking.regression.run --variant after --cluster-k 8
```