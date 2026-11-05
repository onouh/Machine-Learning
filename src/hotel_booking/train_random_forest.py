#!/usr/bin/env python3
"""Train a RandomForest regressor to predict ADR (Average Daily Rate).

Uses the hotel booking dataset and the RandomForest model with seed 42.

Pipeline:
1. Load data
2. Prepare features
3. Split into train/test (80/20)
4. Fit RandomForest on training data
5. Evaluate on test set (RMSE, MAE, R²)
6. Save the trained model

Usage:
    python train_random_forest.py [--data PATH] [--output DIR] [--test-size RATIO]

Example:
    python train_random_forest.py --data data/processed/H2_cleaned.csv --output models/rf_model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature definitions
NUMERIC_FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
]
CATEGORICAL_FEATURES = [
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "ReservedRoomType",
    "DepositType",
    "Agent",
    "CustomerType",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "ADR"

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_ESTIMATORS = 200


def build_preprocessor(
    scale_numeric: bool = False,
    min_frequency: int | None = 10,
) -> ColumnTransformer:
    """Build a ColumnTransformer for preprocessing features."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    other_categoricals = [c for c in CATEGORICAL_FEATURES if c not in ["Country", "Agent"]]
    fold_columns = ["Country", "Agent"]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat_other",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=min_frequency),
                other_categoricals,
            ),
            (
                "cat_fold",
                OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=True, min_frequency=min_frequency),
                fold_columns,
            ),
        ]
    )


def load_and_prepare_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV and prepare features + target."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Raw shape: {df.shape}")

    # Derive arrival features from date column if present
    if "date" in df.columns:
        print("  Deriving arrival features from 'date' column...")
        df["ArrivalDateYear"] = pd.to_datetime(df["date"]).dt.year
        df["ArrivalDateMonth"] = pd.to_datetime(df["date"]).dt.month_name()

    # Get features and target
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Clean feature columns
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Clean categorical columns
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str).str.strip()
        X[col] = X[col].replace(["", "NULL", "nan", "None"], "Unknown")

    print(f"  Features: {len(FEATURES)} columns")
    print(f"  Target (ADR): {y.name}")
    print(f"  ADR stats - mean: {y.mean():.2f}, median: {y.median():.2f}")
    print(f"  ADR==0 rows: {(y == 0).sum()}")

    return X, y


def clean_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    drop_adr_zero: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean data for regression."""
    df = pd.concat([X, y.rename(TARGET)], axis=1)

    initial_rows = len(df)
    print(f"\nCleaning data...")
    print(f"  Initial rows: {initial_rows}")

    # Remove rows with missing target
    missing_target = df[TARGET].isna().sum()
    if missing_target > 0:
        df = df.dropna(subset=[TARGET])
        print(f"  Removed {missing_target} rows with missing ADR")

    # Remove rows with invalid ADR
    invalid_adr = (df[TARGET] < 0).sum()
    if invalid_adr > 0:
        df = df[df[TARGET] >= 0]
        print(f"  Removed {invalid_adr} rows with negative ADR")

    # Optionally drop ADR==0 rows
    if drop_adr_zero:
        adr_zero = (df[TARGET] == 0).sum()
        df = df[df[TARGET] > 0]
        print(f"  Removed {adr_zero} rows with ADR == 0")

    print(f"  Final rows: {len(df)}")
    print(f"  Removed {initial_rows - len(df)} invalid rows total")

    return df[FEATURES], df[TARGET]


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int | None = None,
    max_features: str | float = "sqrt",
    min_samples_leaf: int = 2,
    seed: int = DEFAULT_SEED,
    n_jobs: int = -1,
) -> Pipeline:
    """Train a RandomForest regressor with preprocessing pipeline."""
    print(f"\nTraining RandomForest model...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  max_features: {max_features}")
    print(f"  min_samples_leaf: {min_samples_leaf}")
    print(f"  random_state (seed): {seed}")
    print(f"  Training samples: {len(X_train)}")

    pipeline = Pipeline(
        [
            ("prep", build_preprocessor(scale_numeric=False, min_frequency=10)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    print(f"  Training completed in {fit_time:.2f}s")

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate the trained model on test data."""
    print(f"\nEvaluating model on test set...")
    print(f"  Test samples: {len(X_test)}")

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time = time.perf_counter() - t0

    y_pred_clipped = np.clip(y_pred, 0, None)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_clipped)))
    mae = float(mean_absolute_error(y_test, y_pred_clipped))
    r2 = float(r2_score(y_test, y_pred_clipped))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "test_adr_mean": float(y_test.mean()),
        "test_adr_median": float(y_test.median()),
        "test_adr_max": float(y_test.max()),
        "pred_adr_mean": float(y_pred_clipped.mean()),
        "predict_time_seconds": round(predict_time, 3),
    }

    print(f"\n  Evaluation Results:")
    print(f"  {'='*40}")
    print(f"  RMSE:           {rmse:.2f}")
    print(f"  MAE:            {mae:.2f}")
    print(f"  R²:             {r2:.4f}")
    print(f"  Test ADR mean:  {metrics['test_adr_mean']:.2f}")
    print(f"  Pred ADR mean:  {metrics['pred_adr_mean']:.2f}")
    print(f"  Prediction time: {predict_time:.3f}s")

    return metrics


def save_model(
    pipeline: Pipeline,
    metrics: dict[str, Any],
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> Path:
    """Save the trained model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {output_dir}...")

    model_path = output_dir / "random_forest_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"  Model saved: {model_path}")

    metadata = {
        "model_type": "RandomForestRegressor",
        "seed": seed,
        "n_estimators": n_estimators,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "feature_names": FEATURES,
        "target": TARGET,
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {metadata_path}")

    features_path = output_dir / "feature_names.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(FEATURES, f, indent=2)
    print(f"  Feature names saved: {features_path}")

    return model_path


def train(
    data_path: Path,
    output_dir: Path,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int | None = None,
    max_features: str | float = "sqrt",
    min_samples_leaf: int = 2,
    drop_adr_zero: bool = False,
    save_model_flag: bool = True,
) -> dict[str, Any]:
    """Full training pipeline."""
    total_start = time.perf_counter()

    X, y = load_and_prepare_data(data_path)
    X_clean, y_clean = clean_data(X, y, drop_adr_zero=drop_adr_zero)

    print(f"\nSplitting data (test_size={test_size}, seed={seed})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_clean,
        test_size=test_size,
        random_state=seed,
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    pipeline = train_model(
        X_train,
        y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )

    metrics = evaluate_model(pipeline, X_test, y_test)

    model_path = None
    if save_model_flag:
        model_path = save_model(
            pipeline,
            metrics,
            output_dir,
            seed=seed,
            n_estimators=n_estimators,
        )

    total_time = time.perf_counter() - total_start

    results = {
        "seed": seed,
        "test_size": test_size,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_time_seconds": round(total_time, 2),
        "metrics": metrics,
        "model_path": str(model_path) if model_path else None,
        "output_dir": str(output_dir),
    }

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Output directory: {output_dir}")

    return results


def main() -> int:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a RandomForest regressor to predict hotel booking ADR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults
  python train_random_forest.py

  # Specify custom data and output paths
  python train_random_forest.py --data data/processed/H2_cleaned.csv --output models/rf_model

  # Use different hyperparameters
  python train_random_forest.py --n-estimators 400 --max-depth 20

  # Drop ADR==0 rows (complimentary stays)
  python train_random_forest.py --drop-adr-zero
        """,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/H2_cleaned.csv"),
        help="Path to the CSV data file (default: data/processed/H2_cleaned.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("models/random_forest"),
        help="Directory to save the trained model (default: models/random_forest)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Proportion of data for testing (default: {DEFAULT_TEST_SIZE})",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--n-estimators",
        "-n",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help=f"Number of trees in the forest (default: {DEFAULT_N_ESTIMATORS})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of trees (default: None = unlimited)",
    )
    parser.add_argument(
        "--max-features",
        type=str,
        default="sqrt",
        help='Features to consider at each split (default: "sqrt")',
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="Minimum samples per leaf (default: 2)",
    )
    parser.add_argument(
        "--drop-adr-zero",
        action="store_true",
        help="Drop rows where ADR == 0 (complimentary stays)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the model to disk (for testing)",
    )
    parser.add_argument(
        "--no-pretty-print",
        action="store_true",
        help="Suppress detailed progress output",
    )

    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        return 1

    try:
        results = train(
            data_path=args.data,
            output_dir=args.output,
            test_size=args.test_size,
            seed=args.seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_features=args.max_features,
            min_samples_leaf=args.min_samples_leaf,
            drop_adr_zero=args.drop_adr_zero,
            save_model_flag=not args.no_save,
        )

        if not args.no_pretty_print:
            print(f"\n{'='*50}")
            print("SUMMARY")
            print(f"{'='*50}")
            print(f"Model: RandomForestRegressor")
            print(f"Seed: {results['seed']}")
            print(f"n_estimators: {args.n_estimators}")
            print(f"max_depth: {args.max_depth}")
            print(f"max_features: {args.max_features}")
            print(f"min_samples_leaf: {args.min_samples_leaf}")
            print(f"Train samples: {results['train_samples']:,}")
            print(f"Test samples: {results['test_samples']:,}")
            print(f"RMSE: {results['metrics']['rmse']:.2f}")
            print(f"MAE: {results['metrics']['mae']:.2f}")
            print(f"R²: {results['metrics']['r2']:.4f}")
            print(f"Total time: {results['total_time_seconds']:.2f}s")

            if results['model_path']:
                print(f"\nModel saved to: {results['model_path']}")

        return 0

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
