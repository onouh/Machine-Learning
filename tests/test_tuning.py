"""Tests for the CV-based hyperparameter tuning module (tiny synthetic search)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hotel_booking.leakage.checker import ManifestFitter
from hotel_booking.regression.pipeline import clean_for_regression
from hotel_booking.tuning.search import (
    SCORING,
    build_full_pipeline,
    regressors,
    tune_model,
)
from tests.conftest import synthetic_adr_frame


def _train_frame() -> tuple[pd.DataFrame, np.ndarray]:
    cleaned, _ = clean_for_regression(synthetic_adr_frame(90), variant="after")
    X = cleaned.drop(columns=["ADR"])
    y = cleaned["ADR"].to_numpy(dtype=float)
    return X, y


def test_search_space_registry_covers_all_grid_models() -> None:
    specs = regressors(seed=42)
    expected = {
        "LinearRegression", "Ridge", "Lasso", "ElasticNet", "DecisionTree",
        "RandomForest", "HistGradientBoosting", "GradientBoosting",
        "SVR_RBF", "MLP_small", "MLP_deep",
    }
    assert set(specs) == expected
    for key, spec in specs.items():
        assert spec.param_distributions, key
        assert all(
        param.startswith(("model__", "prep__", "select__"))
        for param in spec.param_distributions
    ), key


def test_tune_model_ridge_tiny_search() -> None:
    X, y = _train_frame()
    spec = regressors(seed=42)["Ridge"]
    log = tune_model(spec, X, y, seed=42, n_iter=3, cv=2)
    assert "alpha" in log["best_params"]
    assert 0 < float(log["best_params"]["alpha"]) < np.inf
    assert any(
        k in log["best_params"] for k in ("strategy", "min_frequency", "with_mean", "k")
    ), "preprocessing params should be in best_params"
    assert log["best_cv_rmse"] > 0
    assert log["cv_folds"] == 2
    assert log["n_iter"] == 3
    assert log["search_seconds"] >= 0
    assert log["best_estimator"] is not None


def test_tune_model_decision_tree_has_full_space_logged() -> None:
    X, y = _train_frame()
    spec = regressors(seed=42)["DecisionTree"]
    log = tune_model(spec, X, y, seed=42, n_iter=3, cv=2)
    assert set(log["search_space_as_params"]) >= {"max_depth", "min_samples_leaf"}
    assert any(
        k in log["search_space_as_params"]
        for k in ("strategy", "min_frequency", "with_mean", "k")
    ), "preprocessing params should be in search space"
    assert log["best_params"].get("max_depth") in (
        4, 6, 8, 10, 12, None
    )


def test_tune_runs_inside_leakage_checked_pipeline() -> None:
    """The tuned best estimator must keep the manifest (proof of fold purity)."""
    X, y = _train_frame()
    spec = regressors(seed=42)["Ridge"]
    log = tune_model(spec, X, y, seed=42, n_iter=2, cv=2)
    best = log["best_estimator"]
    assert "manifest" in best.named_steps
    assert isinstance(best.named_steps["manifest"], ManifestFitter)
    assert best.named_steps["prep"].transformers[0][0] == "num"
    assert SCORING == "neg_root_mean_squared_error"


def test_build_full_pipeline_shape() -> None:
    X, y = _train_frame()
    spec = regressors(seed=42)["ElasticNet"]
    pipe = build_full_pipeline(spec, variant="after")
    pipe.fit(X.iloc[:40], y[:40])
    assert pipe.predict(X.iloc[40:50]).shape == (10,)
    assert "ADR" not in X.columns
    assert "select" in pipe.named_steps
    assert "prep" in pipe.named_steps
