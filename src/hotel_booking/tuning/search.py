"""Per-model hyperparameter search spaces with CV-based tuning.

The search runs inside cross-validation over the *training split only* — the
held-out test set is never touched by the search, the folds, or the final
refit. Every searched model shares the same leakage-free preprocessing
pipeline built by the runner, so the tuning optimizes (preprocess + model)
jointly and the manifest leakage checker can prove fold purity
``hotel_booking.leakage``.

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable  # noqa: F401

import numpy as np
from scipy.stats import loguniform, uniform
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from hotel_booking.leakage.checker import ManifestFitter

SCORING = "neg_root_mean_squared_error"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    make_estimator: Callable[[], Any]
    param_distributions: dict[str, Any] | None
    n_iter: int
    cv: int
    subsample: int | None = None  # cap on training rows used for THIS model only
    note: str = ""


def regressors(seed: int) -> dict[str, ModelSpec]:
    """Model families with their search spaces (indices use target units)."""
    return {
        "LinearRegression": ModelSpec(
            "LinearRegression", lambda: LinearRegression(),
            {"model__fit_intercept": [True, False]}, n_iter=2, cv=3,
            note="trivial grid (baseline-only model per plan)",
        ),
        "Ridge": ModelSpec(
            "Ridge", lambda: Ridge(),
            {"model__alpha": loguniform(1e-3, 100)}, n_iter=20, cv=5,
        ),
        "Lasso": ModelSpec(
            "Lasso", lambda: Lasso(max_iter=100000, tol=1e-4),
            {"model__alpha": loguniform(1e-4, 10)}, n_iter=20, cv=5,
        ),
        "ElasticNet": ModelSpec(
            "ElasticNet", lambda: ElasticNet(max_iter=100000, tol=1e-4),
            {"model__alpha": loguniform(1e-4, 1),
             "model__l1_ratio": uniform(0.05, 0.95)},  # 0.05..1.0
            n_iter=20, cv=5,
        ),
        "DecisionTree": ModelSpec(
            "DecisionTree", lambda: DecisionTreeRegressor(random_state=seed),
            {"model__max_depth": [4, 6, 8, 10, 12, None],
             "model__min_samples_leaf": [1, 2, 5, 10, 20],
             "model__min_impurity_decrease": [0.0, 1e-4, 1e-3, 5e-3]},
            n_iter=30, cv=3,
        ),
        "RandomForest": ModelSpec(
            "RandomForest",
        lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed),
            {"model__n_estimators": [100, 200, 400],
             "model__max_depth": [12, 20, None],
             "model__max_features": ["sqrt", 0.3, 0.5],
             "model__min_samples_leaf": [1, 2, 5]},
            n_iter=8, cv=3, note="budgeted: 24 fits",
        ),
        "HistGradientBoosting": ModelSpec(
            "HistGradientBoosting", lambda: HistGradientBoostingRegressor(random_state=seed),
            {"model__max_iter": [200, 400, 600],
             "model__learning_rate": [0.03, 0.06, 0.1],
             "model__max_leaf_nodes": [31, 63],
             "model__l2_regularization": [0.0, 0.1]},
            n_iter=10, cv=5,
        ),
        "GradientBoosting": ModelSpec(
            "GradientBoosting",
        lambda: GradientBoostingRegressor(n_estimators=250, random_state=seed),
            {"model__learning_rate": [0.03, 0.05, 0.08],
             "model__max_depth": [2, 3],
             "model__subsample": [0.7, 1.0]},
            n_iter=4, cv=3,
        ),
        "SVR_RBF": ModelSpec(
            "SVR_RBF", lambda: SVR(kernel="rbf"),
            {"model__C": loguniform(0.1, 100),
             "model__epsilon": [0.05, 0.1, 0.2, 0.3],
             "model__gamma": ["scale", 0.001, 0.01]},
            n_iter=10, cv=3, subsample=8000,
            note="SVR is O(n^2) — fit budgeted to 8k subsampled training rows",
        ),
        "MLP_small": ModelSpec(
            "MLP_small",
        lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=seed
        ),
            {"model__hidden_layer_sizes": [(48, 24), (64, 32), (96, 48)],
             "model__alpha": loguniform(1e-5, 1e-2),
             "model__learning_rate_init": loguniform(1e-4, 5e-3)},
            n_iter=6, cv=3, subsample=15000,
        ),
        "MLP_deep": ModelSpec(
            "MLP_deep",
        lambda: MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), max_iter=200, early_stopping=True, random_state=seed
        ),
            {"model__hidden_layer_sizes": [(96, 48, 24), (128, 64, 32)],
             "model__alpha": loguniform(1e-5, 1e-2),
             "model__learning_rate_init": loguniform(1e-4, 3e-3)},
            n_iter=6, cv=3, subsample=15000,
        ),
    }


def build_full_pipeline(spec: ModelSpec, variant: str) -> Any:
    """Leakage-checked pipeline: manifest → preprocessing → estimator."""
    from sklearn.pipeline import Pipeline

    from hotel_booking.regression.pipeline import build_preprocessor

    return Pipeline(
        [
            ("manifest", ManifestFitter()),
            ("prep", build_preprocessor(variant)),
            ("model", spec.make_estimator()),
        ]
    )


def tune_model(
    spec: ModelSpec,
    X_train: Any,
    y_train: np.ndarray,
    *,
    seed: int,
    n_iter: int | None = None,
    cv: int | None = None,
    variant: str = "after",
    return_estimator: bool = True,
) -> dict[str, Any]:
    """RandomizedSearchCV over the training split; returns log + fitted search.

    The returned dict contains best_params_ (model__-prefixed keys already
    stripped), best CV score (neg RMSE), the search space, n_iter, cv, and
    search+refit seconds. The fitted best estimator (refit on all training
    rows) is included when ``return_estimator`` is set.
    """
    rng = np.random.default_rng(seed)
    X_use, y_use = X_train, y_train
    subsampled = 0
    if spec.subsample is not None and len(X_train) > spec.subsample:
        idx = np.sort(rng.choice(len(X_train), size=spec.subsample, replace=False))
        X_use, y_use = X_train.iloc[idx], y_train[idx]
        subsampled = int(spec.subsample)

    clf = RandomizedSearchCV(
        estimator=build_full_pipeline(spec, variant),
        param_distributions=spec.param_distributions or {},
        n_iter=n_iter if n_iter is not None else spec.n_iter,
        cv=cv if cv is not None else spec.cv,
        scoring=SCORING,
        n_jobs=1,  # search parallelism kept off; models parallelize internally
        refit=True,
        return_train_score=False,
        random_state=seed,
    )
    t0 = time.perf_counter()
    clf.fit(X_use, y_use)
    elapsed = time.perf_counter() - t0

    best_params = {
        k.removeprefix("model__"): v for k, v in clf.best_params_.items()
    }
    return {
        "best_estimator": clf.best_estimator_ if return_estimator else None,
        "best_params": best_params,
        "best_cv_neg_rmse": float(clf.best_score_),
        "best_cv_rmse": float(-clf.best_score_),
        "search_space_as_params": {
            k.split("__")[-1]: list(v) if isinstance(v, list) else v
            for k, v in (spec.param_distributions or {}).items()
        },
        "n_iter": int(clf.n_iter),
        "cv_folds": int(clf.cv) if isinstance(clf.cv, int) else int(clf.cv.get_n_splits()),
        "search_seconds": round(elapsed, 3),
        "refit_seconds": round(float(clf.refit_time_ or 0.0), 3),
        "subsample_used": subsampled,
        "rows_searched_on": int(len(X_use)),
    }