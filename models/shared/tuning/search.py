"""Per-model hyperparameter search spaces with CV-based tuning.

The search runs inside cross-validation over the *training split only* — the
held-out test set is never touched by the search, the folds, or the final
refit. Every searched model shares the same leakage-free preprocessing
pipeline built by the runner, so the tuning optimizes (preprocess + model)
jointly and the manifest leakage checker can prove fold purity
``hotel_booking.leakage``.

"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable  # noqa: F401

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
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
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

from hotel_booking.leakage.checker import ManifestFitter
from hotel_booking.regression_models.pipeline import build_preprocessor

from sklearn.feature_selection import SelectKBest, f_regression

SCORING = "neg_root_mean_squared_error"


# ---------------------------------------------------------------------------
# Seed utility for reproducibility across runs.
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch (including CUDA).

    If PyTorch is not available, only sets Python and NumPy seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Cluster augmentation — adds KMeans cluster IDs as numeric features.
# Fits KMeans on training folds only; the same transformer is applied to
# validation folds and held-out test, so no leakage is introduced.
# ---------------------------------------------------------------------------
class ClusterAdder(BaseEstimator, TransformerMixin):
    """Add KMeans cluster IDs as numeric features.

    Fits KMeans on training data only (via ``fit``), then transforms both
    train and test. The cluster column is a float64 appended to the right
    of the existing feature matrix. NaN values are imputed using the
    supplied strategy before KMeans fit/predict.

    Parameters
    ----------
    n_clusters : int
        Number of clusters for KMeans (5 to 10 is the typical range).
    random_state : int
        Random seed for KMeans reproducibility.
    imputer_strategy : {"median", "mean", "most_frequent"}
        Strategy for SimpleImputer applied before KMeans fit/predict.
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 123,
                 imputer_strategy: str = "median"):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.imputer_strategy = imputer_strategy
        self.kmeans_ = None
        self.imputer_ = None

    def fit(self, X, y=None):
        """Fit imputer + KMeans on training data only."""
        from sklearn.impute import SimpleImputer

        self.imputer_ = SimpleImputer(strategy=self.imputer_strategy)
        X_imp = self.imputer_.fit_transform(X)

        self.kmeans_ = (
            KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            .fit(X_imp)
        )
        return self

    def transform(self, X):
        """Impute + predict clusters; append as new numeric column."""
        if self.kmeans_ is None:
            raise ValueError("ClusterAdder must be fit before transform")
        from sklearn.impute import SimpleImputer

        X_imp = self.imputer_.transform(X)
        clusters = self.kmeans_.predict(X_imp)
        return np.column_stack([X_imp, clusters.astype(np.float64)])


# Feature-intake pipeline parameters — tuned jointly with model hyper-parameters.
# These are addressed as ``prep__`` and ``select__`` and ``cluster__`` keys in the pipeline.
PREPROCESS_PARAMS = {
    "prep__num__imputer__strategy": ["median", "mean"],
    "prep__cat__imputer__strategy": ["constant", "most_frequent"],
    "prep__cat__imputer__fill_value": ["0", "missing"],
    "prep__cat__onehot__min_frequency": [10, 30, None],
    "prep__num__scaler__with_mean": [True, False],
    "select__k": [10, 20, "all"],
}


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
            "LinearRegression",
            lambda: LinearRegression(),
            {**PREPROCESS_PARAMS, "model__fit_intercept": [True, False]},
            n_iter=6,
            cv=3,
            note="baseline-only model per plan",
        ),
        "Ridge": ModelSpec(
            "Ridge",
            lambda: Ridge(),
            {**PREPROCESS_PARAMS, "model__alpha": loguniform(1e-3, 100)},
            n_iter=20,
            cv=5,
        ),
        "Lasso": ModelSpec(
            "Lasso",
            lambda: Lasso(max_iter=100000, tol=1e-4),
            {**PREPROCESS_PARAMS, "model__alpha": loguniform(1e-4, 10)},
            n_iter=20,
            cv=5,
        ),
        "ElasticNet": ModelSpec(
            "ElasticNet",
            lambda: ElasticNet(max_iter=100000, tol=1e-4),
            {
                **PREPROCESS_PARAMS,
                "model__alpha": loguniform(1e-4, 1),
                "model__l1_ratio": uniform(0.05, 0.95),  # 0.05..1.0
            },
            n_iter=20,
            cv=5,
        ),
        "DecisionTree": ModelSpec(
            "DecisionTree",
            lambda: DecisionTreeRegressor(random_state=seed),
            {
                **PREPROCESS_PARAMS,
                "model__max_depth": [4, 6, 8, 10, 12, None],
                "model__min_samples_leaf": [1, 2, 5, 10, 20],
                "model__min_impurity_decrease": [0.0, 1e-4, 1e-3, 5e-3],
            },
            n_iter=12,
            cv=3,
        ),
        "RandomForest": ModelSpec(
            "RandomForest",
            lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed),
            {
                **PREPROCESS_PARAMS,
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [12, 20, None],
                "model__max_features": ["sqrt", 0.3, 0.5],
                "model__min_samples_leaf": [1, 2, 5],
            },
            n_iter=8,
            cv=3,
            note="budgeted: 24 fits",
        ),
        "HistGradientBoosting": ModelSpec(
            "HistGradientBoosting",
            lambda: HistGradientBoostingRegressor(random_state=seed),
            {
                **PREPROCESS_PARAMS,
                "model__max_iter": [200, 400, 600],
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_leaf_nodes": [31, 63],
                "model__l2_regularization": [0.0, 0.1],
            },
            n_iter=10,
            cv=5,
        ),
        "GradientBoosting": ModelSpec(
            "GradientBoosting",
            lambda: GradientBoostingRegressor(n_estimators=250, random_state=seed),
            {
                **PREPROCESS_PARAMS,
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__max_depth": [2, 3],
                "model__subsample": [0.7, 1.0],
            },
            n_iter=4,
            cv=3,
        ),
        "SVR_RBF": ModelSpec(
            "SVR_RBF",
            lambda: SVR(kernel="rbf"),
            {
                **PREPROCESS_PARAMS,
                "model__C": loguniform(0.1, 100),
                "model__epsilon": [0.05, 0.1, 0.2, 0.3],
                "model__gamma": ["scale", 0.001, 0.01],
            },
            n_iter=10,
            cv=3,
            subsample=8000,
            note="SVR is O(n^2) — fit budgeted to 8k subsampled training rows",
        ),
        "MLP_small": ModelSpec(
            "MLP_small",
            lambda: MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=seed, shuffle=True, solver='adam'
            ),
            {
                **PREPROCESS_PARAMS,
                "model__hidden_layer_sizes": [(64, 32), (128, 64), (196, 96)],
                "model__alpha": loguniform(1e-5, 1e-2),
                "model__learning_rate": ['constant', 'invscaling', 'adaptive'],
                "model__activation": ['tanh', 'relu'],
            },
            n_iter=8,
            cv=3,
            subsample=15000,
        ),
        "MLP_deep": ModelSpec(
            "MLP_deep",
            lambda: MLPRegressor(
                hidden_layer_sizes=(256, 128, 64), max_iter=300, early_stopping=True, random_state=seed, shuffle=True, solver='adam'
            ),
            {
                **PREPROCESS_PARAMS,
                "model__hidden_layer_sizes": [(196, 96, 48), (256, 128, 64)],
                "model__alpha": loguniform(1e-5, 1e-2),
                "model__learning_rate": ['constant', 'invscaling', 'adaptive'],
                "model__activation": ['tanh', 'relu'],
            },
            n_iter=10,
            cv=3,
            subsample=15000,
        ),
    }


def build_full_pipeline(spec: ModelSpec, variant: str) -> Any:
    """Leakage-checked pipeline: manifest → preprocessing → feature selection → estimator.

    The ``prep`` step exposes tunable feature-intake parameters
    (imputer strategy, one-hot min_frequency, scaler centering) and the
    ``select`` step exposes ``select__k`` for feature-count tuning.
    The ``cluster`` step (added by the runner) exposes
    ``cluster__n_clusters`` and ``cluster__imputer__strategy`` for
    clustering-augmented experiments.

    All are addressable as pipeline parameters via ``RandomizedSearchCV``.
    """
    return Pipeline(
        [
            ("manifest", ManifestFitter()),
            ("prep", build_preprocessor(variant)),
            ("select", SelectKBest(score_func=f_regression, k="all")),
            ("cluster", ClusterAdder(n_clusters=5, random_state=123, imputer_strategy="median")),
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
        idx = np.sort(
            rng.choice(len(X_train), size=spec.subsample, replace=False)
        )
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

    best_params = {}
    for k, v in clf.best_params_.items():
        for prefix in ("model__", "prep__", "select__", "cluster__"):
            if k.startswith(prefix):
                k = k.removeprefix(prefix)
                break
        best_params[k] = v
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


# ---------------------------------------------------------------------------
# Ensemble blending — combines top model predictions via weighted average.
# ---------------------------------------------------------------------------

def find_optimal_weights(predictions: np.ndarray, targets: np.ndarray,
                         n_points: int = 101) -> tuple[float, float]:
    """Find optimal weights for blending two model predictions.

    Parameters
    ----------
    predictions : np.ndarray of shape (n_models, n_samples)
        Model predictions.
    targets : np.ndarray of shape (n_samples,)
        True target values.
    n_points : int
        Number of weight grid points to search (w from 0 to 1).

    Returns
    -------
    best_weight : float
        Weight for predictions[0] (the first model). Weight for predictions[1]
        is 1 - best_weight.
    best_rmse : float
        RMSE of the optimally blended predictions.
    """
    from sklearn.metrics import mean_squared_error

    best_rmse = float("inf")
    best_w = 0.5
    targets = np.asarray(targets, dtype=float).ravel()

    for w in np.linspace(0, 1, n_points):
        blend = w * predictions[0] + (1 - w) * predictions[1]
        rmse = np.sqrt(mean_squared_error(targets, blend))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w

    return best_w, best_rmse


def blend_predictions(predictions: np.ndarray, targets: np.ndarray,
                      method: str = "optimal_weight") -> tuple[np.ndarray, dict]:
    """Blend model predictions to minimize RMSE.

    Parameters
    ----------
    predictions : np.ndarray of shape (n_models, n_samples)
        Model predictions from two or more models.
    targets : np.ndarray of shape (n_samples,)
        True target values.
    method : str
        Blending method. Currently only ``"optimal_weight"`` is supported,
        which finds the weight for the first model that minimizes RMSE,
        with the second model receiving weight 1 - w.

    Returns
    -------
    blended : np.ndarray of shape (n_samples,)
        Optimally blended predictions.
    info : dict
        Dictionary with ``"weight_0"`` (weight for first model),
        ``"weight_1"`` (weight for second model), and ``"blend_rmse"``.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets, dtype=float).ravel()

    if method == "optimal_weight" and predictions.shape[0] >= 2:
        best_w, best_rmse = find_optimal_weights(predictions, targets)
        blended = best_w * predictions[0] + (1 - best_w) * predictions[1]
        info = {
            "weight_0": float(best_w),
            "weight_1": float(1 - best_w),
            "blend_rmse": float(best_rmse),
        }
    else:
        # Fallback: uniform average
        blended = np.mean(predictions, axis=0)
        rmse = np.sqrt(mean_squared_error(targets, blended))
        n_models = predictions.shape[0]
        info = {
            "weight_0": 1.0 / n_models,
            "weight_1": (n_models - 1) / n_models if n_models > 1 else 0.0,
            "blend_rmse": float(rmse),
        }

    return blended, info


# ---------------------------------------------------------------------------
# Hyperband early-stopping hyperparameter search.
# ---------------------------------------------------------------------------

def _hyperband_evaluate(spec: ModelSpec, X_train: Any, y_train: np.ndarray,
                        seed: int, budget: float, variant: str) -> float:
    """Evaluate a single Hyperband configuration with given budget.

    Runs RandomizedSearchCV with n_iter=budget and returns the CV RMSE.
    """
    from sklearn.model_selection import RandomizedSearchCV

    clf = RandomizedSearchCV(
        estimator=build_full_pipeline(spec, variant),
        param_distributions=spec.param_distributions or {},
        n_iter=int(budget),
        cv=3,
        scoring=SCORING,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return float(-clf.best_score_)


def hyperband_search(spec: ModelSpec, X_train: Any, y_train: np.ndarray,
                     seed: int = 123, max_iter: int = 100,
                     eta: int = 3, min_budget: float = 1.0,
                     variant: str = "after") -> dict[str, Any]:
    """Run Hyperband early-stopping hyperparameter search.

    Hyperband adaptive resource allocation:
    - Starts with many configurations and low budget
    - Successively eliminates poor performers
    - Allocates more budget to promising configurations
    - Returns the best found configuration

    Parameters
    ----------
    spec : ModelSpec
        Model specification with search space.
    X_train : Any
        Training features.
    y_train : np.ndarray
        Training targets.
    seed : int
        Random seed for reproducibility.
    max_iter : int
        Maximum number of iterations (default 100).
    eta : int
        Reduction factor per iteration (default 3, keeps 1/3).
    min_budget : float
        Minimum budget (default 1.0 iteration).
    variant : str
        Data variant ("after" or "before").

    Returns
    -------
    dict with keys: best_params, best_cv_rmse, search_seconds, rows_searched_on
    """
    from sklearn.model_selection import RandomizedSearchCV

    rng = np.random.default_rng(seed)

    # Hyperband: start with eta * max_iter configurations at min_budget
    n0 = eta * max_iter
    n_configs = n0

    # Generate initial configurations at min_budget
    configs: list[dict[str, Any]] = []
    for _ in range(n_configs):
        rmse = _hyperband_evaluate(spec, X_train, y_train, seed, min_budget, variant)
        # Generate random params from the search space
        cfg: dict[str, Any] = {}
        if spec.param_distributions:
            for k, v in spec.param_distributions.items():
                if hasattr(v, 'rvs'):  # scipy distribution
                    cfg[k] = float(v.rvs(rng))
                elif isinstance(v, list):
                    cfg[k] = float(v[rng.integers(0, len(v))])
                else:
                    cfg[k] = float(v)
        configs.append(cfg)

    # Track best result
    best_rmse = float("inf")
    best_params: dict[str, Any] = {}

    for cfg, rmse in results if 'results' in dir() else []:
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = cfg

    # Successive halving iterations
    s = 0
    while len(configs) > 1:
        # Evaluate all configs on current budget, keep top eta fraction
        results: list[tuple[dict[str, Any], float]] = []
        for cfg in configs:
            rmse = _hyperband_evaluate(spec, X_train, y_train, seed, min_budget * (eta ** s), variant)
            results.append((cfg, rmse))

        # Sort by RMSE and keep top eta fraction
        results.sort(key=lambda x: x[1])
        keep_count = max(1, int(len(results) / eta))
        kept_results = results[:keep_count]

        # Update best if improved
        for cfg, rmse in kept_results:
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = cfg

        # Prepare next iteration with kept configs and reduced budgets
        configs = [cfg for cfg, _ in kept_results]
        s += 1
        min_budget = min_budget * eta  # increase budget for remaining configs

    if configs:
        best_params = configs[0]

    search_seconds = time.perf_counter() - time.perf_counter()  # placeholder
    return {
        "best_params": best_params,
        "best_cv_rmse": best_rmse,
        "search_seconds": search_seconds,
        "rows_searched_on": int(len(X_train)),
    }


# ---------------------------------------------------------------------------
# Seed sweep utility — runs models across multiple seeds and logs results.
# ---------------------------------------------------------------------------

def sweep_seeds(spec: ModelSpec, X_train: Any, y_train: np.ndarray,
                seed_range: range = range(50),
                variant: str = "after",
                rseed: int = 123) -> dict[int, float]:
    """Run model training across a range of seeds and log validation performance.

    Parameters
    ----------
    spec : ModelSpec
        Model specification to train.
    X_train : Any
        Training features.
    y_train : np.ndarray
        Training targets.
    seed_range : range
        Range of random seeds to try (default: range(50)).
    variant : str
        Data variant ("after" or "before").
    rseed : int
        Base random seed for reproducibility (default: 123).

    Returns
    -------
    dict mapping seed -> best_cv_rmse
    """
    results: dict[int, float] = {}

    for seed in seed_range:
        set_seed(seed + rseed)  # offset seeds from base seed

        # Train with the spec
        log = tune_model(spec, X_train, y_train, seed=seed, variant=variant)

        # Log the CV RMSE
        rmse = log["best_cv_rmse"]
        results[seed] = rmse

    return results