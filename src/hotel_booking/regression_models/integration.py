"""Integration module for adding RandomForest to the regression experiment suite.

This module bridges the dedicated RandomForest model (random_forest.py) with
the experiment suite runner by extending the regressors() function from
models.shared.tuning.search.

Since models.shared.tuning.search depends on hotel_booking.leakage.checker
(which may not be present in all repository states), this module provides
a self-contained way to include RandomForest in experiment runs.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import RandomForestRegressor

from hotel_booking.regression_models.random_forest import (
    DEFAULT_SEED,
    RANDOM_FOREST_SPEC,
    make_random_forest_regressor,
)


def get_random_forest_spec(seed: int = DEFAULT_SEED) -> Any:
    """Get the RandomForest ModelSpec for the experiment suite.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    ModelSpec
        Model specification compatible with the experiment suite runner.
    """
    return RANDOM_FOREST_SPEC


def regressors_with_random_forest(
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Get all regressors including RandomForest.

    This is a convenience function that returns a dict with RandomForest
    added. In the full experiment suite, this would be merged with other
    models from models.shared.tuning.search.regressors().

    Parameters
    ----------
    seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    dict
        Dictionary mapping model keys to ModelSpec instances.
    """
    return {
        "RandomForest": get_random_forest_spec(seed=seed),
    }


def create_random_forest_pipeline(
    variant: str = "after",
    seed: int = DEFAULT_SEED,
) -> Any:
    """Create a RandomForest pipeline for direct use.

    This creates a simple pipeline with preprocessing and RandomForest
    without the full leakage-checking infrastructure.

    Parameters
    ----------
    variant : str
        Data variant ("after" or "before").
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline with RandomForest.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Feature definitions (matching the project)
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

    # Build preprocessor
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Create pipeline
    pipeline = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )),
    ])

    return pipeline


def run_random_forest_experiment(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    *,
    seed: int = DEFAULT_SEED,
    tune: bool = False,
    n_iter: int = 8,
    cv: int = 3,
) -> dict[str, Any]:
    """Run a RandomForest experiment and return results.

    Parameters
    ----------
    X_train : DataFrame or array
        Training features.
    y_train : Series or array
        Training target.
    X_test : DataFrame or array
        Test features.
    y_test : Series or array
        Test target.
    seed : int
        Random seed (default: 42).
    tune : bool
        Whether to tune hyperparameters (default: False).
    n_iter : int
        Number of iterations for tuning (default: 8).
    cv : int
        Number of CV folds (default: 3).

    Returns
    -------
    dict
        Dictionary with results including RMSE, MAE, R², and predictions.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import numpy as np

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

    # Build preprocessor
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
       transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Base pipeline
    base_pipeline = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )),
    ])

    if tune:
        # Hyperparameter search space
        param_distributions = {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [12, 20, None],
            "model__max_features": ["sqrt", 0.3, 0.5],
            "model__min_samples_leaf": [1, 2, 5],
        }

        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            refit=True,
            random_state=seed,
        )

        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        best_params = search.best_params_
        best_cv_rmse = -search.best_score_
    else:
        base_pipeline.fit(X_train, y_train)
        pipeline = base_pipeline
        best_params = {}
        best_cv_rmse = None

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_clipped = np.clip(y_pred, 0, None)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_clipped)))
    mae = float(mean_absolute_error(y_test, y_pred_clipped))
    r2 = float(r2_score(y_test, y_pred_clipped))

    return {
        "model": "RandomForest",
        "seed": seed,
        "tuned": tune,
        "best_params": best_params,
        "best_cv_rmse": best_cv_rmse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": y_pred_clipped,
        "pipeline": pipeline,
    }
