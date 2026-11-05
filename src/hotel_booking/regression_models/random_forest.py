"""RandomForest regressor model definition for ADR regression.

RandomForest is an ensemble of decision trees that averages their predictions,
reducing variance and overfitting compared to a single tree. It handles
non-linear relationships well and is robust to feature scaling.

Target: ADR (average daily rate) at booking creation.
Seed: 42 (fixed for reproducibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# Minimal ModelSpec-compatible dataclass so this module is self-contained and
# does not depend on hotel_booking.tuning.search (which pulls in the leakage
# checker that may not be present in all repository states).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Model specification for the regression experiment suite.

    Mirrors the ModelSpec from models.shared.tuning.search so this module
    can be used standalone.
    """
    key: str
    make_estimator: Callable[[], Any]
    param_distributions: dict[str, Any] | None
    n_iter: int
    cv: int
    subsample: int | None = None
    note: str = ""


def make_random_forest_regressor(seed: int = 42) -> Callable[[], Any]:
    """Create a RandomForestRegressor estimator factory with fixed seed 42.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    Callable
        A callable that returns a configured RandomForestRegressor instance.
    """
    def _make() -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    return _make


# ---------------------------------------------------------------------------
# Model specification for the regression experiment suite.
# ---------------------------------------------------------------------------

RANDOM_FOREST_SPEC = ModelSpec(
    key="RandomForest",
    make_estimator=make_random_forest_regressor(seed=42),
    param_distributions={
        "n_estimators": [100, 200, 400],
        "max_depth": [12, 20, None],
        "max_features": ["sqrt", 0.3, 0.5],
        "min_samples_leaf": [1, 2, 5],
    },
    n_iter=8,
    cv=3,
    subsample=None,
    note="RandomForest regressor with seed 42; budgeted 24 fits",
)

# ---------------------------------------------------------------------------
# Convenient access for direct use outside the experiment suite.
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42


def get_random_forest_model(seed: int = DEFAULT_SEED) -> RandomForestRegressor:
    """Instantiate a RandomForestRegressor with seed 42.

    Parameters
    ----------
    seed : int
        Random seed (default: 42).

    Returns
    -------
    RandomForestRegressor
        Configured RandomForest regressor instance.
    """
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
