"""ADR regression experiments and inference pipeline.

Target: ADR (average daily rate) at booking creation.
This package houses all regression-model work: data cleaning, preprocessing,
feature selection, clustering, model training, and evaluation — all gated
through the booking-time-only 'after' variant to avoid information leakage.

Modules:
    random_forest: RandomForest regressor with seed 42
    integration: Integration with experiment suite runner
"""

from hotel_booking.regression_models.random_forest import (
    DEFAULT_SEED,
    ModelSpec,
    RANDOM_FOREST_SPEC,
    get_random_forest_model,
    make_random_forest_regressor,
)

from hotel_booking.regression_models.integration import (
    create_random_forest_pipeline,
    get_random_forest_spec,
    regressors_with_random_forest,
    run_random_forest_experiment,
)

__all__ = [
    # From random_forest
    "DEFAULT_SEED",
    "ModelSpec",
    "RANDOM_FOREST_SPEC",
    "get_random_forest_model",
    "make_random_forest_regressor",
    # From integration
    "create_random_forest_pipeline",
    "get_random_forest_spec",
    "regressors_with_random_forest",
    "run_random_forest_experiment",
]
