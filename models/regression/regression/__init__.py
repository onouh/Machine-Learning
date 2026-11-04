"""ADR regression experiments and inference pipeline.

Target: ADR (average daily rate) at booking creation.

.. deprecated::
    Use :mod:`hotel_booking.regression_models` instead. This package is a
    thin backward-compat shim; all new work lives in the enclosing
    ``regression_models`` subdirectory.
"""

from hotel_booking.regression_models.pipeline import (
    DROP_WITH_REASON,
    PREDICTION_POINT,
    VARIANTS,
    BOOKING_TIME_CATEGORICAL,
    BOOKING_TIME_NUMERIC,
    clean_for_regression,
    feature_names,
    build_preprocessor,
)

from hotel_booking.tuning.search import build_full_pipeline

__all__ = [
    "DROP_WITH_REASON",
    "PREDICTION_POINT",
    "VARIANTS",
    "BOOKING_TIME_CATEGORICAL",
    "BOOKING_TIME_NUMERIC",
    "clean_for_regression",
    "build_full_pipeline",
    "feature_names",
    "build_preprocessor",
]

