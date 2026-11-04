"""Backward-compat shim for :mod:`hotel_booking.regression_models.pipeline`.

All new regression work lives in the enclosing :mod:`regression_models`
subdirectory. This module re-exports the real implementation so that any
remaining callers (tests, scripts, docs) continue to resolve without
changes.
"""

from hotel_booking.regression_models.pipeline import (
    RAW_TARGET,
    PREDICTION_POINT,
    VARIANTS,
    BOOKING_TIME_CATEGORICAL,
    BOOKING_TIME_NUMERIC,
    PRE_TRUNCATION_NUMERIC_EXTRA,
    PRE_TRUNCATION_CATEGORICAL_EXTRA,
    DROP_WITH_REASON,
    CATEGORICAL_FILL,
    LEAKAGE_FORBIDDEN_FEATURES,
    MONTH_MAP,
    _month_to_number,
    _arrival_dates,
    invalid_row_mask,
    clean_for_regression,
    build_preprocessor,
    feature_names,
    load_h2_for_regression,
)

__all__ = [
    "RAW_TARGET",
    "PREDICTION_POINT",
    "VARIANTS",
    "BOOKING_TIME_CATEGORICAL",
    "BOOKING_TIME_NUMERIC",
    "PRE_TRUNCATION_NUMERIC_EXTRA",
    "PRE_TRUNCATION_CATEGORICAL_EXTRA",
    "DROP_WITH_REASON",
    "CATEGORICAL_FILL",
    "LEAKAGE_FORBIDDEN_FEATURES",
    "MONTH_MAP",
    "_month_to_number",
    "_arrival_dates",
    "invalid_row_mask",
    "clean_for_regression",
    "build_preprocessor",
    "feature_names",
    "load_h2_for_regression",
]
